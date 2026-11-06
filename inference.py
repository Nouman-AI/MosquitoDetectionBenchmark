"""Run trained deep-learning or selected classical detectors on an image split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2

from benchmark_utils import (ALL_METHODS, YOLO_MODELS, TV_MODELS, CLASSICAL_MODELS,
                             list_images, image_id, selected_rows, weight_path,
                             load_tv_checkpoint, tv_predict, parameter_file)
from classical import detect_nonai

ROOT = Path(__file__).resolve().parent


def classical_state(method, params, images):
    import numpy as np
    state = {}
    grays = [g for p in images[:300] if (g:=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) is not None]
    if method == "N09_medianbg" and grays:
        state["median_bg"] = np.median(np.stack(grays).astype(np.float32), axis=0).astype(np.uint8)
    if method == "N10_mog2":
        bg = cv2.createBackgroundSubtractorMOG2(history=int(params.get("history",300)),
                                                varThreshold=float(params.get("varThreshold",16)), detectShadows=False)
        for g in grays: bg.apply(g, learningRate=.02)
        state["mog2"] = bg
    if method == "N11_knn":
        bg = cv2.createBackgroundSubtractorKNN(history=int(params.get("history",300)),
                                               dist2Threshold=float(params.get("dist2Threshold",400)), detectShadows=False)
        for g in grays: bg.apply(g, learningRate=.02)
        state["knn"] = bg
    return state


def run(args, method):
    import torch
    images = list_images(args.data, args.split)
    rows = selected_rows(parameter_file("selected_thresholds_epochs_and_parameters_23_methods.csv"))
    selected = rows[method]
    device = args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if method in YOLO_MODELS:
        from ultralytics import YOLO
        model = YOLO(str(weight_path(args.weights, method)))
    elif method in TV_MODELS:
        model = load_tv_checkpoint(method, weight_path(args.weights, method), device)
    else:
        model = None
        params = json.loads(selected["params_json"])
        source_images = list_images(args.data, "train") if (args.data / "train" / "images").exists() else images
        state = classical_state(method, params, source_images)

    preds = {}
    for index, path in enumerate(images, start=1):
        rid = image_id(args.data, args.split, path)
        if method in YOLO_MODELS:
            output = model.predict(source=str(path), conf=args.min_score, imgsz=args.image_size,
                                   max_det=100, verbose=False, device=device)[0]
            dets = [{"bbox": box.tolist(), "score": float(score)}
                    for box, score in zip(output.boxes.xyxy.cpu(), output.boxes.conf.cpu())]
        elif method in TV_MODELS:
            dets = tv_predict(model, path, device, min_score=args.min_score, image_size=args.image_size)
        else:
            img = cv2.imread(str(path))
            dets = detect_nonai(img, CLASSICAL_MODELS[method], params, state) if img is not None else []
        preds[rid] = sorted(dets, key=lambda x: x["score"], reverse=True)[:100]
        if index % 100 == 0:
            print(f"{method}: {index}/{len(images)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{method}_{args.split}.json"
    target.write_text(json.dumps({"method_id": method, "split": args.split,
                                  "predictions": preds}, indent=2), encoding="utf-8")
    print("Saved", target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--weights", type=Path, default=Path("model_weights"))
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--method", default="all", choices=["all"] + list(ALL_METHODS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=1056)
    parser.add_argument("--min-score", type=float, default=.001)
    args = parser.parse_args()
    for method in (list(ALL_METHODS) if args.method == "all" else [args.method]):
        run(args, method)

if __name__ == "__main__":
    main()
