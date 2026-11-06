"""Shared helpers for the single-class mosquito detection benchmark."""
from __future__ import annotations
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

YOLO_MODELS = {
    "Y01_yolo11n": "yolo11n.pt",
    "Y02_yolo11s": "yolo11s.pt",
    "Y03_yolov8n": "yolov8n.pt",
    "Y04_yolov8s": "yolov8s.pt",
}
TV_MODELS = {
    "T01_frcnn_r50": "fasterrcnn_resnet50_fpn_v2",
    "T02_frcnn_mob": "fasterrcnn_mobilenet_v3_large_fpn",
    "T03_mask": "maskrcnn_resnet50_fpn_v2",
    "T04_retina": "retinanet_resnet50_fpn_v2",
    "T06_ssd300": "ssd300_vgg16",
    "T07_ssdlite": "ssdlite320_mobilenet_v3_large",
}
CLASSICAL_MODELS = {
    "N01_dog": "dog_patch_box", "N02_log": "log_patch_box",
    "N03_mser": "mser_dark", "N04_otsu": "otsu_dark_cc",
    "N05_adapt": "adaptive_gaussian_cc", "N06_blackhat": "blackhat_cc",
    "N07_localmin": "local_minima_dark", "N08_canny": "canny_contour",
    "N09_medianbg": "median_bg_diff", "N10_mog2": "mog2_bgsub",
    "N11_knn": "knn_bgsub", "N12_framediff": "frame_diff",
    "N13_blob": "simpleblob_dark",
}
ALL_METHODS = {**YOLO_MODELS, **TV_MODELS, **CLASSICAL_MODELS}


def list_images(data: Path, split: str):
    folder = data / split / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported)


def image_id(data: Path, split: str, path: Path):
    return path.relative_to(data / split / "images").with_suffix("").as_posix()


def annotation_path(data: Path, split: str, img: Path):
    subpath = img.relative_to(data / split / "images").with_suffix(".xml")
    for dirname in ("annotations", "labels"):
        p = data / split / dirname / subpath
        if p.is_file():
            return p
    raise FileNotFoundError(f"Pascal VOC XML missing for {img.name}")


def ground_truth(data: Path, split: str):
    records = {}
    for img in list_images(data, split):
        rid = image_id(data, split, img)
        root = ET.parse(annotation_path(data, split, img)).getroot()
        with Image.open(img) as im:
            width, height = im.size
        boxes = []
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            if b is None:
                continue
            vals = [float(b.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
            x1, y1, x2, y2 = vals
            vals = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
                    max(0.0, min(width, x2)), max(0.0, min(height, y2))]
            if vals[2] > vals[0] and vals[3] > vals[1]:
                boxes.append(vals)
        records[rid] = {"image": img, "boxes": boxes, "width": width, "height": height}
    return records


def selected_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["method_id"]: r for r in csv.DictReader(f)}


def parameter_file(filename: str):
    base = Path(__file__).resolve().parent / "hyperparameters"
    preferred = [base / "selected" / filename,
                 base / "selected_parameters" / filename,
                 base / "03_HYPERPARAMETERS_AND_PRIORS" / filename]
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    matches = sorted(base.rglob(filename)) if base.exists() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Selected parameter file not found: {filename}")
    raise ValueError(f"Multiple parameter files named {filename}: {matches}")


def weight_path(folder: Path, method: str):
    for candidate in [folder / f"{method}.pt", folder / f"{method}.pth",
                      folder / method / "best.pt", folder / method / "best.pth",
                      folder / f"{method}_best.pt", folder / f"{method}_best.pth"]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint for {method} not found in {folder}. See model_weights/README.md")


def build_torchvision_model(method: str, pretrained: bool = False):
    from torchvision.models import detection as md
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    name = TV_MODELS[method]
    if name in ("fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn", "maskrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2"):
        weight_enum = {
            "fasterrcnn_resnet50_fpn_v2": md.FasterRCNN_ResNet50_FPN_V2_Weights,
            "fasterrcnn_mobilenet_v3_large_fpn": md.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            "maskrcnn_resnet50_fpn_v2": md.MaskRCNN_ResNet50_FPN_V2_Weights,
            "retinanet_resnet50_fpn_v2": md.RetinaNet_ResNet50_FPN_V2_Weights,
        }[name]
        model = getattr(md, name)(weights=weight_enum.DEFAULT if pretrained else None,
                                  weights_backbone=None)
        if name.startswith("fasterrcnn") or name.startswith("maskrcnn"):
            n = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(n, 2)
            if name.startswith("maskrcnn"):
                n_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                model.roi_heads.mask_predictor = MaskRCNNPredictor(n_mask, 256, 2)
        else:
            old = model.head.classification_head
            model.head.classification_head = RetinaNetClassificationHead(
                old.cls_logits.in_channels, old.num_anchors, 2)
        return model
    return getattr(md, name)(weights=None, weights_backbone="DEFAULT" if pretrained else None, num_classes=2)


def load_tv_checkpoint(method: str, checkpoint_path: Path, device: str):
    import torch
    model = build_torchvision_model(method, pretrained=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def image_tensor(path: Path, image_size=1056):
    import torch
    with Image.open(path) as img:
        img = img.convert("RGB")
        orig = img.size
        if image_size:
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        a = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255.0
    return torch.from_numpy(a), orig


def tv_predict(model, img: Path, device: str, min_score=0.001, image_size=1056, cap=100):
    import torch
    ten, orig = image_tensor(img, image_size)
    with torch.inference_mode():
        out = model([ten.to(device)])[0]
    sx, sy = orig[0] / image_size, orig[1] / image_size
    dets = []
    for box, score, label in zip(out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
        if int(label) != 1 or score < min_score:
            continue
        x1,y1,x2,y2 = box
        dets.append({"bbox": [x1*sx,y1*sy,x2*sx,y2*sy], "score": float(score)})
    return sorted(dets, key=lambda x: x["score"], reverse=True)[:cap]
