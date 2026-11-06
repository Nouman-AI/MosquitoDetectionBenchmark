"""Train a YOLO or TorchVision detector from Pascal VOC annotations."""
from __future__ import annotations
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np

from benchmark_utils import YOLO_MODELS, TV_MODELS, build_torchvision_model, ground_truth, image_tensor, parameter_file

ROOT = Path(__file__).resolve().parent


def to_yolo_dataset(data: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        gt = ground_truth(data, split)
        for rid, record in gt.items():
            target_img = out / "images" / split / (rid + record["image"].suffix.lower())
            target_lbl = out / "labels" / split / (rid + ".txt")
            target_img.parent.mkdir(parents=True, exist_ok=True)
            target_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["image"], target_img)
            lines = []
            for x1, y1, x2, y2 in record["boxes"]:
                w, h = record["width"], record["height"]
                lines.append(f"0 {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
            target_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
    import yaml
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": {0: "mosquito"}}
    yaml_file = out / "dataset.yaml"
    yaml_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return yaml_file


def train_yolo(args):
    from ultralytics import YOLO
    rows = list(csv.DictReader(parameter_file("YOLO_training_settings.csv").open(newline="", encoding="utf-8-sig")))
    cfg = next(r for r in rows if r["method_id"] == args.method)
    yaml_file = to_yolo_dataset(args.data, args.output / "_yolo_dataset")
    model = YOLO(cfg["model"])
    import torch
    device = (0 if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model.train(data=str(yaml_file), epochs=args.epochs or int(cfg["epochs"]),
                imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
                patience=int(cfg["patience"]), seed=args.seed,
                device=device, project=str(args.output.resolve()), name=args.method,
                exist_ok=True)


def training_grid(method):
    grid_folder = ROOT / "hyperparameters"
    for file in grid_folder.rglob("*.json"):
        record = json.loads(file.read_text())
        for cfg in record.get("grid", []):
            if cfg.get("method_id") == method:
                return cfg
    raise ValueError(f"No TorchVision configuration found for {method}")


class VocDataset:
    def __init__(self, data, split, size):
        self.records = list(ground_truth(data, split).values())
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import torch
        rec = self.records[idx]
        image, original = image_tensor(rec["image"], self.size)
        sx, sy = self.size / original[0], self.size / original[1]
        boxes = [[x1*sx,y1*sy,x2*sx,y2*sy] for x1,y1,x2,y2 in rec["boxes"]]
        if not boxes:
            boxes = [[0.,0.,2.,2.]]
        b = torch.tensor(boxes, dtype=torch.float32)
        target = {"boxes": b, "labels": torch.ones(len(b), dtype=torch.long),
                  "image_id": torch.tensor([idx]),
                  "area": (b[:,2]-b[:,0])*(b[:,3]-b[:,1]),
                  "iscrowd": torch.zeros(len(b), dtype=torch.long)}
        # Rectangular target masks are derived from bounding-box annotations.
        masks = torch.zeros((len(b), self.size, self.size), dtype=torch.uint8)
        for i, (x1,y1,x2,y2) in enumerate(b.tolist()):
            a,c,e,f = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            if e>a and f>c:
                masks[i, max(0,c):min(self.size,f+1), max(0,a):min(self.size,e+1)] = 1
        target["masks"] = masks
        return image, target


def train_tv(args):
    import torch
    from torch.utils.data import DataLoader
    from benchmark_utils import tv_predict
    from evaluate import ranked_ap
    cfg = training_grid(args.method)
    size = int(cfg["image_size"])
    batch = int(cfg["batch_size"])
    train_data = VocDataset(args.data, "train", size)
    validation = ground_truth(args.data, "val")
    loader = DataLoader(train_data, batch_size=batch, shuffle=True, drop_last=batch>1,
                        collate_fn=lambda x: (list(t[0] for t in x), list(t[1] for t in x)), num_workers=0)
    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model = build_torchvision_model(args.method, pretrained=True).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    result_dir = args.output / args.method
    result_dir.mkdir(parents=True, exist_ok=True)
    best, stale = -1., 0
    for epoch in range(1, (args.epochs or int(cfg["epochs"]))+1):
        model.train()
        total, count = 0., 0
        for imgs, targets in loader:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            losses = model(imgs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total += float(loss.item())
            count += 1
        model.eval()
        predictions = {rid: tv_predict(model, rec["image"], device, min_score=.001,
                                        image_size=size, cap=100) for rid, rec in validation.items()}
        ap = ranked_ap(validation, predictions)["mAP50_95"]
        print(f"{args.method} epoch={epoch} loss={total/max(count,1):.5f} val_mAP50_95={ap:.5f}", flush=True)
        if ap > best:
            best, stale = ap, 0
            torch.save({"model": model.state_dict(), "model_name": TV_MODELS[args.method],
                        "epoch": epoch, "val_mAP50_95": best}, result_dir / "best.pt")
        else:
            stale += 1
        if stale >= int(cfg["patience"]):
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="VOC dataset root with train/ and val/")
    parser.add_argument("--method", required=True, choices=list(YOLO_MODELS)+list(TV_MODELS))
    parser.add_argument("--output", type=Path, default=Path("training_runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (train_yolo if args.method in YOLO_MODELS else train_tv)(args)


if __name__ == "__main__":
    main()
