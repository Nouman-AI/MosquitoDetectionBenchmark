"""Evaluate ranked detections, a fixed operating point, and known-five localization."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from benchmark_utils import ALL_METHODS, ground_truth, selected_rows, parameter_file

ROOT = Path(__file__).resolve().parent
IOU_THRESHOLDS = np.arange(.50, .951, .05)
CENTROID_RADII = tuple(range(2, 21, 2))


def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.,x2-x1) * max(0.,y2-y1)
    a = max(0.,box1[2]-box1[0])*max(0.,box1[3]-box1[1])
    b = max(0.,box2[2]-box2[0])*max(0.,box2[3]-box2[1])
    return inter / max(a+b-inter, 1.e-10)


def ap_at_iou(gt, preds, threshold):
    used = {rid:set() for rid in gt}
    total = sum(len(rec["boxes"]) for rec in gt.values())
    if not total: return 0.
    scored = [(float(d.get("score",0.)), rid, d["bbox"])
              for rid in gt for d in preds.get(rid, [])]
    scored.sort(key=lambda x:x[0], reverse=True)
    tp, fp = [], []
    for _,rid,box in scored:
        match = [(iou(box, b), j) for j,b in enumerate(gt[rid]["boxes"]) if j not in used[rid]]
        score,j = max(match,default=(0.,-1))
        accepted = j>=0 and score>=threshold
        if accepted: used[rid].add(j)
        tp.append(int(accepted)); fp.append(int(not accepted))
    if not tp: return 0.
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    return float(np.mean([precision[recall>=level].max() if np.any(recall>=level) else 0.
                          for level in np.linspace(0,1,101)]))


def ranked_ap(gt, preds):
    aps = [ap_at_iou(gt, preds, float(t)) for t in IOU_THRESHOLDS]
    result = {"AP50":aps[0], "mAP50_95":float(np.mean(aps))}
    result.update({f"AP{int(round(t*100))}":val for t,val in zip(IOU_THRESHOLDS,aps)})
    return result


def counts_iou(gt, preds, threshold=.5):
    total_tp=total_fp=total_fn=0
    matched=[]
    for rid, record in gt.items():
        boxes=record["boxes"]
        used=set()
        dets=sorted(preds.get(rid, []), key=lambda x:x.get("score",0), reverse=True)
        for detection in dets:
            candidates=[(iou(detection["bbox"],b),j) for j,b in enumerate(boxes) if j not in used]
            score,j=max(candidates,default=(0.,-1))
            if j>=0 and score>=threshold:
                used.add(j); total_tp+=1; matched.append(score)
            else:
                total_fp+=1
        total_fn+=len(boxes)-len(used)
    return metric_summary(total_tp,total_fp,total_fn), matched


def metric_summary(tp,fp,fn):
    precision=tp/max(tp+fp,1)
    recall=tp/max(tp+fn,1)
    f1=2*precision*recall/max(precision+recall,1.e-12)
    return {"TP":tp,"FP":fp,"FN":fn,"precision":precision,"recall":recall,"F1":f1}


def centroid_metrics(gt,preds,radii=CENTROID_RADII):
    all_results={}
    for radius in radii:
        tp=fp=fn=0; distances=[]
        for rid, record in gt.items():
            boxes=record["boxes"]
            detections=preds.get(rid,[])
            gt_centers=[((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
            pred_centers=[((d["bbox"][0]+d["bbox"][2])/2,
                           (d["bbox"][1]+d["bbox"][3])/2) for d in detections]
            pairs=sorted((math.dist(a,b),i,j) for i,a in enumerate(gt_centers)
                        for j,b in enumerate(pred_centers))
            used_g=set(); used_p=set()
            for d,i,j in pairs:
                if d>radius: break
                if i in used_g or j in used_p: continue
                used_g.add(i); used_p.add(j); distances.append(d)
            tp+=len(used_g); fp+=len(pred_centers)-len(used_p); fn+=len(gt_centers)-len(used_g)
        all_results[f"centroid_{radius}px"]={**metric_summary(tp,fp,fn),
                                               "mean_error_px":float(np.mean(distances)) if distances else None}
    return all_results


def apply_threshold(preds, threshold):
    return {rid: [d for d in dets if float(d.get("score",0)) >= threshold]
            for rid,dets in preds.items()}


def known_five(preds):
    return {rid: sorted(dets,key=lambda x:x.get("score",0),reverse=True)[:5] for rid,dets in preds.items()}


def save_csv(path, rows):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--split",default="test")
    p.add_argument("--predictions",type=Path,default=Path("predictions"))
    p.add_argument("--output",type=Path,default=Path("evaluation_results"))
    p.add_argument("--method",default="all",choices=["all"]+list(ALL_METHODS))
    args=p.parse_args()
    gt=ground_truth(args.data,args.split)
    rows=selected_rows(parameter_file("selected_thresholds_epochs_and_parameters_23_methods.csv"))
    args.output.mkdir(parents=True,exist_ok=True)
    summary=[]; ap_rows=[]; centroid_rows=[]
    methods=list(ALL_METHODS) if args.method=="all" else [args.method]
    for method in methods:
        filepath=args.predictions/f"{method}_{args.split}.json"
        if not filepath.is_file():
            print("Missing predictions; skipping:",filepath)
            continue
        raw=json.loads(filepath.read_text(encoding="utf-8"))["predictions"]
        predictions={rid: sorted(raw.get(rid,[]),key=lambda x:x.get("score",0),reverse=True)[:100]
                     for rid in gt}
        ap=ranked_ap(gt,predictions)
        threshold_str=(rows.get(method) or {}).get("conf","")
        threshold=float(threshold_str) if threshold_str not in ("",None) else None
        op_preds=apply_threshold(predictions,threshold) if threshold is not None else predictions
        unknown,_=counts_iou(gt,op_preds)
        fixed,_=counts_iou(gt,known_five(op_preds))
        cent=centroid_metrics(gt,op_preds)
        summary.append({"method_id":method,"images":len(gt),"annotations":sum(len(r["boxes"]) for r in gt.values()),
                        "AP50":round(ap["AP50"],6),"mAP50_95":round(ap["mAP50_95"],6),
                        "operating_threshold":threshold if threshold is not None else "not separately provided",
                        "unknown_P":round(unknown["precision"],6),"unknown_R":round(unknown["recall"],6),
                        "unknown_F1":round(unknown["F1"],6),"known_five_F1":round(fixed["F1"],6),
                        "centroid_F1_20px":round(cent["centroid_20px"]["F1"],6),
                        "centroid_mean_error_20px":cent["centroid_20px"]["mean_error_px"]})
        for t in IOU_THRESHOLDS:
            ap_rows.append({"method_id":method,"IoU":f"{t:.2f}","AP":round(ap[f"AP{int(round(t*100))}"],6)})
        for label,metrics in cent.items():
            centroid_rows.append({"method_id":method,"radius_px":int(label.split("_")[1][:-2]),**metrics})
        print(method,"AP50",round(ap["AP50"],4),"mAP50_95",round(ap["mAP50_95"],4))
    save_csv(args.output/"summary.csv",summary)
    save_csv(args.output/"ap_by_iou.csv",ap_rows)
    save_csv(args.output/"centroid_metrics.csv",centroid_rows)
    print("Saved results to",args.output)

if __name__=="__main__":
    main()
