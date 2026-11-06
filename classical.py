"""Classical detector reference functions derived from the recovered Paper 1 source."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

MAX_DETECTIONS_PER_IMAGE = 100
CENTER_NMS_PX = 8.0

def nms_center(dets, keep=100, center_px=8.0):
    result=[]
    for item in sorted(dets,key=lambda x:float(x.get("score",0)),reverse=True):
        a=item["bbox"]
        cx,cy=(a[0]+a[2])/2,(a[1]+a[3])/2
        if all(math.hypot(cx-(b["bbox"][0]+b["bbox"][2])/2,cy-(b["bbox"][1]+b["bbox"][3])/2)>center_px for b in result):
            result.append(item)
        if len(result)>=keep:
            break
    return result

def norm_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - mn) * 255.0 / (mx - mn), 0, 255).astype(np.uint8)

def local_dark_response(gray: np.ndarray, block: int = 31) -> np.ndarray:
    block = int(block); block += (block % 2 == 0)
    local = cv2.GaussianBlur(gray, (block, block), 0)
    return cv2.subtract(local, gray)

def blackhat_response(gray: np.ndarray, kernel: int = 21) -> np.ndarray:
    kernel = int(kernel); kernel += (kernel % 2 == 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)

def morph_clean(mask: np.ndarray, open_k: int = 0, close_k: int = 0, dilate_iter: int = 0) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if open_k and open_k > 1:
        open_k = int(open_k); open_k += (open_k % 2 == 0)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k and close_k > 1:
        close_k = int(close_k); close_k += (close_k % 2 == 0)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if dilate_iter and dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k, iterations=int(dilate_iter))
    return mask

def patch_box_around_peak(gray: np.ndarray, cx: float, cy: float, response: np.ndarray, params: Dict[str, Any]) -> List[float]:
    """Old-code idea: centroid from response peak, but box from local dark patch around it."""
    h, w = gray.shape[:2]
    rad = int(params.get("patch_radius", 18))
    x1 = max(0, int(round(cx)) - rad); x2 = min(w - 1, int(round(cx)) + rad)
    y1 = max(0, int(round(cy)) - rad); y2 = min(h - 1, int(round(cy)) + rad)
    roi = gray[y1:y2 + 1, x1:x2 + 1]
    resp_roi = response[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        s = float(params.get("fallback_box", 32))
        return [max(0, cx - s/2), max(0, cy - s/2), min(w - 1, cx + s/2), min(h - 1, cy + s/2)]
    # threshold local dark response near the peak; this gives image-derived box size.
    thr = max(float(params.get("patch_min_response", 2)), float(np.percentile(resp_roi, params.get("patch_percentile", 75))))
    mask = (resp_roi >= thr).astype(np.uint8)
    # Keep component that contains/near peak.
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    px = int(round(cx)) - x1; py = int(round(cy)) - y1
    best = None
    if 0 <= px < labels.shape[1] and 0 <= py < labels.shape[0]:
        lab = int(labels[py, px])
        if lab > 0:
            best = lab
    if best is None:
        # choose component closest to peak
        best_d = 1e9
        for i in range(1, n):
            cxx, cyy = cents[i]
            d = math.hypot(float(cxx - px), float(cyy - py))
            if d < best_d:
                best_d = d; best = i
    if best is None or best <= 0:
        s = float(params.get("fallback_box", 32))
        return [max(0, cx - s/2), max(0, cy - s/2), min(w - 1, cx + s/2), min(h - 1, cy + s/2)]
    xx = stats[best, cv2.CC_STAT_LEFT]; yy = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]; bh = stats[best, cv2.CC_STAT_HEIGHT]
    expand = float(params.get("patch_expand", 8))
    bx1 = max(0.0, x1 + xx - expand); by1 = max(0.0, y1 + yy - expand)
    bx2 = min(float(w - 1), x1 + xx + bw + expand); by2 = min(float(h - 1), y1 + yy + bh + expand)
    min_size = float(params.get("min_box_size", 12))
    if (bx2 - bx1) < min_size:
        mid = (bx1 + bx2) / 2; bx1 = max(0.0, mid - min_size/2); bx2 = min(float(w - 1), mid + min_size/2)
    if (by2 - by1) < min_size:
        mid = (by1 + by2) / 2; by1 = max(0.0, mid - min_size/2); by2 = min(float(h - 1), mid + min_size/2)
    return [float(bx1), float(by1), float(bx2), float(by2)]

def fixed_box_around_center(cx: float, cy: float, img_w: int, img_h: int, params: Dict[str, Any]) -> List[float]:
    """Construct a bounding box around a detected center using the recorded size prior."""
    bw = float(params.get("prior_box_w", params.get("fallback_box", 32.0)))
    bh = float(params.get("prior_box_h", params.get("fallback_box", 32.0)))
    scale = float(params.get("prior_box_scale", params.get("box_scale", 1.0)))
    bw *= scale
    bh *= scale
    min_size = float(params.get("min_box_size", 8.0))
    bw = max(bw, min_size)
    bh = max(bh, min_size)
    x1 = max(0.0, float(cx) - bw / 2.0)
    y1 = max(0.0, float(cy) - bh / 2.0)
    x2 = min(float(img_w - 1), float(cx) + bw / 2.0)
    y2 = min(float(img_h - 1), float(cy) + bh / 2.0)
    return [float(x1), float(y1), float(x2), float(y2)]

def cc_dets(mask: np.ndarray, response: np.ndarray, params: Dict[str, Any], gray: Optional[np.ndarray] = None, patch_box: bool = False) -> List[Dict[str, Any]]:
    h, w = mask.shape[:2]
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    dets = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        min_area = float(params.get("min_area", 3)); max_area = float(params.get("max_area", 1200))
        if area < min_area or area > max_area:
            continue
        x = float(stats[i, cv2.CC_STAT_LEFT]); y = float(stats[i, cv2.CC_STAT_TOP])
        bw = float(stats[i, cv2.CC_STAT_WIDTH]); bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw / max(bh, 1.0), bh / max(bw, 1.0))
        if aspect > float(params.get("max_aspect", 6.0)):
            continue
        extent = area / max(bw * bh, 1.0)
        if extent < float(params.get("min_extent", 0.0)) or extent > float(params.get("max_extent", 1.0)):
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        roi = response[int(y):int(min(h, y + bh + 1)), int(x):int(min(w, x + bw + 1))]
        score = float(np.percentile(roi, 95)) if roi.size else area
        if score < float(params.get("min_response", -1e9)):
            continue
        box_mode = str(params.get("box_mode", "patch" if patch_box else "component"))
        if box_mode == "train_prior":
            box = fixed_box_around_center(cx, cy, w, h, params)
        elif patch_box and gray is not None:
            box = patch_box_around_peak(gray, cx, cy, response, params)
        else:
            expand = float(params.get("expand", 8))
            box = [max(0.0, x - expand), max(0.0, y - expand), min(float(w - 1), x + bw + expand), min(float(h - 1), y + bh + expand)]
        dets.append({"bbox": box, "score": score, "label": 1})
    return nms_center(dets, keep=int(params.get("keep", MAX_DETECTIONS_PER_IMAGE)), center_px=float(params.get("nms_center", CENTER_NMS_PX)))

def detect_nonai(img_bgr: np.ndarray, method: str, params: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]

    if method == "dog_patch_box":
        s1 = float(params.get("sigma_small", 0.7)); s2 = float(params.get("sigma_large", 3.0))
        thr = float(params.get("threshold", 3.0))
        g1 = cv2.GaussianBlur(blur, (0, 0), s1).astype(np.float32)
        g2 = cv2.GaussianBlur(blur, (0, 0), s2).astype(np.float32)
        resp = np.maximum(g2 - g1, 0)
        mask = (resp >= thr).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=True)

    if method == "log_patch_box":
        sigma = float(params.get("sigma", 1.4)); thr = float(params.get("threshold", 4.0))
        sm = cv2.GaussianBlur(blur, (0, 0), sigma)
        lap = cv2.Laplacian(sm, cv2.CV_32F, ksize=3)
        resp = np.maximum(lap, 0)
        mask = (resp >= thr).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=True)

    if method == "blackhat_cc":
        resp = blackhat_response(gray, int(params.get("kernel", 21)))
        mask = (resp >= float(params.get("threshold", 8))).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=bool(params.get("patch_box", True)))

    if method == "adaptive_gaussian_cc":
        block = int(params.get("block", 31)); block += (block % 2 == 0)
        C = float(params.get("C", 4)); min_contrast = float(params.get("min_contrast", 4))
        resp = local_dark_response(gray, block)
        inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)
        mask = ((inv > 0) & (resp >= min_contrast)).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=bool(params.get("patch_box", True)))

    if method == "otsu_dark_cc":
        mode = str(params.get("mode", "local_response"))
        if mode == "blackhat":
            resp = blackhat_response(gray, int(params.get("kernel", 21)))
        else:
            resp = local_dark_response(gray, int(params.get("block", 31)))
        resp_u = norm_u8(resp)
        _, th = cv2.threshold(resp_u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = ((th > 0) & (resp >= float(params.get("min_contrast", 2)))).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=bool(params.get("patch_box", True)))

    if method == "local_minima_dark":
        k = int(params.get("kernel", 15)); k += (k % 2 == 0)
        min_dark = float(params.get("min_dark_response", 5))
        eroded = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        resp = cv2.subtract(cv2.GaussianBlur(gray, (k, k), 0), gray)
        mask = ((gray == eroded) & (resp >= min_dark)).astype(np.uint8)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=int(params.get("dilate_iter", 1)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=True)

    if method == "mser_dark":
        delta = int(params.get("delta", 5)); min_area = int(params.get("min_area", 4)); max_area = int(params.get("max_area", 900))
        try:
            mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)
        except TypeError:
            mser = cv2.MSER_create()
            mser.setDelta(delta); mser.setMinArea(min_area); mser.setMaxArea(max_area)
        regions, boxes = mser.detectRegions(255 - gray)
        dets = []
        resp = local_dark_response(gray, 31)
        for x, y, bw, bh in boxes:
            if bw * bh < min_area or bw * bh > max_area:
                continue
            cx, cy = float(x) + float(bw) / 2.0, float(y) + float(bh) / 2.0
            if str(params.get("box_mode", "component")) == "train_prior":
                box = fixed_box_around_center(cx, cy, w, h, params)
            else:
                expand = float(params.get("expand", 4))
                box = [max(0.0, x-expand), max(0.0, y-expand), min(float(w-1), x+bw+expand), min(float(h-1), y+bh+expand)]
            dets.append({"bbox": box, "score": float(bw*bh), "label": 1})
        return nms_center(dets, keep=int(params.get("keep", MAX_DETECTIONS_PER_IMAGE)), center_px=float(params.get("nms_center", CENTER_NMS_PX)))

    if method == "simpleblob_dark":
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = True; p.blobColor = 0
        p.filterByArea = True; p.minArea = float(params.get("min_area", 4)); p.maxArea = float(params.get("max_area", 900))
        p.filterByCircularity = False; p.filterByConvexity = False; p.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(p)
        kps = detector.detect(gray)
        dets = []
        box_size = float(params.get("box_size", 32))
        for kp in kps:
            cx, cy = kp.pt
            if str(params.get("box_mode", "fixed")) == "train_prior":
                box = fixed_box_around_center(cx, cy, w, h, params)
            else:
                box = [max(0.0, cx-box_size/2), max(0.0, cy-box_size/2), min(float(w-1), cx+box_size/2), min(float(h-1), cy+box_size/2)]
            dets.append({"bbox": box, "score": float(kp.response if kp.response else kp.size), "label": 1})
        return nms_center(dets, keep=int(params.get("keep", MAX_DETECTIONS_PER_IMAGE)), center_px=float(params.get("nms_center", CENTER_NMS_PX)))

    if method == "canny_contour":
        lo = int(params.get("low", 30)); hi = int(params.get("high", 90))
        edges = cv2.Canny(gray, lo, hi)
        mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        return cc_dets(mask, mask.astype(np.float32), params, gray=gray, patch_box=bool(params.get("patch_box", True)))

    if method == "median_bg_diff":
        bg = state.get("median_bg")
        if bg is None:
            return []
        resp = cv2.subtract(bg, gray)
        mask = (resp >= float(params.get("threshold", 10))).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp, params, gray=gray, patch_box=True)

    if method == "mog2_bgsub":
        bgsub = state.setdefault("mog2", cv2.createBackgroundSubtractorMOG2(history=int(params.get("history", 300)), varThreshold=float(params.get("varThreshold", 16)), detectShadows=False))
        fg = bgsub.apply(gray, learningRate=float(params.get("learningRate", 0.001)))
        mask = (fg == 255).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 3)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, fg.astype(np.float32), params, gray=gray, patch_box=True)

    if method == "knn_bgsub":
        bgsub = state.setdefault("knn", cv2.createBackgroundSubtractorKNN(history=int(params.get("history", 300)), dist2Threshold=float(params.get("dist2Threshold", 400)), detectShadows=False))
        fg = bgsub.apply(gray, learningRate=float(params.get("learningRate", 0.001)))
        mask = (fg == 255).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 3)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, fg.astype(np.float32), params, gray=gray, patch_box=True)

    if method == "frame_diff":
        prev = state.get("prev_gray")
        state["prev_gray"] = gray.copy()
        if prev is None:
            return []
        resp = cv2.absdiff(prev, gray)
        mask = (resp >= float(params.get("threshold", 8))).astype(np.uint8)
        mask = morph_clean(mask, int(params.get("open_k", 0)), int(params.get("close_k", 0)), int(params.get("dilate_iter", 0)))
        return cc_dets(mask, resp.astype(np.float32), params, gray=gray, patch_box=True)

    raise ValueError(method)
