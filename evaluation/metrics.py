"""Accept/reject every pipeline change with numbers.

    python -m evaluation.metrics pred_dir gt_dir          # mask PNGs: 255/class-id = wall
Ground truth for each plan: <stem>_walls.png (wall mask), optional <stem>.json with
{"rooms": N, "scale_m_per_px": s, "openings": [[x1,y1,x2,y2], ...]}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def wall_f1(pred: np.ndarray, gt: np.ndarray, tol_px: int = 3) -> dict[str, float]:
    """Boundary-tolerant precision/recall: a pixel counts if the other mask is within tol_px."""
    p, g = pred > 0, gt > 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    g_near = cv2.dilate(g.astype(np.uint8), k) > 0
    p_near = cv2.dilate(p.astype(np.uint8), k) > 0
    prec = float((p & g_near).sum() / max(p.sum(), 1))
    rec = float((g & p_near).sum() / max(g.sum(), 1))
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


def opening_recall(pred: list[tuple[int, int, int, int]], gt: list[list[int]], iou: float = 0.3) -> dict[str, float]:
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union else 0.0

    hit = sum(any(_iou(g, p) >= iou for p in pred) for g in gt)
    matched = sum(any(_iou(g, p) >= iou for g in gt) for p in pred)
    return {"recall": hit / max(len(gt), 1), "precision": matched / max(len(pred), 1)}


def scale_error(pred_mpp: float, gt_mpp: float) -> float:
    return abs(pred_mpp - gt_mpp) / gt_mpp


def main() -> int:
    pred_dir, gt_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for gt_path in sorted(gt_dir.glob("*_walls.png")):
        stem = gt_path.name[: -len("_walls.png")]
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        m = wall_f1(cv2.imread(str(pred_path), 0), cv2.imread(str(gt_path), 0))
        meta = gt_dir / f"{stem}.json"
        rows.append((stem, m, json.loads(meta.read_text()) if meta.exists() else {}))
    if not rows:
        print("no matching plans found")
        return 1
    for stem, m, _ in rows:
        print(f"{stem:30s} wall F1 {m['f1']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}")
    print(f"MEAN wall F1 {np.mean([m['f1'] for _, m, _ in rows]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
