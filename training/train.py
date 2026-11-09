"""Train the wall/door/window U-Net, report per-class IoU/F1, export ONNX.

    python -m training.train --data data/cubicasa --epochs 60 --out walls.onnx
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PlanDataset
from .model import CLASSES, Softmaxed, UNet


def dice_loss(logits, target, n):
    p = torch.softmax(logits, 1)
    oh = F.one_hot(target, n).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[1:].mean()  # ignore background


@torch.no_grad()
def evaluate(net, loader, dev):
    n = len(CLASSES)
    tp, fp, fn = [torch.zeros(n) for _ in range(3)]
    net.eval()
    for x, y in loader:
        pred = net(x.to(dev)).argmax(1).cpu()
        for c in range(n):
            tp[c] += ((pred == c) & (y == c)).sum()
            fp[c] += ((pred == c) & (y != c)).sum()
            fn[c] += ((pred != c) & (y == c)).sum()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    iou = tp / (tp + fp + fn).clamp(min=1)
    return {CLASSES[c]: (float(f1[c]), float(iou[c])) for c in range(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None, help="validation root (defaults to --data, NOT recommended)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="walls.onnx")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(dev)
    tr = DataLoader(PlanDataset(a.data, a.crop, True), a.batch, shuffle=True, num_workers=4, drop_last=True)
    va = DataLoader(PlanDataset(a.val or a.data, a.crop, False), 4, num_workers=2)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr))
    # walls/doors/windows are rare pixels: weight them up
    weight = torch.tensor([0.2, 1.0, 3.0, 3.0], device=dev)
    best = -1.0
    for ep in range(a.epochs):
        net.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            out = net(x)
            loss = F.cross_entropy(out, y, weight=weight) + dice_loss(out, y, len(CLASSES))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m = evaluate(net, va, dev)
        score = sum(v[0] for k, v in m.items() if k != "background") / 3
        print(f"epoch {ep}: loss {loss.item():.3f} " + " ".join(f"{k}:F1={v[0]:.3f}" for k, v in m.items()))
        if score > best:
            best = score
            torch.save(net.state_dict(), a.out.replace(".onnx", ".pt"))
    net.load_state_dict(torch.load(a.out.replace(".onnx", ".pt"), map_location=dev))
    export(net, a.out)


def export(net: torch.nn.Module, path: str) -> None:
    net = Softmaxed(net.cpu().eval())
    dummy = torch.zeros(1, 1, 512, 512)
    torch.onnx.export(
        net, dummy, path, input_names=["image"], output_names=["scores"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "scores": {2: "h", 3: "w"}}, opset_version=17,
    )
    print("exported", path, "-> use wall_model_channel=1")


if __name__ == "__main__":
    main()
