"""Dataset of (plan image, label mask) pairs with floor-plan-specific augmentation.

Folder layout:  root/images/*.png   root/labels/*.png  (same stem; pixel value = class id
0 bg, 1 wall, 2 door, 3 window). Convert CubiCasa5k / R2V / your own labels to this layout once
(CubiCasa5k ships SVG annotations; rasterise walls/doors/windows to ids at the image size).
Augmentation is geometric (90-degree rotations, flips, scale jitter) + photometric (blur, noise,
JPEG, contrast) so the net sees scans and screenshots, not only clean CAD exports.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PlanDataset(Dataset):
    def __init__(self, root: str, crop: int = 512, train: bool = True, scale_range=(0.5, 1.5)):
        self.images = sorted(Path(root, "images").glob("*.png"))
        self.labels_dir = Path(root, "labels")
        self.crop, self.train, self.scale_range = crop, train, scale_range

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = cv2.imread(str(self.images[i]), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(str(self.labels_dir / self.images[i].name), cv2.IMREAD_GRAYSCALE)
        if img is None or lab is None:
            raise FileNotFoundError(self.images[i])
        if self.train:
            img, lab = self._augment(img, lab)
        img, lab = self._crop(img, lab)
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        return x, torch.from_numpy(lab.astype(np.int64))

    def _augment(self, img, lab):
        k = random.randint(0, 3)
        img, lab = np.rot90(img, k), np.rot90(lab, k)
        if random.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        s = random.uniform(*self.scale_range)
        img = cv2.resize(np.ascontiguousarray(img), None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        lab = cv2.resize(np.ascontiguousarray(lab), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.5))
        if random.random() < 0.5:
            img += np.random.normal(0, random.uniform(2, 12), img.shape)
        if random.random() < 0.5:
            a, b = random.uniform(0.7, 1.2), random.uniform(-25, 25)
            img = img * a + b
        img = np.clip(img, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)])
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        return img, lab

    def _crop(self, img, lab):
        c = self.crop
        ph, pw = max(0, c - img.shape[0]), max(0, c - img.shape[1])
        img = np.pad(img, ((0, ph), (0, pw)), constant_values=255)
        lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=0)
        y = random.randint(0, img.shape[0] - c) if self.train else 0
        x = random.randint(0, img.shape[1] - c) if self.train else 0
        return np.ascontiguousarray(img[y:y + c, x:x + c]), np.ascontiguousarray(lab[y:y + c, x:x + c])
