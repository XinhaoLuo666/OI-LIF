"""Gen1 event frames with YOLO labels, using explicit spatial/time layout."""
from pathlib import Path

import cv2
import math
import numpy as np
import torch
from torch.utils.data import Dataset


class Gen1(Dataset):
    def __init__(self, root, split="train", image_size=320, layout="whc",
                 reverse_time=True, time_steps=4, limit=0, num_classes=2, rectangular=False, allow_missing_labels=False,
                 return_geometry=False, files=None):
        root = Path(root) / split
        self.files = ([Path(p) for p in files] if files is not None else
                      sorted(root.glob("*.npy")) + sorted((root / "images").glob("*.npy")))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No Gen1 frames in {root}")
        self.image_size, self.layout = image_size, layout
        self.reverse_time, self.time_steps = reverse_time, time_steps
        self.num_classes = num_classes
        self.rectangular, self.allow_missing_labels = rectangular, allow_missing_labels
        self.return_geometry = return_geometry

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        frames = np.load(path, allow_pickle=False)
        if frames.ndim != 4 or frames.shape[0] != self.time_steps or frames.shape[-1] != 3:
            raise ValueError(f"Expected [{self.time_steps},H,W,3] / WHC frames: {path}: {frames.shape}")
        if self.layout == "whc":
            frames = frames.transpose(0, 2, 1, 3)
        h, w = frames.shape[1:3]
        size = self.image_size
        ratio = min(size / h, size / w)
        # The archived reader first resizes the long side, rounding up the
        # short side, then letterboxes without a second resize.
        new_h, new_w = min(math.ceil(h * ratio), size), min(math.ceil(w * ratio), size)
        canvas_h = math.ceil(new_h/32+.5)*32 if self.rectangular else size
        canvas_w = math.ceil(new_w/32+.5)*32 if self.rectangular else size
        top, left = (canvas_h - new_h) // 2, (canvas_w - new_w) // 2
        images = np.full((self.time_steps, canvas_h, canvas_w, 3), 114, dtype=np.uint8)
        for t, frame in enumerate(frames):
            images[t, top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
        label_path = path.with_suffix(".txt")
        if not label_path.exists():
            label_path = Path(str(label_path).replace("/images/", "/labels/"))
        if not label_path.exists() and not self.allow_missing_labels:
            raise FileNotFoundError(f"Missing YOLO labels for {path}")
        labels = (np.loadtxt(label_path, dtype=np.float32, ndmin=2)
                  if label_path.exists() and label_path.stat().st_size else np.zeros((0, 5), dtype=np.float32))
        if labels.shape[1] != 5 or (labels < 0).any() or (labels[:, 1:] > 1).any():
            raise ValueError(f"Invalid normalized YOLO labels: {label_path}")
        if len(labels) and not ((labels[:, 0] >= 0) & (labels[:, 0] < self.num_classes) & (labels[:, 0] == np.floor(labels[:, 0]))).all():
            raise ValueError(f"Invalid zero-based class IDs: {label_path}")
        # Box coordinates always refer to the physical HWC sensor geometry.
        pad_x, pad_y = (canvas_w - new_w) / 2, (canvas_h - new_h) / 2
        labels[:, 1] = (labels[:, 1] * new_w + pad_x) / canvas_w
        labels[:, 2] = (labels[:, 2] * new_h + pad_y) / canvas_h
        labels[:, 3] *= new_w / canvas_w
        labels[:, 4] *= new_h / canvas_h
        if self.reverse_time:
            images = images[::-1]
        image = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).float() / 255
        labels = torch.from_numpy(labels)
        if self.return_geometry:
            # Legacy scale_boxes uses the resized height ratio for both axes.
            geometry = torch.tensor([h, w, new_h / h, left, top], dtype=torch.float32)
            return image, labels, geometry
        return image, labels


def collate(samples):
    images, labels = zip(*(sample[:2] for sample in samples))
    batch = dict(img=torch.stack(images), cls=torch.cat([y[:, :1] for y in labels]),
                bboxes=torch.cat([y[:, 1:] for y in labels]),
                batch_idx=torch.cat([torch.full((len(y),), i) for i, y in enumerate(labels)]))
    if len(samples[0]) == 3:
        batch['geometry'] = torch.stack([sample[2] for sample in samples])
    return batch
