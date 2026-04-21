import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def load_image(path):
    return Image.open(path).convert("RGB")


def to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def ensure_min_side(img, min_size):
    if min_size <= 0:
        return img
    w, h = img.size
    if min(w, h) >= min_size:
        return img
    scale = float(min_size) / float(min(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def random_crop_params(w, h, crop_size):
    if crop_size <= 0:
        return 0, 0, w, h
    if w == crop_size and h == crop_size:
        return 0, 0, crop_size, crop_size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    return x1, y1, crop_size, crop_size


def paired_random_crop(x, y, crop_size):
    if crop_size <= 0:
        return x, y
    x = ensure_min_side(x, crop_size)
    y = ensure_min_side(y, crop_size)
    w, h = x.size
    x1, y1, cw, ch = random_crop_params(w, h, crop_size)
    x = x.crop((x1, y1, x1 + cw, y1 + ch))
    y = y.crop((x1, y1, x1 + cw, y1 + ch))
    return x, y


def paired_random_flip_rotate(x, y):
    if random.random() < 0.5:
        x = x.transpose(Image.FLIP_LEFT_RIGHT)
        y = y.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        x = x.transpose(Image.FLIP_TOP_BOTTOM)
        y = y.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.3:
        k = random.randint(0, 3)
        if k:
            rot_code = {
                1: Image.ROTATE_90,
                2: Image.ROTATE_180,
                3: Image.ROTATE_270,
            }[k]
            x = x.transpose(rot_code)
            y = y.transpose(rot_code)
    return x, y


def resize_to_square(img, size):
    if size <= 0:
        return img
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.BICUBIC)


class PairedImageDataset(Dataset):
    def __init__(
        self,
        degraded_dir,
        clean_dir,
        crop_size=256,
        augment=True,
        resize_eval=True,
    ):
        degraded_list = list_images(degraded_dir)
        clean_list = list_images(clean_dir)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.resize_eval = bool(resize_eval)

        clean_map = {}
        for path in clean_list:
            key = os.path.splitext(os.path.basename(path))[0]
            clean_map[key] = path

        self.pairs = []
        for path in degraded_list:
            key = os.path.splitext(os.path.basename(path))[0]
            if key not in clean_map:
                continue
            self.pairs.append((path, clean_map[key]))

        if not self.pairs:
            raise ValueError(
                "No paired samples found. Check filename matching between degraded and clean folders."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.pairs[idx]
        x = load_image(x_path)
        y = load_image(y_path)

        if self.augment:
            x, y = paired_random_crop(x, y, self.crop_size)
            x, y = paired_random_flip_rotate(x, y)
        elif self.resize_eval and self.crop_size > 0:
            x = resize_to_square(x, self.crop_size)
            y = resize_to_square(y, self.crop_size)

        return to_tensor(x), to_tensor(y)


class UnpairedImageDataset(Dataset):
    def __init__(self, degraded_dir, crop_size=256, augment=True):
        self.degraded = list_images(degraded_dir)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        if not self.degraded:
            raise ValueError(f"No images found in {degraded_dir}")

    def __len__(self):
        return len(self.degraded)

    def __getitem__(self, idx):
        x = load_image(self.degraded[idx])
        if self.augment:
            x = ensure_min_side(x, self.crop_size)
            w, h = x.size
            x1, y1, cw, ch = random_crop_params(w, h, self.crop_size)
            x = x.crop((x1, y1, x1 + cw, y1 + ch))
            if random.random() < 0.5:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                x = x.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.crop_size > 0:
            x = resize_to_square(x, self.crop_size)

        return to_tensor(x)
