import os
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from glob import glob

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)


def find_images(data_dir: str) -> List[str]:
    patterns = [os.path.join(data_dir, '**', '*.jpg'),
                os.path.join(data_dir, '**', '*.png')]
    paths = []
    for p in patterns:
        paths.extend(glob(p, recursive=True))
    return sorted(paths)


def preprocess_image(
    img_path: str,
    crop_factors: Tuple[float, float, float, float] = (0.1, 0.95, 0.05, 0.95),
    size: Tuple[int, int] = (256, 256)
) -> Optional[np.ndarray]:

    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    t, b = int(crop_factors[0]*h), int(crop_factors[1]*h)
    l, r = int(crop_factors[2]*w), int(crop_factors[3]*w)

    cropped = img[t:b, l:r]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size)

    return resized.astype(np.float32) / 255.0


class ThermalDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = preprocess_image(path)
        if img is None:
            img = np.zeros((256, 256), dtype=np.float32)
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor, path


def analyze_thermal_image(
    img_path: str,
    threshold_value: int = 150
) -> Dict[str, Any]:

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}

    _, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {}

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    pixels = img[mask == 255]
    if pixels.size == 0:
        return {}

    return {
        "max_temp": float(np.max(pixels)),
        "mean_temp": float(np.mean(pixels)),
        "std_temp": float(np.std(pixels)),
        "num_pixels": int(pixels.size),
        "mask": mask
    }
