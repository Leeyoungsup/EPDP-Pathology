"""Pairwise SSIM evaluation between real and synthetic patches.

Uses ``skimage.metrics.structural_similarity`` with a Gaussian window
(win_size=11) on RGB images.
"""
from itertools import combinations
from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def _load_array(str_path: str, int_image_size: int) -> np.ndarray:
    img = Image.open(str_path).convert("RGB").resize((int_image_size, int_image_size))
    return np.asarray(img, dtype=np.float32) / 255.0


def compute_ssim_pairs(list_paths_a: List[str], list_paths_b: List[str],
                       int_image_size: int = 256,
                       bool_self_pairs: bool = False) -> Tuple[float, float]:
    list_arrays_a = [_load_array(p, int_image_size) for p in list_paths_a]

    list_scores: List[float] = []
    if bool_self_pairs:
        for i, j in tqdm(list(combinations(range(len(list_arrays_a)), 2)),
                         desc="SSIM (self)"):
            list_scores.append(ssim(list_arrays_a[i], list_arrays_a[j],
                                    channel_axis=2, data_range=1.0))
    else:
        list_arrays_b = [_load_array(p, int_image_size) for p in list_paths_b]
        for arr_a in tqdm(list_arrays_a, desc="SSIM (cross)"):
            for arr_b in list_arrays_b:
                list_scores.append(ssim(arr_a, arr_b, channel_axis=2,
                                        data_range=1.0))

    np_scores = np.asarray(list_scores)
    return float(np_scores.mean()), float(np_scores.std())
