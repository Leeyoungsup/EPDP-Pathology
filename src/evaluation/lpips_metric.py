"""Pairwise 1 - LPIPS evaluation between real and synthetic patches.

Uses the official ``lpips`` package with the AlexNet backbone (default
in Zhang et al., CVPR 2018). Inputs are normalized to [-1, 1] as
required by the LPIPS module.
"""
from itertools import combinations
from typing import List, Tuple

import lpips
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def _build_loader(list_paths: List[str], int_image_size: int) -> torch.Tensor:
    trans = transforms.Compose([
        transforms.Resize((int_image_size, int_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    list_tensors = [trans(Image.open(p).convert("RGB")) for p in list_paths]
    return torch.stack(list_tensors, dim=0)


def compute_lpips_pairs(list_paths_a: List[str], list_paths_b: List[str],
                        device: torch.device, int_image_size: int = 256,
                        bool_self_pairs: bool = False) -> Tuple[float, float]:
    """Return (mean 1-LPIPS, std) across all pairs.

    If ``bool_self_pairs`` is True, computes pairs within ``list_paths_a``.
    Otherwise, computes pairs across A x B.
    """
    model_lpips = lpips.LPIPS(net="alex").to(device).eval()
    tensor_a = _build_loader(list_paths_a, int_image_size).to(device)

    list_scores: List[float] = []
    with torch.no_grad():
        if bool_self_pairs:
            for i, j in tqdm(list(combinations(range(len(tensor_a)), 2)),
                             desc="1-LPIPS (self)"):
                d = model_lpips(tensor_a[i:i + 1], tensor_a[j:j + 1]).item()
                list_scores.append(1.0 - d)
        else:
            tensor_b = _build_loader(list_paths_b, int_image_size).to(device)
            for i in tqdm(range(len(tensor_a)), desc="1-LPIPS (cross)"):
                for j in range(len(tensor_b)):
                    d = model_lpips(tensor_a[i:i + 1], tensor_b[j:j + 1]).item()
                    list_scores.append(1.0 - d)

    tensor_scores = torch.tensor(list_scores)
    return float(tensor_scores.mean()), float(tensor_scores.std())
