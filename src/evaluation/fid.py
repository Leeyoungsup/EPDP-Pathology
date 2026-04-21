"""Frechet Inception Distance (FID) evaluation utilities.

We follow the manuscript's evaluation protocol: FID is computed at the
organ level (breast / stomach), comparing real-vs-real (baseline) and
real-vs-synthetic. The InceptionV3 features are extracted from
299x299 inputs.
"""
from typing import List

import numpy as np
import torch
from PIL import Image
from pytorch_fid import fid_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm


INT_INCEPTION_INPUT_SIZE = 299


class _PathDataset(Dataset):
    def __init__(self, list_paths: List[str], int_image_size: int = INT_INCEPTION_INPUT_SIZE):
        self.list_paths = list_paths
        self.trans = transforms.Compose([transforms.ToTensor()])
        self.int_image_size = int_image_size

    def __len__(self) -> int:
        return len(self.list_paths)

    def __getitem__(self, index: int) -> Tensor:
        img = Image.open(self.list_paths[index]).convert("RGB")
        img = img.resize((self.int_image_size, self.int_image_size))
        return self.trans(img)


def _extract_inception_features(list_paths: List[str], device: torch.device,
                                int_batch_size: int = 8) -> np.ndarray:
    model_inception = inception_v3(pretrained=True).to(device).eval()
    dataset = _PathDataset(list_paths)
    loader = DataLoader(dataset, batch_size=int_batch_size, shuffle=False)
    list_acts = []
    with torch.no_grad():
        for x in tqdm(loader, desc="Inception features"):
            x = x.to(device)
            list_acts.append(model_inception(x).detach().cpu().numpy())
    return np.concatenate(list_acts, axis=0)


def compute_fid_pairs(list_paths_a: List[str], list_paths_b: List[str],
                      device: torch.device) -> float:
    np_act_a = _extract_inception_features(list_paths_a, device)
    np_act_b = _extract_inception_features(list_paths_b, device)
    mu_a, sigma_a = fid_score.calculate_activation_statistics(np_act_a)
    mu_b, sigma_b = fid_score.calculate_activation_statistics(np_act_b)
    return float(fid_score.calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b))
