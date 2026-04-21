"""Train the EfficientNetV2-L subtype classifier with SAM optimizer.

Same script trains both manuscript variants:
- **EffNet-Real** — point ``--data-root`` at real patches.
- **EffNet-Syn**  — point ``--data-root`` at synthetic patches generated
  by the diffusion + CycleGAN pipeline (per-class subfolders).

Validation set is split off from the same root by ``test_size`` (default 0.2),
stratified by class label.

Expected dataset layout (see ``docs/DATA_FORMAT.md``)::

    <data_root>/
    ├── BRNT/*.png
    ├── BRLC/*.png
    └── ...

Usage::

    python scripts/train_classifier.py \\
        --config configs/classifier.yaml \\
        --organ breast \\
        --data-root /path/to/breast_patches \\
        --output-dir runs/classifier_real_breast
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier.efficientnet_v2 import (
    SAM,
    EfficientNetV2Classifier,
    FeatureExtractor,
)
from src.data.datasets import ClassFolderDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--organ", choices=["breast", "stomach"], required=True,
                        help="Selects breast_classes or stomach_classes from the YAML")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def stratified_split(dataset_full: ClassFolderDataset,
                     float_test_size: float,
                     int_random_state: int) -> tuple[Subset, Subset]:
    list_labels = [int_label for _, int_label in dataset_full.list_samples]
    list_indices = list(range(len(dataset_full)))
    list_train_idx, list_val_idx = train_test_split(
        list_indices,
        test_size=float_test_size,
        random_state=int_random_state,
        stratify=list_labels,
    )
    return Subset(dataset_full, list_train_idx), Subset(dataset_full, list_val_idx)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        dict_cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_ckpt_dir = args.output_dir / "checkpoints"
    path_ckpt_dir.mkdir(exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "config.snapshot.yaml")

    if args.organ == "breast":
        list_classes = list(dict_cfg["breast_classes"])
    else:
        list_classes = list(dict_cfg["stomach_classes"])
    int_num_classes = len(list_classes)
    print(f"[train_classifier] organ={args.organ}, classes={list_classes}")

    dataset_full_train = ClassFolderDataset(
        path_root=args.data_root,
        list_classes=list_classes,
        int_image_size=dict_cfg["image_size"],
        bool_augment=True,
        tuple_normalize_mean=tuple(dict_cfg["normalize_mean"]),
        tuple_normalize_std=tuple(dict_cfg["normalize_std"]),
    )
    dataset_full_val = ClassFolderDataset(
        path_root=args.data_root,
        list_classes=list_classes,
        int_image_size=dict_cfg["image_size"],
        bool_augment=False,
        tuple_normalize_mean=tuple(dict_cfg["normalize_mean"]),
        tuple_normalize_std=tuple(dict_cfg["normalize_std"]),
    )

    dataset_train, _ = stratified_split(
        dataset_full_train, dict_cfg["test_size"], dict_cfg["random_state"]
    )
    _, dataset_val = stratified_split(
        dataset_full_val, dict_cfg["test_size"], dict_cfg["random_state"]
    )
    print(f"[train_classifier] train={len(dataset_train)}, val={len(dataset_val)}")

    loader_train = DataLoader(
        dataset_train,
        batch_size=dict_cfg["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=dict_cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    feature_extractor = FeatureExtractor(str_backbone=dict_cfg["backbone"])
    model = EfficientNetV2Classifier(
        int_num_classes=int_num_classes,
        int_image_feature_dim=dict_cfg["image_feature_dim"],
        feature_extractor=feature_extractor,
    ).to(device)

    base_optimizer_cls = torch.optim.SGD if dict_cfg["base_optimizer"].upper() == "SGD" \
        else torch.optim.Adam
    optimizer = SAM(
        model.parameters(),
        base_optimizer=base_optimizer_cls,
        rho=dict_cfg["rho"],
        adaptive=dict_cfg["adaptive"],
        lr=dict_cfg["lr"],
        momentum=dict_cfg.get("momentum", 0.9),
    )

    int_start_epoch = 0
    float_best_val_acc = 0.0
    if args.resume is not None and args.resume.is_file():
        dict_ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(dict_ckpt["model"])
        optimizer.load_state_dict(dict_ckpt["optimizer"])
        int_start_epoch = int(dict_ckpt.get("epoch", 0))
        float_best_val_acc = float(dict_ckpt.get("best_val_acc", 0.0))
        print(f"[train_classifier] resumed from epoch {int_start_epoch}, "
              f"best val acc so far = {float_best_val_acc:.4f}")

    path_log_csv = args.output_dir / "log.csv"
    bool_log_existed = path_log_csv.is_file()
    file_log = path_log_csv.open("a", newline="", encoding="utf-8")
    writer_log = csv.writer(file_log)
    if not bool_log_existed:
        writer_log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    int_epochs = dict_cfg["epochs"]
    for int_epoch in range(int_start_epoch, int_epochs):
        # ───── Train ─────
        model.train()
        float_train_loss = 0.0
        int_train_correct = 0
        int_train_total = 0
        with tqdm(loader_train, dynamic_ncols=True,
                  desc=f"clf epoch {int_epoch + 1}/{int_epochs}") as bar:
            for tensor_img, tensor_lab in bar:
                tensor_img = tensor_img.to(device, non_blocking=True)
                tensor_lab = tensor_lab.to(device, non_blocking=True)

                # SAM step 1: ascent
                tensor_logits = model(tensor_img)
                tensor_loss = F.cross_entropy(tensor_logits, tensor_lab)
                tensor_loss.backward()
                optimizer.first_step(zero_grad=True)

                # SAM step 2: descent at perturbed weights
                tensor_logits2 = model(tensor_img)
                tensor_loss2 = F.cross_entropy(tensor_logits2, tensor_lab)
                tensor_loss2.backward()
                optimizer.second_step(zero_grad=True)

                float_train_loss += tensor_loss.item() * tensor_img.size(0)
                int_train_correct += (tensor_logits.argmax(dim=1) == tensor_lab).sum().item()
                int_train_total += tensor_img.size(0)
                bar.set_postfix(
                    loss=f"{float_train_loss / int_train_total:.4f}",
                    acc=f"{int_train_correct / int_train_total:.4f}",
                )

        float_train_loss = float_train_loss / max(int_train_total, 1)
        float_train_acc = int_train_correct / max(int_train_total, 1)

        # ───── Validate ─────
        model.eval()
        float_val_loss = 0.0
        int_val_correct = 0
        int_val_total = 0
        with torch.no_grad():
            for tensor_img, tensor_lab in loader_val:
                tensor_img = tensor_img.to(device, non_blocking=True)
                tensor_lab = tensor_lab.to(device, non_blocking=True)
                tensor_logits = model(tensor_img)
                tensor_loss = F.cross_entropy(tensor_logits, tensor_lab)
                float_val_loss += tensor_loss.item() * tensor_img.size(0)
                int_val_correct += (tensor_logits.argmax(dim=1) == tensor_lab).sum().item()
                int_val_total += tensor_img.size(0)
        float_val_loss = float_val_loss / max(int_val_total, 1)
        float_val_acc = int_val_correct / max(int_val_total, 1)

        writer_log.writerow([
            int_epoch + 1,
            f"{float_train_loss:.6f}", f"{float_train_acc:.6f}",
            f"{float_val_loss:.6f}", f"{float_val_acc:.6f}",
        ])
        file_log.flush()

        dict_ckpt = {
            "epoch": int_epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": float_best_val_acc,
            "list_classes": list_classes,
        }
        torch.save(dict_ckpt, path_ckpt_dir / "ckpt_last.pt")
        if float_val_acc > float_best_val_acc:
            float_best_val_acc = float_val_acc
            dict_ckpt["best_val_acc"] = float_best_val_acc
            torch.save(dict_ckpt, path_ckpt_dir / "ckpt_best.pt")
            print(f"[train_classifier] new best val acc = {float_best_val_acc:.4f} "
                  f"@ epoch {int_epoch + 1}")

    file_log.close()
    print(f"[train_classifier] done. best val acc = {float_best_val_acc:.4f}, "
          f"outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
