"""Train the class-conditional pixel-space DDPM (EPDP).

This is a **training-only** script. No external checkpoints are required and
no test-set evaluation is run. Optional DDIM sampling can be toggled via the
config file (``sample_every_epochs``).

Expected dataset layout (see ``docs/DATA_FORMAT.md``)::

    <data_root>/
    ├── <class_0>/*.png
    ├── <class_1>/*.png
    └── ...

Class names come from the ``class_list`` field of the YAML config.

Usage::

    python scripts/train_diffusion.py \\
        --config configs/diffusion_breast.yaml \\
        --data-root /path/to/breast_patches \\
        --output-dir runs/diffusion_breast
"""
from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import ClassFolderDataset
from src.diffusion.beta_schedule import get_named_beta_schedule
from src.diffusion.diffusion import GaussianDiffusion
from src.diffusion.embedding import ConditionalEmbedding
from src.diffusion.scheduler import GradualWarmupScheduler
from src.diffusion.unet import Unet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML config (e.g. configs/diffusion_breast.yaml)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root directory containing one subfolder per class")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write checkpoints, samples, and logs into")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device string (default: cuda:0)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Optional checkpoint path to resume training from")
    return parser.parse_args()


def load_config(path_config: Path) -> dict:
    with path_config.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(dict_cfg: dict, int_num_classes: int, device: torch.device):
    model_unet = Unet(
        in_ch=dict_cfg["inch"],
        mod_ch=dict_cfg["modch"],
        out_ch=dict_cfg["outch"],
        ch_mul=dict_cfg["chmul"],
        num_res_blocks=dict_cfg["numres"],
        cdim=dict_cfg["cdim"],
        use_conv=dict_cfg["useconv"],
        droprate=dict_cfg["droprate"],
        dtype=torch.float32,
    ).to(device)

    model_cemb = ConditionalEmbedding(
        int_num_classes, dict_cfg["cdim"], dict_cfg["cdim"]
    ).to(device)

    np_betas = get_named_beta_schedule(
        schedule_name=dict_cfg.get("beta_schedule", "linear"),
        num_diffusion_timesteps=dict_cfg["T"],
    )
    model_diffusion = GaussianDiffusion(
        dtype=torch.float32,
        model=model_unet,
        betas=np_betas,
        w=dict_cfg["w"],
        v=dict_cfg["v"],
        device=device,
    )
    return model_unet, model_cemb, model_diffusion


def build_optimizer(model_diffusion, model_cemb, dict_cfg: dict):
    optimizer = torch.optim.AdamW(
        itertools.chain(model_diffusion.model.parameters(),
                        model_cemb.parameters()),
        lr=dict_cfg["lr"],
        weight_decay=dict_cfg.get("weight_decay", 1e-6),
    )
    scheduler_after = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=dict_cfg.get("exponential_gamma", 0.95)
    )
    scheduler_warmup = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=dict_cfg.get("multiplier", 1),
        warm_epoch=dict_cfg.get("warm_epoch", 3),
        after_scheduler=scheduler_after,
        last_epoch=0,
    )
    return optimizer, scheduler_warmup


@torch.no_grad()
def sample_and_save(
    model_diffusion: GaussianDiffusion,
    model_cemb: ConditionalEmbedding,
    list_classes: list[str],
    dict_cfg: dict,
    path_samples_dir: Path,
    int_epoch: int,
    device: torch.device,
) -> None:
    """Generate one sample per class and write it under ``samples/<class>/<epoch>.png``."""
    model_diffusion.model.eval()
    model_cemb.eval()
    int_n = len(list_classes)
    tensor_labels = torch.arange(int_n, device=device)
    tensor_cemb = model_cemb(tensor_labels)
    tuple_shape = (int_n, dict_cfg["outch"], dict_cfg["image_size"], dict_cfg["image_size"])

    if dict_cfg.get("ddim", True):
        tensor_generated = model_diffusion.ddim_sample(
            tuple_shape,
            dict_cfg.get("ddim_steps", 100),
            dict_cfg.get("ddim_eta", 0.0),
            dict_cfg.get("ddim_select", "quadratic"),
            cemb=tensor_cemb,
        )
    else:
        tensor_generated = model_diffusion.sample(tuple_shape, cemb=tensor_cemb)

    tensor_generated = tensor_generated / 2 + 0.5  # [-1,1] → [0,1]
    to_pil = torchvision.transforms.ToPILImage()
    for i, str_class in enumerate(list_classes):
        path_class_dir = path_samples_dir / str_class
        path_class_dir.mkdir(parents=True, exist_ok=True)
        to_pil(tensor_generated[i].clamp(0, 1).cpu()).save(
            path_class_dir / f"epoch_{int_epoch:04d}.png"
        )


def main() -> None:
    args = parse_args()
    dict_cfg = load_config(args.config)

    device = torch.device(args.device)
    print(f"[train_diffusion] device={device}, available GPUs={torch.cuda.device_count()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_ckpt_dir = args.output_dir / "checkpoints"
    path_samples_dir = args.output_dir / "samples"
    path_ckpt_dir.mkdir(exist_ok=True)
    path_samples_dir.mkdir(exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "config.snapshot.yaml")

    list_classes: list[str] = list(dict_cfg["class_list"])
    int_num_classes = len(list_classes)
    print(f"[train_diffusion] classes={list_classes}")

    dataset_train = ClassFolderDataset(
        path_root=args.data_root,
        list_classes=list_classes,
        int_image_size=dict_cfg["image_size"],
        bool_augment=True,
    )
    print(f"[train_diffusion] dataset size = {len(dataset_train)} patches")
    loader_train = DataLoader(
        dataset_train,
        batch_size=dict_cfg["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model_unet, model_cemb, model_diffusion = build_model(
        dict_cfg, int_num_classes, device
    )
    optimizer, scheduler = build_optimizer(model_diffusion, model_cemb, dict_cfg)
    bool_amp = bool(dict_cfg.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=bool_amp)

    int_start_epoch = 0
    if args.resume is not None and args.resume.is_file():
        print(f"[train_diffusion] resuming from {args.resume}")
        dict_ckpt = torch.load(args.resume, map_location=device)
        model_diffusion.model.load_state_dict(dict_ckpt["net"])
        model_cemb.load_state_dict(dict_ckpt["cemblayer"])
        optimizer.load_state_dict(dict_ckpt["optimizer"])
        scheduler.load_state_dict(dict_ckpt["scheduler"])
        int_start_epoch = int(dict_ckpt.get("epoch", 0))

    path_log_csv = args.output_dir / "log.csv"
    bool_log_existed = path_log_csv.is_file()
    file_log = path_log_csv.open("a", newline="", encoding="utf-8")
    writer_log = csv.writer(file_log)
    if not bool_log_existed:
        writer_log.writerow(["epoch", "mean_loss", "lr"])

    int_save_every = int(dict_cfg.get("save_every_epochs", 10))
    int_sample_every = int(dict_cfg.get("sample_every_epochs", 25))
    float_drop_threshold = float(dict_cfg["threshold"])

    for int_epoch in range(int_start_epoch, dict_cfg["epochs"]):
        model_diffusion.model.train()
        model_cemb.train()
        float_total_loss = 0.0
        int_steps = 0
        with tqdm(loader_train, dynamic_ncols=True,
                  desc=f"epoch {int_epoch + 1}/{dict_cfg['epochs']}") as bar:
            for tensor_img, tensor_lab in bar:
                tensor_img = tensor_img.to(device, non_blocking=True)
                tensor_lab = tensor_lab.to(device, non_blocking=True)

                tensor_cemb = model_cemb(tensor_lab)
                np_drop_mask = np.random.rand(tensor_img.shape[0]) < float_drop_threshold
                if np_drop_mask.any():
                    tensor_cemb[np_drop_mask] = 0

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool_amp):
                    tensor_loss = model_diffusion.trainloss(tensor_img, cemb=tensor_cemb)
                scaler.scale(tensor_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                float_total_loss += tensor_loss.item()
                int_steps += 1
                bar.set_postfix(
                    loss=f"{float_total_loss / int_steps:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        scheduler.step()

        float_mean_loss = float_total_loss / max(int_steps, 1)
        writer_log.writerow([int_epoch + 1, f"{float_mean_loss:.6f}",
                             f"{optimizer.param_groups[0]['lr']:.6e}"])
        file_log.flush()

        if int_sample_every > 0 and (int_epoch + 1) % int_sample_every == 0:
            sample_and_save(
                model_diffusion, model_cemb, list_classes, dict_cfg,
                path_samples_dir, int_epoch + 1, device,
            )

        if int_save_every > 0 and (int_epoch + 1) % int_save_every == 0:
            dict_ckpt = {
                "epoch": int_epoch + 1,
                "net": model_diffusion.model.state_dict(),
                "cemblayer": model_cemb.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(dict_ckpt, path_ckpt_dir / f"ckpt_{int_epoch + 1:04d}.pt")

        torch.cuda.empty_cache()

    file_log.close()
    print(f"[train_diffusion] done. outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
