"""Train CycleGAN for stain normalization (Stage 1: unpaired A ↔ B).

Two ResNet generators and two PatchGAN discriminators are trained jointly with
- adversarial LSGAN loss (MSE on patch logits),
- cycle-consistency L1 loss (λ_cycle = 10),
- VGG16-based perceptual / style loss (weight ``style_weight``, default 1e4),
- a Shrivastava-style image history pool of size ``image_pool_size``.

Expected dataset layout (see ``docs/DATA_FORMAT.md``)::

    <data_root>/
    ├── trainA/*.png    # source domain
    └── trainB/*.png    # target / reference stain

Usage::

    python scripts/train_cyclegan.py \\
        --config configs/cyclegan.yaml \\
        --data-root /path/to/cyclegan_root \\
        --output-dir runs/cyclegan
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import UnpairedFolderDataset
from src.stain_norm.cyclegan import Discriminator, Generator, weights_init_normal


class ImagePool:
    """Shrivastava et al. image-history buffer for stabilizing GAN training."""

    def __init__(self, int_pool_size: int):
        self.int_pool_size = int_pool_size
        self.list_images: list[torch.Tensor] = []

    def query(self, tensor_images: torch.Tensor) -> torch.Tensor:
        if self.int_pool_size == 0:
            return tensor_images
        list_returned: list[torch.Tensor] = []
        for tensor_img in tensor_images.detach():
            tensor_img = tensor_img.unsqueeze(0)
            if len(self.list_images) < self.int_pool_size:
                self.list_images.append(tensor_img.clone())
                list_returned.append(tensor_img)
            elif random.random() < 0.5:
                int_idx = random.randrange(self.int_pool_size)
                tensor_old = self.list_images[int_idx].clone()
                self.list_images[int_idx] = tensor_img.clone()
                list_returned.append(tensor_old)
            else:
                list_returned.append(tensor_img)
        return torch.cat(list_returned, dim=0)


class VGGFeatureExtractor(nn.Module):
    """ImageNet-normalized VGG16 conv features used for perceptual / style loss."""

    def __init__(self, list_layer_indices: tuple[int, ...] = (3, 8, 15, 22)):
        super().__init__()
        model_vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features
        for p in model_vgg.parameters():
            p.requires_grad = False
        self.model_vgg = model_vgg.eval()
        self.list_layer_indices = list_layer_indices

        self.register_buffer(
            "tensor_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "tensor_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _normalize(self, tensor_x: torch.Tensor) -> torch.Tensor:
        # tensor_x is in [-1, 1] (Tanh output). Bring to [0, 1] then ImageNet-normalize.
        tensor_01 = (tensor_x + 1.0) / 2.0
        return (tensor_01 - self.tensor_mean) / self.tensor_std

    def forward(self, tensor_x: torch.Tensor) -> list[torch.Tensor]:
        tensor_x = self._normalize(tensor_x)
        list_features: list[torch.Tensor] = []
        for int_idx, layer in enumerate(self.model_vgg):
            tensor_x = layer(tensor_x)
            if int_idx in self.list_layer_indices:
                list_features.append(tensor_x)
                if int_idx == self.list_layer_indices[-1]:
                    break
        return list_features


def gram_matrix(tensor_x: torch.Tensor) -> torch.Tensor:
    int_b, int_c, int_h, int_w = tensor_x.shape
    tensor_flat = tensor_x.view(int_b, int_c, int_h * int_w)
    return torch.bmm(tensor_flat, tensor_flat.transpose(1, 2)) / (int_c * int_h * int_w)


def style_loss(model_vgg: VGGFeatureExtractor,
               tensor_pred: torch.Tensor,
               tensor_target: torch.Tensor) -> torch.Tensor:
    list_feat_pred = model_vgg(tensor_pred)
    list_feat_target = model_vgg(tensor_target)
    tensor_loss = tensor_pred.new_zeros(())
    for tensor_p, tensor_t in zip(list_feat_pred, list_feat_target):
        tensor_loss = tensor_loss + nn.functional.l1_loss(
            gram_matrix(tensor_p), gram_matrix(tensor_t.detach())
        )
    return tensor_loss


def linear_decay_lambda(int_epoch: int, int_decay_start: int, int_total_epochs: int) -> float:
    if int_epoch < int_decay_start:
        return 1.0
    return max(0.0, 1.0 - (int_epoch - int_decay_start) / max(1, int_total_epochs - int_decay_start))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing trainA/ and trainB/ subfolders")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        dict_cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_ckpt_dir = args.output_dir / "checkpoints"
    path_samples_dir = args.output_dir / "samples"
    path_ckpt_dir.mkdir(exist_ok=True)
    path_samples_dir.mkdir(exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "config.snapshot.yaml")

    dataset_train = UnpairedFolderDataset(
        path_root=args.data_root,
        int_image_size=dict_cfg.get("crop_size", 1024),
        bool_horizontal_flip=bool(dict_cfg.get("fliplr", False)),
    )
    print(f"[train_cyclegan] |A|={len(dataset_train.list_files_a)}, "
          f"|B|={len(dataset_train.list_files_b)}")

    loader_train = DataLoader(
        dataset_train,
        batch_size=dict_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    G_A2B = Generator(3, 3, n_residual_blocks=dict_cfg.get("num_resnet", 9)).to(device)
    G_B2A = Generator(3, 3, n_residual_blocks=dict_cfg.get("num_resnet", 9)).to(device)
    D_A = Discriminator(3).to(device)
    D_B = Discriminator(3).to(device)
    for net in (G_A2B, G_B2A, D_A, D_B):
        net.apply(weights_init_normal)

    model_vgg = VGGFeatureExtractor().to(device).eval()

    float_lr_g = dict_cfg.get("lrG", 2e-4)
    float_lr_d = dict_cfg.get("lrD", 2e-4)
    tuple_betas = (dict_cfg.get("beta1", 0.5), dict_cfg.get("beta2", 0.999))
    optim_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                               lr=float_lr_g, betas=tuple_betas)
    optim_D_A = torch.optim.Adam(D_A.parameters(), lr=float_lr_d, betas=tuple_betas)
    optim_D_B = torch.optim.Adam(D_B.parameters(), lr=float_lr_d, betas=tuple_betas)

    int_total_epochs = dict_cfg.get("num_epochs", 100)
    int_decay_start = dict_cfg.get("decay_epoch", 50)
    sched_G = torch.optim.lr_scheduler.LambdaLR(
        optim_G, lr_lambda=lambda e: linear_decay_lambda(e, int_decay_start, int_total_epochs))
    sched_D_A = torch.optim.lr_scheduler.LambdaLR(
        optim_D_A, lr_lambda=lambda e: linear_decay_lambda(e, int_decay_start, int_total_epochs))
    sched_D_B = torch.optim.lr_scheduler.LambdaLR(
        optim_D_B, lr_lambda=lambda e: linear_decay_lambda(e, int_decay_start, int_total_epochs))

    fake_A_pool = ImagePool(dict_cfg.get("image_pool_size", 10))
    fake_B_pool = ImagePool(dict_cfg.get("image_pool_size", 10))

    crit_gan = nn.MSELoss()
    crit_cycle = nn.L1Loss()
    crit_identity = nn.L1Loss()

    float_lambda_a = dict_cfg.get("lambdaA", 10.0)
    float_lambda_b = dict_cfg.get("lambdaB", 10.0)
    float_style_w = dict_cfg.get("style_weight", 1e4)
    float_identity_w = 0.5  # standard CycleGAN identity weight wrt λ_cycle

    int_start_epoch = 0
    if args.resume is not None and args.resume.is_file():
        dict_ckpt = torch.load(args.resume, map_location=device)
        G_A2B.load_state_dict(dict_ckpt["G_A2B"])
        G_B2A.load_state_dict(dict_ckpt["G_B2A"])
        D_A.load_state_dict(dict_ckpt["D_A"])
        D_B.load_state_dict(dict_ckpt["D_B"])
        optim_G.load_state_dict(dict_ckpt["optim_G"])
        optim_D_A.load_state_dict(dict_ckpt["optim_D_A"])
        optim_D_B.load_state_dict(dict_ckpt["optim_D_B"])
        int_start_epoch = int(dict_ckpt.get("epoch", 0))
        print(f"[train_cyclegan] resumed from epoch {int_start_epoch}")

    path_log_csv = args.output_dir / "log.csv"
    bool_log_existed = path_log_csv.is_file()
    file_log = path_log_csv.open("a", newline="", encoding="utf-8")
    writer_log = csv.writer(file_log)
    if not bool_log_existed:
        writer_log.writerow(["epoch", "loss_G", "loss_D_A", "loss_D_B", "lr"])

    save_pil = transforms.ToPILImage()

    for int_epoch in range(int_start_epoch, int_total_epochs):
        G_A2B.train(); G_B2A.train(); D_A.train(); D_B.train()
        float_sum_g = float_sum_da = float_sum_db = 0.0
        int_steps = 0

        with tqdm(loader_train, dynamic_ncols=True,
                  desc=f"cyclegan epoch {int_epoch + 1}/{int_total_epochs}") as bar:
            for tensor_real_a, tensor_real_b in bar:
                tensor_real_a = tensor_real_a.to(device, non_blocking=True)
                tensor_real_b = tensor_real_b.to(device, non_blocking=True)

                # ───── Generators ─────
                optim_G.zero_grad(set_to_none=True)

                tensor_idt_a = G_B2A(tensor_real_a)
                tensor_idt_b = G_A2B(tensor_real_b)
                loss_idt = (crit_identity(tensor_idt_a, tensor_real_a) * float_lambda_a
                            + crit_identity(tensor_idt_b, tensor_real_b) * float_lambda_b
                            ) * float_identity_w

                tensor_fake_b = G_A2B(tensor_real_a)
                tensor_fake_a = G_B2A(tensor_real_b)

                tensor_pred_fake_b = D_B(tensor_fake_b)
                tensor_pred_fake_a = D_A(tensor_fake_a)
                loss_g_adv = (crit_gan(tensor_pred_fake_b, torch.ones_like(tensor_pred_fake_b))
                              + crit_gan(tensor_pred_fake_a, torch.ones_like(tensor_pred_fake_a)))

                tensor_rec_a = G_B2A(tensor_fake_b)
                tensor_rec_b = G_A2B(tensor_fake_a)
                loss_cycle = (crit_cycle(tensor_rec_a, tensor_real_a) * float_lambda_a
                              + crit_cycle(tensor_rec_b, tensor_real_b) * float_lambda_b)

                loss_style = (style_loss(model_vgg, tensor_fake_b, tensor_real_b)
                              + style_loss(model_vgg, tensor_fake_a, tensor_real_a)) * float_style_w

                loss_g = loss_g_adv + loss_cycle + loss_idt + loss_style
                loss_g.backward()
                optim_G.step()

                # ───── Discriminator A (real B → fake A) ─────
                optim_D_A.zero_grad(set_to_none=True)
                tensor_fake_a_pool = fake_A_pool.query(tensor_fake_a)
                tensor_pred_real = D_A(tensor_real_a)
                tensor_pred_fake = D_A(tensor_fake_a_pool.detach())
                loss_d_a = 0.5 * (
                    crit_gan(tensor_pred_real, torch.ones_like(tensor_pred_real))
                    + crit_gan(tensor_pred_fake, torch.zeros_like(tensor_pred_fake))
                )
                loss_d_a.backward()
                optim_D_A.step()

                # ───── Discriminator B (real A → fake B) ─────
                optim_D_B.zero_grad(set_to_none=True)
                tensor_fake_b_pool = fake_B_pool.query(tensor_fake_b)
                tensor_pred_real = D_B(tensor_real_b)
                tensor_pred_fake = D_B(tensor_fake_b_pool.detach())
                loss_d_b = 0.5 * (
                    crit_gan(tensor_pred_real, torch.ones_like(tensor_pred_real))
                    + crit_gan(tensor_pred_fake, torch.zeros_like(tensor_pred_fake))
                )
                loss_d_b.backward()
                optim_D_B.step()

                float_sum_g += loss_g.item()
                float_sum_da += loss_d_a.item()
                float_sum_db += loss_d_b.item()
                int_steps += 1
                bar.set_postfix(
                    G=f"{float_sum_g / int_steps:.3f}",
                    D_A=f"{float_sum_da / int_steps:.3f}",
                    D_B=f"{float_sum_db / int_steps:.3f}",
                )

        sched_G.step(); sched_D_A.step(); sched_D_B.step()

        writer_log.writerow([
            int_epoch + 1,
            f"{float_sum_g / max(int_steps, 1):.6f}",
            f"{float_sum_da / max(int_steps, 1):.6f}",
            f"{float_sum_db / max(int_steps, 1):.6f}",
            f"{optim_G.param_groups[0]['lr']:.6e}",
        ])
        file_log.flush()

        # save one A→B preview
        with torch.no_grad():
            G_A2B.eval()
            tensor_preview = G_A2B(tensor_real_a[:1])
            tensor_preview = (tensor_preview.clamp(-1, 1) + 1) / 2
            save_pil(tensor_preview[0].cpu()).save(
                path_samples_dir / f"epoch_{int_epoch + 1:04d}_A2B.png"
            )

        torch.save({
            "epoch": int_epoch + 1,
            "G_A2B": G_A2B.state_dict(),
            "G_B2A": G_B2A.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "optim_G": optim_G.state_dict(),
            "optim_D_A": optim_D_A.state_dict(),
            "optim_D_B": optim_D_B.state_dict(),
        }, path_ckpt_dir / f"ckpt_{int_epoch + 1:04d}.pt")

    file_log.close()
    print(f"[train_cyclegan] done. outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
