"""
HW4 - Image Restoration with PromptIR (v2)
Changes vs v1:
  - base_c default 48->64, num_blocks default 4->6
  - patch_size default 128->256
  - PromptBlock: added Soft Degradation Router (2 prompt sets, auto-weighted)
  - CombinedLoss: added Frequency Loss (FFT amplitude L1) for rain streak
  - batch_size default 4->2 (larger patch needs more VRAM)
  - patience default 20->25

Usage:
    Train:     python train.py --mode train
    Inference: python train.py --mode inference --ckpt checkpoints/best_v2.pth
"""

import os
import random
import argparse
import logging
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# ------------------------------------------------------------------ Config ---

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_v2.pth')
    parser.add_argument('--output_npz', type=str, default='pred.npz')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--prompt_dim', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--base_c', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ----------------------------------------------------------------- Logging ---

def setup_logger(log_path: str, name: str = 'hw4') -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            '[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ------------------------------------------------------------------ Dataset --

class RestorationDataset(Dataset):
    """Loads paired (degraded, clean) images for train/val."""

    def __init__(self, degraded_paths, clean_paths, patch_size=256,
                 augment=True):
        self.degraded_paths = degraded_paths
        self.clean_paths = clean_paths
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.degraded_paths)

    def __getitem__(self, idx):
        deg = Image.open(self.degraded_paths[idx]).convert('RGB')
        clean = Image.open(self.clean_paths[idx]).convert('RGB')

        i, j, h, w = transforms.RandomCrop.get_params(
            deg, (self.patch_size, self.patch_size)
        )
        deg = TF.crop(deg, i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)

        if self.augment:
            if torch.rand(1) > 0.5:
                deg, clean = TF.hflip(deg), TF.hflip(clean)
            if torch.rand(1) > 0.5:
                deg, clean = TF.vflip(deg), TF.vflip(clean)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                deg = TF.rotate(deg, 90 * k)
                clean = TF.rotate(clean, 90 * k)

        return self.to_tensor(deg), self.to_tensor(clean)


class TestDataset(Dataset):
    """Loads test images for inference (no GT)."""

    def __init__(self, test_dir):
        self.paths = sorted(
            [
                os.path.join(test_dir, f)
                for f in os.listdir(test_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        name = os.path.basename(self.paths[idx])
        return self.to_tensor(img), name


# -------------------------------------------------------------------- Model --

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, padding=p, groups=in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResBlock(nn.Module):
    """Residual block with channel attention."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            LayerNorm2d(c),
            DepthwiseSeparableConv(c, c),
            nn.GELU(),
            DepthwiseSeparableConv(c, c),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c // 4),
            nn.ReLU(),
            nn.Linear(c // 4, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.ca(x).unsqueeze(-1).unsqueeze(-1)
        return x + self.body(x) * att


class PromptBlock(nn.Module):
    """
    PromptIR-style cross-attention block with Soft Degradation Router.

    Key modification (our contribution):
      Instead of a single shared prompt, we maintain TWO prompt sets
      (one biased toward rain, one toward snow). A lightweight router
      network predicts a soft mixing weight from the input feature map,
      so the model automatically selects the right prompt mixture at
      inference time without needing any degradation-type label.

    Reference: Potlapalli et al., PromptIR (NeurIPS 2023)
    """

    def __init__(self, feat_dim, prompt_dim=64, num_prompts=5):
        super().__init__()
        self.num_prompts = num_prompts

        self.prompt_A = nn.Parameter(torch.randn(1, num_prompts, prompt_dim))
        self.prompt_B = nn.Parameter(torch.randn(1, num_prompts, prompt_dim))
        self.prompt_proj = nn.Linear(prompt_dim, feat_dim)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(feat_dim * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )

        self.norm_feat = nn.LayerNorm(feat_dim)
        self.norm_prompt = nn.LayerNorm(feat_dim)

        self.q = nn.Linear(feat_dim, feat_dim)
        self.k = nn.Linear(feat_dim, feat_dim)
        self.v = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5

        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        w_route = self.router(x)
        prompt_mix = (
            w_route[:, 0:1].unsqueeze(-1) * self.prompt_A
            + w_route[:, 1:2].unsqueeze(-1) * self.prompt_B
        )
        prompt = self.prompt_proj(prompt_mix)
        prompt = self.norm_prompt(prompt)

        feat = x.flatten(2).permute(0, 2, 1)
        feat = self.norm_feat(feat)

        q = self.q(feat)
        k = self.k(prompt)
        v = self.v(prompt)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = self.out_proj(torch.bmm(attn, v))

        feat = feat + out
        feat = feat + self.ffn(feat)

        return feat.permute(0, 2, 1).reshape(b, c, h, w) + x


class PromptIR(nn.Module):
    """
    U-Net style encoder-decoder with PromptBlocks at every scale.
    base_c=64, num_blocks=6 for improved capacity.
    """

    def __init__(self, in_c=3, base_c=64, num_blocks=6, prompt_dim=64):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.enc0 = nn.Conv2d(in_c, c1, 3, padding=1)

        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.enc1 = nn.Sequential(*[ResBlock(c2) for _ in range(num_blocks)])
        self.pb1 = PromptBlock(c2, prompt_dim)

        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[ResBlock(c3) for _ in range(num_blocks)])
        self.pb2 = PromptBlock(c3, prompt_dim)

        self.down3 = nn.Conv2d(c3, c4, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            *[ResBlock(c4) for _ in range(num_blocks)]
        )
        self.pb3 = PromptBlock(c4, prompt_dim)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c4, c3, 3, padding=1),
        )
        self.dec2 = nn.Sequential(
            *[ResBlock(c3 * 2) for _ in range(num_blocks)]
        )
        self.fuse2 = nn.Conv2d(c3 * 2, c3, 1)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c3, c2, 3, padding=1),
        )
        self.dec1 = nn.Sequential(
            *[ResBlock(c2 * 2) for _ in range(num_blocks)]
        )
        self.fuse1 = nn.Conv2d(c2 * 2, c2, 1)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c2, c1, 3, padding=1),
        )
        self.dec0 = nn.Sequential(
            *[ResBlock(c1 * 2) for _ in range(num_blocks)]
        )
        self.fuse0 = nn.Conv2d(c1 * 2, c1, 1)

        self.head = nn.Conv2d(c1, in_c, 3, padding=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.pb1(self.enc1(self.down1(e0)))
        e2 = self.pb2(self.enc2(self.down2(e1)))
        b = self.pb3(self.bottleneck(self.down3(e2)))

        d2 = self.fuse2(self.dec2(torch.cat([self.up3(b), e2], dim=1)))
        d1 = self.fuse1(self.dec1(torch.cat([self.up2(d2), e1], dim=1)))
        d0 = self.fuse0(self.dec0(torch.cat([self.up1(d1), e0], dim=1)))

        return torch.clamp(self.head(d0) + x, 0., 1.)


# --------------------------------------------------------------------- Loss --

class CombinedLoss(nn.Module):
    """
    L1 + SSIM + Frequency Loss.

    Frequency Loss (our addition):
      Rain streaks are high-frequency vertical patterns. By supervising
      the FFT amplitude spectrum we force the model to correctly
      reconstruct frequency content, not just pixel values.
    """

    def __init__(self):
        super().__init__()

    def ssim_loss(self, pred, target):
        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_t = F.avg_pool2d(target, 3, 1, 1)
        sig_p = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_p ** 2
        sig_t = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_t ** 2
        sig_pt = F.avg_pool2d(pred * target, 3, 1, 1) - mu_p * mu_t
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / (
            (mu_p ** 2 + mu_t ** 2 + c1) * (sig_p + sig_t + c2)
        )
        return 1 - ssim.mean()

    def freq_loss(self, pred, target):
        pred_f = torch.fft.rfft2(pred, norm='ortho')
        target_f = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(pred_f.abs(), target_f.abs())

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        freq = self.freq_loss(pred, target)
        return l1 + 0.1 * ssim + 0.05 * freq


# --------------------------------------------------------------------- PSNR --

def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * np.log10(1.0 / mse)


# ----------------------------------------------------------- Early Stopping --

class EarlyStopping:
    def __init__(self, patience=25, ckpt_path='checkpoints/best.pth',
                 logger=None):
        self.patience = patience
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.best_psnr = -float('inf')
        self.counter = 0
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    def step(self, val_psnr: float, model: nn.Module) -> bool:
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            self.counter = 0
            torch.save(model.state_dict(), self.ckpt_path)
            if self.logger:
                self.logger.info(
                    f'  [EarlyStopping] New best PSNR={val_psnr:.4f}'
                    ' -> checkpoint saved.'
                )
        else:
            self.counter += 1
            if self.logger:
                self.logger.info(
                    f'  [EarlyStopping] No improvement for '
                    f'{self.counter}/{self.patience} epochs.'
                )
            if self.counter >= self.patience:
                if self.logger:
                    self.logger.info(
                        f'  [EarlyStopping] Triggered. '
                        f'Best PSNR={self.best_psnr:.4f}'
                    )
                return True
        return False


# ------------------------------------------------------------- Data helpers --

def collect_pairs(data_dir):
    deg_dir = os.path.join(data_dir, 'train', 'degraded')
    clean_dir = os.path.join(data_dir, 'train', 'clean')
    pairs = []
    for fname in sorted(os.listdir(deg_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        deg_path = os.path.join(deg_dir, fname)
        prefix, num = fname.rsplit('-', 1)
        clean_name = f'{prefix}_clean-{num}'
        clean_path = os.path.join(clean_dir, clean_name)
        if os.path.exists(clean_path):
            pairs.append((deg_path, clean_path))
    return pairs


def split_pairs(pairs, val_ratio=0.1, seed=42):
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


# -------------------------------------------------------------------  Train --

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_logger = setup_logger('logs/train_v2.log', 'train_v2')
    val_logger = setup_logger('logs/val_v2.log', 'val_v2')

    train_logger.info(
        f'=== HW4 Training Start (v2) | device={device} ==='
    )
    train_logger.info(f'Args: {vars(args)}')

    pairs = collect_pairs(args.data_dir)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    train_logger.info(
        f'Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}'
    )

    train_deg, train_clean = zip(*train_pairs)
    val_deg, val_clean = zip(*val_pairs)

    train_ds = RestorationDataset(
        list(train_deg), list(train_clean),
        patch_size=args.patch_size, augment=True
    )
    val_ds = RestorationDataset(
        list(val_deg), list(val_clean),
        patch_size=args.patch_size, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = PromptIR(
        base_c=args.base_c, num_blocks=args.num_blocks,
        prompt_dim=args.prompt_dim
    ).to(device)
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    train_logger.info(f'Model params: {num_params / 1e6:.2f}M')

    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    early_stop = EarlyStopping(
        patience=args.patience,
        ckpt_path=args.ckpt,
        logger=train_logger
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (deg, clean) in enumerate(train_loader, 1):
            deg, clean = deg.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(deg)
            loss = criterion(pred, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if step % 50 == 0:
                train_logger.info(
                    f'Epoch [{epoch}/{args.epochs}] '
                    f'Step [{step}/{len(train_loader)}] '
                    f'Loss={loss.item():.4f}'
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        train_logger.info(
            f'Epoch [{epoch}/{args.epochs}] AvgLoss={avg_loss:.4f} '
            f'LR={scheduler.get_last_lr()[0]:.2e} Time={elapsed:.1f}s'
        )

        model.eval()
        val_psnr_total = 0.0
        with torch.no_grad():
            for deg, clean in val_loader:
                deg, clean = deg.to(device), clean.to(device)
                pred = model(deg)
                val_psnr_total += calc_psnr(pred, clean)

        val_psnr = val_psnr_total / len(val_loader)
        val_logger.info(
            f'Epoch [{epoch}/{args.epochs}] Val PSNR={val_psnr:.4f} dB'
        )

        scheduler.step()

        if early_stop.step(val_psnr, model):
            train_logger.info(
                '=== Early stopping triggered. Training ended. ==='
            )
            break

    train_logger.info(
        f'=== Training done. Best Val PSNR={early_stop.best_psnr:.4f} ==='
    )


# ---------------------------------------------------------------- TTA helpers -

def tta_forward(model, x):
    """
    Test-Time Augmentation over the 8-element dihedral group D4
    (4 rotations x 2 flips). Each variant is passed through the model
    and the inverse transform is applied before averaging, yielding a
    more robust prediction without any additional training.
    """
    preds = []
    for flip in [False, True]:
        for k in range(4):
            t = x
            if flip:
                t = torch.flip(t, dims=[-1])
            if k > 0:
                t = torch.rot90(t, k, dims=[-2, -1])

            out = model(t)

            if k > 0:
                out = torch.rot90(out, 4 - k, dims=[-2, -1])
            if flip:
                out = torch.flip(out, dims=[-1])

            preds.append(out)

    return torch.stack(preds, dim=0).mean(dim=0)


# --------------------------------------------------------------- Inference ---

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf_logger = setup_logger('logs/inference_v2.log', 'inference_v2')
    inf_logger.info(f'=== Inference Start (TTA) | device={device} ===')
    inf_logger.info(f'Checkpoint: {args.ckpt}')

    model = PromptIR(
        base_c=args.base_c, num_blocks=args.num_blocks,
        prompt_dim=args.prompt_dim
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    inf_logger.info('Model loaded.')

    test_dir = os.path.join(args.data_dir, 'test', 'degraded')
    test_ds = TestDataset(test_dir)
    inf_logger.info(
        f'Test images: {len(test_ds)} | TTA: 8 augmentations per image'
    )

    images_dict = {}
    with torch.no_grad():
        for deg_tensor, name in DataLoader(
            test_ds, batch_size=1, num_workers=2
        ):
            deg_tensor = deg_tensor.to(device)
            pred = tta_forward(model, deg_tensor)
            pred_np = (
                pred.squeeze(0).cpu().numpy() * 255
            ).clip(0, 255).astype(np.uint8)
            images_dict[name[0]] = pred_np
            inf_logger.info(
                f'  Processed {name[0]} | shape={pred_np.shape}'
            )

    np.savez(args.output_npz, **images_dict)
    inf_logger.info(
        f'Saved {len(images_dict)} images -> {args.output_npz}'
    )
    print(f'\nDone. Submission saved to: {args.output_npz}')


# ---------------------------------------------------------------------- Main -

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
print("test")