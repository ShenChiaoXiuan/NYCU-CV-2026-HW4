# HW4 - Image Restoration with PromptIR

**NYCU Visual Recognition using Deep Learning — Spring 2026**
**Student ID:** 314551095

---

## Introduction

This project tackles **blind all-in-one image restoration** for two degradation types — rain streaks and snow particles — using a single unified model based on **PromptIR** (Potlapalli et al., NeurIPS 2023).

### Our Contributions on Top of PromptIR

| Modification | Description |
|---|---|
| **Soft Degradation Router** | Each `PromptBlock` maintains two learnable prompt sets (rain-biased / snow-biased). A lightweight router network predicts a soft mixing weight directly from the input feature map, enabling automatic degradation-aware prompting without requiring any type label at inference time. |
| **Frequency Loss** | Added an FFT amplitude L1 term to the training loss. Rain streaks are high-frequency vertical patterns; supervising the frequency domain forces the model to correctly reconstruct frequency content, not just pixel values. |
| **Test-Time Augmentation (TTA)** | At inference, each image is processed under all 8 transforms of the dihedral group D4 (4 rotations × 2 flips). The inverse-transformed predictions are averaged, improving robustness without retraining. |

### Results

| Version | Modifications | Public PSNR |
|---|---|---|
| v1 | Baseline PromptIR (base_c=48, patch=128) | 29.63 dB |
| v2 | + Soft Router, Frequency Loss, base_c=64, patch=256 | 30.32 dB |
| v3 | + TTA at inference | **30.78 dB** |

---

## Environment Setup

```bash
# Python 3.9+
pip install torch torchvision numpy pillow
```

Tested with:
- Python 3.11
- PyTorch 2.x
- CUDA 12.x
- GPU: NVIDIA 24 GB

---

## Usage

### Training

```bash
python train.py --mode train --data_dir ./data --batch_size 4
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./data` | Root folder containing `train/` and `test/` |
| `--epochs` | `200` | Maximum training epochs |
| `--batch_size` | `4` | Training batch size |
| `--patch_size` | `256` | Random crop size during training |
| `--base_c` | `64` | Base channel width of PromptIR |
| `--num_blocks` | `6` | Number of ResBlocks per scale |
| `--patience` | `25` | Early stopping patience (epochs) |
| `--ckpt` | `checkpoints/best_v2.pth` | Path to save best checkpoint |

Expected data structure:

```
data/
├── train/
│   ├── degraded/
│   │   ├── rain-1.png ... rain-1600.png
│   │   └── snow-1.png ... snow-1600.png
│   └── clean/
│       ├── rain_clean-1.png ... rain_clean-1600.png
│       └── snow_clean-1.png ... snow_clean-1600.png
└── test/
    └── degraded/
        └── 0.png ... 99.png
```

### Inference (with TTA)

```bash
python train.py --mode inference --ckpt checkpoints/best_v2.pth --output_npz pred.npz
```

Then package for submission:

```bash
cp pred.npz pred.npz
zip submission.zip pred.npz
```

### Monitor training

```bash
tail -f logs/val_v2.log
```

---

## Performance Snapshot

Public leaderboard score on CodaBench (PSNR):

![leaderboard](snapshot.png)

> v3 (with TTA): **30.78 dB** — above the strong baseline (~30 dB)