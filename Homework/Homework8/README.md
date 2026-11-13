# Homework 8 — Neural Networks and Deep Learning

A reproducible PyTorch binary image classifier for curly and straight hair, developed for Machine Learning Zoomcamp 2026.

## Objective

Train a convolutional neural network on the fixed hair-image dataset, evaluate a 10-epoch baseline, then continue training the same model for another 10 epochs with data augmentation.

## Dataset

Download the fixed dataset:

```bash
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
sha256sum data.zip
unzip -q data.zip
```

Expected SHA-256: `9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67`.

The dataset contains 800 training images and 201 evaluation images supported by `ImageFolder`. Class mapping: `curly=0`, `straight=1`.

## Reproducibility

Use Python 3.11 and the reference CPU dependency versions: PyTorch 2.9.0+cpu, torchvision 0.24.0+cpu, NumPy 2.3.3, Pillow 11.3.0. Fix all random seeds to 42, use deterministic PyTorch algorithms and `num_workers=0`, and retain the supplied train/evaluation split.

## Model

- Convolution: 3 input channels, 32 output channels, 3×3 kernel
- ReLU and 2×2 max pooling
- Fully connected layer: `32 * 99 * 99` → 64
- Output layer: 64 → 1 logit
- Loss: `BCEWithLogitsLoss`
- Optimizer: SGD (`lr=0.002`, `momentum=0.8`)
- Trainable parameters: **20,073,473**

## Experiment protocol

1. Train the baseline model for 10 epochs and save `history_baseline.json`.
2. Continue training **the same model and optimizer** for 10 more epochs with rotation, random resized crop, and horizontal flip applied only to the training data.
3. Save the additional 10 epochs to `history_augmented.json`.
4. Use the unchanged evaluation transform and held-out evaluation split for both phases.
5. Compute the requested metrics from the recorded JSON histories.

## Homework answers

| Question | Answer |
| --- | --- |
| Q1 — Loss | `nn.BCEWithLogitsLoss()` |
| Q2 — Trainable parameters | `20073473` |
| Q3 — Median baseline train accuracy | Pending CPU training |
| Q4 — Population std. dev. baseline train loss | Pending CPU training |
| Q5 — Mean augmented evaluation loss | Pending CPU training |
| Q6 — Mean evaluation accuracy, last 5 augmented epochs | Pending CPU training |

## Acknowledgments

Machine Learning Zoomcamp by [DataTalksClub](https://github.com/DataTalksClub/ml-zoomcamp), led by Alexey Grigorev.
