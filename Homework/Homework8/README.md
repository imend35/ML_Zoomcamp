# Homework 8 — Neural Networks and Deep Learning

**ML Zoomcamp 2026 | DataTalksClub**

## Project Overview

In this homework, I developed and trained a Convolutional Neural Network (CNN) using PyTorch to classify hair images into two categories: curly and straight.

The project explores binary image classification, CNN architecture design, model training, evaluation metrics, and data augmentation.

## Dataset

The dataset contains 1,001 supported image files organized into training and evaluation directories.

| Dataset    | Images |
| ---------- | -----: |
| Training   |    800 |
| Evaluation |    201 |
| Total      |  1,001 |

Class labels:

- Curly: `0`
- Straight: `1`

Dataset source: [Straight vs. Curly Hair Dataset](https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip)

## Technologies

- Python 3.11
- PyTorch 2.9.0 (CPU)
- Torchvision 0.24.0
- NumPy 2.3.3
- Pillow 11.3.0

## Model Architecture

I implemented a CNN with the following architecture:

1. Input: RGB images resized to 200 × 200
2. Convolution: 32 filters, 3 × 3 kernel
3. ReLU activation
4. Max pooling: 2 × 2
5. Flatten
6. Fully connected hidden layer: 64 neurons
7. ReLU activation
8. Output layer: 1 logit

**Total trainable parameters: 20,073,473**

The model uses `BCEWithLogitsLoss` for binary classification and SGD optimization with a learning rate of `0.002` and momentum of `0.8`.

## Training Process

I trained the model in two consecutive phases.

**Phase 1 — Baseline Training**

- 10 epochs
- Batch size: 20
- Input normalization using ImageNet mean and standard deviation
- Fixed random seed: 42

**Phase 2 — Data Augmentation**

I continued training the same model and optimizer for another 10 epochs using:

- Random rotation (±50 degrees)
- Random resized crop
- Random horizontal flip

The evaluation dataset used the same preprocessing throughout both phases, without random augmentation.

## Homework Results

| Question                                                           | Result            |
| ------------------------------------------------------------------ | ----------------- |
| Q1 — Loss function                                                 | BCEWithLogitsLoss |
| Q2 — Trainable parameters                                          | 20,073,473        |
| Q3 — Median training accuracy                                      | 0.81              |
| Q4 — Standard deviation of training loss                           | 0.139             |
| Q5 — Mean evaluation loss with augmentation                        | 0.591             |
| Q6 — Mean evaluation accuracy over the final five augmented epochs | 0.73              |

## Observations

The baseline model achieved a final training accuracy of 93.87%, but its evaluation accuracy was 63.68%, suggesting overfitting.

After the augmentation phase, the final evaluation accuracy increased to 69.65%, while evaluation loss decreased from 1.0212 to 0.5790.

The highest evaluation accuracy during the augmentation phase was 77.11%, achieved in its ninth epoch.

These results illustrate how data augmentation can help improve generalization in image classification.

## How to Run

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements-cpu.txt
```

Download and extract the dataset into the `data/` directory.

Verify the model architecture:

```bash
python model.py
```

Run both training phases:

```bash
python train.py
```

The training script generates:

- `history_baseline.json`
- `history_augmented.json`

These files contain the training and evaluation metrics for each phase.

## Acknowledgments

Thanks to **DataTalksClub** and **Alexey Grigorev** for providing the ML Zoomcamp course and practical machine learning assignments.
