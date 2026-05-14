# Aesthetic Rating Pretraining

Train an image aesthetic scoring model on the AVA Aesthetic Visual Assessment dataset. The training script converts AVA vote distributions into expected rating scores from 1 to 10, then fine-tunes an EfficientViT-B3 regressor with PyTorch.

## Overview

This repository contains a compact training pipeline for aesthetic image rating:

- Loads AVA-style image labels from `data/ground_truth_dataset.csv`
- Reads matching images from `data/images`
- Computes a continuous aesthetic score from the 1-10 vote histogram
- Trains an `efficientvit_b3` model from `timm`
- Uses Albumentations for image augmentation
- Saves the best validation checkpoint to `checkpoints/best_model.pt`

## Dataset

The project was built for the AVA Aesthetic Visual Assessment dataset:

https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment/data

Expected local structure:

```text
data/
  ground_truth_dataset.csv
  images/
    1.jpg
    2.jpg
    ...
```

The CSV is expected to include an `image_num` column and vote columns named:

```text
vote_1, vote_2, vote_3, vote_4, vote_5, vote_6, vote_7, vote_8, vote_9, vote_10
```

Images that are missing from `data/images` are skipped automatically.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install torch torchvision timm albumentations opencv-python tqdm
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation guide.

## Training

After placing the dataset under `data/`, run:

```bash
python train.py
```

The script will:

1. Load and validate local image/label pairs
2. Split the data into train and validation sets
3. Train for 10 epochs by default
4. Save the best checkpoint by validation loss

Best model output:

```text
checkpoints/best_model.pt
```

## Configuration

Training settings are defined in the `Config` class inside `train.py`.

Default values:

| Setting | Value |
| --- | --- |
| Image size | `256` |
| Validation split | `0.1` |
| Batch size | `40` |
| Epochs | `10` |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| Workers | `4` |
| Device | CUDA if available, otherwise CPU |

Adjust these values directly in `train.py` to match your hardware or experiment setup.

## Model

The model uses `efficientvit_b3` from `timm` with pretrained weights. The classifier head is replaced with a small regression head that outputs one continuous aesthetic score.

Loss function:

```text
Mean Squared Error
```

Optimizer:

```text
AdamW
```

## Notes

- The `data/` and `checkpoints/` directories are ignored by Git.
- The dataset itself is not included in this repository.
- The checkpoint contains the model state, optimizer state, validation loss, and epoch.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
