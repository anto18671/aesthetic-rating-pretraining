# Import dependencies
import csv
import os
from pathlib import Path

import albumentations as A
import cv2
import timm
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


# Define configuration
class Config:
    # Dataset
    data_dir = "data"
    labels_file = "ground_truth_dataset.csv"
    images_dir = "images"
    image_size = 256
    val_split = 0.1

    # Training
    batch_size = 40
    epochs = 10
    learning_rate = 5e-5
    weight_decay = 1e-4
    num_workers = 4

    # Output
    output_dir = "checkpoints"

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# Define dataset wrapper
class AestheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, samples=None):
        # Store paths
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / Config.images_dir

        # Store transform
        self.transform = transform

        # Load local AVA samples
        if samples is None:
            self.samples = self._load_samples(
                self.data_dir / Config.labels_file
            )
        else:
            self.samples = samples

    def _load_samples(self, labels_path):
        # Initialize sample list
        samples = []

        # Define vote columns
        vote_columns = [
            f"vote_{score}"
            for score in range(1, 11)
        ]

        # Read labels
        with labels_path.open(
            newline="",
            encoding="utf-8"
        ) as labels_file:
            reader = csv.DictReader(labels_file)

            # Iterate over rows
            for row in reader:
                # Build image path
                image_path = (
                    self.images_dir
                    / f"{row['image_num']}.jpg"
                )

                # Skip missing images
                if not image_path.exists():
                    continue

                # Compute expected aesthetic score from vote distribution
                votes = [
                    float(row[column])
                    for column in vote_columns
                ]
                total_votes = sum(votes)

                if total_votes == 0:
                    continue

                score = sum(
                    vote * rating
                    for rating, vote in enumerate(
                        votes,
                        start=1
                    )
                ) / total_votes

                # Add sample
                samples.append((
                    image_path,
                    score
                ))

        # Validate samples
        if not samples:
            raise RuntimeError(
                f"No images found for labels in {labels_path}"
            )

        return samples

    def __len__(self):
        # Return dataset length
        return len(self.samples)

    def __getitem__(self, index):
        # Get sample
        image_path, score = self.samples[index]

        # Get image
        image = cv2.imread(str(image_path))

        # Validate image
        if image is None:
            raise RuntimeError(
                f"Unable to read image: {image_path}"
            )

        # Convert image to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert score to tensor
        score = torch.tensor(
            score,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, score


def split_samples(samples, val_split, seed=42):
    # Create shuffled index order
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(samples),
        generator=generator
    ).tolist()

    # Compute split sizes
    val_size = int(
        len(samples) * val_split
    )
    train_size = len(samples) - val_size

    # Split samples
    train_samples = [
        samples[index]
        for index in indices[:train_size]
    ]

    val_samples = [
        samples[index]
        for index in indices[train_size:]
    ]

    return train_samples, val_samples


# Define regression model
class Regressor(nn.Module):
    def __init__(self):
        # Initialize module
        super().__init__()

        # Create EfficientViT backbone
        self.backbone = timm.create_model(
            "efficientvit_b3",
            pretrained=True,
            num_classes=1,
        )

        # Replace classifier head
        self.backbone.head.classifier = nn.Sequential(
            nn.Linear(2304, 768, bias=False),
            nn.LayerNorm(768),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(768, 512, bias=False),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 1),
        )

        # Print model
        print(self.backbone)

    def forward(self, x):
        # Forward pass
        return self.backbone(x)


# Define train function
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    # Set train mode
    model.train()

    # Initialize loss
    running_loss = 0.0

    # Create progress bar
    progress_bar = tqdm(
        loader,
        desc="Train",
        leave=False
    )

    # Iterate over batches
    for images, targets in progress_bar:
        # Move tensors to device
        images = images.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss = criterion(
            predictions,
            targets
        )

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Compute average loss
        average_loss = (
            running_loss /
            (progress_bar.n + 1)
        )

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}"
        )

    # Return average loss
    return running_loss / len(loader)


# Define validation function
def validate(
    model,
    loader,
    criterion,
    device
):
    # Set eval mode
    model.eval()

    # Initialize loss
    running_loss = 0.0

    # Disable gradients
    with torch.no_grad():
        # Create progress bar
        progress_bar = tqdm(
            loader,
            desc="Val",
            leave=False
        )

        # Iterate over batches
        for images, targets in progress_bar:
            # Move tensors to device
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(
                predictions,
                targets
            )

            # Update running loss
            running_loss += loss.item()

            # Compute average loss
            average_loss = (
                running_loss /
                (progress_bar.n + 1)
            )

            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{average_loss:.4f}"
            )

    # Return validation loss
    return running_loss / len(loader)


# Define main function
def main():
    # Create output directory
    os.makedirs(
        Config.output_dir,
        exist_ok=True
    )

    # Define train transforms
    train_transform = A.Compose([
        A.Resize(
            Config.image_size,
            Config.image_size
        ),

        A.HorizontalFlip(p=0.5),

        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=0.5
        ),

        A.Affine(
            translate_percent={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05)
            },
            scale=(0.90, 1.10),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),

        A.GaussianBlur(
            blur_limit=(3, 5),
            p=0.2
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        ToTensorV2()
    ])

    # Define validation transforms
    val_transform = A.Compose([
        A.Resize(
            Config.image_size,
            Config.image_size
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        ToTensorV2()
    ])

    # Load local samples
    full_dataset = AestheticDataset(
        Config.data_dir
    )

    # Create train/validation split
    train_samples, val_samples = split_samples(
        full_dataset.samples,
        Config.val_split,
        seed=42
    )

    # Print dataset sizes
    print(
        f"Train samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_samples)}"
    )

    # Create datasets
    train_dataset = AestheticDataset(
        Config.data_dir,
        transform=train_transform,
        samples=train_samples
    )

    val_dataset = AestheticDataset(
        Config.data_dir,
        transform=val_transform,
        samples=val_samples
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Create model
    model = Regressor()

    # Move model to device
    model = model.to(Config.device)

    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # Define loss function
    criterion = nn.MSELoss()

    # Initialize best loss
    best_loss = float("inf")

    # Train epochs
    for epoch in range(Config.epochs):
        # Print epoch
        print(
            f"\nEpoch "
            f"{epoch + 1}/{Config.epochs}"
        )

        # Train model
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            Config.device
        )

        # Validate model
        val_loss = validate(
            model,
            val_loader,
            criterion,
            Config.device
        )

        # Print metrics
        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_loss:
            # Update best loss
            best_loss = val_loss

            # Define checkpoint path
            checkpoint_path = (
                Path(Config.output_dir)
                / "best_model.pt"
            )

            # Save checkpoint
            torch.save(
                {
                    "model_state_dict":
                    model.state_dict(),

                    "optimizer_state_dict":
                    optimizer.state_dict(),

                    "val_loss":
                    val_loss,

                    "epoch":
                    epoch
                },
                checkpoint_path
            )

            # Print save message
            print(
                f"Saved best model "
                f"to {checkpoint_path}"
            )


# Run script
if __name__ == "__main__":
    main()
