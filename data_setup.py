import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count() or 0

def create_train_val_dataloaders(
    train_dir: Path,
    val_dir: Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """
    Creates training and validation dataloaders.

    Takes in a training directory and validattion directory path
    and turns them into PyTorch Datasets and then into PyTorch
    DataLoaders.

    Args:
        train_dir: Path to training directory
        val_dir: Path to validation directory
        transform: torchvision transforms to perform on training
                    and validation data
        batch_size: Number of samples per batch in each of the
                    DataLoaders.
        num_workers: Number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, val_dataloader, class_names).
        Where class_names is a list of target classes.
        Example usage:
        `train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=path/to/train_dir,
            val_dir=path/to/val_dir,
            batch_size=32,
            num_workers=4
        )`
    """
    # create datasets using ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)

    # get class names
    class_names = train_data.classes

    # Make DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader, val_dataloader, class_names
