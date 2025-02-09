import math
import shutil
import tarfile
from typing import List, Tuple
from pathlib import Path

import torch

def extract_all_tar_images(data_path: str)-> Path:
    """
    Function used to extract all items inside a .tgz file and
    return a pathlib.PosixPath object that denoted the extracted path.
    """
    path = Path(data_path)
    image_path = path / "all_images"
    tar_path = path / "test_images.tgz"

    if not tar_path.exists():
        print(f"The tar file {tar_path} does not exist.")
        return image_path
        try:
            # Extract the tar file into the data_path
            with tarfile.open(tar_path, "r:gz") as tf:
                print(f"Extracting {tar_path} into {data_path}")
                tf.extractall(path=data_path, filter=lambda tarinfo, _: tarinfo) #to avoid deprecation warning
            # Rename the extracted 'test_images' directory to 'all_images'to avoid confusion
            extracted_dir = data_path / "test_images"
            if extracted_dir.exists():
                extracted_dir.rename(image_path)
                print(f"Renamed '{extracted_dir}' to '{image_path}'")
            else:
                image_path.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            print(f"An error occurred during extraction: {e}")
            return image_path
        # Remove the tar file
        tar_path.unlink()
    return image_path

def move_files_using_paths_list(
    file_paths: List[Path], dest_dir: Path
)->None:
    for file in file_paths:
        target = dest_dir / file.name
        shutil.move(str(file), str(target))

def separate_images_by_class(
    root: str|Path, train_ratio: float=0.75
)->Tuple[Path, Path]:
    """
    Separate images into two folders - one that has images with tickers
    and another that does not. We will be using
    `torchvision.datasets.ImageFolder` to create datasets, so our images
    should be organized into folders.
    We will split the data into the following subdirectories:
    - train/ticker
    - train/no_ticker
    - val/ticker
    - val/no_ticker
    An unseen test dataset will be provided for testing.

    Args:

        root (str): The path to the folder containing all images
        train_ratio: Proportion of images to use in the training

    Returns:
        Tuple[Path, Path]: A tuple containing the paths to the training
        and validation directories.
    """
    root_path = Path(root)
    tickers = [
        f for f in root_path.iterdir() if f.is_file()
        and f.name.startswith("ticker")
    ]
    no_tickers = [
        f for f in root_path.iterdir() if f.is_file()
        and f.name.startswith("no_ticker")
    ]
    print(f"Number of images with tickers: {len(tickers)}")
    print(f"Number of images without tickers: {len(no_tickers)}")

    # split the ticker class into train and validation
    num_ticker_train = math.floor(train_ratio*len(tickers))
    train_tickers = tickers[: num_ticker_train]
    val_tickers = tickers[num_ticker_train: ]

    # split the no_ticker class
    num_no_ticker_train = math.floor(train_ratio*len(no_tickers))
    train_no_tickers = no_tickers[: num_no_ticker_train]
    val_no_tickers = no_tickers[num_no_ticker_train: ]

    # create train and val directories
    train_dir = root_path / "train"
    val_dir = root_path / "val"

    # subdirs per class for train dir
    train_ticker_dir = train_dir / "ticker"
    train_no_ticker_dir = train_dir / "no_ticker"

    # create subdirs per class for the val dir
    val_ticker_dir = val_dir / "ticker"
    val_no_ticker_dir = val_dir / "no_ticker"

    # create all the subdirectories with `.mkdir()`
    for sub_dir in [
    train_ticker_dir, val_ticker_dir, train_no_ticker_dir, val_no_ticker_dir
    ]:
        sub_dir.mkdir(parents=True, exist_ok=True)

    # move files into the correct subdirectory
    move_files_using_paths_list(train_tickers, train_ticker_dir)
    move_files_using_paths_list(val_tickers, val_ticker_dir)
    move_files_using_paths_list(train_no_tickers, train_no_ticker_dir)
    move_files_using_paths_list(val_no_tickers, val_no_ticker_dir)

    return train_dir, val_dir

def separate_test_images_by_labels(root: str)-> Tuple[Path, Path]:
    root_path = Path(root)
    ticker_dir = root_path / "ticker"
    no_ticker_dir = root_path / "no_ticker"
    ticker_dir.mkdir(exist_ok=True)
    no_ticker_dir.mkdir(exist_ok=True)
    # Iterate over all files in the root folder
    for file in root_path.iterdir():
        # Process only files (skip directories)
        if file.is_file():
            if file.name.startswith("ticker"):
                target_path = ticker_dir / file.name
                file.rename(target_path)
            elif file.name.startswith("no_ticker"):
                target_path = no_ticker_dir / file.name
                file.rename(target_path)

    return ticker_dir, no_ticker_dir

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
)-> None:
    """
    Saves a PyTorch model to a target dictionary.
    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(
            model=model_0,
            target_dir="models",
            model_name="going_modular_tinyvgg_model.pth"
        )
    """
    models_dir = Path(target_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), \
        "Model should end with '.pt' or '.pth'"
    model_path = models_dir / model_name
    print(f"Saving model to: {model_path}")
    torch.save(obj=model.state_dict(), f=model_path)
