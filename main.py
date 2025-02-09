from typing import Tuple, List
import os
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from model_builder import TinyVGG

from utils import separate_test_images_by_labels
# script that tests if an image belongs to a ticker class or a no_ticker class
# scirpt also tests a folder of multiple images

def test(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
)-> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    "Tests the model on an unseen dataset"
    model.eval()
    test_loss, test_acc = 0, 0
    preds, targets, probs = [], [], []

    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)

            test_loss += loss.item()
            test_pred_class = test_pred.argmax(dim=1)
            test_acc += (test_pred_class == y).sum().item()/len(test_pred_class)
            # store predictions
            preds.append(test_pred_class.cpu())
            targets.append(y.cpu())
            probs.append(torch.softmax(test_pred, dim=1).cpu())

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    return test_loss, test_acc, torch.cat(preds), torch.cat(targets), torch.cat(probs)

def test_model_on_sample_image(
    model:torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    class_names: List[str]
)->Tuple[str, str]:
    image_path_obj = Path(image_path)
    filename = image_path_obj.name
    image = Image.open(image_path)
    if filename.startswith("token"):
        true_label = "token"
    else:
        true_label = "no_token"
    image = transform(image)
    # add batch dimension
    assert isinstance(image, torch.Tensor), \
    "Image not tensor, use `transforms.ToTensor`"
    image = image.unsqueeze(0).to(device)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        prediction = model(image)
        predicted_class = torch.argmax(prediction, dim=1)
        predicted_index = int(predicted_class.item())

    return class_names[predicted_index], true_label

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    test_path = input("Please enter the path of data for testing:\n")
    model_path = BASE_DIR / "models" / "optimized_model.pth"
    if not model_path.exists():
        print(f"Error: The model file {model_path} does not exist.\n"
                "Please run train.py to train the model and create this file.")
        exit(1)

    class_names = ["ticker", "no_ticker"]

    model = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(class_names)
    )

    # Load the model state dictionary from file.
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Ask the user whether to test on a folder of images or a single image.
    mode = input("Test on (1) a folder of images or (2) a single image? \nEnter 1 or 2:\n ").strip()

    if mode == "1":
        separate_test_images_by_labels(test_path)
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

        # Create a test dataset and dataloader using torchvision.datasets.ImageFolder.
        test_data = datasets.ImageFolder(test_path, transform=test_transform)
        test_dataloader = DataLoader(
            test_data,
            batch_size=16,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            pin_memory=True
        )

        # Define the loss function (even though we might be mostly interested in accuracy).
        loss_fn = torch.nn.CrossEntropyLoss()

        test_loss, test_acc, preds, targets, probs = test(
            model, test_dataloader, loss_fn, device
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        # Display results
        preds_list = preds.tolist()
        targets_list = targets.tolist()

        # Get filenames and true labels
        filenames = [s[0] for s in test_data.samples]

        # Print results table
        print("\nDetailed Results:")
        print("-" * 60)
        print(f"{'Image':<20} | {'True Label':<15} | {'Predicted Label':<15}")
        print("-" * 60)
        for filename, true_idx, pred_idx in zip(filenames, targets_list, preds_list):
            image_name = os.path.basename(filename)
            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]
            print(f"{image_name:<20} | {true_label:<15} | {pred_label:<15}")

    elif mode == "2":
        # Test on a single image.
        sample_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        predicted_label, true_label = test_model_on_sample_image(
            model, test_path, sample_transform, device, class_names
        )
        print("\nTest Result:")
        print("-" * 30)
        print(f"Image: {os.path.basename(test_path)}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {predicted_label}")
        print("-" * 30)
