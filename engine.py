from typing import List, Dict, Tuple
import torch
from tqdm.auto import tqdm

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    device:torch.device
)->Tuple[float, float]:
    """
    Trains a PyTorch model for a single epoch.

    Turns a target Pytorch model to training mode and then runs through all of
    the training steps.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on
        loss_fn: A PyTorch loss function to minimize
        optimizer: a PyTorch optimizer to help minimize the loss function
        device: A target device to compute on

    Returns:
        A tuple of training loss and training accuracy metrics, in the form of
        (train_loss, train_accuracy). For example: (0.1112, 0.8743)
    """
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device
)-> Tuple[float, float]:
    """
    Validates a PyTorch model for a single epoch.

    Turns a target PyTorch model to 'eval' model and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested
        dataloader: A DataLoader instance for the model to be tested on
        loss_fn: A Pytorch loss function to calculate loss on the test data
        device: A target device to compute on

    Returns:
        a tuple of testing loss and testing accuracy metrics in the form
        (test_loss, test_accuracy). For example:
        (0.0223, 0.8985)
    """
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            val_pred = model(X)
            loss = loss_fn(val_pred, y)

            val_loss += loss.item()
            val_pred_class = val_pred.argmax(dim=1)
            val_acc += (val_pred_class == y).sum().item()/len(val_pred_class)
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device
)-> Dict[str, List[float]]:
    """
    Trains and validates a Pytorch model.

    Passes a target PyTorch model through a train_step() and a test_step()
    function for a number of epochs, training and testing the model in the
    same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on

    Returns:
    A dictionary of training and testing loss as well as training and testing accuracy
    metrics. Each metric has a value ina list for each epoch.
    In the form: {
        train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]
    }
    For example if training for epochs=2:
        {
            train_loss: [2.0616, 1.0537],
            train_acc: [0,3934, 0.4125],
            test_loss: [1,2537, 1.5706],
            test_acc: [0.3400, 0.2973]
        }
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        val_loss, val_acc = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        print(f"Epoch: {epoch+1}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}\n")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    return results
