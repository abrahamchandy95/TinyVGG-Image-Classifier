from pathlib import Path
import torch
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils

# setup hyperparams
NUM_EPOCHS = 5
BATCH_SIZE = 16
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    data_path = BASE_DIR / "data"
    images_path = utils.extract_all_tar_images(str(data_path))
    train_dir, val_dir = utils.separate_images_by_class(images_path)

    device = torch.device("cpu")

    train_transform_trivial = transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    val_transform_simple = transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    train_dataloader, val_dataloader, class_names = (
        data_setup.create_train_val_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            train_transform=train_transform_trivial,
            val_transform=val_transform_simple,
            batch_size=BATCH_SIZE
        )
    )
    # create model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = timer()
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    utils.save_model(
        model=model,
        target_dir=str("models"),
        model_name="optimized_model.pth"
    )
