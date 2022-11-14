### TODO:
# We should replace the library code from torch-reid with a natural one
# Maybe use pythorch-lightning
# The saving and loading of the checkpoint is based on
# https://web.archive.org/web/20201106104658/https://chshih2.github.io/blog/2020/10/06/Pytorch-continue-training/

from convnext_impl.build_models import build_convnext
import imagenet_data_loader.imagenet_preprocessed_data_loader as imagenet
import torch
from tqdm import tqdm
import glob
import re


def main():
    NUM_EPOCHS = 140
    MODEL_PATH = "convnext_imagenet.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    imagenet_loader_args = {
        # Path to the data (with folders imagenet21k_train and imagenet21k_val)
        "data_path": "/mnt/data1/gonzalo.corral/imagenet21k_resized/",
        "image_size": 224,
        "batch_size": 64,
        # How many subprocess to use to load the data (0 load in main process).
        "num_workers": 0,
    }
    train_loader, validation_loader = imagenet.create_imagenet_data_loaders(
        imagenet_loader_args
    )

    print("Building model: {}")
    convnext_config = {
        "num_classes": imagenet.imagenet_data().get("num_training_classes", 10450),
        "pretrained": False,
    }
    convnext_model = build_convnext(**convnext_config)
    convnext_model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(convnext_model.parameters())

    # Determine if there is a training checkpoint and load it.
    latest_epoch = None
    TRAINING_EPOCH_PATH_GLOB = r"convnext_imagenet_epoch_*.pth"
    training_epochs = glob.glob(TRAINING_EPOCH_PATH_GLOB)
    if len(training_epochs) > 0:
        training_epochs.sort(
            key=lambda name: [int(s) for s in re.findall(r"\b\d+\b", name)][0],
            reverse=True,
        )
        latest_epoch = training_epochs[0]


    best_validation_loss = None
    # Starting epoch (in case we need to restart training from a previous epoch)
    start_epoch = 0
    if latest_epoch is not None:
        checkpoint = torch.load(latest_epoch)
        start_epoch = checkpoint["epoch"] + 1
        convnext_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_validation_loss = checkpoint["best_loss"]
        convnext_model = convnext_model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Training loop
    for epoch in tqdm(range(start_epoch, NUM_EPOCHS)):
        if best_validation_loss is None:
            # Bogus high number to record the best loss in an epoch. This is
            # used if we aren't continuing training from an saved epoch
            best_validation_loss = 1_000_000.0
        else:
            pass

        EPOCH_PATH = f"convnext_imagenet_epoch_{epoch}.pth"
        # Make sure gradient tracking is on, and do a pass over the data
        convnext_model.train(True)
        current_epoch_loss = None

        total_train_loss = 0.0
        for cur_train_iter, training_data in enumerate(train_loader):
            data_inputs, data_labels = training_data

            # Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # Run the model on the input data
            training_outputs = convnext_model(data_inputs)
            # Calculate the loss
            loss = loss_function(training_outputs, data_labels)
            current_epoch_loss = loss

            # Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()

            #  Update the parameters
            optimizer.step()

            # Calculate the total and average loss
            total_train_loss += loss
            avg_training_loss = total_train_loss / (cur_train_iter + 1)

        convnext_model.train(False)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": convnext_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": current_epoch_loss,
                "best_loss": best_validation_loss,
            },
            EPOCH_PATH,
        )

        total_validation_loss = 0.0
        for cur_validation_iter, validation_data in enumerate(validation_loader):
            validation_inputs, validation_labels = validation_data
            validation_inputs = validation_inputs.to(device)
            validation_labels = validation_labels.to(device)
            validation_outputs = convnext_model(validation_inputs)
            validation_loss = loss_function(validation_outputs, validation_labels)
            total_validation_loss += validation_loss

            avg_validation_loss = total_validation_loss / (cur_validation_iter + 1)
            print(f"Validation loss = {avg_validation_loss}")

            # Track best performance, and save the model's state
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                # Save the final model
                torch.save(convnext_model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
