### TODO:
# We should replace the library code from torch-reid with a natural one
# Maybe use pythorch-lightning
# We have another option using a data_manager from torchreid to do the work
# - The data loaders should work
# - The engine won't work as we haven't wrapped it in a data-manager

from convnext_impl.build_models import build_convnext
import imagenet_data_loader.imagenet_preprocessed_data_loader as imagenet
import torch
from tqdm import tqdm


def main():
    NUM_EPOCHS = 140
    MODEL_PATH = "convnext_imagenet.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    imagenet_loader_args = {
        # Path to the data (with folders imagenet21k_train and imagenet21k_val)
        "data_path": "/mnt/data1/gonzalo.corral/imagenet21k/",
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
    model = build_convnext(**convnext_config)
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Bogus high number to record the best loss in any epoch
    best_validation_loss = 1_000_000.0
    # Training loop
    for epoch in tqdm(range(NUM_EPOCHS)):
        EPOCH_PATH = f"convnext_imagenet_epoch_{epoch}.pth"
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        current_epoch_loss = None

        total_train_loss = 0.0
        avg_training_loss = 0.0
        for cur_train_iter, training_data in enumerate(train_loader):
            data_inputs, data_labels = training_data

            # Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # Run the model on the input data
            training_outputs = model(data_inputs)
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

        model.train(False)

        # Save the current state of the epoch, to allow to resume training later. To load do:
        # checkpoint = torch.load(PATH)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # model.to(device)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        #
        # model.eval()
        # or
        # model.train()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": current_epoch_loss,
            },
            EPOCH_PATH,
        )

        total_validation_loss = 0.0
        for cur_validation_iter, validation_data in enumerate(validation_loader):
            validation_inputs, validation_labels = validation_data
            validation_inputs = validation_inputs.to(device)
            validation_labels = validation_labels.to(device)
            validation_outputs = model(validation_inputs)
            validation_loss = loss_function(validation_outputs, validation_labels)
            total_validation_loss += validation_loss

            avg_validation_loss = total_validation_loss / (cur_validation_iter + 1)
            print(
                "LOSS train {} valid {}".format(avg_training_loss, avg_validation_loss)
            )

            # Track best performance, and save the model's state
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                # Save the final model
                torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
