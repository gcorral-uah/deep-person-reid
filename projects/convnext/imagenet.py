### TODO:
# We should replace the library code from torch-reid with a natural one
# Maybe use pythorch-lightning
# The saving and loading of the checkpoint is based on
# https://web.archive.org/web/20201106104658/https://chshih2.github.io/blog/2020/10/06/Pytorch-continue-training/

from convnext_impl.convnext_osnet import build_convnext_osnet
import imagenet_data_loader.imagenet_preprocessed_data_loader as imagenet
import torch
from tqdm import tqdm
import glob
import re
import imagenet_data_loader.early_stopping as early_stopping
import gc
from projects.convnext.utils_imagnet.imaginet_loops import (
    training_loop,
    validation_loop,
    do_early_stopping,
)
from utils_imagnet.logging_imagenet import init_gpu_logging, DEBUG

## Note: This imports are for debugging.
if DEBUG:
    from loguru import logger

    # Remove the stdout logging and only log to a file. The pre-configured
    # handler is guaranteed to have the index 0.
    logger.remove(0)

    # If there is already an existing file with the same name that the file to
    # be created, then the existing file is renamed by appending the date to
    # its basename to prevent file overwriting.
    logger.add("imagenet_training.log")
    init_gpu_logging()
else:
    logger = None


def main():
    NUM_EPOCHS = 140
    MODEL_PATH = "convnext_imagenet.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if DEBUG:
        assert logger is not None
        logger.debug(f"Using Device {device}\n")

    imagenet_22k_loader_args = {
        # Path to the data (with folders imagenet21k_train and imagenet21k_val)
        "data_path": "/mnt/data1/gonzalo.corral/imagenet21k_resized/",
        "image_height": 224,
        "image_width": 224,
        "batch_size": 32,
        # How many subprocess to use to load the data (0 load in main process).
        "num_workers": 1,
    }
    # train_loader_22k, validation_loader_22k = imagenet.create_imagenet_22k_data_loaders( imagenet_22k_loader_args)

    imagenet_1k_loader_args = {
        # Path to the data (with files ILSVRC2012_devkit_t12.tar.gz,
        # ILSVRC2012_img_train.tar and
        # ILSVRC2012_img_val.tar )
        "data_path": "/data1/gonzalo.corral/ILSVRC2012/",
        "image_height": 224,
        "image_width": 224,
        "batch_size": 32,
        # How many subprocess to use to load the data (0 load in main process).
        # The optimum is 4*num_gpus. This __leaks__ memory iff >0.
        "num_workers": 1,
    }

    train_loader_1k, validation_loader_1k = imagenet.create_imagenet_1k_data_loaders(
        imagenet_1k_loader_args
    )

    # imagenet_loader_args = imagenet_22k_loader_args
    imagenet_loader_args = imagenet_1k_loader_args

    train_loader, validation_loader = train_loader_1k, validation_loader_1k

    if DEBUG:
        assert logger is not None
        if "train_loader_1k" in vars():
            dataset_name = "imagenet_1k"
        elif "train_loader_22k" in vars():
            dataset_name = "imagenet_22k"
        else:
            dataset_name = ""
        logger.debug(f"Loaded dataset {dataset_name}")

    print("Building model: {}")
    convnext_config_22k = {
        "num_classes": imagenet.imagenet_data_22k().get("num_training_classes", 10450),
        "pretrained": False,
    }

    convnext_config_1k = {
        "num_classes": imagenet.imagenet_data_1k().get("num_training_classes", 1000),
        "pretrained": False,
    }

    # convnext_config = convnext_config_22k
    convnext_config = convnext_config_1k

    convnext_model = build_convnext_osnet(**convnext_config)
    convnext_model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(convnext_model.parameters())

    # Determine if there is a training checkpoint and load it.
    latest_epoch = None
    TRAINING_EPOCH_PATH_GLOB = r"convnext_imagenet_epoch_*.pth"
    training_epochs = glob.glob(TRAINING_EPOCH_PATH_GLOB)
    if len(training_epochs) > 0:
        training_epochs.sort(
            key=lambda name: [int(s) for s in re.findall(r"\d+", name)][0],
            reverse=True,
        )
        latest_epoch = training_epochs[0]
        print(f"Using saved epoch {latest_epoch}")

    if DEBUG and latest_epoch is not None:
        assert logger is not None
        logger.debug(f"Loading epoch: {latest_epoch}\n")

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

    early_stopper = early_stopping.EarlyStopping(patience=7, verbose=True)
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
        total_training_loss = training_loop(
            train_loader=train_loader,
            current_epoch=epoch,
            model=convnext_model,
            optimizer=optimizer,
            loss_fn=loss_function,
            device=device,
            data_dir=imagenet_loader_args["data_path"],
            logger=logger,
        )

        convnext_model.train(False)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": convnext_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_training_loss,
                "best_loss": best_validation_loss,
            },
            EPOCH_PATH,
        )

        # Try to free gpu_memory
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            print(f"I am going to start validation of epoch: {epoch}")
            validation_loss = validation_loop(
                validation_loader=validation_loader,
                current_epoch=epoch,
                model=convnext_model,
                loss_fn=loss_function,
                device=device,
                data_dir=imagenet_loader_args["data_path"],
                logger=logger,
            )
            stop_early = do_early_stopping(
                early_stopper=early_stopper,
                validation_loss=validation_loss,
                model=convnext_model,
                logger=logger,
                current_epoch=epoch,
                data_dir=imagenet_loader_args["data_path"],
            )
            if stop_early:
                print(f"Early stopping in epoch {epoch}")
                break

            print(f"I reached the end of validation in epoch {epoch}")

        print(f"I reached the end of training in epoch {epoch}")
        # Try to free gpu_memory
        gc.collect()
        torch.cuda.empty_cache()

    print(f"I have finished training")


if __name__ == "__main__":
    main()
