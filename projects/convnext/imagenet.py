### TODO:
# We should replace the library code from torch-reid with a natural one
# Maybe use pythorch-lightning
# The saving and loading of the checkpoint is based on
# https://web.archive.org/web/20201106104658/https://chshih2.github.io/blog/2020/10/06/Pytorch-continue-training/

DEBUG = True
from convnext_impl.convnext_osnet import build_convnext_osnet
import imagenet_data_loader.imagenet_preprocessed_data_loader as imagenet
import torch
from tqdm import tqdm
import glob
import re
import imagenet_data_loader.early_stopping as early_stopping
import gc

## Note: This imports are for debugging.
if DEBUG:
    # Requires psutil loguru nvidia-ml-py3
    from loguru import logger
    import os
    import sys
    import psutil
    import atexit
    import nvidia_smi

    # Remove the stdout logging and only log to a file. The pre-configured
    # handler is guaranteed to have the index 0.
    logger.remove(0)

    # If there is already an existing file with the same name that the file to
    # be created, then the existing file is renamed by appending the date to
    # its basename to prevent file overwriting.
    logger.add("imagenet_training.log")

    nvidia_smi.nvmlInit()
    atexit.register(nvidia_smi.nvmlShutdown)

    def memory_use():
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        available_memory = mem.available
        used_memory_percent = mem.percent
        available_swap = swap.free
        used_swap_percent = swap.percent
        mes = (
            f"The system has {available_memory} free memory (used {used_memory_percent}%),"
            + f"and {available_swap} swap (used {used_swap_percent}%)"
        )
        return mes

    def disk_use(path):
        disk_usage = psutil.disk_usage(path)
        free_space = disk_usage.free
        percent_used = disk_usage.percent
        mes = f"The disk that contains {path} has {free_space} free_space and is {percent_used}% full\n"
        return mes

    def cpu_use_detailed_nonblocking():
        """Update and/or return the per CPU list using the psutil library."""
        # The first value is bogus, as it meassures the cpu activity since the last call.
        percpu_percent = []
        for cpu_number, cputimes in enumerate(
            psutil.cpu_times_percent(interval=0.0, percpu=True)
        ):
            cpu = {
                "key": "f{cpu_number}",
                "cpu_number": cpu_number,
                "total": round(100 - cputimes.idle, 1),
                "user": cputimes.user,
                "system": cputimes.system,
                "idle": cputimes.idle,
            }
            if hasattr(cputimes, "nice"):
                cpu["nice"] = cputimes.nice
            if hasattr(cputimes, "iowait"):
                cpu["iowait"] = cputimes.iowait
            if hasattr(cputimes, "irq"):
                cpu["irq"] = cputimes.irq
            if hasattr(cputimes, "softirq"):
                cpu["softirq"] = cputimes.softirq
            if hasattr(cputimes, "steal"):
                cpu["steal"] = cputimes.steal
            if hasattr(cputimes, "guest"):
                cpu["guest"] = cputimes.guest
            if hasattr(cputimes, "guest_nice"):
                cpu["guest_nice"] = cputimes.guest_nice
            percpu_percent.append(cpu)
        return percpu_percent

    def cpu_use_detailed_blocking_for_1sec():
        """Update and/or return the per CPU list using the psutil library."""
        # Carefull this may sleep for one second. I think this has too much overhead.
        percpu_percent = []
        for cpu_number, cputimes in enumerate(
            psutil.cpu_times_percent(interval=1.0, percpu=True)
        ):
            cpu = {
                "key": "f{cpu_number}",
                "cpu_number": cpu_number,
                "total": round(100 - cputimes.idle, 1),
                "user": cputimes.user,
                "system": cputimes.system,
                "idle": cputimes.idle,
            }
            if hasattr(cputimes, "nice"):
                cpu["nice"] = cputimes.nice
            if hasattr(cputimes, "iowait"):
                cpu["iowait"] = cputimes.iowait
            if hasattr(cputimes, "irq"):
                cpu["irq"] = cputimes.irq
            if hasattr(cputimes, "softirq"):
                cpu["softirq"] = cputimes.softirq
            if hasattr(cputimes, "steal"):
                cpu["steal"] = cputimes.steal
            if hasattr(cputimes, "guest"):
                cpu["guest"] = cputimes.guest
            if hasattr(cputimes, "guest_nice"):
                cpu["guest_nice"] = cputimes.guest_nice
            percpu_percent.append(cpu)
        return percpu_percent

    def cpu_use_blocking_1sec():
        # Carefull this may sleep for one second. I think this has too much overhead.
        lst = psutil.cpu_percent(interval=1.0, percpu=True)
        usage_list = []
        for idx, elem in enumerate(lst):
            usage = f"Cpu {idx} has a usage of {elem}\n"
            usage_list.append(usage)
        res = "".join(usage_list)
        return res

    def cpu_use_nonblocking():
        # The first value is bogus, as it meassures the cpu activity since the last call.
        lst = psutil.cpu_percent(interval=0.0, percpu=True)
        usage_list = []
        for idx, elem in enumerate(lst):
            usage = f"Cpu {idx} has a usage of {elem}\n"
            usage_list.append(usage)
        res = "".join(usage_list)
        return res

    def cpu_use():
        return cpu_use_nonblocking()

    def gpu_use():
        if torch.cuda.is_available():
            deviceCount = nvidia_smi.nvmlDeviceGetCount()
            gpu_usage = []
            for i in range(deviceCount):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                usage = "Device {}: {}, Memory : ({:.2f}% free): {} (total), {} (free), {} (used) \n".format(
                    i,
                    nvidia_smi.nvmlDeviceGetName(handle),
                    100 * info.free / info.total,
                    info.total,
                    info.free,
                    info.used,
                )
                gpu_usage.append(usage)

            if torch.cuda.device_count() == 1:
                used_device = torch.cuda.current_device()
                current_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(used_device)
                current_info = nvidia_smi.nvmlDeviceGetMemoryInfo(current_handle)
                current_usage = (
                    "The GPU used is {} {}: {}, Memory : ({:.2f}% free)\n".format(
                        used_device,
                        nvidia_smi.nvmlDeviceGetName(current_handle),
                        100 * current_info.free / current_info.total,
                    )
                )
                gpu_usage.append(current_usage)
            else:
                pass

            return "".join(gpu_usage)
        else:
            return "No NVIDIA GPU avalible, using CPU."


def main():
    NUM_EPOCHS = 140
    MODEL_PATH = "convnext_imagenet.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if DEBUG:
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
    # train_loader_22k, validation_loader_22k = imagenet.create_imagenet_22k_data_loaders(
    #     imagenet_22k_loader_args
    # )

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
        logger.debug(f"Loading epoch: {latest_epoch}\n")

    best_validation_loss = None
    # Starting epoch (in case we need to restart training from a previous epoch)
    start_epoch = 0
    if latest_epoch is not None:
        torch.cuda.empty_cache()
        checkpoint = torch.load(latest_epoch)
        start_epoch = checkpoint["epoch"] + 1
        convnext_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_validation_loss = checkpoint["best_loss"]
        convnext_model = convnext_model.to(device)
        torch.cuda.empty_cache()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        torch.cuda.empty_cache()

    earlystopping = early_stopping.EarlyStopping(patience=7, verbose=True)
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
            print(
                f"In iteration: {cur_train_iter} and epoch: {epoch} of training loader"
            )

            if DEBUG:
                logger.debug("##START OF TRAINING ITERATION. \n\n\n ")
                logger.debug(
                    f"Entering iteration: {cur_train_iter} and epoch: {epoch} of training loader\n"
                )
                logger.debug(cpu_use())
                logger.debug(memory_use())
                logger.debug(gpu_use())
                logger.debug("Current disk:" + disk_use(os.getcwd()))
                logger.debug("Data disk:" + disk_use(imagenet_loader_args["data_path"]))

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
            if DEBUG:
                logger.debug(
                    f"Exiting iteration: {cur_train_iter} and epoch: {epoch} of training loader with avg_loss: {avg_training_loss}\n"
                )
                logger.debug(cpu_use())
                logger.debug(memory_use())
                logger.debug(gpu_use())
                logger.debug("Current disk:" + disk_use(os.getcwd()))
                logger.debug("Data disk:" + disk_use(imagenet_loader_args["data_path"]))
                logger.debug("##STOP OF TRAINING ITERATION. \n\n\n ")

        print(f"I reached the end of training in epoch {epoch}")
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

        # Try to free gpu_memory
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            print(f"I am going to start validation of epoch: {epoch}")
            total_validation_loss = 0.0
            for cur_validation_iter, validation_data in enumerate(validation_loader):
                print(
                    f"In iteration: {cur_validation_iter} and epoch: {epoch} of validation loader"
                )
                if DEBUG:
                    logger.debug("##START OF VALIDATION ITERATION. \n\n\n ")
                    logger.debug(
                        f"Entering iteration: {cur_validation_iter} and epoch: {epoch} of validation loader\n"
                    )
                    logger.debug(cpu_use())
                    logger.debug(memory_use())
                    logger.debug(gpu_use())
                    logger.debug("Current disk:" + disk_use(os.getcwd()))
                    logger.debug(
                        "Data disk:" + disk_use(imagenet_loader_args["data_path"])
                    )

                validation_inputs, validation_labels = validation_data
                validation_inputs = validation_inputs.to(device)
                validation_labels = validation_labels.to(device)
                validation_outputs = convnext_model(validation_inputs)
                validation_loss = loss_function(validation_outputs, validation_labels)
                total_validation_loss += validation_loss

                avg_validation_loss = total_validation_loss / (cur_validation_iter + 1)
                print(f"Validation loss = {avg_validation_loss}")

                if DEBUG:
                    logger.debug(
                        f"Exiting iteration: {cur_validation_iter} and epoch: {epoch} of validation loader\n"
                    )
                    logger.debug(cpu_use())
                    logger.debug(memory_use())
                    logger.debug(gpu_use())
                    logger.debug("Current disk:" + disk_use(os.getcwd()))
                    logger.debug(
                        "Data disk:" + disk_use(imagenet_loader_args["data_path"])
                    )
                    logger.debug("##STOP OF VALIDATION ITERATION. \n\n\n ")

            # Track best performance, and save the model's state
            if total_validation_loss < best_validation_loss:
                best_validation_loss = total_validation_loss
                # Save the final model
                torch.save(convnext_model.state_dict(), MODEL_PATH)
            print(f"I reached the end of validation in epoch {epoch}")

            # early_stopping needs the validation loss to check if it has
            # decresed, and if it has, it will make a checkpoint of the current
            # model. If the best validation value hasn't improved in some time
            # it sets the early_stop instance variable, so we know that we need
            # to stop.
            earlystopping(total_validation_loss, convnext_model)
            if earlystopping.early_stop:
                if DEBUG:
                    logger.debug(f"Early stopping in epoch {epoch}\n")
                    logger.debug(cpu_use())
                    logger.debug(memory_use())
                    logger.debug(gpu_use())
                    logger.debug("Current disk:" + disk_use(os.getcwd()))
                    logger.debug(
                        "Data disk:" + disk_use(imagenet_loader_args["data_path"])
                    )

                print(f"Early stopping in epoch {epoch}")
                break

            if DEBUG:
                logger.debug(f"Finished epoch {epoch}")
                logger.debug(cpu_use())
                logger.debug(memory_use())
                logger.debug(gpu_use())
                logger.debug("Current disk:" + disk_use(os.getcwd()))
                logger.debug("Data disk:" + disk_use(imagenet_loader_args["data_path"]))

        # Try to free gpu_memory
        gc.collect()
        torch.cuda.empty_cache()

    print(f"I have finished training")


if __name__ == "__main__":
    main()
