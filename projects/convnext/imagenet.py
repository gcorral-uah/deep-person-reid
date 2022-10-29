### TODO:
# We should replace the library code from torch-reid with a natural one
# Maybe use pythorch-lightning
# We have another option using a data_manager from torchreid to do the work
# - The data loaders should work
# - The engine won't work as we haven't wrapped it in a data-manager

from convnext_impl.build_models import build_convnext
import imagenet_data_loader.imagenet_preprocessed_data_loader as imagenet


def main():
    imagenet_loader_args = {
        # Path to the data (with folders imagenet21k_train and imagenet21k_val)
        "data_path": "",
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


if __name__ == "__main__":
    main()
