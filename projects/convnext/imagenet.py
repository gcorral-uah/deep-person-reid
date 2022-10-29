### TODO:
# We should replace the library code from torch-reid with a natural one
# Maybe use pythorch-lightning
# We have another option using a data_manager from torchreid to do the work
# - The data loaders should work
# - The engine won't work as we haven't wrapped it in a data-manager

from convnext_impl.build_models import build_convnext
import imagenet_data_loader.imagenet_preprocessed_data_loader as imagenet


def main():
    imagenet_args = {}
    rain_loader, val_loader = imagenet.create_imagenet_data_loaders(imagenet_args)
    imagenet_dataset_parameters = imagenet.imagenet_data()

    print("Building model: {}")
    is_pretrained = False
    model = build_convnext(
        num_classes=imagenet_dataset_parameters["num_training_classes"],
        pretrained=is_pretrained,
    )


if __name__ == "__main__":
    main()
