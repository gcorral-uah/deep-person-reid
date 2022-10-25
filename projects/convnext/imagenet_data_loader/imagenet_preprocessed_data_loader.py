import os

import torch
from randagument import RandAugment
from torchvision import transforms
from torchvision.datasets import ImageFolder

from augmentations import CutoutPIL


def imagenet_data():
    return {"num_training_classes": 10450, "num_test_classes": 10450}


def create_imagenet_data_loaders(args):
    data_path_train = os.path.join(args.data_path, "imagenet21k_train")
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ]
    )

    data_path_val = os.path.join(args.data_path, "imagenet21k_val")
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageFolder(data_path_train, transform=train_transform)
    val_dataset = ImageFolder(data_path_val, transform=val_transform)
    print("length train dataset: {}".format(len(train_dataset)))
    print("length val dataset: {}".format(len(val_dataset)))

    sampler_train = None
    sampler_val = None

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler_train is None,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler_train,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=sampler_val,
    )

    train_loader = PrefetchLoader(train_loader)
    val_loader = PrefetchLoader(val_loader)
    return train_loader, val_loader


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        first = True
        for batch in self.loader:
            with torch.cuda.stream(self.stream):  # stream - parallel
                self.next_input = batch[0].cuda(
                    non_blocking=True
                )  # note - (0-1) normalization in .ToTensor()
                self.next_target = batch[1].cuda(non_blocking=True)

            if not first:
                yield input, target  # prev
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target

            # Ensures that the tensor memory is not reused for another tensor until all current work queued on stream
            # are complete.
            input.record_stream(torch.cuda.current_stream())
            target.record_stream(torch.cuda.current_stream())

        # final batch
        yield input, target

        # cleaning at the end of the epoch
        del self.next_input
        del self.next_target
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    def set_epoch(self, epoch):
        self.loader.sampler.set_epoch(epoch)
