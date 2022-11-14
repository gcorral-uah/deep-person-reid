# --------------------------------------------------------
# ImageNet-21K Pretraining for The Masses
# Copyright 2021 Alibaba MIIL (c)
# Licensed under MIT License [see the LICENSE file for details]
# Written by Tal Ridnik
# --------------------------------------------------------
import os
import argparse
import time
import torch
import torch.optim
from torch.optim import lr_scheduler

from imagenet_preprocessed_data_loader import create_imagenet_data_loaders
from src_files.helper_functions.general_helper_functions import (
    accuracy,
    AverageMeter,
    silence_PIL_warnings,
)
from src_files.models import create_model
from src_files.loss_functions.losses import CrossEntropyLS
from torch.cuda.amp import GradScaler, autocast
from src_files.optimizers.create_optimizer import create_optimizer

parser = argparse.ArgumentParser(
    description="PyTorch ImageNet21K Single-label Training"
)
parser.add_argument("--data_path", type=str)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--model_name", default="tresnet_m")
parser.add_argument("--model_path", default="./tresnet_m.pth", type=str)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--image_size", default=224, type=int)
parser.add_argument("--num_classes", default=11221, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=80, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--label_smooth", default=0.2, type=float)


def num_distrib():
    return int(os.environ.get("WORLD_SIZE", 0))


def main():
    # arguments
    args = parser.parse_args()

    # EXIF warning silent
    silence_PIL_warnings()

    # Setup model
    model = create_model(args).cuda()

    # create optimizer
    optimizer = create_optimizer(model, args)

    # Data loading
    train_loader, val_loader = create_imagenet_data_loaders(args)

    # Actuall Training
    train_21k(model, train_loader, val_loader, optimizer, args)


def train_21k(model, train_loader, val_loader, optimizer, args):
    # set loss
    loss_fn = CrossEntropyLS(args.label_smooth)

    # set scheduler
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        cycle_momentum=False,
        div_factor=20,
    )

    # set scalaer
    scaler = GradScaler()

    # training loop
    for epoch in range(args.epochs):
        # train epoch
        print("\nEpoch {}".format(epoch))
        epoch_start_time = time.time()
        for i, (input, target) in enumerate(train_loader):
            with autocast():  # mixed precision
                output = model(input)
                loss = loss_fn(output, target)  # note - loss also in fp16
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        print(
            "\nFinished Epoch, Training Rate: {:.1f} [img/sec]".format(
                len(train_loader) * args.batch_size / epoch_time * max(1, 1)
            )
        )

        # validation epoch
        validate_21k(val_loader, model)


def validate_21k(val_loader, model):
    print("starting validation")
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # mixed precision
            with autocast():
                logits = model(input).float()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

    print("Validation results:")
    print("Acc_Top1 [%] {:.2f},  Acc_Top5 [%] {:.2f} ".format(top1.avg, top5.avg))
    model.train()


if __name__ == "__main__":
    main()
