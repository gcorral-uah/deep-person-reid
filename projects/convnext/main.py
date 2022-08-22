import os
import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger,
    check_isfile,
    set_random_seed,
    collect_env_info,
    resume_from_checkpoint,
    compute_model_complexity,
)

from default_config import (
    imagedata_kwargs,
    optimizer_kwargs,
    engine_run_kwargs,
    lr_scheduler_kwargs,
)
from default_config import get_default_config


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        nargs="+",
        help="source datasets (delimited by space)",
    )
    parser.add_argument(
        "-t",
        "--targets",
        type=str,
        nargs="+",
        help="target datasets (delimited by space)",
    )
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
    parser.add_argument("--root", type=str, default="", help="path to data root")
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default="",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    if cfg.use_gpu and args.gpu_devices:
        # if gpu_devices is not specified, all available gpus will be used
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    log_name = "test.log" if cfg.test.evaluate else "train.log"
    log_name += time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print("Show configuration\n{}\n".format(cfg))
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))

    print("Building model: {}".format(cfg.model.name))
    # FIXME(Gonzalo): We can't use torchreid.models.build_model with a custom model.
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print("Model complexity: params={:,} flops={:,}".format(num_params, flops))

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer
        )

    print("Building NAS engine")
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        use_gpu=cfg.use_gpu,
        label_smooth=cfg.loss.softmax.label_smooth,
    )
    engine.run(**engine_run_kwargs(cfg))


if __name__ == "__main__":
    main()
