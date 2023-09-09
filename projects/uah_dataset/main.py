import argparse
import os.path as osp
import sys
import time

import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    check_isfile,
    collect_env_info,
    compute_model_complexity,
    load_pretrained_weights,
    Logger,
    resume_from_checkpoint,
    set_random_seed,
)

from uah import UAHDataset

from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = "resnet50"
    cfg.model.pretrained = (
        True  # automatically load pretrained model weights if available
    )
    cfg.model.load_weights = ""  # path to model weights
    cfg.model.resume = ""  # path to checkpoint for resume training

    # data
    cfg.data = CN()
    cfg.data.type = "image"
    cfg.data.root = "reid-data"
    cfg.data.sources = ["market1501"]
    cfg.data.targets = ["market1501"]
    cfg.data.workers = 4  # number of data loading workers
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ["random_flip"]  # data augmentation
    cfg.data.k_tfm = (
        1  # number of times to apply augmentation to an image independently
    )
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = "log"  # path to save log
    cfg.data.load_train_targets = False  # load training set from target dataset

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = (
        False  # add 500k distractors to the gallery set for market1501
    )
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = (
        False  # use labeled images, if False, use detected images
    )
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation

    cfg.uah = CN()
    cfg.uah.crop_images = False # Crop the images to only have one person per image.

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = "RandomSampler"  # sampler for source train loader
    cfg.sampler.train_sampler_t = "RandomSampler"  # sampler for target train loader
    cfg.sampler.num_instances = (
        4  # number of instances per identity for RandomIdentitySampler
    )
    cfg.sampler.num_cams = (
        1  # number of cameras to sample in a batch (for RandomDomainSampler)
    )
    cfg.sampler.num_datasets = (
        1  # number of datasets to sample in a batch (for RandomDatasetSampler)
    )

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15  # number of images to sample in a tracklet
    cfg.video.sample_method = "evenly"  # how to sample images from a tracklet
    cfg.video.pooling_method = "avg"  # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = "adam"
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 60
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 32
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = [
        "classifier"
    ]  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = ["classifier"]  # newly added layers with default lr
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = "single_step"
    cfg.train.stepsize = [20]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.print_freq = 20  # print frequency
    cfg.train.seed = 1  # random seed

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.0  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = "softmax"
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.0  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.0  # weight to balance cross entropy loss

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = "euclidean"  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = (
        False  # normalize feature vectors before computing distance
    )
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = (
        -1
    )  # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = (
        False  # visualize ranked results (only available when cfg.test.evaluate=True)
    )
    cfg.test.visrank_topk = 10  # top-k ranks to visualize

    return cfg


def imagedata_kwargs(cfg):
    return {
        "root": cfg.data.root,
        "sources": cfg.data.sources,
        "targets": cfg.data.targets,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms": cfg.data.transforms,
        "k_tfm": cfg.data.k_tfm,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "split_id": cfg.data.split_id,
        "combineall": cfg.data.combineall,
        "load_train_targets": cfg.data.load_train_targets,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "num_cams": cfg.sampler.num_cams,
        "num_datasets": cfg.sampler.num_datasets,
        "train_sampler": cfg.sampler.train_sampler,
        "train_sampler_t": cfg.sampler.train_sampler_t,
        # image dataset specific
        "cuhk03_labeled": cfg.cuhk03.labeled_images,
        "cuhk03_classic_split": cfg.cuhk03.classic_split,
        "market1501_500k": cfg.market1501.use_500k_distractors,
        "uah_dataset_crop_images": cfg.uah.crop_images,
    }


def videodata_kwargs(cfg):
    return {
        "root": cfg.data.root,
        "sources": cfg.data.sources,
        "targets": cfg.data.targets,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms": cfg.data.transforms,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "split_id": cfg.data.split_id,
        "combineall": cfg.data.combineall,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "num_cams": cfg.sampler.num_cams,
        "num_datasets": cfg.sampler.num_datasets,
        "train_sampler": cfg.sampler.train_sampler,
        # video dataset specific
        "seq_len": cfg.video.seq_len,
        "sample_method": cfg.video.sample_method,
    }


def optimizer_kwargs(cfg):
    return {
        "optim": cfg.train.optim,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "momentum": cfg.sgd.momentum,
        "sgd_dampening": cfg.sgd.dampening,
        "sgd_nesterov": cfg.sgd.nesterov,
        "rmsprop_alpha": cfg.rmsprop.alpha,
        "adam_beta1": cfg.adam.beta1,
        "adam_beta2": cfg.adam.beta2,
        "staged_lr": cfg.train.staged_lr,
        "new_layers": cfg.train.new_layers,
        "base_lr_mult": cfg.train.base_lr_mult,
    }


def lr_scheduler_kwargs(cfg):
    return {
        "lr_scheduler": cfg.train.lr_scheduler,
        "stepsize": cfg.train.stepsize,
        "gamma": cfg.train.gamma,
        "max_epoch": cfg.train.max_epoch,
    }


def engine_run_kwargs(cfg):
    return {
        "save_dir": cfg.data.save_dir,
        "max_epoch": cfg.train.max_epoch,
        "start_epoch": cfg.train.start_epoch,
        "fixbase_epoch": cfg.train.fixbase_epoch,
        "open_layers": cfg.train.open_layers,
        "start_eval": cfg.test.start_eval,
        "eval_freq": cfg.test.eval_freq,
        "test_only": cfg.test.evaluate,
        "print_freq": cfg.train.print_freq,
        "dist_metric": cfg.test.dist_metric,
        "normalize_feature": cfg.test.normalize_feature,
        "visrank": cfg.test.visrank,
        "visrank_topk": cfg.test.visrank_topk,
        "use_metric_cuhk03": cfg.cuhk03.use_metric_cuhk03,
        "ranks": cfg.test.ranks,
        "rerank": cfg.test.rerank,
    }


def build_datamanager(cfg):
    if cfg.data.type == "image":
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == "image":
        if cfg.loss.name == "softmax":
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
            )

    else:
        if cfg.loss.name == "softmax":
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method,
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == "triplet" and cfg.loss.triplet.weight_x == 0:
        assert (
            cfg.train.fixbase_epoch == 0
        ), "The output of classifier is not included in the computational graph"


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
    check_cfg(cfg)

    log_name = "test.log" if cfg.test.evaluate else "train.log"
    log_name += time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print("Show configuration\n{}\n".format(cfg))
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print("Building model: {}".format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu,
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print("Model complexity: params={:,} flops={:,}".format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print("Building {}-engine for {}-reid".format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == "__main__":
    torchreid.data.register_image_dataset("UAHDataset", UAHDataset)
    main()
