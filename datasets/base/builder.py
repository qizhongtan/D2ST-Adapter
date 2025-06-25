#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

""" Builder for the dataloader."""

import torch
import utils.misc as misc
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from datasets.base.few_shot_dataset import Few_shot


def get_sampler(cfg, dataset, split, shuffle):
    if misc.get_num_gpus(cfg) > 1:
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None


def build_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (Configs): global config object. details in utils/config.py
        split (str): the split of the data loader. Options include `train`,
            `val`, `test`, and `submission`.
    Returns:
        loader object.
    """
    assert split in ["train", "val", "test", "submission"]
    if split in ["train"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    else:
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(cfg, split)

    # Create a sampler for multi-process training
    sampler = get_sampler(cfg, dataset, split, shuffle)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    sampler = loader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(cur_epoch)


def build_dataset(cfg, split):
    return Few_shot(cfg, split)
