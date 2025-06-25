#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

import torch
import torch.nn as nn
import utils.logging as logging
from models.base.models import BaseVideoModel

def build_model(cfg, gpu_id=None):
    model = BaseVideoModel(cfg)

    if torch.cuda.is_available():
        assert (cfg.NUM_GPUS <= torch.cuda.device_count()), "Cannot use more GPU devices than available"
    else:
        assert (cfg.NUM_GPUS == 0), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        model = model.cuda(device=cur_device)

    try:
        # convert batchnorm to be synchronized across 
        # different GPUs if needed
        sync_bn = cfg.BN.SYNC_BN
        if sync_bn == True and cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    except:
        sync_bn = None

    if cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
            # module=model, device_ids=[cur_device], output_device=cur_device
        )

    return model
