#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

import torch.nn as nn
from models.base.adapter import HEAD_REGISTRY


class BaseVideoModel(nn.Module):
    def __init__(self, cfg):
        super(BaseVideoModel, self).__init__()
        self.cfg = cfg
        self.model = HEAD_REGISTRY.get(cfg.ADAPTER.NAME)(cfg=cfg)

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        self.training = mode
        super(BaseVideoModel, self).train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)

        return self
