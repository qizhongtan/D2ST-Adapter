#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.meters import TrainMeter, ValMeter
from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, val_meter, val_loader):
    # Enable train mode.
    model.train()
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    train_meter.iter_tic()

    for cur_iter, task_dict in enumerate(train_loader):
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', 'real_support_labels']'''
        if cur_iter >= cfg.TRAIN.NUM_TRAIN_TASKS:
            break
        # Save a checkpoint.
        cur_epoch = cur_iter // cfg.SOLVER.STEPS_ITER
        if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0:
            model_bucket = None
            cur_epoch_save = cur_iter // cfg.TRAIN.VAL_FRE_ITER
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg, model_bucket)
            val_meter.set_model_ema_enabled(False)
            eval_epoch(val_loader, model, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg)
            model.train()

        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(float(cur_iter) / cfg.SOLVER.STEPS_ITER, cfg)
        optim.set_lr(optimizer, lr)

        model_dict = model(task_dict)
        target_logits = model_dict['logits']

        if hasattr(cfg.TRAIN, "USE_CLASSIFICATION_VALUE"):
            loss = (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE *
                    F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) / cfg.TRAIN.BATCH_SIZE
        else:
            loss = F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) / cfg.TRAIN.BATCH_SIZE

        # check Nan Loss.
        if math.isnan(loss):
            loss.backward(retain_graph=False)
            optimizer.zero_grad()
            continue
        loss.backward(retain_graph=False)

        # optimize
        if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
            optimizer.step()
            optimizer.zero_grad()

        # Compute the errors.
        preds = target_logits
        num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
        top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item())

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(top1_err, top5_err, loss, lr, train_loader.batch_size * max(misc.get_num_gpus(cfg), 1))
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch + cfg.TRAIN.NUM_FOLDS - 1)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= cfg.TRAIN.NUM_TEST_TASKS:
            break
        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        # preds, logits = model(inputs)
        model_dict = model(task_dict)

        target_logits = model_dict['logits']
        loss = F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) / cfg.TRAIN.BATCH_SIZE

        # Compute the errors.
        labels = task_dict['target_labels']
        preds = target_logits
        num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
        top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item())
        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(top1_err, top5_err, val_loader.batch_size * max(misc.get_num_gpus(cfg), 1))
        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def train_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, model_bucket)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.SOLVER.MAX_EPOCH - start_epoch) % cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    cur_epoch = 0
    shuffle_dataset(train_loader, cur_epoch)

    # freeze some parameters
    for name, param in model.named_parameters():
        if 'class_embedding' not in name and 'temporal_embedding' not in name and 'Adapter' not in name and 'ln_post' not in name and 'classification_layer' not in name:
            param.requires_grad = False

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        for name, param in model.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, val_meter, val_loader)
