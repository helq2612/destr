# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# this is used to draw curves for train and test.

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from datasets.coco_eval import CocoEvaluator
from models.conditional_detr import PostProcess

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default="data/coco/", type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='trained_weights',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    return parser


def main(args):

    print(args)

    device = torch.device(args.device)

    log_paths = [("offitial_6_1.0_1.0_6_1.0_1.0_.txt", 50), 
                 ("cdetr_6_0.5_0.5_6_0.5_0.5_.txt", 50), 
                 ("cdetr_6_0.5_0.5_6_1.0_0.5_.txt", 50),
                 ("svct_6_0.5_0.5_6_1.0_0.5_.txt", 50)]
    x_axis = np.arange(50)
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    num_col = len(log_paths)
    fig, axes = plt.subplots(4,2, figsize=(12,5*4))
    
    loss_types = ["train_loss", "test_loss", "train_loss_bbox", "test_loss_bbox", "train_loss_ce",  "test_loss_ce", "train_loss_giou",  "test_loss_giou"]

    for (log_path, num_epochs) in log_paths:
        losses = get_from_log("log_files/"+log_path, num_epochs)
        log_labels = log_path.split("_")
        log_label = "{}_{}({}+{})_{}({}+{})".format(log_labels[0], log_labels[1], log_labels[2], log_labels[3], log_labels[4], log_labels[5], log_labels[6])
        for idx in range(8):
            row, col = int(idx/2), int(idx%2)
            len_res = len(losses[loss_types[idx]])
            axes[row, col].plot(x_axis[:len_res], losses[loss_types[idx]], label = log_label)
            

    for idx in range(8):
        row, col = int(idx/2), int(idx%2)
        axes[row, col].title.set_text(loss_types[idx])
        axes[row, col].legend(loc="upper right")
        
    # plt.title("Training losses")
    plt.savefig('log_files/foo.png')


def get_from_log(file_name, num_epochs):
    train_losses, test_losses = [], []
    train_loss_bbox, train_loss_ce, train_loss_giou = [], [], []
    test_loss_bbox,  test_loss_ce,  test_loss_giou = [], [], []
    res = defaultdict(lambda: [])
    with open (file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[-num_epochs:]:
            epoch = json.loads(line)
            for key in epoch:
                res[key].append(epoch[key])
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
