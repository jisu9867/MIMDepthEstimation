import argparse
from pprint import pprint

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                                 count_parameters)
from seg.models.builder import EncoderDecoder as segmodel
from seg.utils.pyt_utils import load_model
from seg.config import config
from resnet_encoder import Custom_resnet
from segformer_encoder import Custom_segformer
from transformers import AutoImageProcessor, ResNetForImageClassification, SegformerForImageClassification, SegformerForSemanticSegmentation

import torch.nn as nn
from zoedepth.utils.misc import colorize
import os
from PIL import Image
import torchvision.transforms as transforms


@torch.no_grad()
def evaluate(test_loader, model, config, round_vals=True, round_precision=3):
    model.eval()
    metrics_1 = RunningAverageDict()
    metrics_2 = RunningAverageDict()
    zoe_list, ait_list = [], []
    # zoe_hist = torch.zeros(100).cuda()
    # ait_hist = torch.zeros(100).cuda()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth, ait, zoe = sample['image'], sample['depth'], sample['ait'], sample['zoe']
        image, depth, ait, zoe = image.cuda(), depth.cuda(), ait.cuda(), zoe.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        # out = model(image, hha)
        out = model(image).logits
        # out = nn.functional.interpolate(out, depth.shape[-2:], mode='bilinear', align_corners=False)
        # print("out:", out)
        softmax = nn.Softmax(dim=1)
        proba_out = softmax(out)
        print("proba_out:", proba_out)
        zoe_list.append(proba_out[0][0].cpu())
        ait_list.append(proba_out[0][1].cpu())


        pred1 = zoe*proba_out[0][0] + ait*proba_out[0][1]
        mean = (zoe+ait) / 2


        pred2 = zoe*(mask==0) + ait*(mask==1)
        # mean_mask = (zoe+ait)/2


        # print(depth.shape, pred.shape)
        metrics_1.update(compute_metrics(depth, pred1, config=config))
        metrics_2.update(compute_metrics(depth, pred2, config=config))

    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(zoe_list, color = 'red', bins = 100, range = [0,1], label='ZoeDepth')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(ait_list, color = 'blue', bins = 100, range = [0,1], label='AiT')
    plt.xlabel('probability')
    plt.legend()
    plt.savefig(f'/media/gwlee/Data/gwlee/shortcuts/zoe_ait_histogram_epoch3.png')
    plt.close()

    if round_vals:
        def r(m):
            return round(m, round_precision)
    else:
        def r(m):
            return m
    metrics1 = {k: r(v) for k, v in metrics_1.get_value().items()}
    metrics2 = {k: r(v) for k, v in metrics_2.get_value().items()}
    return metrics1, metrics2


def main(config_zoe):
    # network = segmodel(cfg=config, criterion=None, norm_layer = nn.BatchNorm2d)
    pretrained_network = SegformerForImageClassification.from_pretrained("nvidia/mit-b4")
    network = Custom_segformer(pretrained_network)
    model = load_model(network, "/media/gwlee/Data/gwlee/segformer/log_NYUDepthv2_mit_b4/checkpoint/epoch-3.pth")
    test_loader = DepthDataLoader(config_zoe, 'online_eval').data
    model = model.cuda()
    metrics1, metrics2 = evaluate(test_loader, model, config_zoe)
    print(f"{colors.fg.green}")
    print(metrics1)
    print(metrics2)
    print(f"{colors.reset}")
    # metrics['#params'] = f"{round(count_parameters(, include_all=True)/1e6, 2)}M"
    return metrics1, metrics2


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):
    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config_zoe = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config_zoe)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics1, metrics2 = main(config_zoe)
    return metrics1, metrics2
