# -*- coding:utf-8 -*-
# author: Xinge
# @file: loss_builder.py

from enum import Enum

import torch

from utils.lovasz_losses import lovasz_softmax
from utils.regression_losses import CombinedLoss


def build(wce=True, lovasz=True, num_class=20, ignore_label=0):

    loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError

class Losses(Enum):
    MSE = 0,
    MAE = 1,
    Huber = 2,
    WeightedCombinedLoss = 3,
    DynamicCombinedLoss = 4,
    SwitchingLoss = 5,
    CustomCombinedLoss = 6,
    UndefinedLoss = 7,

def build_regression(model: str):
    loss: Losses = Losses.UndefinedLoss
    huber_names = ["huber", "smooth", "smoothl1", "huberloss", "smoothloss", "smoothl1loss",
                   "huber_loss", "smooth_loss", "smoothl1_loss", "smooth_l1_loss", "smooth_l1"]
    mse_names = ["mse", "squared", "mean_squared", "mean_squared_error"]
    mae_names = ["mae", "absolute", "mean_absolute", "mean_absolute_error"]
    weighted_combined_names = ["weighted_combined", "weighted", "w", "wc"]
    dynamic_combined_names = ["dynamic_combined", "dynamic_weighted_combined", "dynamic_weighted", "learnable_weighted", "dc", "dwc", "dw", "lw", "lc"]
    switching_names = ["switching", "switching_loss", "s", "sl"]
    custom_combined_names = ["custom_combined", "custom", "cc", "c"]

    names_dict = {
        Losses.MSE: mse_names,
        Losses.MAE: mae_names,
        Losses.Huber: huber_names,
        Losses.WeightedCombinedLoss: weighted_combined_names,
        Losses.DynamicCombinedLoss: dynamic_combined_names,
        Losses.SwitchingLoss: switching_names,
        Losses.CustomCombinedLoss: custom_combined_names,
    }

    for key, value in names_dict.items():
        if model.lower() in value:
            loss = key

    if loss == Losses.MSE:
        return loss, torch.nn.MSELoss()
    elif loss == Losses.MAE:
        return loss, torch.nn.L1Loss()
    elif loss == Losses.Huber:
        return loss, torch.nn.SmoothL1Loss()
    elif loss == Losses.WeightedCombinedLoss:
        return loss, CombinedLoss().weighted_combined_loss
    elif loss == Losses.DynamicCombinedLoss:
        return loss, CombinedLoss().dynamic_combined_loss
    elif loss == Losses.SwitchingLoss:
        return loss, CombinedLoss().switching_loss
    elif loss == Losses.CustomCombinedLoss:
        return loss, CombinedLoss().custom_combined_loss
    elif loss == Losses.UndefinedLoss:
        raise NotImplementedError
    else:
        raise NotImplementedError