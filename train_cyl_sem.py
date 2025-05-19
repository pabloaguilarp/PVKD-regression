# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
from typing import List

import numpy as np
import torch
import torch.optim as optim
from numba.scripts.generate_lower_listing import description
from torch import nn
from tqdm import tqdm
import re

from builder.loss_builder import Losses
from utils.metric_util import *
from utils.utils import fast_occupancy, voxel_to_point_wise, get_metrics, TensorboardHelper, compute_classwise_metrics
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device(args.device)

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0 and epoch > 0:
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten,
                                                  val_label_tensor.shape[0])  #val_batch_size)
                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                              ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten=train_pt_fea_ten, train_vox_ten=train_vox_ten,
                               batch_size=point_label_tensor.shape[0])
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(
                outputs, point_label_tensor)

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
        pbar.close()
        epoch += 1


def main_regression(args):
    pytorch_device = torch.device(args.device)

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    embedding_dim = model_config['embedding_dim']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    model_save_dir = os.path.dirname(model_save_path)

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    unique_label_str.insert(0, 'unlabeled')

    # Override params
    model_config['loss_func'] = args.loss
    train_hypers["learning_rate"] = args.learning_rate
    lr_num = re.sub(r'\D', '', str(train_hypers["learning_rate"]))
    model_name = f"{model_config['loss_func']}_{lr_num}"
    if args.use_labels:
        model_name = f"{model_name}_label"
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pt")
    train_hypers['model_save_path'] = model_save_path
    train_hypers['tensorboard_log_dir'] = os.path.join(train_hypers['tensorboard_log_dir'], f"{model_name}")
    if args.tensorboard_path != '':
        train_hypers['tensorboard_log_dir'] = args.tensorboard_path

    print("Overridden parameters:")
    print(f" - Loss function: {model_config['loss_func']}")
    print(f" - Learning rate: {train_hypers['learning_rate']}")
    print(f" - Output path: {model_save_path}")
    print(f" - Tensorboard log path: {train_hypers['tensorboard_log_dir']}")
    print("\n")

    use_labels = args.use_labels
    if use_labels:
        print("Using labels as additional input")

    ################ Initialize loss ################
    loss_enum, loss_func = loss_builder.build_regression(model_config['loss_func'])
    print(f"Using loss function: {type(loss_func).__name__}")

    # default weights tensor for combined weighted loss function (non-learnable)
    weights_tensor = torch.ones(num_class) * 0.5
    # default weights tensor for dynamic weighted loss function (learnable)
    dynamic_weights_alpha = nn.Parameter(torch.ones(num_class) * 0.5)
    # preferences dict for custom combined loss function
    label_preferences = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 0,
        15: 1,
        16: 1,
        17: 1,
        18: 0,
        19: 0,
    }

    def compute_loss(f_pred_flat, f_data_flat, f_labels_array_flat):
        if loss_enum in [Losses.MAE, Losses.MSE, Losses.Huber]:  # Class-agnostic loss functions
            l_loss = loss_func(f_pred_flat, f_data_flat)
        elif loss_enum == Losses.WeightedCombinedLoss:  # Class-aware combined loss functions
            l_loss = loss_func(f_pred_flat, f_data_flat, f_labels_array_flat, weights_tensor)
        elif loss_enum == Losses.DynamicCombinedLoss:
            l_loss = loss_func(f_pred_flat, f_data_flat, f_labels_array_flat, dynamic_weights_alpha)
        elif loss_enum == Losses.SwitchingLoss:
            l_loss = loss_func(f_pred_flat, f_data_flat, f_labels_array_flat, threshold=0.5)
        elif loss_enum == Losses.CustomCombinedLoss:
            l_loss = loss_func(f_pred_flat, f_data_flat, f_labels_array_flat, label_preferences)
        else:
            l_loss = loss_func(f_pred_flat, f_data_flat)
        return l_loss

    if args.device == 'cpu':
        train_dataloader_config['num_workers'] = 1
        val_dataloader_config['num_workers'] = 1

    train_dataset_loader, val_dataset_loader = data_builder.build_regression(dataset_config,
                                                                             train_dataloader_config,
                                                                             val_dataloader_config,
                                                                             grid_size=grid_size)

    # Set up tensorboard
    writer = TensorboardHelper(train_hypers['use_tensorboard'], train_hypers['tensorboard_log_dir'],
                               train_hypers['tensorboard_comment'], args.use_labels, model_config['loss_func'],
                               train_hypers['learning_rate'])

    my_model = model_builder.build(model_config, use_labels, embedding_dim)
    if args.load_model:
        # Override load path with save path
        model_load_path = model_save_path
        # Check if exists
        if os.path.exists(model_load_path):
            print(f"Loading model from {model_load_path}")
            my_model = load_checkpoint(model_load_path, my_model)
        else:
            print(f"Model path {model_load_path} does not exist. This is the first training execution or no .pt file exists.")
    else:
        print(f"Loading model option is disabled. If file exists, it will be overwritten!")

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])
    if loss_enum == Losses.DynamicCombinedLoss:
        optimizer.add_param_group({'params': [dynamic_weights_alpha]})

    # training
    epoch = 0
    best_val = {
        "mae": np.inf,
        "mse": np.inf,
        "rmse": np.inf,
    }

    # Init metrics engine
    metrics_engine = get_metrics('numpy')   # Numpy apparently is much faster than torch

    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    def prepare_data(grid: List[torch.Tensor], data_tuple: tuple):
        """
        Generate point-wise data tensors from grid

        grid: List of tensor containing coordinates in [output_shape] for each point.
              Shape of each tensor of the list: [N, 3]
        data_tuple: Tuple of lists of tensors (or tensors) containing the data to be
                    prepared. If it is a list of tensors, the shape of each tensor
                    must be [output_shape]. If it is a tensor, its shape must be
                    [batch_size, output_shape]. The lists of tensors and the first
                    dimension must be the same.
        """
        out_tuple = ()
        for v in data_tuple:
            out = []
            assert len(v) == len(grid)
            for i in range(len(grid)):
                pi = fast_occupancy(grid[i], v[i])
                out.append(pi)
            out_tuple += (out,)
        return out_tuple

    print("Starting training loop")
    while epoch < train_hypers['max_num_epochs']:
        print(f"epoch {epoch}")

        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)

        for i_iter, (_, processed_voxel_label, _, voxel_intensity, train_grid, train_pt_label, train_pt_fea) in enumerate(train_dataset_loader):
            print(f"epoch {epoch}, iter {i_iter}")

            if global_iter % check_iter == 0 and epoch > 0:
                my_model.eval()
                metrics_engine.clear()
                val_loss_list = []
                cw_metrics_iter = [[[] for _ in range(5)] for _ in range(num_class)]
                with torch.no_grad():
                    for i_iter_val, (_, val_processed_voxel_label, val_point_intensity, val_voxel_intensity, val_grid, val_pt_label, val_pt_fea) in enumerate(
                            val_dataset_loader):
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_voxel_intensity_tensor = val_voxel_intensity.type(torch.FloatTensor).to(pytorch_device)
                        val_pt_label_ten = [torch.from_numpy(i).type(torch.IntTensor).to(pytorch_device) for i in val_pt_label]

                        if use_labels:
                            predict_intensity = my_model(train_pt_fea_ten=val_pt_fea_ten, train_vox_ten=val_grid_ten,
                                                         batch_size=val_voxel_intensity_tensor.shape[0],
                                                         train_pt_label_ten=val_pt_label_ten, num_class=num_class)
                        else:
                            predict_intensity = my_model(val_pt_fea_ten, val_grid_ten, val_voxel_intensity_tensor.shape[0])

                        (predictions, data, labels_array) = prepare_data(val_grid_ten, (predict_intensity.squeeze(dim=1),
                                                                                        val_voxel_intensity_tensor, val_processed_voxel_label))
                        pred_flat = torch.cat(predictions, dim=0)
                        data_flat = torch.cat(data, dim=0)
                        labels_array_flat = torch.cat(labels_array, dim=0)

                        # tensorboard
                        writer.add_histograms_if(i_iter_val == 0, [
                            "Values/val_predictions_voxel_wise", "Values/val_intensities_voxel_wise"
                        ], [pred_flat, data_flat], global_step=epoch)

                        # loss = loss_func(pred_flat, data_flat)
                        loss = compute_loss(pred_flat, data_flat, labels_array_flat)

                        # Generic metric calculation
                        metrics_engine.compute_metrics(data[0].cpu().numpy(), predictions[0].cpu().numpy())   # Optimized method for metrics calculation
                        # Class-wise metric calculation
                        cw_metrics = compute_classwise_metrics(data[0].cpu().numpy(), predictions[0].cpu().numpy(), labels_array[0].cpu().numpy())
                        for cls, dct in cw_metrics.items():
                            """
                            "mse": mse,   0
                            "mae": mae,   1
                            "rmse": rmse, 2
                            "r2": r2,     3
                            "mape": mape  4
                            """
                            cw_metrics_iter[cls][0].append(dct["mse"])
                            cw_metrics_iter[cls][1].append(dct["mae"])
                            cw_metrics_iter[cls][2].append(dct["rmse"])
                            cw_metrics_iter[cls][3].append(dct["r2"])
                            cw_metrics_iter[cls][4].append(dct["mape"])

                        val_loss_list.append(loss.detach().cpu().numpy())
                my_model.train()

                print("Validation metrics:")
                for key, val in metrics_engine.metrics_dict.items():
                    print(f"{key}: {np.mean(val):.4f}")
                    writer.add_scalar(f"Metrics/{key}", np.mean(val), global_step=epoch)
                print("Class-wise metrics:")
                for cls, dta in enumerate(cw_metrics_iter):
                    print(f" Class {cls} ({unique_label_str[cls]}):")
                    print(f"  mse: {np.mean(dta[0]):.4f}")
                    print(f"  mae: {np.mean(dta[1]):.4f}")
                    print(f"  rmse: {np.mean(dta[2]):.4f}")
                    print(f"  r2: {np.mean(dta[3]):.4f}")
                    print(f"  mape: {np.mean(dta[4]):.4f}")
                    writer.add_scalar(f"Metrics_cw/class_{cls}_{unique_label_str[cls]}/mse", np.mean(dta[0]), global_step=epoch)
                    writer.add_scalar(f"Metrics_cw/class_{cls}_{unique_label_str[cls]}/mae", np.mean(dta[1]), global_step=epoch)
                    writer.add_scalar(f"Metrics_cw/class_{cls}_{unique_label_str[cls]}/rmse", np.mean(dta[2]), global_step=epoch)
                    writer.add_scalar(f"Metrics_cw/class_{cls}_{unique_label_str[cls]}/r2", np.mean(dta[3]), global_step=epoch)
                    writer.add_scalar(f"Metrics_cw/class_{cls}_{unique_label_str[cls]}/mape", np.mean(dta[4]), global_step=epoch)
                del val_point_intensity, val_voxel_intensity, val_pt_fea, val_grid
                print(f"Current val loss is {np.mean(val_loss_list):.6f}")
                writer.add_scalar("Loss/mean_loss_list", np.mean(val_loss_list), global_step=epoch)
                writer.flush()

                def check_and_save(k: str):
                    if best_val[k] > np.mean(metrics_engine.metrics_dict[k]):
                        best_val[k] = np.mean(metrics_engine.metrics_dict[k])
                        return True
                    return False

                def check_metrics_and_save(metrs: [str]):
                    save = []
                    for metr in metrs:
                        save.append(check_and_save(metr))
                    if any(save):
                        os.makedirs(model_save_dir, exist_ok=True)
                        torch.save(my_model.state_dict(), model_save_path)
                        reasons = []
                        for string, bools in zip(metrs, save):
                            if bools:
                                reasons.append(string)
                        print(f"Model saved at {model_save_path}. Reasons: {reasons}")

                # Check only MAE, MSE and RMSE
                check_metrics_and_save(["mae", "mse", "rmse"])

                metrics_engine.clear()

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            train_pt_label_ten = [torch.from_numpy(i).type(torch.IntTensor).to(pytorch_device) for i in train_pt_label]

            voxel_intensity_tensor = voxel_intensity.type(torch.FloatTensor).to(pytorch_device)

            # forward + backward + optimize
            if use_labels:
                outputs = my_model(train_pt_fea_ten=train_pt_fea_ten, train_vox_ten=train_vox_ten,
                                   batch_size=voxel_intensity_tensor.shape[0],
                                   train_pt_label_ten=train_pt_label_ten, num_class=num_class)
            else:
                outputs = my_model(train_pt_fea_ten=train_pt_fea_ten, train_vox_ten=train_vox_ten,
                                   batch_size=voxel_intensity_tensor.shape[0])

            (predictions, data, labels_array) = prepare_data(train_vox_ten, (outputs.squeeze(dim=1), voxel_intensity_tensor, processed_voxel_label))
            pred_flat = torch.cat(predictions, dim=0)
            data_flat = torch.cat(data, dim=0)
            labels_array_flat = torch.cat(labels_array, dim=0)

            loss = compute_loss(pred_flat, data_flat, labels_array_flat)

            writer.add_histograms_if(i_iter == 0, [
                "Values/train_predictions_voxel_wise"
                "Values/train_intensities_voxel_wise"
            ], [pred_flat, data_flat], global_step=epoch)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            print(f"loss: {loss.item()}")

            if loss_enum == Losses.DynamicCombinedLoss:
                print(f"Loss function alpha weights: {dynamic_weights_alpha}")

            def print_mean_loss(check_point):
                if global_iter % check_point == 0:
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                              (epoch, i_iter, np.mean(loss_list)))
                    else:
                        print('loss error')

            print_mean_loss(check_point=1000)

            optimizer.zero_grad()
            pbar.update(1)
            global_iter += 1

            print_mean_loss(check_point=check_iter)

        writer.add_scalar("Loss/train", np.mean(loss_list), global_step=epoch)
        writer.flush()

        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('-r', '--regression', action='store_true', default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-l', '--loss', type=str, default='huber', help='Loss function')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-t', '--tensorboard_path', type=str, default='', help='Tensorboard path')
    parser.add_argument('-lb', '--use_labels', action='store_true', default=True, help='Use labels as additional input layer')
    parser.add_argument('--load_model', action='store_true', default=True, help='Load model from specified path (if exists)')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)

    if args.regression:
        print("Using regression pipeline")
        main_regression(args)
    else:
        main(args)
