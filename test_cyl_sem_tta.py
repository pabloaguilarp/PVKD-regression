# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py
import errno
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

from utils.utils import get_metrics

warnings.filterwarnings("ignore")

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def main(args):
    # pytorch_device = torch.device('cuda:0')
    pytorch_device = torch.device('cpu')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']

    model_load_path = train_hypers['model_load_path']

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    train_dataset_loader, test_dataset_loader, test_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=True)

    output_path = 'out_cyl/test'
    voting_num = 4

    if True:
        print('*'*80)
        print('Generate predictions for test split')
        print('*'*80)
        pbar = tqdm(total=len(test_dataset_loader))
        time.sleep(10)
        if True:
            if True:
                my_model.eval()
                with torch.no_grad():
                    for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index) in enumerate(
                            test_dataset_loader):
                        test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          test_pt_fea]
                        test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]

                        predict_labels = my_model(test_pt_fea_ten, test_grid_ten, val_batch_size, test_grid, voting_num, use_tta=True)
                        predict_labels = torch.argmax(predict_labels, dim=0).type(torch.uint8)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        test_pred_label = np.expand_dims(predict_labels,axis=1)
                        save_dir = test_pt_dataset.im_idx[test_index[0]]
                        _,dir2 = save_dir.split('/sequences/',1)
                        new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
                        if not os.path.exists(os.path.dirname(new_save_dir)):
                            try:
                                os.makedirs(os.path.dirname(new_save_dir))
                            except OSError as exc:
                                if exc.errno != errno.EEXIST:
                                    raise
                        test_pred_label = test_pred_label.astype(np.uint32)
                        test_pred_label.tofile(new_save_dir)
                        pbar.update(1)
                del test_grid, test_pt_fea, test_grid_ten, test_index
        pbar.close()
        print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
        print('Remapping script can be found in semantic-kitti-api.')


def main_regression(args):
    pytorch_device = torch.device(args.device)

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    test_dataloader_config = configs['test_data_loader']

    test_batch_size = test_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    embedding_dim = model_config['embedding_dim']

    use_labels = args.use_labels

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    model_load_path = args.model    # Load model from specified path in params

    if args.device == 'cpu':
        train_dataloader_config['num_workers'] = 1
        test_dataloader_config['num_workers'] = 1

    _, test_dataset_loader, test_pt_dataset = data_builder.build_regression(dataset_config,
                                                                  train_dataloader_config,
                                                                  test_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=True)

    my_model = model_builder.build(model_config, use_labels, embedding_dim)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model, use_cpu=(args.device == 'cpu'))
    else:
        raise FileNotFoundError(f"File '{model_load_path}' does not exist")

    my_model.to(pytorch_device)

    def get_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    output_path = os.path.join(test_dataloader_config['output_path'], get_name(args.model))
    voting_num = args.voting_num
    print(f"Using voting num: {voting_num}")

    if args.output_path != "":
        output_path = args.output_path
        output_path = os.path.join(output_path, get_name(args.model))

    def check_and_return(path):
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        return path

    def check_dirs_and_save(save_dir):
        _, dir2 = save_dir.split('/sequences/', 1)
        new_save_dir = os.path.join(output_path, "sequences", f"{dir2.replace('velodyne', 'predictions')[:-3]}ref") # ref stands for 'reflectivity'
        return check_and_return(new_save_dir)

    def save_unstructured(save_dir):
        filename = os.path.basename(save_dir).replace('.bin', '.ref')
        new_save_dir = os.path.join(output_path, filename)
        return check_and_return(new_save_dir)

    files_counter = 0
    max_files = args.num_samples

    metrics_engine = None
    if args.calculate_metrics:
        metrics_engine = get_metrics('numpy')
        print("Calculating metrics using 'numpy' engine")

    print('*'*80)
    print('Generate predictions for test split')
    print('*'*80)
    pbar = tqdm(total=len(test_dataset_loader))
    time.sleep(10)
    my_model.eval()
    with (torch.no_grad()):
        for i_iter_test, (
        _, _, test_point_intensity, test_voxel_intensity, test_grid, test_pt_label, test_pt_fea, test_index) in enumerate(
                test_dataset_loader):
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              test_pt_fea]
            test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]
            test_pt_label_ten = [torch.from_numpy(i).type(torch.IntTensor).to(pytorch_device) for i in test_pt_label]

            # Generate predictions
            if use_labels:
                predict_intensity = my_model(train_pt_fea_ten=test_pt_fea_ten, train_vox_ten=test_grid_ten,
                                             batch_size=test_batch_size, val_grid=test_grid,
                                             train_pt_label_ten=test_pt_label_ten, num_class=num_class, voting_num=voting_num, use_tta=True)
            else:
                predict_intensity = my_model(train_pt_fea_ten=test_pt_fea_ten, train_vox_ten=test_grid_ten,
                                             batch_size=test_batch_size, val_grid=test_grid,
                                             voting_num=voting_num, use_tta=True)
            predict_intensity = predict_intensity.cpu().detach().numpy()

            # Save intensities to file
            if files_counter < max_files or max_files == -1:
                if args.keep_structure:
                    new_save_dir = check_dirs_and_save(test_pt_dataset.im_idx[test_index[0]])
                else:
                    new_save_dir = save_unstructured(test_pt_dataset.im_idx[test_index[0]])
                predict_intensity = predict_intensity.astype(np.float32)
                predict_intensity.tofile(new_save_dir)
                files_counter += 1

            if args.calculate_metrics:
                # Class-wise metrics cannot be computed because test set does not have labels
                metrics_engine.compute_metrics(test_point_intensity[0].astype(np.float32), predict_intensity.squeeze(axis=0).transpose())

            pbar.update(1)
        del test_grid, test_pt_fea, test_grid_ten, test_index
    pbar.close()
    print("Test metrics:")
    for key, val in metrics_engine.metrics_dict.items():
        print(f"{key}: {np.mean(val):.4f}")

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('-r', '--regression', action='store_true', default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-lb', '--use_labels', action='store_true', default=True, help='Use labels as additional input layer')
    parser.add_argument('-c', '--calculate_metrics', action='store_true', default=True, help='Calculate metrics after inference')
    parser.add_argument('-n', '--num_samples', type=int, default=-1, help='Number of samples to save (-1 for no limit)')
    parser.add_argument('-v', '--voting_num', type=int, default=4, help='Voting num')
    parser.add_argument('-o', '--output_path', type=str, default="", help='Output directory')
    parser.add_argument('-s', '--keep_structure', action='store_true', default=True, help='Keep sequences structure ('
                                                                                          'true) or save all output '
                                                                                          'files in the same directory')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)

    if args.regression:
        print("Using regression pipeline")
        main_regression(args)
    else:
        main(args)
