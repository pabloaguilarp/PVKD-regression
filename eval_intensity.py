import argparse
import os
import sys

import warnings

import yaml
import numpy as np
from tqdm import tqdm

from config.config import load_config_data
from utils.utils import get_metrics

warnings.filterwarnings("ignore")


def main(args):
    configs = load_config_data(args.config_path)
    label_mapping = configs['dataset_params']['label_mapping']
    dataset_path = os.path.abspath(configs['train_data_loader']['data_path'])
    predictions_path = os.path.abspath(configs['test_data_loader']['output_path'])
    predictions_path = os.path.join(predictions_path, args.model)

    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    split = semkittiyaml['split'][args.split]

    metrics_engine = get_metrics(backend='numpy')

    def check_sequences(path):
        if os.path.basename(path) == "sequences":
            return path
        else:
            return os.path.join(path, "sequences")

    metrics_dict_mean = metrics_engine.get_metrics_dict()
    for seq in split:
        dataset_path = check_sequences(dataset_path)
        predictions_path = check_sequences(predictions_path)
        seq_data_path = os.path.join(dataset_path, str(seq).zfill(2))
        seq_pred_path = os.path.join(predictions_path, str(seq).zfill(2))
        preds_path = os.path.join(seq_pred_path, "predictions")
        scans_path = os.path.join(seq_data_path, "velodyne")
        if not os.path.exists(preds_path):
            continue
        print(f"Processing sequence {str(seq).zfill(2)}")
        metrics_engine.clear()
        for scan in tqdm(os.listdir(preds_path)):
            scan_path = os.path.join(scans_path, f"{scan.replace('ref', 'bin')}")
            pred_path = os.path.join(preds_path, scan)
            point_cloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            data_intensity = point_cloud[:, 3]
            pred_intensity = np.fromfile(pred_path, dtype=np.float32).reshape(-1)
            metrics_engine.compute_metrics(data_intensity, pred_intensity)

        for key, value in metrics_engine.metrics_dict.items():
            metrics_dict_mean[key].append(np.mean(value))

    print("*" * 50)
    print("*" * 50)
    print("All sequences completed\nResults:")
    for key, value in metrics_dict_mean.items():
        print(f"{key}: {str(np.mean(value)).replace('.', ',')}")
    print("*" * 50)
    print("*" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-s', '--split', default='test')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)

    main(args)
