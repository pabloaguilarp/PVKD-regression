# This script is used to identify and list all files within a dataset that contain a specific label value.
# It processes sequences of labeled point cloud data, checks for the presence of the specified label in each file,
# and generates a dictionary mapping sequences to the files containing the label.
# The results are saved as a pickle file for further analysis and are also printed to the console for quick reference.

import argparse
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm


def main(args):
    sequences = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    sequences_list = {}
    for sequence in sequences:
        seq_path = os.path.join(args.dataset_path, sequence, "labels")
        files = [f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f)) and f.endswith(".label")]

        files_list = []
        for file in tqdm(files):
            label = np.fromfile(os.path.join(seq_path, file), dtype=np.uint32)
            label = label.reshape((-1))
            if args.value in label:
                files_list.append(file)
        if len(files_list) > 0:
            sequences_list[sequence] = files_list

    # Save to the pickle file
    with open(f"./scans_with_label_{args.value}.pkl", "wb") as file:
        pickle.dump(sequences_list, file)

    print("Dictionary saved to data.pkl")

    print(f"======== List of files that contain label {args.value} ========")
    for key, value in sequences_list.items():
        print(f"Sequence: {key} contains {len(value)} files with label {args.value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find scans with certain label')
    parser.add_argument('-d', '--dataset_path', help='Path to dataset (sequences folder)',
                        default="/path/to/SynLiDAR/")
    parser.add_argument('-v', '--value', help='Value to look for', default=1, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)

    main(args)

