# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py 

import os
import numpy as np
from torch.utils import data
import yaml
import pickle

REGISTERED_PC_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(self, data_path, imageset='demo',
                 return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == 'val':
            print(demo_label_path)
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'demo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'val':
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

@register_dataset
class SemKITTI_synlidar(data.Dataset):
    def __init__(self, data_path, imageset='all',
                 return_ref=False, label_mapping="synlidar.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            annotations_yaml = yaml.safe_load(stream)
        self.map_2_semantickitti_map = annotations_yaml['map_2_semantickitti']

        split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = self.read_points(self.im_idx[index])

        # annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.uint32).reshape((-1, 1))
        annotated_data = self.read_label(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label')
        annotated_data.reshape((-1, 1))
        annotated_data = np.vectorize(self.map_2_semantickitti_map.__getitem__)(annotated_data)
        annotated_data = annotated_data[:, np.newaxis]

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

    @staticmethod
    def cartesian_to_spherical(point_cloud, range_noise=0.05):
        """
        Converts a point cloud from Cartesian (x, y, z) to spherical coordinates (r, theta, phi).

        Parameters:
            point_cloud (numpy.ndarray): A Nx3 array of Cartesian coordinates.

        Returns:
            numpy.ndarray: A Nx3 array of spherical coordinates (r, theta, phi).
        """
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)  # azimuthal angle
        phi = np.arccos(z / np.maximum(r, np.finfo(float).eps))  # polar angle, avoid division by zero

        noise = np.random.uniform(-range_noise, range_noise, size=r.shape)
        r_noisy = np.maximum(r + noise, 0)  # Ensure r remains non-negative

        return np.stack((r_noisy, theta, phi), axis=-1)

    @staticmethod
    def spherical_to_cartesian(spherical_coords):
        """
        Converts a point cloud from spherical (r, theta, phi) to Cartesian (x, y, z) coordinates.

        Parameters:
            spherical_coords (numpy.ndarray): A Nx3 array of spherical coordinates (r, theta, phi).

        Returns:
            numpy.ndarray: A Nx3 array of Cartesian coordinates (x, y, z).
        """
        r, theta, phi = spherical_coords[:, 0], spherical_coords[:, 1], spherical_coords[:, 2]
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.stack((x, y, z), axis=-1)

    def read_points(self, path, add_noise=True):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # [x,y,z,intensity]

        # Add noise
        if add_noise:
            noisy_spherical = self.cartesian_to_spherical(scan, range_noise=0.03)
            noisy_xyz = self.spherical_to_cartesian(noisy_spherical)
            scan[:, :3] = noisy_xyz

        return scan

    @staticmethod
    def read_label(path):
        label = np.fromfile(path, dtype=np.uint32)
        label = label.reshape((-1))
        return label

@register_dataset
class Waymo(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="waymo.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.im_idx = []
        if imageset =='train':
            with open('/nvme/yuenan/train-0-31.txt', 'r') as f:
                for line in f.readlines():
                    self.im_idx.append(line.strip())
        else:
            with open('/nvme/yuenan/val-0-7.txt', 'r') as f: #val-0-7-label.txt
                for line in f.readlines():
                    self.im_idx.append(line.strip())

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):

        raw_data = np.load(self.im_idx[index])[:, 3:6].reshape((-1,3))
        len_first = raw_data.shape[0]
        use_extra = True #False
        use_sec_return = True #True

        if use_extra:
            intensity = np.load(self.im_idx[index])[:, 1].reshape((-1,1))
            intensity = np.tanh(intensity) 
            #elongation = np.load(self.im_idx[index])[:, 2].reshape((-1,1))
            extra_data = intensity 
        if 'no_label' in self.im_idx[index]:
            annotated_data = np.load(self.im_idx[index].replace('test/', 'test_nolabel/').replace('train/', 'train_nolabel/').replace('no_label_point_clouds/', 'train_nolabel/')).reshape((-1,1)) #
            #annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif 'valid' in self.im_idx[index] and self.imageset == 'train':
            #annotated_data = np.load(self.im_idx[index].replace('valid/', 'valid_nolabel/')).reshape((-1,1)) #
            base_path = '/home/sysuser/cylinder3d/val_submit'#+self.imageset
            frame_id = self.im_idx[index].split('/')[-1]
            annotated_data = np.load(base_path+'/first/'+frame_id).reshape((-1, 1))
        else:
            #annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
            #                             dtype=np.uint32).reshape((-1, 1))
            #annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.load(self.im_idx[index])[:, -1].reshape((-1,1)) #annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            #annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)

        if use_sec_return:
            if True: #self.imageset == 'train': #and 'test' not in self.im_idx[index]:
                sec_path = self.im_idx[index].replace('first', 'second') #'validation%2F', 'validation_').replace('training%2F', 'training_')
                #sec_path = sec_path.replace('labels_', 'labels')
                sec_data = np.load(sec_path)[:, 3:6].reshape((-1,3))
                #len_first = raw_data.shape[0]
                assert len_first != 3
                raw_data = np.concatenate((raw_data, sec_data), axis=0)
                sec_annotated_data = np.load(sec_path)[:, -1].reshape((-1,1))
                annotated_data = np.concatenate((annotated_data, sec_annotated_data), axis=0)
                sec_intensity = np.load(sec_path)[:, 1].reshape((-1,1))
                sec_intensity = np.tanh(sec_intensity)
                extra_data = np.concatenate((intensity, sec_intensity), axis=0)
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        data_tuple += (extra_data,)
        if use_sec_return:
            data_tuple += (len_first,)
        #data_tuple += (self.im_idx[index],)
        #data_tuple += (self.im_idx[index].split('/')[-1],)
        #assert 'segment' in self.im_idx[index].split('/')[-1]
        return data_tuple


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label

from os.path import join
@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 2 # additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        pose0 = self.poses[dir_idx][number_idx]

        if number_idx - self.multiscan >= 0:

            for fuse_idx in range(self.multiscan):
                plus_idx = fuse_idx + 1

                pose = self.poses[dir_idx][number_idx - plus_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                if self.imageset == 'test':
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                  dtype=np.int32).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan


        return data_tuple


# load Semantic KITTI class info

def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name
