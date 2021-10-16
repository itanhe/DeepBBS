#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

# Part of the code is referred from: https://github.com/charlesq34/pointnet
# Part of the code is referred from: https://github.com/WangYueFt/dcp

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768, random_spherical=False):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)

    if random_spherical:
        random_p1 = np.random.randn(1,3)
        random_p1 /= np.linalg.norm(random_p1, axis=1)
        random_p1 *= 500
    else:
        random_p1 = pointcloud1[np.random.randint(0, num_points, size=(1)), :]
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    if random_spherical:
        random_p2 = np.random.randn(1, 3)
        random_p2 /= np.linalg.norm(random_p2, axis=1)
        random_p2 *= 500
    else:
        random_p2 = pointcloud2[np.random.randint(0, num_points, size=(1)), :]
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T

class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train', gaussian_noise=False, unseen=False, factor=4, src_unbalance=False, tgt_unbalance=False, random_point_order=True, different_pc=False):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        if different_pc:
            self.num_points *= 2
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        self.src_unbalance = src_unbalance
        self.tgt_unbalance = tgt_unbalance
        self.random_point_order = random_point_order
        self.different_pc = different_pc
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])

        if self.different_pc:
            ind = np.random.permutation(self.num_points)
            pointcloud1 = pointcloud1[:,ind[:round(self.num_points/2)]]
            pointcloud2 = pointcloud2[:,ind[round(self.num_points/2):]]
        if self.random_point_order:
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T
        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)
        if self.src_unbalance:
            pointcloud1 = pointcloud1[:,:512]
        if self.tgt_unbalance:
            pointcloud2 = pointcloud2[:, :512]
        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)
        if self.random_point_order:
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), euler_ab.astype('float32')

    def __len__(self):
        return self.data.shape[0]