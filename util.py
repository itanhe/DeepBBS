#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import numpy as np
from scipy.spatial.transform import Rotation


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
# Part of the code is referred from: https://github.com/WangYueFt/dcp

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def square_dist_torch(A, B):
    AA = (A**2).sum(dim=1, keepdim=True)
    BB = (B**2).sum(dim=1, keepdim=True)
    inner = torch.matmul(A.float(), B.float().T)

    R = AA + (-2)*inner + BB.T

    return R

def new_cdist(x1, x2):
        x1 = x1.float()
        x2 = x2.float()
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True).float()
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True).float()
        res = -2*torch.matmul(x1, x2.transpose(-2, -1)) + x2_norm.transpose(-2, -1) + x1_norm
        res = res.clamp_min_(1e-30).sqrt_()
        return res

def dist_torch(A,B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """
    s = square_dist_torch(A,B)
    s[s<0]=0
    return torch.sqrt(s)

def cdist_torch(A,B,points_dim=None):
    num_features = 512
    if points_dim is not None:
        num_features = points_dim
    if (A.shape[-1] != num_features):
        A = torch.transpose(A, dim0=-2, dim1=-1)
    if (B.shape[-1] != num_features):
        B = torch.transpose(B, dim0=-2, dim1=-1)
    assert A.shape[-1] == num_features
    assert B.shape[-1] == num_features
    A = A.double().contiguous()
    B = B.double().contiguous()
    C = new_cdist(A,B)
    return C

def min_without_self_per_row_torch(D):
    """
    Accepts a distance matrix between all points in a set. For each point,
    returns its distance from the closest point that is not itself.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: vector of distances to nearest neighbor for each point.
    """
    E = D.clone()
    diag_ind = range(E.shape[0])
    E[diag_ind,diag_ind] = np.inf
    m = E.min(dim=1).values
    return m

def representative_neighbor_dist_torch(D):
    """
    Accepts a distance matrix between all points in a set,
    returns a number that is representative of the distances in this set.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: The representative distance in this set
    """

    assert D.shape[0] == D.shape[1], "Input to representative_neighbor_dist should be a matrix of distances from a point cloud to itself"
    m = min_without_self_per_row_torch(D)
    neighbor_dist = m.median()
    return neighbor_dist.cpu().detach().numpy()

def guess_best_alpha_torch(A,dim_num=3, transpose=None):
    """
        A good guess for the temperature of the soft argmin (alpha) can
        be calculated as a linear function of the representative (e.g. median)
        distance of points to their nearest neighbor in a point cloud.

        :param A: Point Cloud of size Nx3
        :return: Estimated value of alpha
        """

    COEFF = 0.1
    EPS = 1e-8
    if transpose is None:
        assert A.shape[0] != A.shape[1], 'Number of points and number of dimensions can''t be same'
    if (A.shape[1] != dim_num and transpose is None) or transpose:
        A = A.T
    assert A.shape[1]==dim_num
    rep = representative_neighbor_dist_torch(dist_torch(A, A))
    return COEFF * rep + EPS

def soft_BBS_loss_torch(T, S, t, points_dim=None, return_mat=False, transpose=None):
    num_features = 512
    if transpose is None:
        assert S.shape[0] != S.shape[1] and T.shape[0] != T.shape[1], 'Number of points and number of dimensions can''t be same'
    if points_dim is not None:
        num_features = points_dim
    if (T.shape[1] is not num_features and transpose is None) or transpose:
        T = torch.transpose(T, dim0=0, dim1=1)
    if (S.shape[1] is not num_features and transpose is None) or transpose:
        S = torch.transpose(S, dim0=0, dim1=1)
    assert S.shape[1] == num_features and T.shape[1] == num_features, 'Points dimension dismatch'

    T_num_samples = T.shape[0]
    S_num_samples = S.shape[0]
    mean_num_samples = np.mean([T_num_samples, S_num_samples])
    D = cdist_torch(T, S, points_dim)
    R = torch.squeeze(softargmin_rows_torch(D, t))
    C = torch.squeeze(softargmin_rows_torch(torch.transpose(D, dim0=0, dim1=1), t))
    C = torch.transpose(C, dim0=0, dim1=1)
    B = torch.mul(R, C)
    loss = torch.div(-torch.sum(B), mean_num_samples).view(1)
    if return_mat:
        return B
    else:
        return loss

def my_softmax(x, eps=1e-12, dim=0):
    x_exp = torch.exp(x - x.max())
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp/(x_exp_sum + eps)

def softargmin_rows_torch(X, t, eps=1e-12):
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=eps, dim=1)
    return weights
