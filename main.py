#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import gc
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from model import DCP, SVDHead_no_network
from util import npmat2euler, transform_point_cloud, cdist_torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from sklearn.metrics import r2_score

# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL
# Part of the code is referred from: https://github.com/WangYueFt/dcp


def _init_(args):
    if not os.path.exists(args.checkpoint_dir + 'checkpoints'):
        os.makedirs(args.checkpoint_dir+'checkpoints')
    if not os.path.exists(args.checkpoint_dir+'checkpoints/' + args.exp_name):
        os.makedirs(args.checkpoint_dir+'checkpoints/' + args.exp_name)
    if not os.path.exists(args.checkpoint_dir+'checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs(args.checkpoint_dir+'checkpoints/' + args.exp_name + '/' + 'models')
    shutil.copyfile('main.py', os.path.join(args.checkpoint_dir+'checkpoints', args.exp_name, 'main.py.backup'))
    shutil.copyfile('model.py', os.path.join(args.checkpoint_dir+'checkpoints', args.exp_name, 'model.py.backup'))
    shutil.copyfile('data.py', os.path.join(args.checkpoint_dir+'checkpoints', args.exp_name, 'data.py.backup'))


def test_one_epoch(args, net, test_loader, epoch=0):
    with torch.no_grad():
        net.eval()
        total_loss = 0
        num_examples = 0

        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []

        if args.DeepBBS_pp:
            spatial_step = SVDHead_no_network()

        for src, target, rotation_ab, translation_ab, euler_ab in tqdm(test_loader):
            if not args.no_cuda:
                src = src.cuda()
                target = target.cuda()
                rotation_ab = rotation_ab.cuda()
                translation_ab = translation_ab.cuda()

            batch_size = src.size(0)
            num_examples += batch_size
            target_copy = target.clone()

            if not args.DeepBBS_pp:
                not_converged = True; iter_num=1; transformed_src = src; R=[]; T=[]
                while not_converged:
                    rotation_ab_pred, translation_ab_pred, src_corr = net(transformed_src, target_copy, iter_num)
                    diff = np.mean(np.abs((npmat2euler(rotation_ab_pred.detach().cpu()))))
                    if diff < args.iterative_convergence['deep_min_diff'] or iter_num >= args.iterative_convergence['deep_max_iter']:
                        not_converged = False
                    transformed_src = transform_point_cloud(transformed_src.detach(), rotation_ab_pred.detach(), translation_ab_pred.detach()).detach()
                    R.append(rotation_ab_pred.detach()); T.append(translation_ab_pred.detach())
                    iter_num += 1
                rotation_ab_pred = R[0]
                for i in range(len(R)-1):
                    rotation_ab_pred = torch.matmul(R[i+1],rotation_ab_pred)
                translation_ab_pred = T[-1]; temp_R = torch.eye(3).unsqueeze(0).cuda()
                for i in range(len(T)-1):
                    temp_R = torch.matmul(temp_R,R[-1-i])
                    translation_ab_pred = translation_ab_pred + torch.matmul(temp_R,T[-i-2].unsqueeze(2)).squeeze(2)

            else:
                not_converged_spatial = True; not_converged_deep = True; iter_num = 1; transformed_src = src; R = []; T = []
                while not_converged_spatial:
                    if not_converged_deep:
                        rotation_ab_pred, translation_ab_pred, src_corr = net(transformed_src, target_copy, iter_num)
                    else:
                        rotation_ab_pred, translation_ab_pred, src_corr = spatial_step(transformed_src, target_copy, iter=1)
                    diff = np.mean(np.abs((npmat2euler(rotation_ab_pred.detach().cpu()))))
                    if diff < args.iterative_convergence['deep_min_diff'] or iter_num >= args.iterative_convergence['deep_max_iter']:
                        not_converged_deep = False
                    if (iter_num > args.iterative_convergence['spatial_min_iter'] and diff<args.iterative_convergence['spatial_min_diff']) or iter_num>args.iterative_convergence['spatial_max_iter']:
                        not_converged_spatial = False
                    transformed_src = transform_point_cloud(transformed_src.detach(), rotation_ab_pred.detach(), translation_ab_pred.detach()).detach()
                    R.append(rotation_ab_pred.detach()); T.append(translation_ab_pred.detach())
                    iter_num += 1
                rotation_ab_pred = R[0]
                for i in range(len(R) - 1):
                    rotation_ab_pred = torch.matmul(R[i + 1], rotation_ab_pred)
                translation_ab_pred = T[-1]; temp_R = torch.eye(3).unsqueeze(0).cuda()
                for i in range(len(T) - 1):
                    temp_R = torch.matmul(temp_R, R[-1 - i])
                    translation_ab_pred = translation_ab_pred + torch.matmul(temp_R, T[-i - 2].unsqueeze(2)).squeeze(2)
            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.numpy())

            identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)
            ind_mask = (cdist_torch(transform_point_cloud(src, rotation_ab, translation_ab), target, points_dim=3).min(dim=2).values < 0.05)
            loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                   + F.mse_loss(translation_ab_pred, translation_ab) \
                   + 0.95**epoch * ((src_corr - transform_point_cloud(src, rotation_ab, translation_ab)) ** 2).sum(dim=1).view(-1)[ind_mask.view(-1)].mean()

            total_loss += loss.item() * batch_size

        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.concatenate(eulers_ab, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab


def train_one_epoch(args, net, train_loader, opt, epoch):
    net.train()
    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    eulers_ab = []

    for src, target, rotation_ab, translation_ab, euler_ab in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred, src_corr = net(src, target, 1)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())

        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        ind_mask = (cdist_torch(transform_point_cloud(src, rotation_ab, translation_ab), target, points_dim=3).min(dim=2).values < 0.05)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab) \
               + 0.95**epoch * ((src_corr - transform_point_cloud(src, rotation_ab, translation_ab)) ** 2).sum(dim=1).view(-1)[ind_mask.view(-1)].mean()



        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab


def test(args, net, test_loader):
    test_loss, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, test_translations_ab_pred, test_eulers_ab = test_one_epoch(args, net, test_loader)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))
    test_r_ab_r2_score = r2_score(np.degrees(test_eulers_ab), test_rotations_ab_pred_euler)
    test_t_ab_r2_score = r2_score(test_translations_ab, test_translations_ab_pred)

    print('==FINAL TEST==')
    print('Loss: %f, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, rot_R2: %f, trans_R2: %f'
                  % (test_loss, test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab,
                     test_t_mae_ab, test_r_ab_r2_score, test_t_ab_r2_score))

def train(args, net, train_loader, test_loader):
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[90, 130, 150], gamma=0.1)
    best_test_loss = np.inf

    for epoch in range(args.epochs):
        train_loss, train_rotations_ab, train_translations_ab, train_rotations_ab_pred, train_translations_ab_pred, \
        train_eulers_ab = train_one_epoch(args, net, train_loader, opt, epoch)

        scheduler.step()

        test_loss, test_rotations_ab, test_translations_ab, test_rotations_ab_pred, \
        test_translations_ab_pred, test_eulers_ab = test_one_epoch(args, net, test_loader, epoch)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

        if best_test_loss >= test_loss:
            best_test_loss = test_loss

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), args.checkpoint_dir+'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), args.checkpoint_dir+'checkpoints/%s/models/model.best.t7' % args.exp_name)

        print('==TRAIN==')
        print('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_r_mse_ab, train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))

        print('==TEST==')
        print('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, test_r_mse_ab, test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), args.checkpoint_dir+'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), args.checkpoint_dir+'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')

    ######################## Network Parameters ########################
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')

    ######################## Model Parameters ########################
    parser.add_argument('--alpha_factor', type=float, default=4)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--DeepBBS_pp', dest='DeepBBS_pp', action='store_true')
    parser.add_argument('--DeepBBS', dest='DeepBBS_pp', action='store_false')
    parser.set_defaults(DeepBBS_pp=True)

    ######################## Training Parameters ########################
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoint_dir', type=str, default='')

    ######################## Dataset Parameters ########################
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--different_pc', type=bool, default=False)
    parser.add_argument('--n_subsampled_points', type=int, default=1024, metavar='N',
                        help='Num of subsampled points to use')

    ######################## Testing Parameters ########################
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--deep_min_diff', type=float, default=0.4, metavar='N')
    parser.add_argument('--deep_max_iter', type=int, default=5, metavar='N')
    parser.add_argument('--spatial_min_diff', type=float, default=0.01, metavar='N')
    parser.add_argument('--spatial_min_iter', type=int, default=30, metavar='N')
    parser.add_argument('--spatial_max_iter', type=int, default=45, metavar='N')

    args = parser.parse_args()

    args.iterative_convergence = {'deep_min_diff': args.deep_min_diff, 'deep_max_iter': args.deep_max_iter,
                                  'spatial_min_diff': args.spatial_min_diff, 'spatial_min_iter': args.spatial_min_iter,
                                  'spatial_max_iter': args.spatial_max_iter}

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    _init_(args)

    train_loader = DataLoader(
        ModelNet40(num_points=args.num_points, num_subsampled_points=args.n_subsampled_points, partition='train',
                   gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor,
                   random_point_order=True, different_pc=args.different_pc), batch_size=args.batch_size,
                   shuffle=True, drop_last=True)
    test_loader = DataLoader(
        ModelNet40(num_points=args.num_points, num_subsampled_points=args.n_subsampled_points, partition='test',
                   gaussian_noise=args.gaussian_noise, unseen=args.unseen, factor=args.factor,
                   random_point_order=True, different_pc=args.different_pc), batch_size=1,
                   shuffle=False, drop_last=False)

    net = DCP(args)
    if not args.no_cuda:
        net.cuda()

    if args.eval:
        model_path = args.model_path
        if not os.path.exists(model_path):
            print("can't find pretrained model")
            raise FileNotFoundError
        if args.no_cuda:
            net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        else:
            net.load_state_dict(torch.load(model_path), strict=False)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Using", torch.cuda.device_count(), "GPUs.")

    if args.eval:
        test(args, net, test_loader)
    else:
        train(args, net, train_loader, test_loader)


if __name__ == '__main__':
    main()
