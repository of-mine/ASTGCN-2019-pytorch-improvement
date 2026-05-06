#!/usr/bin/env python
# coding: utf-8
"""
这个脚本用于训练 ASTGCN 的注意力消融模型。
也是整合版,为了从同一个入口进入减少变量

支持三种模式：
1. full
   完整 ASTGCN：时间注意力 + 空间注意力
2. temporal_only
   仅保留时间注意力：去掉空间注意力
3. spatial_only
   仅保留空间注意力：去掉时间注意力

使用方法：
python train_ASTGCN_ablation.py --config configurations/PEMS04_astgcn_temporal_only.conf
python train_ASTGCN_ablation.py --config configurations/PEMS04_astgcn_spatial_only.conf
python train_ASTGCN_ablation.py --config configurations/PEMS08_astgcn_temporal_only.conf
python train_ASTGCN_ablation.py --config configurations/PEMS08_astgcn_spatial_only.conf
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
import csv

from model.ASTGCN_ablation import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mae, masked_mse


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_astgcn_temporal_only.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']
ablation_mode = training_config['ablation_mode']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (
    model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)

log_path = f"logs/{dataset_name}/{folder_dir}.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])


train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

net = make_model(
    DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
    adj_mx, num_for_predict, len_input, num_of_vertices, ablation_mode=ablation_mode
)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('ablation_mode\t', ablation_mode)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)

    masked_flag = 0
    criterion = None
    criterion_masked = None
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function in ('mse', 'rmse'):
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    else:
        raise ValueError('Unsupported loss_function: %s' % loss_function)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    start_time = time()

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)
        print('load weight from: ', params_filename)

    for epoch in range(start_epoch, epochs):
        epoch_train_loss = 0
        batch_count = 0
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()

        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data

            optimizer.zero_grad()
            outputs = net(encoder_inputs)

            # 消融版模型和原 ASTGCN 一样，forward 返回的是：
            # (预测结果, 空间注意力列表, 时间注意力列表)
            # 训练时只需要预测结果，所以这里取 outputs[0]。
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss = loss.item()
            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)
            sw.add_scalar('val_loss', val_loss, epoch)
            epoch_train_loss += training_loss
            batch_count += 1

            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs'
                      % (global_step, training_loss, time() - start_time))

        epoch_train_loss /= batch_count
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_train_loss, val_loss])

    print('best epoch:', best_epoch)
    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, metric_method, _mean, _std, type):
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step,
                                    metric_method, _mean, _std, params_path, type)


if __name__ == "__main__":
    train_main()
