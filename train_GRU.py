#!/usr/bin/env python
# coding: utf-8
"""
train_GRU.py 的用途:
1. 读取配置文件
2. 加载 prepareData.py 处理好的交通流样本
3. 构建纯时间序列 GRU 基准模型
4. 训练并记录 train_loss / val_loss 到 logs
5. 保存最佳模型参数, 并在测试集上输出预测结果

终端运行指令:
python train_GRU.py --config configurations/PEMS04_gru.conf
python train_GRU.py --config configurations/PEMS08_gru.conf
"""

import argparse
import configparser
import csv
import os
import shutil
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.metrics import masked_mae, masked_mse
from lib.utils import compute_val_loss_mstgcn, load_graphdata_channel1, predict_and_save_results_mstgcn
from model.GRU import make_model


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configurations/PEMS04_gru.conf", type=str,
                    help="configuration file path")
args = parser.parse_args()

config = configparser.ConfigParser()
print("Read configuration file: %s" % args.config)
config.read(args.config)
data_config = config["Data"]
training_config = config["Training"]

graph_signal_matrix_filename = data_config["graph_signal_matrix_filename"]
num_for_predict = int(data_config["num_for_predict"])
dataset_name = data_config["dataset_name"]

model_name = training_config["model_name"]
ctx = training_config["ctx"]
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config["learning_rate"])
epochs = int(training_config["epochs"])
start_epoch = int(training_config["start_epoch"])
batch_size = int(training_config["batch_size"])
num_of_weeks = int(training_config["num_of_weeks"])
num_of_days = int(training_config["num_of_days"])
num_of_hours = int(training_config["num_of_hours"])
in_channels = int(training_config["in_channels"])
hidden_size = int(training_config.get("hidden_size", 32))
num_layers = int(training_config.get("num_layers", 1))
dropout = float(training_config.get("dropout", 0.0))
loss_function = training_config["loss_function"]
metric_method = training_config["metric_method"]
missing_value = float(training_config["missing_value"])

folder_dir = "%s_h%dd%dw%d_channel%d_%e" % (
    model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print("folder_dir:", folder_dir)
params_path = os.path.join("experiments", dataset_name, folder_dir)
print("params_path:", params_path)

log_path = "logs/%s/%s.csv" % (dataset_name, folder_dir)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)

net = make_model(
    DEVICE=DEVICE,
    in_channels=in_channels,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_for_predict=num_for_predict,
    dropout=dropout,
)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print("create params directory %s" % params_path)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print("delete the old one and create params directory %s" % params_path)
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print("train from params directory %s" % params_path)
    else:
        raise SystemExit("Wrong type of model!")

    print("param list:")
    print("CUDA\t", DEVICE)
    print("in_channels\t", in_channels)
    print("hidden_size\t", hidden_size)
    print("num_layers\t", num_layers)
    print("dropout\t", dropout)
    print("batch_size\t", batch_size)
    print("graph_signal_matrix_filename\t", graph_signal_matrix_filename)
    print("start_epoch\t", start_epoch)
    print("epochs\t", epochs)

    masked_flag = 0
    criterion = None
    criterion_masked = None
    if loss_function == "masked_mse":
        criterion_masked = masked_mse
        masked_flag = 1
    elif loss_function == "masked_mae":
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == "mae":
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function in ("mse", "rmse"):
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    else:
        raise ValueError("Unsupported loss_function: %s" % loss_function)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    start_time = time()

    if start_epoch > 0:
        params_filename = os.path.join(params_path, "epoch_%s.params" % start_epoch)
        net.load_state_dict(torch.load(params_filename, map_location=DEVICE))
        print("start epoch:", start_epoch)
        print("load weight from: ", params_filename)

    for epoch in range(start_epoch, epochs):
        epoch_train_loss = 0
        batch_count = 0
        params_filename = os.path.join(params_path, "epoch_%s.params" % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print("save parameters to file: %s" % params_filename)

        net.train()
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data
            optimizer.zero_grad()
            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss = loss.item()
            global_step += 1
            sw.add_scalar("training_loss", training_loss, global_step)
            sw.add_scalar("val_loss", val_loss, epoch)
            epoch_train_loss += training_loss
            batch_count += 1

            if global_step % 1000 == 0:
                print("global step: %s, training loss: %.2f, time: %.2fs"
                      % (global_step, training_loss, time() - start_time))

        epoch_train_loss /= batch_count
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_train_loss, val_loss])

    print("best epoch:", best_epoch)
    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, "test")


def predict_main(global_step, data_loader, data_target_tensor, metric_method, _mean, _std, type):
    params_filename = os.path.join(params_path, "epoch_%s.params" % global_step)
    print("load weight from:", params_filename)
    net.load_state_dict(torch.load(params_filename, map_location=DEVICE))
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step,
                                    metric_method, _mean, _std, params_path, type)


if __name__ == "__main__":
    train_main()
