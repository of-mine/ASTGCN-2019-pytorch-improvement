#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """
    纯时间序列 GRU 基准模型。

    输入仍然使用当前项目统一的形状: (B, N, F, T)
    B: batch size
    N: 节点数
    F: 输入特征通道数
    T: 历史时间步长度

    GRU 不使用邻接矩阵, 因此它只学习单个节点自己的历史变化规律,
    不显式建模道路节点之间的空间依赖关系。
    """

    def __init__(self, in_channels, hidden_size, num_layers, num_for_predict, dropout=0.0):
        super(GRUBaseline, self).__init__()
        self.num_for_predict = int(num_for_predict)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        # num_layers=1 时 PyTorch 会忽略 GRU 内部 dropout, 这里显式置 0 避免无效配置警告。
        gru_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=int(in_channels),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidden_size, self.num_for_predict)

    def forward(self, x):
        # (B, N, F, T) -> (B, N, T, F), 让每个节点形成一条时间序列。
        batch_size, num_nodes, num_features, num_steps = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()

        # 合并 B 和 N, 对所有节点共享同一套 GRU 参数。
        x = x.view(batch_size * num_nodes, num_steps, num_features)
        _, hidden = self.gru(x)

        # hidden[-1] 是最后一层 GRU 在最后时间步的隐藏状态。
        output = self.projection(hidden[-1])
        return output.view(batch_size, num_nodes, self.num_for_predict)


def make_model(DEVICE, in_channels, hidden_size, num_layers, num_for_predict, dropout=0.0):
    model = GRUBaseline(
        in_channels=in_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_for_predict=num_for_predict,
        dropout=dropout,
    )
    return model.to(DEVICE)
