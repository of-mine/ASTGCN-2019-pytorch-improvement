# -*- coding:utf-8 -*-
"""
这个文件是 ASTGCN 的消融版本。

用途：
1. 保留原 ASTGCN 的整体结构
2. 通过开关控制是否启用时间注意力、空间注意力
3. 用于做下面三类实验：
   - 完整模型：时间注意力 + 空间注意力
   - 仅时间注意力：去掉空间注意力
   - 仅空间注意力：去掉时间注意力

建议的阅读顺序：
1. 先看 make_model(...)
2. 再看 ASTGCN_block.__init__(...)
3. 最后看 ASTGCN_block.forward(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import scaled_Laplacian, cheb_polynomial


class Spatial_Attention_layer(nn.Module):
    """
    空间注意力层。

    输入：
    x -> (B, N, F, T)

    输出：
    空间注意力矩阵 -> (B, N, N)

    含义：
    对每个 batch，学习“节点与节点之间”当前应该关注的关系强度。
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)
        return S_normalized


class Temporal_Attention_layer(nn.Module):
    """
    时间注意力层。

    输入：
    x -> (B, N, F, T)

    输出：
    时间注意力矩阵 -> (B, T, T)

    含义：
    对每个 batch，学习“时刻与时刻之间”当前应该关注的关系强度。
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        _, num_of_vertices, _, _ = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class cheb_conv(nn.Module):
    """
    普通的 K 阶切比雪夫图卷积。

    这个版本不使用空间注意力，适合“去掉空间注意力”的消融实验。
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])

    def forward(self, x):
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                theta_k = self.Theta[k]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class cheb_conv_withSAt(nn.Module):
    """
    带空间注意力的 K 阶切比雪夫图卷积。

    这个版本会把空间注意力矩阵乘到图传播过程里，
    适合“启用空间注意力”的模型。
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])

    def forward(self, x, spatial_attention):
        batch_size, num_of_vertices, _, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k.mul(spatial_attention)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    """
    ASTGCN 的核心时空块。

    这个版本增加了两个消融开关：
    - use_temporal_attention：是否启用时间注意力
    - use_spatial_attention：是否启用空间注意力

    三种常见用法：
    1. 完整模型
       use_temporal_attention=True, use_spatial_attention=True
    2. 仅时间注意力
       use_temporal_attention=True, use_spatial_attention=False
    3. 仅空间注意力
       use_temporal_attention=False, use_spatial_attention=True
    """

    def __init__(
        self,
        DEVICE,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_of_vertices,
        num_of_timesteps,
        use_temporal_attention=True,
        use_spatial_attention=True,
    ):
        super(ASTGCN_block, self).__init__()
        self.use_temporal_attention = use_temporal_attention
        self.use_spatial_attention = use_spatial_attention

        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)

        # 同时保留两种图卷积：
        # 1. 普通图卷积：用于“去掉空间注意力”
        # 2. 带空间注意力图卷积：用于“启用空间注意力”
        self.cheb_conv_plain = cheb_conv(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)

        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1)
        )
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x):
        """
        x: (B, N, F, T)

        返回：
        1. 当前 block 的输出特征
        2. 空间注意力矩阵（如果当前模式禁用了空间注意力，则返回 None）
        3. 时间注意力矩阵（如果当前模式禁用了时间注意力，则返回 None）
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # 1. 时间注意力开关：
        # 如果启用时间注意力，就先算 temporal_At，再用它重加权输入时间维。
        # 如果禁用时间注意力，就直接让后续空间注意力吃原始输入 x。
        if self.use_temporal_attention:
            temporal_At = self.TAt(x)
            x_TAt = torch.matmul(
                x.reshape(batch_size, -1, num_of_timesteps), temporal_At
            ).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        else:
            temporal_At = None
            x_TAt = x

        # 2. 空间注意力开关：
        # 如果启用空间注意力，就从 x_TAt 中算出空间注意力矩阵，
        # 并把它用于带空间注意力的图卷积。
        # 如果禁用空间注意力，就直接走普通图卷积。
        if self.use_spatial_attention:
            spatial_At = self.SAt(x_TAt)
            spatial_gcn = self.cheb_conv_SAt(x, spatial_At)
        else:
            spatial_At = None
            spatial_gcn = self.cheb_conv_plain(x)

        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual, spatial_At, temporal_At


class ASTGCN_submodule(nn.Module):
    """
    把多个 ASTGCN_block 叠起来，并在最后接一个卷积层输出未来预测。

    这里把 use_temporal_attention / use_spatial_attention 一路传下去，
    保证整个模型的每个 block 都处在同一种消融模式下。
    """

    def __init__(
        self,
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices,
        use_temporal_attention=True,
        use_spatial_attention=True,
    ):
        super(ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([
            ASTGCN_block(
                DEVICE, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                cheb_polynomials, num_of_vertices, len_input,
                use_temporal_attention=use_temporal_attention,
                use_spatial_attention=use_spatial_attention,
            )
        ])

        self.BlockList.extend([
            ASTGCN_block(
                DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                cheb_polynomials, num_of_vertices, len_input // time_strides,
                use_temporal_attention=use_temporal_attention,
                use_spatial_attention=use_spatial_attention,
            )
            for _ in range(nb_block - 1)
        ])

        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        spatial_at_list = []
        temporal_at_list = []

        for block in self.BlockList:
            x, spatial_At, temporal_At = block(x)
            spatial_at_list.append(spatial_At)
            temporal_at_list.append(temporal_At)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output, spatial_at_list, temporal_at_list


def make_model(
    DEVICE,
    nb_block,
    in_channels,
    K,
    nb_chev_filter,
    nb_time_filter,
    time_strides,
    adj_mx,
    num_for_predict,
    len_input,
    num_of_vertices,
    ablation_mode='full',
):
    """
    构建消融版 ASTGCN。

    ablation_mode 的可选值：
    - full：完整 ASTGCN，时间注意力 + 空间注意力
    - temporal_only：只保留时间注意力，去掉空间注意力
    - spatial_only：只保留空间注意力，去掉时间注意力

    这个函数里会把不同模式翻译成两个布尔开关，
    再传给下层 block。
    """
    if ablation_mode == 'full':
        use_temporal_attention = True
        use_spatial_attention = True
    elif ablation_mode == 'temporal_only':
        use_temporal_attention = True
        use_spatial_attention = False
    elif ablation_mode == 'spatial_only':
        use_temporal_attention = False
        use_spatial_attention = True
    else:
        raise ValueError("Unsupported ablation_mode: %s" % ablation_mode)

    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]

    model = ASTGCN_submodule(
        DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
        cheb_polynomials, num_for_predict, len_input, num_of_vertices,
        use_temporal_attention=use_temporal_attention,
        use_spatial_attention=use_spatial_attention,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
