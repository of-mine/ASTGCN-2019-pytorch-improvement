# ASTGCN Traffic Flow Forecasting Dashboard

本项目基于 ASTGCN（Attention Based Spatial-Temporal Graph Convolutional Networks）实现交通流预测，并在原始模型基础上补充了 MSTGCN、GRU、ASTGCN 注意力消融实验，以及 PyQt5 可视化结果面板。整体流程是：准备原始数据 -> 生成模型输入样本 -> 训练模型并保存日志/参数 -> 启动可视化页面查看 loss、节点预测和未来路网预测。

![ASTGCN architecture](fig/ASTGCN%20architecture.png)

## 项目结构

```text
configurations/          模型和数据配置文件
data/                    PEMS04 / PEMS08 原始数据和预处理结果
experiments/             训练后的模型参数和测试输出
logs/                    每个模型的 train_loss / val_loss 日志，以便后面的可视化loss曲线
lib/                     数据加载、邻接矩阵、指标计算等工具函数
model/                   ASTGCN、MSTGCN、GRU 和消融模型定义
prepareData.py           数据预处理入口
train_MSTGCN_r.py        MSTGCN 训练入口
train_GRU.py             GRU 训练入口
train_ASTGCN_ablation.py ASTGCN 完整模型和消融实验训练入口
visualize.py             单独绘制 loss 曲线
pyqt_result_dashboard.py PyQt5 可视化总面板
requirements-py311-cu128.txt 当前 Windows + CUDA 12.8 环境依赖快照
```

## 环境准备

当前开发环境：

```text
Python: 3.11.3
OS: Windows-10-10.0.26200-SP0
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
NVIDIA Driver: 596.36
PyTorch: 2.11.0+cu128
CUDA runtime used by PyTorch: 12.8
```

在 Windows PowerShell 中，从项目根目录开始构建虚拟环境并安装全部依赖：

```powershell
py -3.11 -m venv .venv-py311-cu128
.\.venv-py311-cu128\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-py311-cu128.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

如果 PowerShell 提示不允许执行激活脚本，先临时放开当前窗口的脚本执行权限，再重新激活：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-py311-cu128\Scripts\Activate.ps1
```

安装完成后验证 Python、PyTorch、CUDA 和 GPU 是否可用：

```powershell
python --version
python -m pip list
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果本机没有 NVIDIA GPU，也可以使用 CPU 环境，但训练和可视化里的滚动预测会明显变慢。正常cpu跑完经计算要跑近十天，gpu跑完大概俩小时。

## 数据准备

项目使用 PEMS04 和 PEMS08 数据集。原始数据应放在如下位置：

```text
data/PEMS04/pems04.npz
data/PEMS04/distance.csv
data/PEMS08/PEMS08.npz
data/PEMS08/distance.csv
```

其中 `.npz` 文件需要包含 key `data`，形状为：

```text
(sequence_length, num_of_vertices, num_of_features)
```

配置文件里的关键字段位于 `[Data]` 和 `[Training]`：

```ini
graph_signal_matrix_filename = data/PEMS04/pems04.npz
adj_filename = data/PEMS04/distance.csv
num_of_vertices = 307
points_per_hour = 12
num_for_predict = 12
```

不同模型和数据集对应不同配置文件，例如：

```text
configurations/PEMS04_mstgcn.conf
configurations/PEMS04_gru.conf
configurations/PEMS04_astgcn_full.conf
configurations/PEMS04_astgcn_temporal_only.conf
configurations/PEMS04_astgcn_spatial_only.conf
configurations/PEMS08_mstgcn.conf
configurations/PEMS08_gru.conf
```

## 数据预处理

训练前需要先把原始交通流序列切分成模型可读取的样本。预处理入口是 `prepareData.py`：

```powershell
python prepareData.py --config configurations/PEMS04_astgcn_full.conf
python prepareData.py --config configurations/PEMS08_astgcn_full.conf
```

脚本会根据配置里的 `num_of_hours`、`num_of_days`、`num_of_weeks`、`num_for_predict` 生成历史窗口和预测目标，并按 6:2:2 划分训练集、验证集、测试集。

输出文件示例：

```text
data/PEMS04/pems04_r1_d0_w0_astcgn.npz
data/PEMS08/pems08_r1_d0_w0_astcgn.npz
```

这个文件会被训练脚本自动读取，里面包含：

```text
train_x, train_target, train_timestamp
val_x, val_target, val_timestamp
test_x, test_target, test_timestamp
mean, std
```

## 模型训练

训练脚本会读取配置文件，加载预处理后的 `.npz` 数据，训练模型，并把每个 epoch 的 `train_loss` 和 `val_loss` 写入 `logs/`。验证集效果更优时会保存对应参数到 `experiments/`。

MSTGCN：

```powershell
python train_MSTGCN_r.py --config configurations/PEMS04_mstgcn.conf
python train_MSTGCN_r.py --config configurations/PEMS08_mstgcn.conf
```

GRU：

```powershell
python train_GRU.py --config configurations/PEMS04_gru.conf
python train_GRU.py --config configurations/PEMS08_gru.conf
```

ASTGCN 消融实验：

完整 ASTGCN 也统一放在 `train_ASTGCN_ablation.py` 中训练，其中 `*_astgcn_full.conf` 表示完整模型，`*_temporal_only.conf` 表示仅保留时间注意力，`*_spatial_only.conf` 表示仅保留空间注意力。

```powershell
python train_ASTGCN_ablation.py --config configurations/PEMS04_astgcn_full.conf
python train_ASTGCN_ablation.py --config configurations/PEMS04_astgcn_temporal_only.conf
python train_ASTGCN_ablation.py --config configurations/PEMS04_astgcn_spatial_only.conf
python train_ASTGCN_ablation.py --config configurations/PEMS08_astgcn_full.conf
python train_ASTGCN_ablation.py --config configurations/PEMS08_astgcn_temporal_only.conf
python train_ASTGCN_ablation.py --config configurations/PEMS08_astgcn_spatial_only.conf
```

训练完成后，典型输出如下：

```text
logs/PEMS04/astgcn_full_h1d0w0_channel1_1.000000e-03.csv
experiments/PEMS04/astgcn_full_h1d0w0_channel1_1.000000e-03/epoch_*.params
experiments/PEMS04/astgcn_full_h1d0w0_channel1_1.000000e-03/output_epoch_*_test.npz
```

## 单独查看 Loss 曲线

如果只想查看某个模型的训练/验证 loss，可以使用 `visualize.py`：

```powershell
python visualize.py --config configurations/PEMS04_astgcn_full.conf
python visualize.py --config configurations/PEMS08_astgcn_full.conf
```

保存图片：

```powershell
python visualize.py --config configurations/PEMS04_astgcn_full.conf --save fig/pems04_loss.png
```

## 启动可视化页面

完整可视化入口是 `pyqt_result_dashboard.py`：

```powershell
python pyqt_result_dashboard.py
```

页面包含三个主要视图：

1. `Loss 曲线`
   展示当前数据集下各模型的训练 loss、验证 loss，并标出验证集最优 epoch。

2. `节点纯预测`
   选择数据集、节点索引和预测区间后，系统会先用校准窗口比较多个模型，自动选择效果最好的模型。图像分成左右两半：

   左半是验证区间，显示真实值 `Actual` 和模型预测值 `Pred`。

   右半是未来区间，使用同一个最优模型做滚动预测，只显示未来预测值 `Future Pred`。

   下方表格会列出验证段和未来段的时间索引、展示时间、模型、真实值和预测值。

3. `未来路网预测`
   选择目标路段和未来时间段后，系统会根据最近三周该路段的校准误差选择最优模型，再对未来 0-24 小时进行滚动预测。路网图中颜色和线宽表示潜在拥堵风险，点击路段可以查看对应预测值和模型指标。



如果需要别人完整复现实验，可以把数据集下载地址、模型参数文件或训练结果通过网盘/Release 单独提供，并在 README 中说明放置路径。

## Reference

```latex
@inproceedings{guo2019attention,
  title={Attention based spatial-temporal graph convolutional networks for traffic flow forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={922--929},
  year={2019}
}
```
