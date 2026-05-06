# -*- coding:utf-8 -*-
"""
visualize.py 的用途:
从 logs 目录下的 CSV 日志文件中读取 epoch、train_loss、val_loss，
然后画出训练损失曲线和验证损失曲线，并标出验证集最优的 epoch。

使用方法:
1. 根据配置文件自动推断日志路径:
   python visualize.py --config configurations/PEMS04_astgcn.conf

2. 直接指定日志文件:
   python visualize.py --log logs/PEMS04/astgcn_r_h1d0w0_channel1_1.000000e-03.csv

3. 保存图片到本地:
   python visualize.py --config configurations/PEMS04_astgcn.conf --save fig/pems04_loss.png

4. 自定义图标题:
   python visualize.py --config configurations/PEMS04_astgcn.conf --title "PEMS04 Loss Curve"
"""

import argparse
import configparser
import csv
import os

import matplotlib.pyplot as plt


def build_log_path_from_config(config_path):
    """
    读取配置文件中的关键训练参数，并按照 train_ASTGCN_r.py 中的命名规则，
    自动拼出对应的日志文件路径。

    这样做的好处是:
    只要你训练和可视化使用的是同一个 .conf 文件，
    就不用手动再去 logs 目录里找 CSV 文件名。
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    data_config = config["Data"]
    training_config = config["Training"]

    dataset_name = data_config["dataset_name"]
    model_name = training_config["model_name"]
    num_of_hours = int(training_config["num_of_hours"])
    num_of_days = int(training_config["num_of_days"])
    num_of_weeks = int(training_config["num_of_weeks"])
    in_channels = int(training_config["in_channels"])
    learning_rate = float(training_config["learning_rate"])

    folder_dir = "%s_h%dd%dw%d_channel%d_%e" % (
        model_name,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        in_channels,
        learning_rate,
    )
    log_path = os.path.join("logs", dataset_name, folder_dir + ".csv")
    return log_path,dataset_name,model_name     #自此返回元组


def read_loss_csv(log_path):
    """
    从 CSV 日志中读取三列数据:
    - epoch
    - train_loss
    - val_loss

    返回三个列表，后面 matplotlib 画图时会直接使用。
    """
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_path, "r", newline="") as f:
        # DictReader 会把每一行读成字典，例如:
        # {"epoch": "0", "train_loss": "114.91", "val_loss": "217.96"}
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not epochs:
        raise ValueError("Log file is empty: %s" % log_path)
    #小tips:Python支持序列解包,按理来讲传回去的是个元组,但是这里在调用函数时给三个变量则是对应的传回数据
    return epochs, train_losses, val_losses


def plot_loss_curve(epochs, train_losses, val_losses, title, save_path=None):
    """
    使用 matplotlib 绘制 loss 曲线。

    参数说明:
    - epochs: 横坐标，对应训练轮数
    - train_losses: 训练集损失
    - val_losses: 验证集损失
    - title: 图标题
    - save_path: 如果不为 None，则把图片保存到指定位置
    """
    # 找到验证集损失最小值的位置，用来标注“最佳 epoch”。
    #这里找的是val_losses列表中最小的索引,可以理解为epoch,后面用这个索引锁定两个最佳.
    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_epoch = epochs[best_idx]
    best_val_loss = val_losses[best_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    plt.scatter(#标记最好的点
        [best_epoch],
        [best_val_loss],
        color="red",
        s=60,
        label="Best Val Loss",
        zorder=5,
    )
    # annotate 用来给图上的某个点加文字说明。
    plt.annotate(
        "best epoch=%d\nval_loss=%.4f" % (best_epoch, best_val_loss),#这里有个转义字符,\n,表示换行
        xy=(best_epoch, best_val_loss),
        xytext=(10, 10),
        textcoords="offset points",
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            # 如果保存目录不存在，就先自动创建。exist_ok=True 的意思是如果目录已经存在了就不报错。
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print("Saved figure to:", save_path)

    plt.show()


def main():
    """
    命令行入口函数。
    正常往里面传参数也是这样写的,parser.add_argument("--config", default="configurations/PEMS04_astgcn.conf", help="Configuration file used to infer the CSV log path.")

    运行脚本时，程序会先解析命令行参数；
    然后确定日志路径；
    再读取 CSV；
    最后调用 plot_loss_curve 画图。
    """
    parser = argparse.ArgumentParser(description="Visualize training and validation loss from CSV logs.")
    parser.add_argument(
        "--config",
        default="configurations/PEMS04_astgcn.conf",#默认
        help="Configuration file used to infer the CSV log path.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Direct path to a CSV log file. If provided, it overrides --config.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the plotted figure, for example fig/pems04_loss.png",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    args = parser.parse_args()#这里的parse_args()函数是怎么用的???

    # 如果你手动传了 --log，就优先使用它；
    # 否则就根据配置文件自动推断日志路径。

    if args.log:
        log_path = args.log
        dataset_name = ""
        model_name = ""
    else:
        log_path, dataset_name, model_name = build_log_path_from_config(args.config)

    if not os.path.exists(log_path):
        raise FileNotFoundError("Log file not found: %s" % log_path)
    
    #小tips:Python里面索引直接加[]即可,不用像数据结构一样引用加'.'
    epochs, train_losses, val_losses = read_loss_csv(log_path)

    # 如果没有传 --title，就使用日志文件名作为默认标题的一部分。
    # os.path.basename()作用是“只取路径里的最后一个文件名”。
    title = args.title if args.title else "Loss Curve - %s - %s" % (dataset_name,model_name) #这里的%s%s是字符串格式化,前一个%s对应dataset_name,后一个%s对应model_name,因为是两个所以要传入元组,所以要加小括号
    print("Using log file:", log_path)
    print("Total epochs found:", len(epochs))
    print("Epoch range: %d -> %d" % (epochs[0], epochs[-1]))

    plot_loss_curve(epochs, train_losses, val_losses, title, args.save)


if __name__ == "__main__":
    main()
