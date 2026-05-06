# -*- coding: utf-8 -*-
"""
Generate UML-style diagrams for the ASTGCN traffic forecasting project.

Outputs:
  fig/uml/analysis_class_diagram.png
  fig/uml/analysis_sequence_diagram.png
  fig/uml/design_class_diagram.png
  fig/uml/design_sequence_diagram.png
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = os.path.join("fig", "uml")


def configure_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def setup_ax(figsize=(16, 10), title=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if title:
        ax.text(
            50,
            97,
            title,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#172033",
        )
    return fig, ax


def class_box(ax, x, y, w, h, title, attrs=None, methods=None, fill="#f8fafc", edge="#334155"):
    attrs = attrs or []
    methods = methods or []
    rect = Rectangle((x, y), w, h, linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(rect)
    title_sep_y = y + h - 5.8
    ax.plot([x, x + w], [title_sep_y, title_sep_y], color=edge, linewidth=1.0, zorder=3)
    ax.text(
        x + w / 2,
        y + h - 2.9,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
        zorder=4,
    )
    total_lines = len(attrs) + len(methods) + (1 if methods else 0)
    available = max(1.8, h - 8.8)
    gap = min(3.0, max(1.25, available / max(total_lines, 1)))
    font_size = 8.2 if gap >= 2.7 else 7.4 if gap >= 2.15 else 6.2
    text_y = y + h - 7.5
    for item in attrs:
        ax.text(x + 1.6, text_y, item, ha="left", va="top", fontsize=font_size, color="#1f2937", zorder=4)
        text_y -= gap
    if methods:
        sep_y = text_y + gap * 0.2
        if y + 2.0 < sep_y < y + h - 7.0:
            ax.plot([x, x + w], [sep_y, sep_y], color=edge, linewidth=0.8, zorder=3)
        text_y -= gap * 0.45
        for item in methods:
            ax.text(x + 1.6, text_y, item, ha="left", va="top", fontsize=font_size, color="#1f2937", zorder=4)
            text_y -= gap


def arrow(ax, start, end, label=None, color="#334155", style="-|>", rad=0.0, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=1.35,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-",
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 1.3,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
            zorder=5,
        )


def note(ax, x, y, text, w=22, h=8, fill="#fff7ed"):
    rect = Rectangle((x, y), w, h, linewidth=1.1, edgecolor="#fb923c", facecolor=fill, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#7c2d12", zorder=3)


def save(fig, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def draw_analysis_class_diagram():
    fig, ax = setup_ax((16, 10), "分析类图：交通流预测系统概念模型")

    class_box(
        ax,
        39,
        76,
        22,
        17,
        "交通流预测系统",
        ["- 数据集集合", "- 模型实验集合", "- 预测与评估结果"],
        ["+ 训练模型()", "+ 展示预测结果()"],
        "#e0f2fe",
        "#0369a1",
    )
    class_box(
        ax,
        5,
        58,
        22,
        15,
        "交通数据集",
        ["- dataset_name", "- graph_signal_matrix", "- distance_matrix"],
        ["+ 加载原始数据()", "+ 生成样本窗口()"],
        "#ecfeff",
        "#0891b2",
    )
    class_box(
        ax,
        5,
        32,
        22,
        15,
        "交通流样本",
        ["- history_window", "- target_horizon", "- normalized_value"],
        ["+ 提供训练输入()", "+ 提供预测标签()"],
        "#f0fdf4",
        "#16a34a",
    )
    class_box(
        ax,
        33,
        58,
        18,
        15,
        "检测节点",
        ["- node_id", "- flow_series", "- spatial_position"],
        ["+ 获取历史流量()"],
        "#f8fafc",
        "#475569",
    )
    class_box(
        ax,
        33,
        32,
        18,
        15,
        "道路边",
        ["- source_node", "- target_node", "- distance"],
        ["+ 表示路网连接()"],
        "#f8fafc",
        "#475569",
    )
    class_box(
        ax,
        61,
        58,
        20,
        15,
        "实验配置",
        ["- model_name", "- learning_rate", "- nb_block/K", "- batch_size/epochs"],
        ["+ 决定训练参数()"],
        "#fefce8",
        "#ca8a04",
    )
    class_box(
        ax,
        61,
        32,
        20,
        15,
        "预测模型",
        ["- ASTGCN", "- MSTGCN", "- GRU"],
        ["+ train()", "+ predict()"],
        "#fff1f2",
        "#e11d48",
    )
    class_box(
        ax,
        75,
        9,
        20,
        15,
        "评估指标",
        ["- MAE", "- RMSE", "- MAPE"],
        ["+ 衡量误差()"],
        "#eef2ff",
        "#4f46e5",
    )
    class_box(
        ax,
        46,
        9,
        21,
        15,
        "预测结果",
        ["- prediction", "- target", "- best_epoch"],
        ["+ 保存npz()", "+ 计算指标()"],
        "#f5f3ff",
        "#7c3aed",
    )
    class_box(
        ax,
        16,
        9,
        22,
        15,
        "结果展示界面",
        ["- loss曲线", "- 节点预测曲线", "- 未来路网风险"],
        ["+ 展示实验结果()"],
        "#fdf4ff",
        "#a21caf",
    )

    arrow(ax, (50, 78), (16, 73), "管理")
    arrow(ax, (16, 58), (16, 47), "切分为")
    arrow(ax, (27, 65), (33, 65), "包含多个")
    arrow(ax, (42, 58), (42, 47), "连接形成")
    arrow(ax, (61, 65), (51, 65), "配置数据")
    arrow(ax, (71, 58), (71, 47), "实例化")
    arrow(ax, (27, 39), (61, 39), "训练/测试输入")
    arrow(ax, (71, 32), (56, 24), "输出")
    arrow(ax, (67, 16), (75, 16), "评价")
    arrow(ax, (46, 16), (38, 16), "展示")
    arrow(ax, (50, 78), (71, 73), "管理")
    note(ax, 75, 78, "分析类图强调业务对象与职责，\n不绑定具体 Python 类名。", 20, 10)
    save(fig, "analysis_class_diagram.png")


def draw_sequence(ax, participants, messages, title):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 97, title, ha="center", va="center", fontsize=20, fontweight="bold", color="#172033")
    xs = [8 + i * (84 / (len(participants) - 1)) for i in range(len(participants))]
    y_top = 88
    y_bottom = 8
    for x, name in zip(xs, participants):
        rect = Rectangle((x - 6.5, y_top), 13, 6.5, linewidth=1.3, edgecolor="#334155", facecolor="#e0f2fe")
        ax.add_patch(rect)
        ax.text(x, y_top + 3.25, name, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.plot([x, x], [y_top, y_bottom], linestyle="--", linewidth=1.0, color="#94a3b8")
    index = {name: xs[i] for i, name in enumerate(participants)}
    for msg in messages:
        sender, receiver, y, text = msg[:4]
        kind = msg[4] if len(msg) > 4 else "call"
        x1 = index[sender]
        x2 = index[receiver]
        if sender == receiver:
            ax.plot([x1, x1 + 5, x1 + 5, x1], [y, y, y - 4, y - 4], color="#334155", linewidth=1.2)
            ax.add_patch(FancyArrowPatch((x1 + 5, y - 4), (x1 + 0.4, y - 4), arrowstyle="-|>", mutation_scale=11, color="#334155"))
            ax.text(x1 + 6, y - 1.6, text, ha="left", va="center", fontsize=8)
            continue
        dashed = kind == "return"
        ax.add_patch(
            FancyArrowPatch(
                (x1, y),
                (x2, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#334155",
                linestyle="--" if dashed else "-",
            )
        )
        ax.text(
            (x1 + x2) / 2,
            y + 1.6,
            text,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )


def draw_analysis_sequence_diagram():
    fig, ax = setup_ax((17, 10))
    participants = ["用户", "配置/数据", "数据预处理", "训练实验", "预测模型", "评估持久化", "可视化界面"]
    messages = [
        ("用户", "配置/数据", 82, "选择数据集、模型和参数"),
        ("配置/数据", "数据预处理", 76, "读取流量序列与距离矩阵"),
        ("数据预处理", "数据预处理", 70, "构造历史窗口和预测标签"),
        ("数据预处理", "训练实验", 64, "输出 train/val/test 样本"),
        ("训练实验", "预测模型", 58, "按配置构建 ASTGCN/MSTGCN/GRU"),
        ("训练实验", "预测模型", 52, "批量训练：前向传播"),
        ("预测模型", "训练实验", 46, "返回预测值"),
        ("训练实验", "训练实验", 40, "计算损失并反向传播"),
        ("训练实验", "评估持久化", 34, "验证集评估并保存最佳权重"),
        ("训练实验", "评估持久化", 28, "测试集预测，保存 npz 和指标"),
        ("用户", "可视化界面", 22, "打开结果仪表盘"),
        ("可视化界面", "评估持久化", 16, "读取日志、权重、预测结果"),
        ("可视化界面", "用户", 10, "展示 loss、节点预测、未来路网风险", "return"),
    ]
    draw_sequence(ax, participants, messages, "分析顺序图：整体业务流程")
    save(fig, "analysis_sequence_diagram.png")


def draw_design_sequence_diagram():
    fig, ax = setup_ax((19, 11))
    participants = [
        "train脚本",
        "lib.utils",
        "make_model",
        "ASTGCN_submodule",
        "ASTGCN_block",
        "Attention/GCN",
        "Optimizer",
        "Dashboard",
    ]
    messages = [
        ("train脚本", "lib.utils", 84, "load_graphdata_channel1(): 构建DataLoader"),
        ("train脚本", "lib.utils", 78, "get_adjacency_matrix(): 读取邻接矩阵"),
        ("train脚本", "make_model", 72, "按配置创建模型"),
        ("make_model", "lib.utils", 66, "scaled_Laplacian() / cheb_polynomial()"),
        ("make_model", "ASTGCN_submodule", 60, "__init__(): 组装BlockList和final_conv"),
        ("train脚本", "ASTGCN_submodule", 54, "for batch: forward(encoder_inputs)"),
        ("ASTGCN_submodule", "ASTGCN_block", 48, "依次执行每个时空块"),
        ("ASTGCN_block", "Attention/GCN", 42, "Temporal_Attention.forward()"),
        ("ASTGCN_block", "Attention/GCN", 36, "Spatial_Attention.forward()"),
        ("ASTGCN_block", "Attention/GCN", 30, "cheb_conv_withSAt.forward() + time_conv"),
        ("ASTGCN_block", "ASTGCN_submodule", 24, "返回特征、空间注意力、时间注意力", "return"),
        ("ASTGCN_submodule", "train脚本", 19, "final_conv输出预测张量", "return"),
        ("train脚本", "Optimizer", 14, "loss.backward(); optimizer.step()"),
        ("train脚本", "lib.utils", 9, "compute_val_loss(); save best; predict_and_save_results()"),
        ("Dashboard", "lib.utils", 4, "ExperimentManager读取日志/权重/npz并绘图"),
    ]
    draw_sequence(ax, participants, messages, "设计顺序图：训练、预测与展示调用链")
    save(fig, "design_sequence_diagram.png")


def draw_design_class_diagram():
    fig, ax = setup_ax((19, 12), "设计类图：交通流预测结果可视化系统")

    class_box(
        ax,
        40,
        88,
        20,
        7,
        "QMainWindow",
        [],
        [],
        "#e2e8f0",
        "#475569",
    )
    class_box(
        ax,
        27,
        66,
        35,
        18,
        "DashboardWindow",
        ["- manager: ExperimentManager", "- current_dataset", "- current_future_snapshot", "- future_zoomed/fullpage_mode"],
        ["+ refresh_all_views()", "+ plot_loss_overview()", "+ run_pure_prediction()", "+ run_future_road_view()"],
        "#fae8ff",
        "#a21caf",
    )
    class_box(
        ax,
        68,
        66,
        27,
        18,
        "可视化控件组",
        ["- dataset_combo / node_spin", "- loss_canvas", "- prediction_canvas", "- future_road_canvas", "- prediction_table"],
        ["+ 展示Loss曲线", "+ 展示节点预测曲线", "+ 展示未来路网风险"],
        "#fdf4ff",
        "#c026d3",
    )
    class_box(
        ax,
        29,
        39,
        38,
        20,
        "ExperimentManager",
        ["- project_root / device", "- config_cache / dataset_cache", "- loss_cache / prediction_cache", "- future_forecast_cache"],
        ["+ available_datasets()", "+ load_loss_series()", "+ pure_predict_best_model()", "+ build_future_road_snapshot()", "+ compute_metrics()"],
        "#ecfccb",
        "#65a30d",
    )
    class_box(
        ax,
        3,
        40,
        21,
        17,
        "<<PretrainedModel>>",
        ["ASTGCN / MSTGCN / GRU", "- 已训练参数: *.params", "- 离线训练完成"],
        ["+ forward(x)", "+ 输出预测结果"],
        "#fee2e2",
        "#dc2626",
    )
    class_box(
        ax,
        72,
        40,
        23,
        17,
        "文件与结果资源",
        ["configurations/*.conf", "logs/*.csv", "data/*.npz", "experiments/*.params"],
        ["+ 提供配置/日志/数据/权重"],
        "#f8fafc",
        "#334155",
    )
    class_box(
        ax,
        3,
        15,
        17,
        15,
        "ModelSpec",
        ["+ label", "+ config_path", "+ color", "用途: 模型显示配置"],
        [],
        "#e0f2fe",
        "#0284c7",
    )
    class_box(
        ax,
        23,
        15,
        17,
        15,
        "LossSeries",
        ["+ epochs", "+ train_losses", "+ val_losses", "+ best_epoch", "用途: Loss曲线数据"],
        [],
        "#e0f2fe",
        "#0284c7",
    )
    class_box(
        ax,
        43,
        15,
        22,
        15,
        "PurePredictionResult",
        ["+ selected_model_label", "+ time_axis", "+ node_target", "+ node_prediction", "用途: 节点预测曲线数据"],
        [],
        "#e0f2fe",
        "#0284c7",
    )
    class_box(
        ax,
        68,
        15,
        27,
        15,
        "FutureRoadSnapshot",
        ["+ selected_edge_nodes", "+ node_flow/risk", "+ edge_flow/risk", "+ top_edges", "用途: 未来路网风险数据"],
        [],
        "#e0f2fe",
        "#0284c7",
    )

    arrow(ax, (50, 88), (45, 84), "继承", style="-|>", dashed=True)
    arrow(ax, (62, 76), (68, 76), "组合界面控件")
    arrow(ax, (45, 67), (48, 59), "组合")
    arrow(ax, (29, 50), (24, 50), "加载并推理", dashed=True)
    arrow(ax, (67, 50), (72, 50), "读取", dashed=True)
    arrow(ax, (48, 39), (12, 30), "生成/使用")
    arrow(ax, (48, 39), (31, 30), "生成/使用")
    arrow(ax, (48, 39), (54, 30), "生成/使用")
    arrow(ax, (55, 39), (81, 30), "生成/使用")
    arrow(ax, (67, 48), (74, 66), "返回可视化数据")

    note(ax, 4, 88, "训练脚本属于离线阶段：\n只负责生成权重和日志，\n不作为可视化系统类图主体。", 24, 8)
    note(ax, 70, 88, "类图依据：\npyqt_result_dashboard.py\nresult_dashboard_core.py", 25, 8, "#eef2ff")
    save(fig, "design_class_diagram.png")


def main():
    configure_matplotlib()
    draw_analysis_class_diagram()
    draw_analysis_sequence_diagram()
    draw_design_class_diagram()
    draw_design_sequence_diagram()


if __name__ == "__main__":
    main()

