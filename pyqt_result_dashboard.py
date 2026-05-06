#!/usr/bin/env python
# coding: utf-8

# 要改的地方:选择正确的模型,要求是已经把最优的值算好了然后再绘画路段,而不是像现在一样每个模型有一个属于自己的图.
# 模型预测要有之前的数据,而不是只画未来的.要求前面有实际数据有预测数据,后面半段只有预测数据.
import os
import sys

# Load PyTorch before Qt/Matplotlib on Windows so CUDA DLLs initialize first.
import torch  # noqa: F401

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    print("PyQt5 未安装，请先执行: pip install PyQt5", file=sys.stderr)
    raise

import matplotlib
from matplotlib.figure import Figure

from result_dashboard_core import ExperimentManager, build_subplot_shape

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


class DashboardWindow(QMainWindow):
    def __init__(self, project_root):
        super(DashboardWindow, self).__init__()
        self.project_root = os.path.abspath(project_root)
        self.manager = ExperimentManager(self.project_root, device="auto")
        self.current_dataset = None
        self.current_future_snapshot = None
        self.future_zoomed = False
        self.future_default_limits = None
        self.future_fullpage_mode = False

        self.setWindowTitle("ASTGCN Result Dashboard")
        self.resize(1500, 980)

        self._build_ui()
        self._initialize_state()

    def _build_ui(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)

        refresh_button = QPushButton("刷新视图")
        refresh_button.clicked.connect(self.refresh_all_views)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("数据集"))
        top_layout.addWidget(self.dataset_combo)
        top_layout.addStretch(1)
        top_layout.addWidget(refresh_button)
        root_layout.addLayout(top_layout)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.loss_tab = QWidget()
        self.compare_tab = QWidget()
        self.future_road_tab = QWidget()

        self.tabs.addTab(self.loss_tab, "Loss 曲线")
        self.tabs.addTab(self.compare_tab, "节点纯预测")
        self.tabs.addTab(self.future_road_tab, "未来路网预测")

        self._build_loss_tab()
        self._build_compare_tab()
        self._build_future_road_tab()

    def _build_loss_tab(self):
        layout = QVBoxLayout(self.loss_tab)

        self.loss_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        layout.addWidget(self.loss_canvas)

        self.loss_info_label = QLabel("")
        self.loss_info_label.setWordWrap(True)
        layout.addWidget(self.loss_info_label)

    def _build_compare_tab(self):
        layout = QVBoxLayout(self.compare_tab)

        control_group = QGroupBox("节点纯预测设置")
        control_layout = QGridLayout(control_group)
        control_layout.setHorizontalSpacing(10)
        control_layout.setVerticalSpacing(8)

        self.node_spin = QSpinBox()
        self.node_spin.setMinimum(0)

        self.pure_start_spin = QSpinBox()
        self.pure_end_spin = QSpinBox()

        run_button = QPushButton("运行纯预测")
        run_button.clicked.connect(self.run_pure_prediction)

        control_layout.addWidget(QLabel("节点索引"), 0, 0)
        control_layout.addWidget(self.node_spin, 0, 1)
        control_layout.addWidget(QLabel("预测起点"), 0, 2)
        control_layout.addWidget(self.pure_start_spin, 0, 3)
        control_layout.addWidget(QLabel("预测终点"), 0, 4)
        control_layout.addWidget(self.pure_end_spin, 0, 5)
        control_layout.addWidget(run_button, 0, 6)
        control_layout.setColumnStretch(7, 1)

        self.case_summary_label = QLabel("")
        self.case_summary_label.setWordWrap(True)
        control_layout.addWidget(self.case_summary_label, 1, 0, 1, 7)

        layout.addWidget(control_group)

        self.prediction_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        layout.addWidget(self.prediction_canvas)

        self.prediction_table = QTableWidget(0, 6)
        self.prediction_table.setHorizontalHeaderLabels([
            "Phase",
            "Index",
            "Time",
            "Model",
            "Actual",
            "Pred",
        ])
        self.prediction_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.prediction_table)

    def _build_future_road_tab(self):
        layout = QVBoxLayout(self.future_road_tab)

        self.future_control_group = QGroupBox("未来时段设置")
        control_layout = QGridLayout(self.future_control_group)
        control_layout.setHorizontalSpacing(10)
        control_layout.setVerticalSpacing(8)

        self.future_edge_combo = QComboBox()
        self.future_edge_combo.setMinimumWidth(220)

        self.future_strategy_label = QLabel("节点级三周 MAE 最优")

        self.future_start_hour_spin = QSpinBox()
        self.future_start_hour_spin.setRange(0, 23)
        self.future_start_hour_spin.setValue(7)
        self.future_start_hour_spin.setSuffix(" h")

        self.future_end_hour_spin = QSpinBox()
        self.future_end_hour_spin.setRange(1, 24)
        self.future_end_hour_spin.setValue(9)
        self.future_end_hour_spin.setSuffix(" h")

        run_button = QPushButton("执行未来路网预测")
        run_button.clicked.connect(self.run_future_road_view)

        control_layout.addWidget(QLabel("目标路段"), 0, 0)
        control_layout.addWidget(self.future_edge_combo, 0, 1)
        control_layout.addWidget(QLabel("选模方式"), 0, 2)
        control_layout.addWidget(self.future_strategy_label, 0, 3)
        control_layout.addWidget(QLabel("未来开始"), 0, 4)
        control_layout.addWidget(self.future_start_hour_spin, 0, 5)
        control_layout.addWidget(QLabel("未来结束"), 0, 6)
        control_layout.addWidget(self.future_end_hour_spin, 0, 7)
        control_layout.addWidget(run_button, 0, 8)
        control_layout.setColumnStretch(9, 1)

        self.future_summary_label = QLabel("")
        self.future_summary_label.setWordWrap(True)
        self.future_summary_label.hide()
        control_layout.addWidget(self.future_summary_label, 1, 0, 1, 7)

        layout.addWidget(self.future_control_group)

        future_content_layout = QHBoxLayout()
        self.future_road_canvas = FigureCanvas(Figure(figsize=(11, 8)))
        future_content_layout.addWidget(self.future_road_canvas, 1)
        self.future_road_canvas.mpl_connect("button_press_event", self.on_future_canvas_click)

        self.future_info_group = QGroupBox("选中路段预测值")
        future_info_layout = QVBoxLayout(self.future_info_group)
        self.future_info_label = QLabel("")
        self.future_info_label.setWordWrap(True)
        self.future_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        future_info_layout.addWidget(self.future_info_label)
        future_info_layout.addStretch(1)
        future_content_layout.addWidget(self.future_info_group, 0)

        layout.addLayout(future_content_layout)
        layout.setStretch(1, 1)

    def _initialize_state(self):
        datasets = self.manager.available_datasets()
        if not datasets:
            raise RuntimeError("没有找到可用且已训练完成的配置，请检查 configurations / logs / experiments 目录。")
        self.dataset_combo.addItems(datasets)
        self.dataset_combo.setCurrentIndex(0)
        self.on_dataset_changed(self.dataset_combo.currentText())

    def on_dataset_changed(self, dataset_name):
        if not dataset_name:
            return
        self.current_dataset = dataset_name

        summary = self.manager.pure_prediction_summary(dataset_name)
        self.node_spin.setMaximum(summary["num_nodes"] - 1)
        self.pure_start_spin.setRange(summary["valid_start"], summary["valid_end"])
        self.pure_end_spin.setRange(summary["valid_start"], summary["valid_end"])

        default_end = summary["case_end"]
        default_window = min(summary["points_per_hour"] * 24, default_end - summary["case_start"] + 1)
        default_start = max(summary["case_start"], default_end - default_window + 1)
        self.pure_start_spin.setValue(default_start)
        self.pure_end_spin.setValue(default_end)

        self.case_summary_label.setText(
            "节点数: {num_nodes}，合法预测起点: {valid_start} -> {valid_end}，"
            "默认展示最近 {points_per_hour} 点/小时下约 24 小时的纯预测曲线。".format(**summary)
        )

        self._initialize_future_road_state(dataset_name)
        self.refresh_all_views()

    def _initialize_future_road_state(self, dataset_name):
        summary = self.manager.future_road_summary(dataset_name)
        edge_options = self.manager.list_edge_options(dataset_name)

        self.future_edge_combo.clear()
        for edge_index, label in edge_options:
            self.future_edge_combo.addItem(label, edge_index)

        self.future_summary_label.setText(
            "当前页展示的是抽象路网的未来拥堵预测。"
            " 系统会先用最近三周历史结果为每个节点固定一个最优模型，"
            " 再把所有节点的未来预测组合到同一张路网上；切换目标路段只改变高亮和右侧信息，不会重新改整张图的模型。"
            " 当前节点数: %(num_nodes)d，边数: %(num_edges)d，模型选择窗口: %(calibration_hours)d 小时。"
            % summary
        )

    def refresh_all_views(self):
        self.plot_loss_overview()
        self.run_pure_prediction()
        self.run_future_road_view()

    def plot_loss_overview(self):
        specs = self.manager.get_model_specs(self.current_dataset)
        rows, cols = build_subplot_shape(len(specs))
        figure = self.loss_canvas.figure
        figure.clear()

        for index, spec in enumerate(specs, start=1):
            axis = figure.add_subplot(rows, cols, index)
            loss_series = self.manager.load_loss_series(spec)
            axis.plot(loss_series.epochs, loss_series.train_losses, label="Train", linewidth=1.8, color=spec.color)
            axis.plot(loss_series.epochs, loss_series.val_losses, label="Val", linewidth=1.8, linestyle="--", color="#333333")
            axis.scatter(
                [loss_series.best_epoch],
                [loss_series.best_val_loss],
                color="#d62728",
                s=28,
                zorder=5,
            )
            axis.set_title(spec.label)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.grid(True, linestyle="--", alpha=0.25)
            axis.legend(loc="best", fontsize=8)

        figure.tight_layout()
        self.loss_canvas.draw()

        info_lines = []
        for spec in specs:
            loss_series = self.manager.load_loss_series(spec)
            info_lines.append("%s: best epoch=%d, best val_loss=%.6f" % (spec.label, loss_series.best_epoch, loss_series.best_val_loss))
        self.loss_info_label.setText(" | ".join(info_lines))

    def run_pure_prediction(self):
        if not self.current_dataset:
            return

        node_index = int(self.node_spin.value())
        start_index = int(self.pure_start_spin.value())
        end_index = int(self.pure_end_spin.value())
        if start_index > end_index:
            QMessageBox.warning(self, "参数错误", "预测起点必须小于或等于预测终点。")
            return

        try:
            result = self.manager.pure_predict_best_model(
                self.current_dataset,
                node_index,
                start_index,
                end_index,
            )
            self.plot_pure_prediction(result)
            self.fill_pure_prediction_table(result)
            self.status_bar.showMessage("节点纯预测已更新", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "节点纯预测失败", str(exc))
            self.status_bar.showMessage("节点纯预测失败", 3000)

    def plot_pure_prediction(self, result):
        figure = self.prediction_canvas.figure
        figure.clear()
        axis = figure.add_subplot(111)

        validation_count = len(result.node_prediction)
        future_count = len(result.future_prediction)
        validation_x = list(result.time_axis)
        future_x = list(result.future_time_axis)
        prediction_x = validation_x + future_x
        prediction_y = list(result.node_prediction) + list(result.future_prediction)

        split_index = int(result.end_index) + 0.5
        axis.axvspan(int(result.start_index) - 0.5, int(result.end_index) + 0.5, color="#e8f2ff", alpha=0.45, zorder=0)
        axis.axvspan(int(result.future_time_axis[0]) - 0.5, int(result.future_time_axis[-1]) + 0.5, color="#fff3dd", alpha=0.45, zorder=0)
        axis.axvline(split_index, color="#666666", linestyle=":", linewidth=1.5)

        axis.plot(
            validation_x,
            result.node_target,
            label="Actual",
            linewidth=2.0,
            linestyle="-",
            color="#1565c0",
        )
        axis.plot(
            prediction_x,
            prediction_y,
            label="Pred + Future Pred",
            linewidth=2.0,
            color="#e07a1f",
        )

        y_min, y_max = axis.get_ylim()
        axis.text(
            (int(result.start_index) + int(result.end_index)) / 2.0,
            y_max,
            "Validation: actual + prediction",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#174a7c",
        )
        axis.text(
            (int(result.future_time_axis[0]) + int(result.future_time_axis[-1])) / 2.0,
            y_max,
            "Future: prediction only",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#8a5a00",
        )

        axis.set_title(
            "Node %d | %s | validation index=%d -> %d | future index=%d -> %d"
            % (
                result.node_index,
                result.selected_model_label,
                result.start_index,
                result.end_index,
                int(result.future_time_axis[0]),
                int(result.future_time_axis[-1]),
            )
        )
        axis.set_xlabel("Time index")
        axis.set_ylabel("Traffic Flow")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        self.prediction_canvas.draw()

    def fill_pure_prediction_table(self, result):
        validation_rows = len(result.node_prediction)
        future_rows = len(result.future_prediction)
        self.prediction_table.setRowCount(validation_rows + future_rows)

        for row_index, prediction_value in enumerate(result.node_prediction):
            time_value = result.time_datetimes[row_index]
            time_text = time_value.strftime("%Y-%m-%d %H:%M") if time_value is not None else "-"
            items = [
                "Validation",
                str(int(result.time_axis[row_index])),
                time_text,
                result.selected_model_label,
                "%.6f" % float(result.node_target[row_index]),
                "%.6f" % float(prediction_value),
            ]
            for col_index, value in enumerate(items):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignCenter)
                self.prediction_table.setItem(row_index, col_index, table_item)

        for future_index, prediction_value in enumerate(result.future_prediction):
            row_index = validation_rows + future_index
            time_value = result.future_time_datetimes[future_index]
            time_text = time_value.strftime("%Y-%m-%d %H:%M") if time_value is not None else "-"
            items = [
                "Future",
                str(int(result.future_time_axis[future_index])),
                time_text,
                result.selected_model_label,
                "-",
                "%.6f" % float(prediction_value),
            ]
            for col_index, value in enumerate(items):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignCenter)
                self.prediction_table.setItem(row_index, col_index, table_item)

        self.prediction_table.resizeColumnsToContents()

    def run_future_road_view(self):
        if not self.current_dataset:
            return

        start_hour = int(self.future_start_hour_spin.value())
        end_hour = int(self.future_end_hour_spin.value())
        if start_hour >= end_hour:
            QMessageBox.warning(self, "参数错误", "未来开始时间必须早于未来结束时间。")
            return

        edge_index = int(self.future_edge_combo.currentData() or 0)

        try:
            snapshot = self.manager.build_future_road_snapshot(
                self.current_dataset,
                edge_index,
                start_hour,
                end_hour,
            )
            self.current_future_snapshot = snapshot
            self.plot_future_road_snapshot(snapshot)

            best_rank = snapshot.calibration_ranking[0]
            self.future_info_label.setText(
                "目标路段\n"
                "%d -> %d\n\n"
                "选中时间\n"
                "%s 至 %s\n\n"
                "选中时段预测值\n"
                "%.6f\n\n"
                "风险分数\n"
                "%.6f\n\n"
                "端点节点模型\n"
                "%d: %s (MAE=%.6f)\n"
                "%d: %s (MAE=%.6f)\n\n"
                "选中路段候选模型参考\n"
                "%s\n\n"
                "三周路段校准指标\n"
                "MSE=%.6f\nMAE=%.6f\nRMSE=%.6f\nMAPE=%.6f"
                % (
                    snapshot.selected_edge_nodes[0],
                    snapshot.selected_edge_nodes[1],
                    snapshot.future_start_datetime.strftime("%Y-%m-%d %H:%M"),
                    snapshot.future_end_datetime.strftime("%Y-%m-%d %H:%M"),
                    snapshot.selected_edge_future_flow,
                    snapshot.selected_edge_future_risk,
                    snapshot.selected_edge_nodes[0],
                    snapshot.selected_edge_model_labels[0],
                    snapshot.selected_edge_node_mae[0],
                    snapshot.selected_edge_nodes[1],
                    snapshot.selected_edge_model_labels[1],
                    snapshot.selected_edge_node_mae[1],
                    best_rank["label"],
                    best_rank["metrics"]["MSE"],
                    best_rank["metrics"]["MAE"],
                    best_rank["metrics"]["RMSE"],
                    best_rank["metrics"]["MAPE"],
                )
            )
            self.status_bar.showMessage("未来路网预测已更新", 3000)
        except Exception as exc:
            self.current_future_snapshot = None
            QMessageBox.critical(self, "未来路网预测失败", str(exc))
            self.status_bar.showMessage("未来路网预测失败", 3000)

    def plot_future_road_snapshot(self, snapshot):
        figure = self.future_road_canvas.figure
        figure.clear()
        axis = figure.add_subplot(111)

        positions = snapshot.positions
        edges = snapshot.edges

        for edge in edges:
            start_node = int(edge[0])
            end_node = int(edge[1])
            axis.plot(
                [positions[start_node, 0], positions[end_node, 0]],
                [positions[start_node, 1], positions[end_node, 1]],
                color="#8c8c8c",
                linewidth=1.8,
                alpha=0.65,
                zorder=1,
            )

        palette = ["#ffd54f", "#ffb300", "#fb8c00", "#e53935", "#7f0000"]
        for edge_index, edge in enumerate(edges):
            start_node = int(edge[0])
            end_node = int(edge[1])
            risk = float(snapshot.edge_risk[edge_index])
            level = int(snapshot.edge_level[edge_index])
            axis.plot(
                [positions[start_node, 0], positions[end_node, 0]],
                [positions[start_node, 1], positions[end_node, 1]],
                color=palette[level],
                linewidth=2.6 + 7.5 * risk,
                alpha=0.92,
                zorder=2,
            )

        selected_edge = snapshot.selected_edge_index
        selected_nodes = snapshot.edges[selected_edge]
        axis.plot(
            [positions[selected_nodes[0], 0], positions[selected_nodes[1], 0]],
            [positions[selected_nodes[0], 1], positions[selected_nodes[1], 1]],
            color="#1565c0",
            linewidth=8.5,
            alpha=0.95,
            zorder=3,
        )

        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=8.0 + 28.0 * snapshot.node_risk,
            c="#2f3640",
            alpha=0.7,
            linewidths=0.0,
            zorder=4,
        )

        for item in snapshot.top_edges[:5]:
            edge = snapshot.edges[item["edge_index"]]
            start_node = int(edge[0])
            end_node = int(edge[1])
            mid_x = (positions[start_node, 0] + positions[end_node, 0]) / 2.0
            mid_y = (positions[start_node, 1] + positions[end_node, 1]) / 2.0
            axis.text(mid_x, mid_y, str(item["rank"]), fontsize=8, ha="center", va="center", color="#111111", zorder=5)

        axis.set_title(
            "%s 抽象路网拥堵预测 | 目标路段 %d -> %d | 节点级最优模型 | 未来 %d-%d 小时"
            % (
                snapshot.dataset_name,
                snapshot.selected_edge_nodes[0],
                snapshot.selected_edge_nodes[1],
                snapshot.start_hour,
                snapshot.end_hour,
            )
        )
        axis.set_aspect("equal")
        axis.margins(x=0.08, y=0.08)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        self.future_default_limits = (axis.get_xlim(), axis.get_ylim())
        self.future_zoomed = False
        figure.tight_layout()
        self.future_road_canvas.draw()

    def on_future_canvas_click(self, event):
        if event.inaxes is None or self.current_future_snapshot is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        snapshot = self.current_future_snapshot
        positions = snapshot.positions
        edges = snapshot.edges
        x_min, x_max = event.inaxes.get_xlim()
        y_min, y_max = event.inaxes.get_ylim()
        span = max(abs(x_max - x_min), abs(y_max - y_min))
        edge_threshold = 0.03 * span
        node_threshold = 0.025 * span

        nearest_edge_index = None
        nearest_distance = None
        for edge_index, edge in enumerate(edges):
            start_node = int(edge[0])
            end_node = int(edge[1])
            start_point = positions[start_node]
            end_point = positions[end_node]
            distance = self._distance_point_to_segment(
                event.xdata,
                event.ydata,
                float(start_point[0]),
                float(start_point[1]),
                float(end_point[0]),
                float(end_point[1]),
            )
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_edge_index = edge_index

        nearest_node_distance = None
        for node_position in positions:
            distance = ((event.xdata - float(node_position[0])) ** 2 + (event.ydata - float(node_position[1])) ** 2) ** 0.5
            if nearest_node_distance is None or distance < nearest_node_distance:
                nearest_node_distance = distance

        if nearest_edge_index is not None and nearest_distance is not None and nearest_distance <= edge_threshold:
            combo_index = self.future_edge_combo.findData(int(nearest_edge_index))
            if combo_index >= 0:
                self.future_edge_combo.setCurrentIndex(combo_index)
                self.run_future_road_view()
        elif nearest_node_distance is None or nearest_node_distance > node_threshold:
            self.toggle_future_zoom(event.inaxes)

    def toggle_future_zoom(self, axis):
        if self.future_default_limits is None:
            return

        default_xlim, default_ylim = self.future_default_limits
        if self.future_zoomed:
            axis.set_xlim(default_xlim)
            axis.set_ylim(default_ylim)
            self.future_zoomed = False
            self._set_future_fullpage_mode(False)
        else:
            axis.set_xlim(default_xlim)
            axis.set_ylim(default_ylim)
            self.future_zoomed = True
            self._set_future_fullpage_mode(True)
        self.future_road_canvas.draw_idle()

    def _set_future_fullpage_mode(self, enabled):
        if self.future_fullpage_mode == enabled:
            return
        self.future_fullpage_mode = enabled
        self.future_control_group.setVisible(not enabled)
        self.future_info_group.setVisible(not enabled)

    def _distance_point_to_segment(self, px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def main():
    app = QApplication(sys.argv)
    window = DashboardWindow(os.path.dirname(os.path.abspath(__file__)))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
