#!/usr/bin/env python
# coding: utf-8

import configparser
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import torch

from lib.metrics import masked_mape_np
from lib.utils import get_adjacency_matrix, re_normalization
from model.ASTGCN_ablation import make_model as make_astgcn_ablation_model
from model.ASTGCN_r import make_model as make_astgcn_model
from model.GRU import make_model as make_gru_model
from model.MSTGCN_r import make_model as make_mstgcn_model


DEFAULT_MODEL_CONFIGS = {
    "PEMS04": [
        ("ASTGCN Full", "configurations/PEMS04_astgcn_full.conf", "#1f77b4"),
        ("MSTGCN", "configurations/PEMS04_mstgcn.conf", "#ff7f0e"),
        ("GRU", "configurations/PEMS04_gru.conf", "#2ca02c"),
        ("ASTGCN Temporal", "configurations/PEMS04_astgcn_temporal_only.conf", "#d62728"),
        ("ASTGCN Spatial", "configurations/PEMS04_astgcn_spatial_only.conf", "#9467bd"),
    ],
    "PEMS08": [
        ("ASTGCN Full", "configurations/PEMS08_astgcn_full.conf", "#1f77b4"),
        ("MSTGCN", "configurations/PEMS08_mstgcn.conf", "#ff7f0e"),
        ("GRU", "configurations/PEMS08_gru.conf", "#2ca02c"),
        ("ASTGCN Temporal", "configurations/PEMS08_astgcn_temporal_only.conf", "#d62728"),
        ("ASTGCN Spatial", "configurations/PEMS08_astgcn_spatial_only.conf", "#9467bd"),
    ],
}

DATASET_START_TIMES = {
    # 这里统一把展示时间迁移到 2026 年，用于毕业设计页面展示。
    # 注意：这只是显示层时间轴映射，不会改变模型训练和预测本身。
    "PEMS04": datetime(2026, 1, 1, 0, 0, 0),
    "PEMS08": datetime(2026, 7, 1, 0, 0, 0),
}

@dataclass
class ModelSpec:
    label: str
    config_path: str
    color: str


@dataclass
class LossSeries:
    epochs: list
    train_losses: list
    val_losses: list
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str


@dataclass
class PredictionResult:
    label: str
    color: str
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    sample_index: int
    node_index: int
    node_history: np.ndarray
    node_target: np.ndarray
    node_prediction: np.ndarray
    node_metrics: dict
    sample_metrics: dict


@dataclass
class PurePredictionRank:
    label: str
    color: str
    best_epoch: int
    best_val_loss: float
    calibration_metrics: dict


@dataclass
class PurePredictionResult:
    dataset_name: str
    node_index: int
    start_index: int
    end_index: int
    calibration_start: int
    calibration_end: int
    selected_model_label: str
    selected_model_color: str
    selected_model_epoch: int
    selected_model_val_loss: float
    time_axis: np.ndarray
    time_datetimes: list
    node_target: np.ndarray
    node_prediction: np.ndarray
    future_time_axis: np.ndarray
    future_time_datetimes: list
    future_prediction: np.ndarray
    ranking: list


@dataclass
class TopologySnapshot:
    dataset_name: str
    model_label: str
    model_color: str
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    sample_index: int
    horizon_index: int
    positions: np.ndarray
    edges: np.ndarray
    predicted_flow: np.ndarray
    actual_flow: np.ndarray
    risk_score: np.ndarray
    risk_level: np.ndarray
    summary_metrics: dict
    top_nodes: list


@dataclass
class FutureRoadSnapshot:
    dataset_name: str
    model_label: str
    model_color: str
    best_epoch: int
    best_val_loss: float
    checkpoint_path: str
    selected_edge_index: int
    selected_edge_nodes: tuple
    start_hour: int
    end_hour: int
    start_step: int
    end_step: int
    last_known_datetime: datetime
    future_start_datetime: datetime
    future_end_datetime: datetime
    positions: np.ndarray
    edges: np.ndarray
    node_flow: np.ndarray
    node_risk: np.ndarray
    node_model_labels: list
    node_model_mae: np.ndarray
    edge_flow: np.ndarray
    edge_risk: np.ndarray
    edge_level: np.ndarray
    selected_edge_model_labels: tuple
    selected_edge_node_mae: tuple
    selected_edge_future_flow: float
    selected_edge_future_risk: float
    calibration_start: int
    calibration_end: int
    calibration_ranking: list
    top_edges: list


class ExperimentManager(object):
    def __init__(self, project_root, device=None):
        self.project_root = os.path.abspath(project_root)
        self.device = self._resolve_device(device)
        self._config_cache = {}
        self._dataset_cache = {}
        self._loss_cache = {}
        self._bundle_cache = {}
        self._prediction_cache = {}
        self._range_prediction_cache = {}
        self._pure_ranking_cache = {}
        self._topology_cache = {}
        self._future_forecast_cache = {}
        self._future_node_forecast_cache = {}

    def _resolve_device(self, device):
        if device and device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def available_datasets(self):
        datasets = []
        for dataset_name, items in DEFAULT_MODEL_CONFIGS.items():
            if self.get_model_specs(dataset_name):
                datasets.append(dataset_name)
        return datasets

    def get_model_specs(self, dataset_name):
        specs = []
        for label, config_path, color in DEFAULT_MODEL_CONFIGS.get(dataset_name, []):
            abs_config_path = self._abs_path(config_path)
            if os.path.exists(abs_config_path) and self._config_has_results(abs_config_path):
                specs.append(ModelSpec(label=label, config_path=abs_config_path, color=color))
        return specs

    def load_loss_series(self, spec):
        cache_key = spec.config_path
        if cache_key in self._loss_cache:
            return self._loss_cache[cache_key]

        config = self.load_config(spec.config_path)
        log_path = self._build_log_path(config)
        if not os.path.exists(log_path):
            raise FileNotFoundError("Log file not found: %s" % log_path)

        epochs = []
        train_losses = []
        val_losses = []
        with open(log_path, "r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                val_losses.append(float(row["val_loss"]))

        if not epochs:
            raise ValueError("Log file is empty: %s" % log_path)

        best_index = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        best_epoch = epochs[best_index]
        params_path = self._build_params_path(config)
        checkpoint_path = os.path.join(params_path, "epoch_%s.params" % best_epoch)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Best checkpoint not found: %s" % checkpoint_path)

        loss_series = LossSeries(
            epochs=epochs,
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=val_losses[best_index],
            checkpoint_path=checkpoint_path,
        )
        self._loss_cache[cache_key] = loss_series
        return loss_series

    def representative_cases(self, dataset_name):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            return []

        dataset = self.load_dataset(specs[0].config_path)
        target = dataset["test_target"]
        mean_scores = target.mean(axis=(1, 2))
        variance_scores = target.var(axis=(1, 2))
        mid_index = len(target) // 2

        candidates = [
            ("低流量案例", int(np.argmin(mean_scores))),
            ("高流量案例", int(np.argmax(mean_scores))),
            ("高波动案例", int(np.argmax(variance_scores))),
            ("中间案例", mid_index),
        ]

        unique_cases = []
        used_indices = set()
        for name, sample_index in candidates:
            if sample_index in used_indices:
                continue
            used_indices.add(sample_index)
            unique_cases.append((name, sample_index))
            if len(unique_cases) >= 3:
                break
        return unique_cases

    def dataset_summary(self, dataset_name):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No configuration found for dataset: %s" % dataset_name)

        dataset = self.load_dataset(specs[0].config_path)
        return {
            "num_samples": int(dataset["test_x"].shape[0]),
            "num_nodes": int(dataset["test_x"].shape[1]),
            "input_steps": int(dataset["test_x"].shape[3]),
            "predict_steps": int(dataset["test_target"].shape[2]),
        }

    def topology_summary(self, dataset_name):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No configuration found for dataset: %s" % dataset_name)

        topology = self.load_topology_bundle(specs[0].config_path)
        dataset = self.load_dataset(specs[0].config_path)
        return {
            "num_nodes": int(topology["positions"].shape[0]),
            "num_edges": int(topology["edges"].shape[0]),
            "num_samples": int(dataset["test_x"].shape[0]),
            "predict_steps": int(dataset["test_target"].shape[2]),
            "layout_note": "节点坐标由邻接矩阵生成的虚拟拓扑布局得到，不对应真实地图位置。",
        }

    def future_road_summary(self, dataset_name):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No configuration found for dataset: %s" % dataset_name)

        topology = self.load_topology_bundle(specs[0].config_path)
        time_bundle = self.load_time_series_bundle(specs[0].config_path)
        return {
            "num_nodes": int(topology["positions"].shape[0]),
            "num_edges": int(topology["edges"].shape[0]),
            "points_per_hour": int(time_bundle["points_per_hour"]),
            "max_future_hours": 24,
            "calibration_hours": 21 * 24,
            "last_known_datetime": self._index_to_datetime(
                self.load_config(specs[0].config_path)["Data"]["dataset_name"],
                time_bundle["raw_channel1"].shape[0] - 1,
            ),
            "layout_note": "路网为抽象拓扑图，边关系来自邻接矩阵，颜色高亮表示未来时段潜在拥堵风险。",
        }

    def list_edge_options(self, dataset_name):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No configuration found for dataset: %s" % dataset_name)

        topology = self.load_topology_bundle(specs[0].config_path)
        options = []
        for edge_index, edge in enumerate(topology["edges"]):
            start_node = int(edge[0])
            end_node = int(edge[1])
            options.append((edge_index, "路段 %d: %d -> %d" % (edge_index, start_node, end_node)))
        return options

    def build_future_road_snapshot(self, dataset_name, edge_index, start_hour, end_hour):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No trained model found for dataset: %s" % dataset_name)

        topology = self.load_topology_bundle(specs[0].config_path)
        time_bundle = self.load_time_series_bundle(specs[0].config_path)
        edges = topology["edges"]
        points_per_hour = int(time_bundle["points_per_hour"])
        max_future_hours = 24

        if start_hour < 0 or end_hour > max_future_hours or start_hour >= end_hour:
            raise ValueError("Future time range must satisfy 0 <= start_hour < end_hour <= 24")
        if edge_index < 0 or edge_index >= edges.shape[0]:
            raise IndexError("edge_index out of range: %s" % edge_index)

        calibration_end = int(time_bundle["valid_end"])
        calibration_window = min(21 * 24 * points_per_hour, calibration_end - int(time_bundle["valid_start"]) + 1)
        calibration_start = calibration_end - calibration_window + 1

        future_steps = max_future_hours * points_per_hour
        node_forecast_bundle = self._build_nodewise_future_forecast(
            dataset_name,
            specs,
            calibration_start,
            calibration_end,
            future_steps,
        )
        forecast = node_forecast_bundle["forecast"]

        start_step = int(start_hour * points_per_hour)
        end_step = int(end_hour * points_per_hour)
        selected_window = forecast[start_step:end_step]
        if selected_window.size == 0:
            raise ValueError("Selected future time range is empty")

        node_flow = np.max(selected_window, axis=0).astype(np.float32)
        node_risk = self._boost_contrast(self._normalize_values(node_flow))

        edge_flow = np.maximum(node_flow[edges[:, 0]], node_flow[edges[:, 1]]).astype(np.float32)
        edge_risk = self._boost_contrast(self._normalize_values(edge_flow))
        edge_level = self._risk_levels(edge_risk)

        top_edge_indices = np.argsort(-edge_risk)[:10]
        top_edges = []
        for rank, edge_idx in enumerate(top_edge_indices, start=1):
            start_node = int(edges[edge_idx, 0])
            end_node = int(edges[edge_idx, 1])
            top_edges.append({
                "rank": rank,
                "edge_index": int(edge_idx),
                "start_node": start_node,
                "end_node": end_node,
                "edge_flow": float(edge_flow[edge_idx]),
                "risk_score": float(edge_risk[edge_idx]),
                "risk_level": int(edge_level[edge_idx]),
            })

        selected_edge_nodes = (int(edges[edge_index, 0]), int(edges[edge_index, 1]))
        calibration_ranking = self._rank_models_for_road_window(
            dataset_name,
            specs,
            selected_edge_nodes,
            calibration_start,
            calibration_end,
        )
        raw_length = time_bundle["raw_channel1"].shape[0]
        last_known_datetime = self._index_to_datetime(dataset_name, raw_length - 1)
        return FutureRoadSnapshot(
            dataset_name=dataset_name,
            model_label="Node-wise best",
            model_color="#1565c0",
            best_epoch=-1,
            best_val_loss=float("nan"),
            checkpoint_path="",
            selected_edge_index=int(edge_index),
            selected_edge_nodes=selected_edge_nodes,
            start_hour=int(start_hour),
            end_hour=int(end_hour),
            start_step=int(start_step),
            end_step=int(end_step),
            last_known_datetime=last_known_datetime,
            future_start_datetime=last_known_datetime + timedelta(minutes=int(60 * start_hour)),
            future_end_datetime=last_known_datetime + timedelta(minutes=int(60 * end_hour)),
            positions=topology["positions"],
            edges=edges,
            node_flow=node_flow,
            node_risk=node_risk,
            node_model_labels=node_forecast_bundle["node_model_labels"],
            node_model_mae=node_forecast_bundle["node_model_mae"],
            edge_flow=edge_flow,
            edge_risk=edge_risk,
            edge_level=edge_level,
            selected_edge_model_labels=(
                node_forecast_bundle["node_model_labels"][selected_edge_nodes[0]],
                node_forecast_bundle["node_model_labels"][selected_edge_nodes[1]],
            ),
            selected_edge_node_mae=(
                float(node_forecast_bundle["node_model_mae"][selected_edge_nodes[0]]),
                float(node_forecast_bundle["node_model_mae"][selected_edge_nodes[1]]),
            ),
            selected_edge_future_flow=float(edge_flow[edge_index]),
            selected_edge_future_risk=float(edge_risk[edge_index]),
            calibration_start=int(calibration_start),
            calibration_end=int(calibration_end),
            calibration_ranking=calibration_ranking,
            top_edges=top_edges,
        )

    def build_topology_snapshot(self, dataset_name, model_label, sample_index, horizon_index):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No trained model found for dataset: %s" % dataset_name)

        selected_spec = None
        for spec in specs:
            if spec.label == model_label:
                selected_spec = spec
                break
        if selected_spec is None:
            selected_spec = specs[0]

        topology = self.load_topology_bundle(selected_spec.config_path)
        bundle = self.load_model_bundle(selected_spec)
        dataset = self.load_dataset(selected_spec.config_path)

        num_samples = dataset["test_x"].shape[0]
        predict_steps = dataset["test_target"].shape[2]
        if sample_index < 0 or sample_index >= num_samples:
            raise IndexError("sample_index out of range: %s" % sample_index)
        if horizon_index < 0 or horizon_index >= predict_steps:
            raise IndexError("horizon_index out of range: %s" % horizon_index)

        sample_x = dataset["test_x"][sample_index:sample_index + 1]
        sample_target = dataset["test_target"][sample_index]

        with torch.no_grad():
            inputs = torch.from_numpy(sample_x).type(torch.FloatTensor).to(self.device)
            outputs = bundle["model"](inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            prediction = outputs.detach().cpu().numpy()[0]

        predicted_flow = prediction[:, horizon_index].astype(np.float32)
        actual_flow = sample_target[:, horizon_index].astype(np.float32)
        risk_score = self._normalize_values(predicted_flow)
        risk_level = self._risk_levels(risk_score)

        top_indices = np.argsort(-risk_score)[:10]
        top_nodes = []
        for rank, node_idx in enumerate(top_indices, start=1):
            top_nodes.append({
                "rank": rank,
                "node_index": int(node_idx),
                "predicted_flow": float(predicted_flow[node_idx]),
                "actual_flow": float(actual_flow[node_idx]),
                "risk_score": float(risk_score[node_idx]),
                "risk_level": int(risk_level[node_idx]),
            })

        return TopologySnapshot(
            dataset_name=dataset_name,
            model_label=selected_spec.label,
            model_color=selected_spec.color,
            best_epoch=bundle["loss"].best_epoch,
            best_val_loss=bundle["loss"].best_val_loss,
            checkpoint_path=bundle["loss"].checkpoint_path,
            sample_index=sample_index,
            horizon_index=horizon_index,
            positions=topology["positions"],
            edges=topology["edges"],
            predicted_flow=predicted_flow,
            actual_flow=actual_flow,
            risk_score=risk_score,
            risk_level=risk_level,
            summary_metrics=self.compute_metrics(actual_flow, predicted_flow),
            top_nodes=top_nodes,
        )

    def pure_prediction_summary(self, dataset_name):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No trained model found for dataset: %s" % dataset_name)

        bundle = self.load_time_series_bundle(specs[0].config_path)
        return {
            "num_nodes": int(bundle["raw_channel1"].shape[1]),
            "points_per_hour": int(bundle["points_per_hour"]),
            "minutes_per_step": int(bundle["minutes_per_step"]),
            "history_steps": int(bundle["input_steps"]),
            "predict_steps": int(bundle["num_for_predict"]),
            "valid_start": int(bundle["valid_start"]),
            "valid_end": int(bundle["valid_end"]),
            "case_start": int(bundle["case_start"]),
            "case_end": int(bundle["case_end"]),
            "calibration_start": int(bundle["calibration_start"]),
            "calibration_end": int(bundle["calibration_end"]),
            "window_points": int(bundle["window_points"]),
            "dataset_start_datetime": bundle["dataset_start_datetime"],
            "valid_start_datetime": bundle["valid_start_datetime"],
            "valid_end_datetime": bundle["valid_end_datetime"],
            "case_start_datetime": bundle["case_start_datetime"],
            "case_end_datetime": bundle["case_end_datetime"],
            "calibration_start_datetime": bundle["calibration_start_datetime"],
            "calibration_end_datetime": bundle["calibration_end_datetime"],
        }

    def pure_predict_best_model(
        self,
        dataset_name,
        node_index,
        start_index,
        end_index,
        calibration_start=None,
        calibration_end=None,
    ):
        specs = self.get_model_specs(dataset_name)
        if not specs:
            raise ValueError("No trained model found for dataset: %s" % dataset_name)

        base_bundle = self.load_time_series_bundle(specs[0].config_path)
        num_nodes = base_bundle["raw_channel1"].shape[1]
        if node_index < 0 or node_index >= num_nodes:
            raise IndexError("node_index out of range: %s" % node_index)

        if start_index > end_index:
            raise ValueError("start_index must be <= end_index")
        if start_index < base_bundle["valid_start"] or end_index > base_bundle["valid_end"]:
            raise ValueError(
                "Selected prediction range [%d, %d] must stay inside valid range [%d, %d]"
                % (start_index, end_index, base_bundle["valid_start"], base_bundle["valid_end"])
            )

        if calibration_start is None:
            calibration_start = base_bundle["calibration_start"]
        if calibration_end is None:
            calibration_end = base_bundle["calibration_end"]

        if calibration_start > calibration_end:
            raise ValueError("calibration_start must be <= calibration_end")
        if calibration_start < base_bundle["valid_start"] or calibration_end > base_bundle["valid_end"]:
            raise ValueError(
                "Selected calibration range [%d, %d] must stay inside valid range [%d, %d]"
                % (calibration_start, calibration_end, base_bundle["valid_start"], base_bundle["valid_end"])
            )
        if calibration_end >= start_index:
            raise ValueError("calibration_end must be earlier than prediction start_index")

        ranking = self._get_pure_ranking(dataset_name, node_index, specs, calibration_start, calibration_end)
        selected = ranking[0]

        selected_spec = next(spec for spec in specs if spec.label == selected.label)
        node_prediction, node_target = self._predict_range_first_step(selected_spec, start_index, end_index, node_index)
        future_steps = len(node_prediction)
        future_start_index = int(end_index + 1)
        future_forecast = self._forecast_future_nodes_from_index(selected_spec, future_start_index, future_steps)
        future_end_index = future_start_index + future_steps

        return PurePredictionResult(
            dataset_name=dataset_name,
            node_index=node_index,
            start_index=start_index,
            end_index=end_index,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            selected_model_label=selected.label,
            selected_model_color=selected.color,
            selected_model_epoch=selected.best_epoch,
            selected_model_val_loss=selected.best_val_loss,
            time_axis=np.arange(start_index, end_index + 1, dtype=np.int32),
            time_datetimes=[self._index_to_datetime(dataset_name, idx) for idx in range(start_index, end_index + 1)],
            node_target=node_target,
            node_prediction=node_prediction,
            future_time_axis=np.arange(future_start_index, future_end_index, dtype=np.int32),
            future_time_datetimes=[self._index_to_datetime(dataset_name, idx) for idx in range(future_start_index, future_end_index)],
            future_prediction=future_forecast[:, node_index].astype(np.float32),
            ranking=ranking,
        )

    def _get_pure_ranking(self, dataset_name, node_index, specs, calibration_start, calibration_end):
        cache_key = (dataset_name, node_index, calibration_start, calibration_end)
        if cache_key in self._pure_ranking_cache:
            return self._pure_ranking_cache[cache_key]

        ranking = []
        for spec in specs:
            loss = self.load_loss_series(spec)
            calibration_pred, calibration_target = self._predict_range_first_step(
                spec, calibration_start, calibration_end, node_index
            )
            ranking.append(
                PurePredictionRank(
                    label=spec.label,
                    color=spec.color,
                    best_epoch=loss.best_epoch,
                    best_val_loss=loss.best_val_loss,
                    calibration_metrics=self.compute_metrics(calibration_target, calibration_pred),
                )
            )

        ranking.sort(key=lambda item: item.calibration_metrics["MAE"])
        self._pure_ranking_cache[cache_key] = ranking
        return ranking

    def predict_case(self, spec, sample_index, node_index):
        bundle = self.load_model_bundle(spec)
        dataset = self.load_dataset(spec.config_path)

        num_samples = dataset["test_x"].shape[0]
        num_nodes = dataset["test_x"].shape[1]
        if sample_index < 0 or sample_index >= num_samples:
            raise IndexError("sample_index out of range: %s" % sample_index)
        if node_index < 0 or node_index >= num_nodes:
            raise IndexError("node_index out of range: %s" % node_index)

        sample_x = dataset["test_x"][sample_index:sample_index + 1]
        sample_target = dataset["test_target"][sample_index]

        with torch.no_grad():
            inputs = torch.from_numpy(sample_x).type(torch.FloatTensor).to(self.device)
            outputs = bundle["model"](inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            prediction = outputs.detach().cpu().numpy()[0]

        history = re_normalization(
            sample_x[:, node_index:node_index + 1, 0:1, :],
            dataset["mean"],
            dataset["std"],
        )[0, 0, 0]

        node_target = sample_target[node_index]
        node_prediction = prediction[node_index]

        return PredictionResult(
            label=spec.label,
            color=spec.color,
            best_epoch=bundle["loss"].best_epoch,
            best_val_loss=bundle["loss"].best_val_loss,
            checkpoint_path=bundle["loss"].checkpoint_path,
            sample_index=sample_index,
            node_index=node_index,
            node_history=history,
            node_target=node_target,
            node_prediction=node_prediction,
            node_metrics=self.compute_metrics(node_target, node_prediction),
            sample_metrics=self.compute_metrics(sample_target.reshape(-1), prediction.reshape(-1)),
        )

    def load_model_bundle(self, spec):
        cache_key = spec.config_path
        if cache_key in self._bundle_cache:
            return self._bundle_cache[cache_key]

        config = self.load_config(spec.config_path)
        loss_series = self.load_loss_series(spec)
        model = self._build_model(config)
        state_dict = torch.load(loss_series.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        bundle = {
            "model": model,
            "config": config,
            "loss": loss_series,
        }
        self._bundle_cache[cache_key] = bundle
        return bundle

    def load_dataset(self, config_path):
        config = self.load_config(config_path)
        processed_path = self._build_processed_data_path(config)
        if processed_path in self._dataset_cache:
            return self._dataset_cache[processed_path]

        if not os.path.exists(processed_path):
            raise FileNotFoundError("Processed dataset not found: %s" % processed_path)

        file_data = np.load(processed_path)
        dataset = {
            "test_x": file_data["test_x"][:, :, 0:1, :].astype(np.float32),
            "test_target": file_data["test_target"].astype(np.float32),
            "test_timestamp": file_data["test_timestamp"].astype(np.int32),
            "mean": file_data["mean"][:, :, 0:1, :].astype(np.float32),
            "std": file_data["std"][:, :, 0:1, :].astype(np.float32),
        }
        self._dataset_cache[processed_path] = dataset
        return dataset

    def load_time_series_bundle(self, config_path):
        config = self.load_config(config_path)
        cache_key = ("time_series", self._abs_path(config_path))
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        data_config = config["Data"]
        training_config = config["Training"]
        raw_path = self._abs_path(data_config["graph_signal_matrix_filename"])
        processed_path = self._build_processed_data_path(config)
        if not os.path.exists(raw_path):
            raise FileNotFoundError("Raw dataset not found: %s" % raw_path)
        if not os.path.exists(processed_path):
            raise FileNotFoundError("Processed dataset not found: %s" % processed_path)

        raw_channel1 = np.load(raw_path)["data"][:, :, 0:1].astype(np.float32)
        processed = np.load(processed_path)
        mean = processed["mean"][:, :, 0:1, :].astype(np.float32)
        std = processed["std"][:, :, 0:1, :].astype(np.float32)

        points_per_hour = int(data_config["points_per_hour"])
        minutes_per_step = int(60 / points_per_hour)
        num_for_predict = int(data_config["num_for_predict"])
        num_of_weeks = int(training_config["num_of_weeks"])
        num_of_days = int(training_config["num_of_days"])
        num_of_hours = int(training_config["num_of_hours"])
        dataset_name = data_config["dataset_name"]

        min_start = max(
            num_of_weeks * 7 * 24 * points_per_hour,
            num_of_days * 24 * points_per_hour,
            num_of_hours * points_per_hour,
        )
        raw_length = raw_channel1.shape[0]
        max_start = raw_length - num_for_predict

        # valid_start / valid_end 表示在当前历史依赖配置下，可用于“预测起点”的合法范围。
        # 页面上的时间选择会基于这个范围展开，不再只限制最后两周。
        valid_start = min_start
        valid_end = max_start

        # 保留一组默认值，页面初始仍然落在最后两周，便于直接展示。
        window_points = min(14 * 24 * points_per_hour, max_start - min_start + 1)
        case_start = max(min_start, raw_length - window_points)
        case_end = max_start
        calibration_end = case_start - 1
        calibration_start = max(min_start, calibration_end - window_points + 1)
        dataset_start_datetime = DATASET_START_TIMES.get(dataset_name)

        bundle = {
            "raw_channel1": raw_channel1,
            "mean": mean,
            "std": std,
            "points_per_hour": points_per_hour,
            "minutes_per_step": minutes_per_step,
            "num_for_predict": num_for_predict,
            "num_of_weeks": num_of_weeks,
            "num_of_days": num_of_days,
            "num_of_hours": num_of_hours,
            "input_steps": int(num_of_weeks * 7 * 24 * points_per_hour + num_of_days * 24 * points_per_hour + num_of_hours * points_per_hour),
            "valid_start": valid_start,
            "valid_end": valid_end,
            "case_start": case_start,
            "case_end": case_end,
            "calibration_start": calibration_start,
            "calibration_end": calibration_end,
            "window_points": window_points,
            "dataset_start_datetime": dataset_start_datetime,
            "valid_start_datetime": self._index_to_datetime(dataset_name, valid_start),
            "valid_end_datetime": self._index_to_datetime(dataset_name, valid_end),
            "case_start_datetime": self._index_to_datetime(dataset_name, case_start),
            "case_end_datetime": self._index_to_datetime(dataset_name, case_end),
            "calibration_start_datetime": self._index_to_datetime(dataset_name, calibration_start),
            "calibration_end_datetime": self._index_to_datetime(dataset_name, calibration_end),
        }
        self._dataset_cache[cache_key] = bundle
        return bundle

    def load_topology_bundle(self, config_path):
        cache_key = ("topology", self._abs_path(config_path))
        if cache_key in self._topology_cache:
            return self._topology_cache[cache_key]

        config = self.load_config(config_path)
        data_config = config["Data"]
        adj_filename = self._abs_path(data_config["adj_filename"])
        num_of_vertices = int(data_config["num_of_vertices"])
        id_filename = self._abs_path(data_config["id_filename"]) if config.has_option("Data", "id_filename") else None

        adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
        adj_mx = np.maximum(adj_mx, adj_mx.T).astype(np.float32)
        np.fill_diagonal(adj_mx, 0.0)

        edges = np.argwhere(np.triu(adj_mx > 0, k=1)).astype(np.int32)
        positions = self._build_virtual_topology_layout(adj_mx)

        bundle = {
            "adj_mx": adj_mx,
            "edges": edges,
            "positions": positions,
        }
        self._topology_cache[cache_key] = bundle
        return bundle

    def _forecast_future_nodes(self, spec, future_steps):
        cache_key = (spec.config_path, future_steps)
        if cache_key in self._future_forecast_cache:
            return self._future_forecast_cache[cache_key]

        bundle = self.load_model_bundle(spec)
        time_bundle = self.load_time_series_bundle(spec.config_path)

        sequence = time_bundle["raw_channel1"].copy()
        num_for_predict = int(time_bundle["num_for_predict"])
        generated_steps = 0
        predicted_chunks = []

        with torch.no_grad():
            while generated_steps < future_steps:
                label_start_idx = sequence.shape[0]
                sample_x = self._extract_forecast_sample(sequence, time_bundle, label_start_idx)
                inputs = torch.from_numpy(sample_x[np.newaxis, ...]).type(torch.FloatTensor).to(self.device)
                outputs = bundle["model"](inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                block_prediction = outputs.detach().cpu().numpy()[0]

                take_steps = min(num_for_predict, future_steps - generated_steps)
                taken_block = block_prediction[:, :take_steps].T.astype(np.float32)
                predicted_chunks.append(taken_block)
                sequence = np.concatenate([sequence, taken_block[:, :, None]], axis=0)
                generated_steps += take_steps

        forecast = np.concatenate(predicted_chunks, axis=0)
        self._future_forecast_cache[cache_key] = forecast
        return forecast

    def _forecast_future_nodes_from_index(self, spec, forecast_start_index, future_steps):
        cache_key = ("from_index", spec.config_path, int(forecast_start_index), int(future_steps))
        if cache_key in self._future_forecast_cache:
            return self._future_forecast_cache[cache_key]

        bundle = self.load_model_bundle(spec)
        time_bundle = self.load_time_series_bundle(spec.config_path)
        raw_sequence = time_bundle["raw_channel1"]

        if forecast_start_index <= 0 or forecast_start_index > raw_sequence.shape[0]:
            raise ValueError("forecast_start_index must be inside known raw sequence")

        sequence = raw_sequence[:forecast_start_index].copy()
        num_for_predict = int(time_bundle["num_for_predict"])
        generated_steps = 0
        predicted_chunks = []

        with torch.no_grad():
            while generated_steps < future_steps:
                label_start_idx = sequence.shape[0]
                sample_x = self._extract_forecast_sample(sequence, time_bundle, label_start_idx)
                inputs = torch.from_numpy(sample_x[np.newaxis, ...]).type(torch.FloatTensor).to(self.device)
                outputs = bundle["model"](inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                block_prediction = outputs.detach().cpu().numpy()[0]

                take_steps = min(num_for_predict, future_steps - generated_steps)
                taken_block = block_prediction[:, :take_steps].T.astype(np.float32)
                predicted_chunks.append(taken_block)
                sequence = np.concatenate([sequence, taken_block[:, :, None]], axis=0)
                generated_steps += take_steps

        forecast = np.concatenate(predicted_chunks, axis=0)
        self._future_forecast_cache[cache_key] = forecast
        return forecast

    def _build_nodewise_future_forecast(self, dataset_name, specs, calibration_start, calibration_end, future_steps):
        cache_key = ("nodewise_future", dataset_name, calibration_start, calibration_end, future_steps)
        if cache_key in self._future_node_forecast_cache:
            return self._future_node_forecast_cache[cache_key]

        first_spec_bundle = self.load_time_series_bundle(specs[0].config_path)
        num_nodes = int(first_spec_bundle["raw_channel1"].shape[1])
        best_mae = np.full(num_nodes, np.inf, dtype=np.float32)
        best_spec_index = np.zeros(num_nodes, dtype=np.int32)
        model_forecasts = []

        for spec_index, spec in enumerate(specs):
            pred_all, target_all = self._predict_range_first_step_all(spec, calibration_start, calibration_end)
            node_mae = np.mean(np.abs(pred_all - target_all), axis=0).astype(np.float32)
            improved = node_mae < best_mae
            best_mae[improved] = node_mae[improved]
            best_spec_index[improved] = spec_index
            model_forecasts.append(self._forecast_future_nodes(spec, future_steps))

        combined_forecast = np.zeros((future_steps, num_nodes), dtype=np.float32)
        node_model_labels = []
        node_model_colors = []
        for node_index in range(num_nodes):
            spec_index = int(best_spec_index[node_index])
            combined_forecast[:, node_index] = model_forecasts[spec_index][:, node_index]
            node_model_labels.append(specs[spec_index].label)
            node_model_colors.append(specs[spec_index].color)

        result = {
            "forecast": combined_forecast,
            "node_model_labels": node_model_labels,
            "node_model_colors": node_model_colors,
            "node_model_mae": best_mae,
            "node_model_indices": best_spec_index,
        }
        self._future_node_forecast_cache[cache_key] = result
        return result

    def _rank_models_for_road_window(self, dataset_name, specs, edge_nodes, calibration_start, calibration_end):
        cache_key = ("road_rank", dataset_name, tuple(edge_nodes), calibration_start, calibration_end)
        if cache_key in self._topology_cache:
            return self._topology_cache[cache_key]

        node_indices = [int(edge_nodes[0]), int(edge_nodes[1])]
        ranking = []
        for spec in specs:
            pred_all, target_all = self._predict_range_all_steps(spec, calibration_start, calibration_end)
            loss = self.load_loss_series(spec)
            road_pred = pred_all[:, node_indices, :].reshape(-1)
            road_target = target_all[:, node_indices, :].reshape(-1)
            ranking.append({
                "label": spec.label,
                "color": spec.color,
                "best_epoch": loss.best_epoch,
                "best_val_loss": loss.best_val_loss,
                "metrics": self.compute_metrics(road_target, road_pred),
            })

        ranking.sort(key=lambda item: item["metrics"]["MSE"])
        self._topology_cache[cache_key] = ranking
        return ranking

    def _extract_forecast_sample(self, sequence, time_bundle, label_start_idx):
        num_for_predict = int(time_bundle["num_for_predict"])
        points_per_hour = int(time_bundle["points_per_hour"])
        num_of_weeks = int(time_bundle["num_of_weeks"])
        num_of_days = int(time_bundle["num_of_days"])
        num_of_hours = int(time_bundle["num_of_hours"])

        feature_parts = []

        if num_of_weeks > 0:
            week_indices = self._search_history_indices_for_forecast(
                sequence.shape[0], num_of_weeks, label_start_idx, num_for_predict, 7 * 24, points_per_hour
            )
            if not week_indices:
                raise ValueError("Invalid week history window for forecast at index=%d" % label_start_idx)
            week_sample = np.concatenate([sequence[i:j] for i, j in week_indices], axis=0)
            feature_parts.append(week_sample.transpose(1, 2, 0))

        if num_of_days > 0:
            day_indices = self._search_history_indices_for_forecast(
                sequence.shape[0], num_of_days, label_start_idx, num_for_predict, 24, points_per_hour
            )
            if not day_indices:
                raise ValueError("Invalid day history window for forecast at index=%d" % label_start_idx)
            day_sample = np.concatenate([sequence[i:j] for i, j in day_indices], axis=0)
            feature_parts.append(day_sample.transpose(1, 2, 0))

        if num_of_hours > 0:
            hour_indices = self._search_history_indices_for_forecast(
                sequence.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour
            )
            if not hour_indices:
                raise ValueError("Invalid hour history window for forecast at index=%d" % label_start_idx)
            hour_sample = np.concatenate([sequence[i:j] for i, j in hour_indices], axis=0)
            feature_parts.append(hour_sample.transpose(1, 2, 0))

        if not feature_parts:
            raise ValueError("No history features available for forecast")

        sample_x = np.concatenate(feature_parts, axis=-1)
        sample_x = (sample_x - time_bundle["mean"][0, 0, 0, 0]) / time_bundle["std"][0, 0, 0, 0]
        return sample_x.astype(np.float32)

    def _search_history_indices_for_forecast(
        self,
        sequence_length,
        num_of_depend,
        label_start_idx,
        num_for_predict,
        units,
        points_per_hour,
    ):
        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0 and end_idx <= sequence_length:
                x_idx.append((start_idx, end_idx))
            else:
                return None
        if len(x_idx) != num_of_depend:
            return None
        return x_idx[::-1]

    def load_config(self, config_path):
        abs_config_path = self._abs_path(config_path)
        if abs_config_path in self._config_cache:
            return self._config_cache[abs_config_path]

        if not os.path.exists(abs_config_path):
            raise FileNotFoundError("Config file not found: %s" % abs_config_path)

        config = configparser.ConfigParser()
        config.read(abs_config_path)
        self._config_cache[abs_config_path] = config
        return config

    def compute_metrics(self, target, prediction):
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
        residual = prediction - target
        mse = float(np.mean(np.square(residual)))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residual)))
        error_var = float(np.var(residual))
        denom = float(np.sum(np.square(target - np.mean(target))))
        if denom <= 1e-12:
            r2 = 0.0
        else:
            r2 = float(1.0 - np.sum(np.square(residual)) / denom)
        mape = float(masked_mape_np(target.reshape(-1, 1), prediction.reshape(-1, 1), 0.0))
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "ErrorVar": error_var,
            "R2": r2,
        }

    def _build_model(self, config):
        data_config = config["Data"]
        training_config = config["Training"]

        adj_filename = self._abs_path(data_config["adj_filename"])
        num_of_vertices = int(data_config["num_of_vertices"])
        id_filename = self._abs_path(data_config["id_filename"]) if config.has_option("Data", "id_filename") else None

        adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

        model_name = training_config["model_name"]
        in_channels = int(training_config["in_channels"])
        num_of_hours = int(training_config["num_of_hours"])
        num_of_days = int(training_config["num_of_days"])
        num_of_weeks = int(training_config["num_of_weeks"])
        time_strides = num_of_hours
        num_for_predict = int(data_config["num_for_predict"])
        len_input = int(data_config["len_input"])

        if model_name == "astgcn_r":
            model = make_astgcn_model(
                self.device,
                int(training_config["nb_block"]),
                in_channels,
                int(training_config["K"]),
                int(training_config["nb_chev_filter"]),
                int(training_config["nb_time_filter"]),
                time_strides,
                adj_mx,
                num_for_predict,
                len_input,
                num_of_vertices,
            )
        elif model_name == "mstgcn_r":
            model = make_mstgcn_model(
                self.device,
                int(training_config["nb_block"]),
                in_channels,
                int(training_config["K"]),
                int(training_config["nb_chev_filter"]),
                int(training_config["nb_time_filter"]),
                time_strides,
                adj_mx,
                num_for_predict,
                len_input,
            )
        elif model_name == "gru":
            model = make_gru_model(
                self.device,
                in_channels,
                int(training_config.get("hidden_size", 32)),
                int(training_config.get("num_layers", 1)),
                num_for_predict,
                float(training_config.get("dropout", 0.0)),
            )
        elif model_name.startswith("astgcn_"):
            model = make_astgcn_ablation_model(
                self.device,
                int(training_config["nb_block"]),
                in_channels,
                int(training_config["K"]),
                int(training_config["nb_chev_filter"]),
                int(training_config["nb_time_filter"]),
                time_strides,
                adj_mx,
                num_for_predict,
                len_input,
                num_of_vertices,
                ablation_mode=training_config.get("ablation_mode", "full"),
            )
        else:
            raise ValueError("Unsupported model_name: %s" % model_name)

        return model.to(self.device)

    def _predict_range_first_step(self, spec, start_index, end_index, node_index, batch_size=64):
        cache_key = (spec.config_path, start_index, end_index, node_index)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        prediction_all, target_all = self._predict_range_first_step_all(
            spec, start_index, end_index, batch_size=batch_size
        )
        result = (prediction_all[:, node_index], target_all[:, node_index])
        self._prediction_cache[cache_key] = result
        return result

    def _predict_range_first_step_all(self, spec, start_index, end_index, batch_size=64):
        cache_key = (spec.config_path, start_index, end_index)
        if cache_key in self._range_prediction_cache:
            return self._range_prediction_cache[cache_key]

        bundle = self.load_model_bundle(spec)
        time_bundle = self.load_time_series_bundle(spec.config_path)

        inputs, targets = self._build_samples_for_range(time_bundle, start_index, end_index)
        predictions = []

        with torch.no_grad():
            for offset in range(0, inputs.shape[0], batch_size):
                batch_x = torch.from_numpy(inputs[offset: offset + batch_size]).type(torch.FloatTensor).to(self.device)
                outputs = bundle["model"](batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())

        prediction = np.concatenate(predictions, axis=0)[:, :, 0].astype(np.float32)
        target = targets[:, :, 0].astype(np.float32)
        result = (prediction, target)
        self._range_prediction_cache[cache_key] = result
        return result

    def _predict_range_all_steps(self, spec, start_index, end_index, batch_size=64):
        cache_key = ("all_steps", spec.config_path, start_index, end_index)
        if cache_key in self._range_prediction_cache:
            return self._range_prediction_cache[cache_key]

        bundle = self.load_model_bundle(spec)
        time_bundle = self.load_time_series_bundle(spec.config_path)

        inputs, targets = self._build_samples_for_range(time_bundle, start_index, end_index)
        predictions = []

        with torch.no_grad():
            for offset in range(0, inputs.shape[0], batch_size):
                batch_x = torch.from_numpy(inputs[offset: offset + batch_size]).type(torch.FloatTensor).to(self.device)
                outputs = bundle["model"](batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions.append(outputs.detach().cpu().numpy())

        prediction = np.concatenate(predictions, axis=0).astype(np.float32)
        target = targets.astype(np.float32)
        result = (prediction, target)
        self._range_prediction_cache[cache_key] = result
        return result

    def _build_samples_for_range(self, time_bundle, start_index, end_index):
        xs = []
        ys = []
        for label_start_idx in range(start_index, end_index + 1):
            sample_x, sample_y = self._extract_single_sample(time_bundle, label_start_idx)
            xs.append(sample_x)
            ys.append(sample_y)
        return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)

    def _extract_single_sample(self, time_bundle, label_start_idx):
        raw = time_bundle["raw_channel1"]
        num_for_predict = time_bundle["num_for_predict"]
        points_per_hour = time_bundle["points_per_hour"]
        num_of_weeks = time_bundle["num_of_weeks"]
        num_of_days = time_bundle["num_of_days"]
        num_of_hours = time_bundle["num_of_hours"]

        feature_parts = []

        if num_of_weeks > 0:
            week_indices = self._search_history_indices(
                raw.shape[0], num_of_weeks, label_start_idx, num_for_predict, 7 * 24, points_per_hour
            )
            if not week_indices:
                raise ValueError("Invalid week window at label_start_idx=%d" % label_start_idx)
            week_sample = np.concatenate([raw[i:j] for i, j in week_indices], axis=0)
            feature_parts.append(week_sample.transpose(1, 2, 0))

        if num_of_days > 0:
            day_indices = self._search_history_indices(
                raw.shape[0], num_of_days, label_start_idx, num_for_predict, 24, points_per_hour
            )
            if not day_indices:
                raise ValueError("Invalid day window at label_start_idx=%d" % label_start_idx)
            day_sample = np.concatenate([raw[i:j] for i, j in day_indices], axis=0)
            feature_parts.append(day_sample.transpose(1, 2, 0))

        if num_of_hours > 0:
            hour_indices = self._search_history_indices(
                raw.shape[0], num_of_hours, label_start_idx, num_for_predict, 1, points_per_hour
            )
            if not hour_indices:
                raise ValueError("Invalid hour window at label_start_idx=%d" % label_start_idx)
            hour_sample = np.concatenate([raw[i:j] for i, j in hour_indices], axis=0)
            feature_parts.append(hour_sample.transpose(1, 2, 0))

        if not feature_parts:
            raise ValueError("No history features available for current configuration")

        sample_x = np.concatenate(feature_parts, axis=-1)
        sample_x = (sample_x - time_bundle["mean"][0, 0, 0, 0]) / time_bundle["std"][0, 0, 0, 0]

        sample_y = raw[label_start_idx: label_start_idx + num_for_predict].transpose(1, 2, 0)[:, 0, :]
        return sample_x, sample_y

    def _search_history_indices(self, sequence_length, num_of_depend, label_start_idx, num_for_predict, units, points_per_hour):
        if label_start_idx + num_for_predict > sequence_length:
            return None

        x_idx = []
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i
            end_idx = start_idx + num_for_predict
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None

        if len(x_idx) != num_of_depend:
            return None
        return x_idx[::-1]

    def _build_log_path(self, config):
        data_config = config["Data"]
        training_config = config["Training"]
        folder_dir = self._folder_dir(training_config)
        return os.path.join(self.project_root, "logs", data_config["dataset_name"], folder_dir + ".csv")

    def _build_params_path(self, config):
        data_config = config["Data"]
        training_config = config["Training"]
        folder_dir = self._folder_dir(training_config)
        return os.path.join(self.project_root, "experiments", data_config["dataset_name"], folder_dir)

    def _build_processed_data_path(self, config):
        data_config = config["Data"]
        training_config = config["Training"]
        graph_signal_matrix_filename = self._abs_path(data_config["graph_signal_matrix_filename"])
        file_stem = os.path.basename(graph_signal_matrix_filename).split(".")[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        file_name = "%s_r%s_d%s_w%s_astcgn.npz" % (
            file_stem,
            training_config["num_of_hours"],
            training_config["num_of_days"],
            training_config["num_of_weeks"],
        )
        return os.path.join(dirpath, file_name)

    def _folder_dir(self, training_config):
        return "%s_h%dd%dw%d_channel%d_%e" % (
            training_config["model_name"],
            int(training_config["num_of_hours"]),
            int(training_config["num_of_days"]),
            int(training_config["num_of_weeks"]),
            int(training_config["in_channels"]),
            float(training_config["learning_rate"]),
        )

    def _abs_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.project_root, path))

    def _config_has_results(self, abs_config_path):
        try:
            config = self.load_config(abs_config_path)
            log_path = self._build_log_path(config)
            params_path = self._build_params_path(config)
            return os.path.exists(log_path) and os.path.exists(params_path)
        except Exception:
            return False

    def _index_to_datetime(self, dataset_name, index_value):
        dataset_start = DATASET_START_TIMES.get(dataset_name)
        if dataset_start is None:
            return None
        config_specs = self.get_model_specs(dataset_name)
        if config_specs:
            config = self.load_config(config_specs[0].config_path)
            points_per_hour = int(config["Data"]["points_per_hour"])
        else:
            points_per_hour = 12
        minutes_per_step = int(60 / points_per_hour)
        return dataset_start + timedelta(minutes=index_value * minutes_per_step)

    def _build_virtual_topology_layout(self, adj_mx):
        num_nodes = adj_mx.shape[0]
        if num_nodes <= 1:
            return np.zeros((num_nodes, 2), dtype=np.float32)

        degree = adj_mx.sum(axis=1)
        if np.allclose(degree, 0.0):
            angles = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)
            return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

        inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-6))
        normalized_adj = inv_sqrt_degree[:, None] * adj_mx * inv_sqrt_degree[None, :]
        laplacian = np.eye(num_nodes, dtype=np.float64) - normalized_adj.astype(np.float64)

        try:
            _, eigenvectors = np.linalg.eigh(laplacian)
            x_axis = eigenvectors[:, 1] if num_nodes > 1 else np.zeros(num_nodes)
            y_axis = eigenvectors[:, 2] if num_nodes > 2 else np.sin(np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False))
            positions = np.stack([x_axis, y_axis], axis=1)
        except np.linalg.LinAlgError:
            angles = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)
            positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        positions = positions - positions.mean(axis=0, keepdims=True)
        max_abs = np.max(np.abs(positions))
        if max_abs < 1e-6:
            angles = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False)
            positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            max_abs = 1.0
        positions = positions / max_abs

        degree_scale = self._normalize_values(degree)
        positions = positions * (0.75 + 0.35 * (1.0 - degree_scale[:, None]))
        positions = self._refine_layout_with_forces(adj_mx, positions)
        return positions.astype(np.float32)

    def _refine_layout_with_forces(self, adj_mx, positions, steps=120):
        num_nodes = positions.shape[0]
        if num_nodes <= 1:
            return positions.astype(np.float32)

        positions = positions.astype(np.float64, copy=True)
        rng = np.random.default_rng(num_nodes)
        positions += rng.normal(loc=0.0, scale=0.01, size=positions.shape)
        adjacency = (adj_mx > 0).astype(np.float64)
        np.fill_diagonal(adjacency, 0.0)

        area = 4.0
        k = np.sqrt(area / max(num_nodes, 1))
        temperature = 0.18

        for _ in range(steps):
            delta = positions[:, None, :] - positions[None, :, :]
            distance = np.linalg.norm(delta, axis=2)
            distance = np.maximum(distance, 1e-4)

            repulsive = (k * k / distance)
            repulsive_force = (delta / distance[:, :, None]) * repulsive[:, :, None]
            displacement = repulsive_force.sum(axis=1)

            for i, j in np.argwhere(np.triu(adjacency > 0, k=1)):
                diff = positions[i] - positions[j]
                dist = max(np.linalg.norm(diff), 1e-4)
                attractive = (dist * dist / k)
                direction = diff / dist
                displacement[i] -= direction * attractive
                displacement[j] += direction * attractive

            # 轻微拉回中心，避免节点在迭代中持续被推向边界。
            displacement -= positions * 0.06
            disp_norm = np.linalg.norm(displacement, axis=1, keepdims=True)
            disp_norm = np.maximum(disp_norm, 1e-6)
            positions += (displacement / disp_norm) * np.minimum(disp_norm, temperature)
            temperature *= 0.96

        positions = positions - positions.mean(axis=0, keepdims=True)
        positions += rng.normal(loc=0.0, scale=0.008, size=positions.shape)
        max_radius = np.max(np.linalg.norm(positions, axis=1))
        if max_radius < 1e-6:
            return positions.astype(np.float32)
        positions = positions / max_radius
        positions = positions * 0.82
        return positions.astype(np.float32)

    def _normalize_values(self, values):
        values = np.asarray(values, dtype=np.float32)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value - min_value < 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return (values - min_value) / (max_value - min_value)

    def _boost_contrast(self, values, gamma=0.55):
        values = np.asarray(values, dtype=np.float32)
        values = np.clip(values, 0.0, 1.0)
        return np.power(values, gamma).astype(np.float32)

    def _risk_levels(self, risk_score):
        risk_score = np.asarray(risk_score, dtype=np.float32)
        thresholds = np.quantile(risk_score, [0.50, 0.75, 0.90])
        levels = np.zeros_like(risk_score, dtype=np.int32)
        levels[risk_score >= thresholds[0]] = 1
        levels[risk_score >= thresholds[1]] = 2
        levels[risk_score >= thresholds[2]] = 3
        return levels


def build_subplot_shape(num_plots):
    if num_plots <= 0:
        return 1, 1
    cols = 2 if num_plots > 1 else 1
    rows = int(math.ceil(float(num_plots) / cols))
    return rows, cols
