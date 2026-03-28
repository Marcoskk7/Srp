"""
训练器基类
定义统一的训练和测试接口
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import logging


class BaseTrainer(ABC):
    """训练器基类 - 所有方法的统一接口"""

    def __init__(self, name: str, config: Any):
        """
        Args:
            name: 方法名称
            config: 配置对象
        """
        self.name = name
        self.config = config
        self.device = config.training.device
        self.logger = logging.getLogger(self.name)

        # 获取数据集信息
        self.dataset_type = config.data.dataset
        self.task_type = config.data.task_type if hasattr(config.data, 'task_type') else "condition"

        # 根据数据集类型设置参数
        if self.dataset_type == "PU":
            self.signal_length = config.data.effective_signal_length  # 使用有效信号长度
            self.data_type = config.data.pu_data_type  # 'fft' or 'time'
            self.num_classes_train = config.data.pu_class_num_train
            self.num_classes_test = config.data.pu_class_num_test
        elif self.dataset_type == "CWRU":
            self.signal_length = config.data.effective_signal_length  # 使用有效信号长度
            self.data_type = config.data.cwru_data_type  # 'time' or 'fft'
            self.num_classes_train = 10  # 默认，会根据任务类型调整
            self.num_classes_test = 10  # 默认，会根据任务类型调整

            if self.task_type == "condition":
                # 工况迁移：训练和测试都是10类
                self.num_classes_train = 10
                self.num_classes_test = 10
            else:  # fault transfer
                # 故障迁移：源域和目标域类别数不同
                self.num_classes_train = len(config.data.cwru_fault_source_codes)
                self.num_classes_test = len(config.data.cwru_fault_target_codes)
        else:
            raise ValueError(f"未知的数据集类型: {self.dataset_type}")

        # 设置模型参数
        if self.dataset_type == "PU":
            self.feature_dim = config.model.feature_dim
            self.adaptive_pool_size = config.model.adaptive_pool_size
        else:
            self.feature_dim = config.model.feature_dim
            self.adaptive_pool_size = config.model.cwru_adaptive_pool_size

        # 增强参数（由外部调用方设置，默认不增强）
        self.augment_type: str = 'none'
        self.augment_shot: int = 0
        self.noise_level: float = 0.05
        self.aug_data: list = []   # list[ndarray | None]，与 metatest_data 类别顺序对齐

        # 记录配置信息
        self.logger.info(f"Data configuration: dataset={self.dataset_type}, task_type={self.task_type}")
        self.logger.info(f"  data_type={self.data_type}, signal_length={self.signal_length}")
        self.logger.info(f"  train_classes={self.num_classes_train}, test_classes={self.num_classes_test}")

    @abstractmethod
    def train(self, metatrain_data: list) -> Tuple[nn.Module, float]:
        """训练阶段 - 抽象方法，子类必须实现
        Returns:
            model: 训练好的模型（不同类型方法返回不同结构）
            train_time: 训练耗时（用于性能分析）
        """
        pass

    @abstractmethod
    def test(self, model: nn.Module, metatest_data: list) -> Dict[str, Any]:
        """测试阶段 - 抽象方法，子类必须实现
        Returns:
            results: 结构化测试结果，包含各shot配置的统计信息
        """
        pass

    def run_experiment(self, metatrain_data: list, metatest_data: list,
                       run_id: int = 0) -> Dict[str, Any]:
        """运行完整实验（训练+测试） - 模板方法"""
        self._run_id = run_id  # 供子类在 test() 中使用
        self.logger.info(f"Run {run_id + 1} started")
        self.logger.info(f"  Train classes: {self.num_classes_train}, Test classes: {self.num_classes_test}")

        # 阶段1: 训练 - 调用子类具体实现
        model, train_time = self.train(metatrain_data)

        # 阶段2: 测试 - 调用子类具体实现
        test_results = self.test(model, metatest_data)

        # 阶段3: 结果合并 - 统一的结果格式
        test_results['train_time'] = train_time  # 记录训练耗时
        test_results['run_id'] = run_id  # 标识运行次数

        return test_results

    def _get_augment_num(self, shot: int) -> int:
        """返回当前 shot 下需要补充的增强样本数。"""
        return max(0, self.augment_shot - shot) if self.augment_shot > 0 else 0

    def _to_device(self, *tensors):
        """将张量移动到设备，并确保类型正确 - 实用工具方法"""
        result = []
        for t in tensors:
            if t is None:
                result.append(None)
            else:
                # 转换为torch tensor
                if not isinstance(t, torch.Tensor):
                    t = torch.from_numpy(t)

                # 移动到设备并设置正确的数据类型
                if t.dtype == torch.float64:
                    t = t.float()  # float64 → float32
                elif t.dtype in [torch.int32, torch.int16, torch.int8]:
                    t = t.long()  # 其他整数类型 → int64

                result.append(t.to(self.device))

        return result


class EpisodeMetrics:
    """Episode级别的指标计算器 - 统计聚合工具"""

    def __init__(self):
        self.accuracies = []
        self.losses = []

    def update(self, accuracy: float, loss: float = None):
        """更新指标 - 累积统计信息"""
        self.accuracies.append(accuracy)
        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, Any]:
        """计算汇总统计 - 生成最终评估报告"""
        if not self.accuracies:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0,
                    'median': 0.0, 'all_values': []}
        acc_array = np.array(self.accuracies)

        results = {
            'mean': np.mean(acc_array),
            'std': np.std(acc_array),
            'max': np.max(acc_array),
            'min': np.min(acc_array),
            'median': np.median(acc_array),
            'all_values': self.accuracies
        }

        if self.losses:
            results['avg_loss'] = np.mean(self.losses)

        return results