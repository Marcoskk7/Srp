"""
融合数据集加载器
支持PU和CWRU数据集，不同的任务类型（Direct, Finetune, Meta-learning）

支持两种数据预处理方式：
1. FFT：将时域信号转换为频域特征（默认）
2. Time：直接使用时域原始信号
"""
import os
import random
import numpy as np
from scipy.io import loadmat
from scipy.fftpack import fft
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from typing import List, Tuple, Optional, Dict, Any
import logging


class DataLoaderFactory:
    """数据加载器工厂 - 根据配置创建对应的数据加载器"""

    @staticmethod
    def create_loader(config) -> Any:
        """创建数据加载器"""
        dataset_type = config.data.dataset

        if dataset_type == "PU":
            return PULoader(config)
        elif dataset_type == "CWRU":
            return CWRULoader(config)
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")


class PULoader:
    """PU数据集加载器 - 统一接口

    核心职责：
    1. 加载原始.mat文件数据
    2. 根据配置划分训练/测试类别
    3. 提供统一的数据访问接口
    """

    def __init__(self, config):
        self.config = config
        self.root_dir = config.data.pu_root_dir
        self.train_labels = config.data.pu_train_labels
        self.test_labels = config.data.pu_test_labels
        self.seed = config.data.random_seed
        random.seed(self.seed)

    def load_folders(self, class_num: int) -> Tuple[list, list]:
        """
        加载训练和测试文件夹 - 核心数据划分方法
        """
        all_labels = self.train_labels + self.test_labels
        train_labels = all_labels[:-class_num]  # 前N-class_num个为训练
        test_labels = all_labels[-class_num:]  # 后class_num个为测试

        metatrain_folders = self._load_label_data(train_labels)
        metatest_folders = self._load_label_data(test_labels)

        return metatrain_folders, metatest_folders

    def _load_label_data(self, labels: List[str]) -> list:
        """加载指定标签的所有数据"""
        folders = []
        for label in labels:
            folder_path = os.path.join(self.root_dir, label)
            data = self._load_single_folder(folder_path, label)
            folders.append((data, label))
        return folders

    def _load_single_folder(self, folder_path: str, label: str) -> np.ndarray:
        """加载单个文件夹的所有.mat文件 - 数据解析核心"""
        name_prefix = f'N09_M07_F10_{label}_'
        file_paths = [
            os.path.join(folder_path, name_prefix + str(i))
            for i in range(1, 21)  # 每个类别20个文件
        ]

        all_data = []
        for file_path in file_paths:
            mat_data = loadmat(file_path)
            # ⚠️ 关键：复杂的数组索引，依赖特定的.mat文件结构
            signal = mat_data[file_path.split('\\')[-1]][0][0][2][0][6][2][0]
            # 重塑为固定长度样本：2048点/样本
            signal = signal[:signal.size // 2048 * 2048].reshape(-1, 2048)
            all_data.append(signal)

        return np.vstack(all_data)  # 合并所有文件的样本


class CWRULoader:
    """CWRU数据集加载器 - 支持工况迁移和故障迁移"""

    def __init__(self, config):
        self.config = config
        self.root_dir = config.data.cwru_root_dir
        self.fault_types = config.data.cwru_fault_types
        self.fault_to_label = config.data.cwru_fault_to_label
        self.train_domains = config.data.cwru_train_domains
        self.test_domain = config.data.cwru_test_domain
        self.signal_length = config.data.cwru_signal_length
        self.seed = config.data.random_seed

        # 故障迁移配置
        self.fault_source_codes = config.data.cwru_fault_source_codes
        self.fault_target_codes = config.data.cwru_fault_target_codes
        self.fault_load_condition = config.data.cwru_fault_load_condition

        random.seed(self.seed)

        logging.info(f"CWRU Loader initialized - Task type: {config.data.task_type}")
        logging.info(f"  Data type: {config.data.cwru_data_type}")  # 新增日志
        if config.data.cwru_condition_transfer:
            logging.info(f"  Condition transfer: Train domains: {self.train_domains}, Test domain: {self.test_domain}")
        elif config.data.cwru_fault_transfer:
            logging.info(f"  Fault transfer: Load condition: {self.fault_load_condition}")
            logging.info(f"    Source codes: {self.fault_source_codes}")
            logging.info(f"    Target codes: {self.fault_target_codes}")

    def load_folders(self) -> Tuple[list, list]:
        """
        加载CWRU数据集的源域和目标域数据

        支持两种迁移任务：
        1. 工况迁移：不同负载条件，相同故障类型
        2. 故障迁移：相同负载条件，不同故障类型
        """
        if self.config.data.cwru_condition_transfer:
            return self._load_condition_transfer_data()
        elif self.config.data.cwru_fault_transfer:
            return self._load_fault_transfer_data()
        else:
            raise ValueError("CWRU数据集必须指定迁移类型: condition_transfer 或 fault_transfer")

    def _load_condition_transfer_data(self) -> Tuple[list, list]:
        """加载工况迁移数据（不同负载条件）"""
        logging.info(
            f"Loading CWRU condition transfer data - Train domains: {self.train_domains}, Test domain: {self.test_domain}")

        # 加载训练域数据（多个域合并）
        metatrain_folders = []
        for domain in self.train_domains:
            domain_data = self._load_domain_data(domain)
            metatrain_folders.extend(domain_data)

        # 加载测试域数据
        metatest_folders = self._load_domain_data(self.test_domain)

        # 确保类别标签是连续的（0到9）
        for i, (data, label) in enumerate(metatest_folders):
            metatest_folders[i] = (data, i)

        for i, (data, label) in enumerate(metatrain_folders):
            metatrain_folders[i] = (data, i % 10)  # CWRU工况迁移固定10类

        return metatrain_folders, metatest_folders

    def _load_fault_transfer_data(self) -> Tuple[list, list]:
        """加载故障迁移数据（T2任务）- 相同负载，不同故障类型"""
        logging.info(f"Loading CWRU fault transfer data (T2 task)")
        logging.info(f"  Load condition: {self.fault_load_condition}")
        logging.info(f"  Source fault codes: {self.fault_source_codes}")
        logging.info(f"  Target fault codes: {self.fault_target_codes}")

        # 加载源域故障数据（6-way）
        metatrain_folders = []
        for fault_code in self.fault_source_codes:
            data = self._load_fault_data(self.fault_load_condition, fault_code)
            # 使用故障代码映射到标签
            label = self.fault_to_label.get(fault_code, fault_code)
            metatrain_folders.append((data, label))
            logging.debug(f"  Source: fault_code={fault_code}, label={label}, samples={len(data)}")

        # 加载目标域故障数据（4-way）
        metatest_folders = []
        for fault_code in self.fault_target_codes:
            data = self._load_fault_data(self.fault_load_condition, fault_code)
            # 使用故障代码映射到标签
            label = self.fault_to_label.get(fault_code, fault_code)
            metatest_folders.append((data, label))
            logging.debug(f"  Target: fault_code={fault_code}, label={label}, samples={len(data)}")

        # 重新映射目标域标签为0-3连续值
        for i, (data, label) in enumerate(metatest_folders):
            metatest_folders[i] = (data, i)

        # 重新映射源域标签为0-5连续值
        for i, (data, label) in enumerate(metatrain_folders):
            metatrain_folders[i] = (data, i)

        logging.info(
            f"Fault transfer data loaded - Source classes: {len(metatrain_folders)}, Target classes: {len(metatest_folders)}")
        return metatrain_folders, metatest_folders

    def _load_domain_data(self, domain: int) -> list:
        """加载指定域的所有数据"""
        domain_folders = []

        # 确保域在有效范围内
        if domain not in self.fault_types:
            raise ValueError(f"无效的域: {domain}。有效域: {list(self.fault_types.keys())}")

        fault_codes = self.fault_types[domain]

        for label_idx, fault_code in enumerate(fault_codes):
            # 加载该故障类型的数据
            data = self._load_fault_data(domain, fault_code, label_idx)
            domain_folders.append((data, label_idx))

        return domain_folders

    def _load_fault_data(self, domain: int, fault_code: int, label: Optional[int] = None) -> np.ndarray:
        """加载单个故障类型的数据"""
        axis = "_DE_time"  # 驱动端水平方向

        # 构建文件路径
        if fault_code < 100:
            realaxis = "X0" + str(fault_code) + axis
        else:
            realaxis = "X" + str(fault_code) + axis

        # 构建完整路径
        data_path = os.path.join(self.root_dir, "12k", f"Drive_end_{domain}", f"{fault_code}.mat")

        if not os.path.exists(data_path):
            # 尝试备用路径格式
            data_path = os.path.join(self.root_dir, f"{fault_code}.mat")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"CWRU数据文件不存在: {data_path}")

        # 加载MAT文件
        mat_data = loadmat(data_path)

        # 提取数据
        if realaxis in mat_data:
            data_array = mat_data[realaxis].reshape(-1)
        else:
            # 尝试其他可能的键名
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if len(available_keys) > 0:
                data_array = mat_data[available_keys[0]].reshape(-1)
            else:
                raise KeyError(f"找不到数据键: {realaxis}，可用键: {list(mat_data.keys())}")

        # 归一化
        if self.config.data.normalization:
            data_array = self._normalize(data_array)

        # 滑动窗口采样
        stride = int(self.signal_length * (1 - self.config.data.cwru_overlap_ratio))
        sample_number = (len(data_array) - self.signal_length) // stride + 1

        # 限制样本数量，避免内存问题
        max_samples = 500  # 每个故障类型最多500个样本
        sample_number = min(sample_number, max_samples)

        all_samples = []
        for i in range(sample_number):
            start = i * stride
            end = start + self.signal_length
            sub_data = data_array[start:end]

            # FFT预处理（仅在配置为'fft'时进行）
            if self.config.data.cwru_data_type == 'fft':
                # 执行FFT变换
                sub_data = np.fft.fft(sub_data)
                sub_data = np.abs(sub_data) / len(sub_data)
                sub_data = sub_data[:int(len(sub_data) / 2)]
                target_length = self.config.data.cwru_fft_length
            else:
                # 时域信号，使用原始长度
                target_length = self.signal_length

            # 确保长度一致
            if len(sub_data) > target_length:
                sub_data = sub_data[:target_length]
            elif len(sub_data) < target_length:
                sub_data = np.pad(sub_data, (0, target_length - len(sub_data)))

            all_samples.append(sub_data)

        return np.array(all_samples)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化数据"""
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)


# ========== 任务类（保持不变） ==========
class BaseTask:
    """任务基类 - 定义统一的任务接口"""

    def __init__(self, character_folders: list, seed: Optional[int] = None):
        self.character_folders = character_folders
        if seed is not None:
            np.random.seed(seed)

    def _shuffle_data(self, data: np.ndarray) -> np.ndarray:
        """打乱数据 - 确保任务多样性"""
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        return data[indices]


class DirectTask(BaseTask):
    """Direct Transfer任务 - 用于DTN方法"""

    def __init__(self, character_folders: list, train_num: int,
                 seed: Optional[int] = None):
        super().__init__(character_folders, seed)
        self.train_num = train_num
        self.train_files = []
        self.test_files = []
        self.train_labels = []
        self.test_labels = []
        self._build_task()

    def _build_task(self):
        """构建任务数据 - 固定划分策略"""
        for class_idx, (data, label) in enumerate(self.character_folders):
            data = self._shuffle_data(data)
            # 固定划分：前train_num个训练，其余测试
            self.train_files.extend(data[:self.train_num])
            self.test_files.extend(data[self.train_num:])

            self.train_labels.extend([class_idx] * self.train_num)
            self.test_labels.extend([class_idx] * (len(data) - self.train_num))


class FinetuneTask(BaseTask):
    """Finetune任务 - 用于FTN方法"""

    def __init__(self, character_folders: list, support_num: int,
                 seed: Optional[int] = None,
                 aug_data: Optional[list] = None,
                 augment_num: int = 0):
        super().__init__(character_folders, seed)
        self.support_num = support_num
        self.aug_data = aug_data      # list[ndarray|None], one per class
        self.augment_num = augment_num
        self.support_files = []
        self.query_files = []
        self.support_labels = []
        self.query_labels = []
        self._build_task()

    def _build_task(self):
        """构建支持集和查询集"""
        for class_idx, (data, label) in enumerate(self.character_folders):
            data = self._shuffle_data(data)
            # 支持集：前 support_num 个真实样本
            self.support_files.extend(data[:self.support_num])
            self.support_labels.extend([class_idx] * self.support_num)
            # 查询集：仅使用真实样本（不含增强样本）
            self.query_files.extend(data[self.support_num:])
            self.query_labels.extend([class_idx] * (len(data) - self.support_num))
            # 增强：将 GAN/噪声样本追加到支持集
            if self.aug_data is not None and self.augment_num > 0:
                aug_for_class = self.aug_data[class_idx] if class_idx < len(self.aug_data) else None
                if aug_for_class is not None and len(aug_for_class) > 0:
                    n = min(self.augment_num, len(aug_for_class))
                    indices = np.random.choice(len(aug_for_class), n, replace=False)
                    self.support_files.extend(aug_for_class[indices])
                    self.support_labels.extend([class_idx] * n)


class MetaTask(BaseTask):
    """Meta-learning任务（Episode采样） - 用于MRN/MAML方法"""

    def __init__(self, character_folders: list, num_classes: int,
                 support_num: int, query_num: int, seed: Optional[int] = None,
                 aug_data: Optional[list] = None,
                 augment_num: int = 0):
        super().__init__(character_folders, seed)
        self.num_classes = num_classes  # N-way
        self.support_num = support_num  # K-shot（支持集）
        self.query_num = query_num  # 查询集样本数
        self.aug_data = aug_data      # list[ndarray|None], one per class (aligned with character_folders)
        self.augment_num = augment_num
        self.support_files = []
        self.query_files = []
        self.support_labels = []
        self.query_labels = []
        self._sample_task()

    def _sample_task(self):
        """随机采样N-way K-shot任务"""

        # 确保有足够的类别
        if len(self.character_folders) < self.num_classes:
            self.num_classes = len(self.character_folders)

        # 随机选择N个类别（同时跟踪原始索引以找到对应的增强数据）
        all_indices = list(range(len(self.character_folders)))
        sampled_indices = random.sample(all_indices, self.num_classes)

        for class_idx, orig_idx in enumerate(sampled_indices):
            data, label = self.character_folders[orig_idx]
            data = self._shuffle_data(data)
            # 支持集：前K个真实样本
            self.support_files.extend(data[:self.support_num])
            # 查询集：接下来的query_num个真实样本
            self.query_files.extend(
                data[self.support_num:self.support_num + self.query_num]
            )

            self.support_labels.extend([class_idx] * self.support_num)
            self.query_labels.extend([class_idx] * self.query_num)

            # 增强：将 GAN/噪声样本追加到支持集
            if self.aug_data is not None and self.augment_num > 0:
                aug_for_class = self.aug_data[orig_idx] if orig_idx < len(self.aug_data) else None
                if aug_for_class is not None and len(aug_for_class) > 0:
                    n = min(self.augment_num, len(aug_for_class))
                    indices = np.random.choice(len(aug_for_class), n, replace=False)
                    self.support_files.extend(aug_for_class[indices])
                    self.support_labels.extend([class_idx] * n)


# ========== 数据集类 ==========
# class SignalDataset(Dataset):
#     """信号数据集 - PyTorch Dataset封装
#
#     支持两种数据预处理方式：
#     1. 'fft': 对时域信号进行FFT变换
#     2. 'time': 直接使用时域信号
#     3. 'none': 不做任何处理（数据已经预处理过）
#     """
#
#     def __init__(self, files: list, labels: list, data_type: str = 'fft', signal_length: int = 1024):
#         self.files = files
#         self.labels = np.array(labels)
#         self.data_type = data_type
#         self.signal_length = signal_length
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         signal = self.files[idx]
#
#         # 数据预处理
#         if self.data_type == 'fft':
#             # FFT变换：时域 → 频域
#             # 去均值 + FFT + 取绝对值
#             signal = abs(fft(signal - np.mean(signal)))[:self.signal_length]
#         elif self.data_type == 'time':
#             # 时域信号：只去均值
#             signal = signal - np.mean(signal)
#             # 确保信号长度正确
#             if len(signal) > self.signal_length:
#                 signal = signal[:self.signal_length]
#             elif len(signal) < self.signal_length:
#                 signal = np.pad(signal, (0, self.signal_length - len(signal)))
#         # 如果 data_type 是 'none' 或其他值，直接使用原始信号
#
#         signal = signal.reshape(1, -1)  # [1, signal_length]
#         label = self.labels[idx]
#
#         # 修复：确保返回正确的类型
#         return signal.astype(np.float32), label.astype(np.int64)
# ========== 数据集类（MODIFIED：对 'time' 类型不做去均值）==========
class SignalDataset(Dataset):
    """信号数据集 - PyTorch Dataset封装

    支持两种数据预处理方式：
    1. 'fft': 对时域信号进行FFT变换
    2. 'time': 直接使用时域信号（假设已全局归一化，不做额外处理）
    3. 'none': 不做任何处理（数据已经预处理过）
    """

    def __init__(self, files: list, labels: list, data_type: str = 'fft', signal_length: int = 1024):
        self.files = files
        self.labels = np.array(labels)
        self.data_type = data_type
        self.signal_length = signal_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal = self.files[idx]

        if self.data_type == 'fft':
            # FFT变换：时域 → 频域
            signal = abs(fft(signal - np.mean(signal)))[:self.signal_length]
        elif self.data_type == 'time':
            # 时域信号：只保证长度，不去均值（假设信号已经过全局归一化）
            if len(signal) > self.signal_length:
                signal = signal[:self.signal_length]
            elif len(signal) < self.signal_length:
                signal = np.pad(signal, (0, self.signal_length - len(signal)))
        else:   # 'none' 或其他，仅保证长度
            if len(signal) > self.signal_length:
                signal = signal[:self.signal_length]
            elif len(signal) < self.signal_length:
                signal = np.pad(signal, (0, self.signal_length - len(signal)))

        signal = signal.reshape(1, -1)  # [1, signal_length]
        label = self.labels[idx]
        return signal.astype(np.float32), label.astype(np.int64)




class BalancedSampler(Sampler):
    """类别平衡采样器 - 用于元学习任务"""

    def __init__(self, num_per_class: int, num_classes: int,
                 num_instances: int, shuffle: bool = True):
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [
                [i + j * self.num_instances
                 for i in torch.randperm(self.num_instances)[:self.num_per_class]]
                for j in range(self.num_classes)
            ]
        else:
            batch = [
                [i + j * self.num_instances
                 for i in range(self.num_per_class)]
                for j in range(self.num_classes)
            ]

        batch = [item for sublist in batch for item in sublist]
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1  # 采样器长度，这里返回1因为每次迭代生成一个batch




# ========== 数据加载器工厂函数 ==========
def get_direct_loader(task: DirectTask, batch_size: int,
                      split: str = 'train', shuffle: bool = True,
                      data_type: str = 'fft', signal_length: int = 1024) -> DataLoader:
    """获取Direct任务的DataLoader"""
    files = task.train_files if split == 'train' else task.test_files
    labels = task.train_labels if split == 'train' else task.test_labels

    dataset = SignalDataset(files, labels, data_type, signal_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_finetune_loader(task: FinetuneTask, batch_size: int,
                        split: str = 'support', shuffle: bool = True,
                        data_type: str = 'fft', signal_length: int = 1024) -> DataLoader:
    """获取Finetune任务的DataLoader"""
    files = task.support_files if split == 'support' else task.query_files
    labels = task.support_labels if split == 'support' else task.query_labels

    dataset = SignalDataset(files, labels, data_type, signal_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_meta_loader(task: MetaTask, num_per_class: int,
                    split: str = 'support', shuffle: bool = False,
                    data_type: str = 'fft', signal_length: int = 1024) -> DataLoader:
    """获取Meta-learning任务的DataLoader - 使用平衡采样器"""
    files = task.support_files if split == 'support' else task.query_files
    labels = task.support_labels if split == 'support' else task.query_labels
    # 增强后每类样本数可能不平衡（某些类 aug_data 为 None），取最大值确保覆盖
    from collections import Counter
    label_counts = Counter(labels)
    num_instances = max(label_counts.values()) if label_counts else 0

    dataset = SignalDataset(files, labels, data_type, signal_length)
    sampler = BalancedSampler(
        num_per_class, task.num_classes, num_instances, shuffle
    )

    return DataLoader(
        dataset,
        batch_size=num_per_class * task.num_classes,  # 确保batch包含所有类别
        sampler=sampler
    )