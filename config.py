"""
GenerationConfig: generation 项目原生配置类

将 argparse.Namespace 包装为 methods/ 下各训练器所需的层次化配置接口。
外部只需 `from config import GenerationConfig`，传入 argparse 解析后的 args 即可。
"""
import argparse
import torch
from types import SimpleNamespace
from typing import List


class GenerationConfig:
    """
    generation 项目配置，暴露 config.data / config.model / config.training 接口。

    场景设定（固定，与 generation 的 CWRU 数据对齐）:
        dataset      : CWRU
        task_type    : fault  —— 源域 6 类 (IR/OR 故障) → 目标域 4 类 (Normal/Ball)
        data_type    : time   —— 时域信号，长度 2400
        feature_dim  : 64
        pool_size    : 64     —— CWRU 专用自适应池化输出长度
    """

    def __init__(self, args: argparse.Namespace):
        self.data = SimpleNamespace(
            dataset="CWRU",
            task_type="fault",
            # CWRU 信号参数
            cwru_data_type="time",
            cwru_signal_length=2400,
            cwru_fft_length=512,
            effective_signal_length=2400,   # 时域直接使用原始长度
            # 用 len() 推导类别数；具体值无关紧要
            cwru_fault_source_codes=list(range(6)),   # num_classes_train = 6
            cwru_fault_target_codes=list(range(4)),   # num_classes_test  = 4
            # PU 占位（generation 不使用 PU，但 base_trainer 需要这些字段存在）
            pu_data_type="fft",
            pu_class_num_train=8,
            pu_class_num_test=5,
            pu_signal_length=2048,
            pu_fft_length=1024,
        )

        self.model = SimpleNamespace(
            feature_dim=64,
            relation_dim=8,         # MRN 专用
            adaptive_pool_size=25,  # PU 占位
            cwru_adaptive_pool_size=64,
        )

        self.training = SimpleNamespace(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            learning_rate=0.001,
            train_episode=200,
            test_episode=100,
            finetune_episode=50,
            batch_size_train=64,
            batch_size_test=256,
            batch_num_per_class=15,
            random_seed=getattr(args, "seed", 42),
            force_retrain=getattr(args, "force_regenerate", False),
            # MAML 超参数（fault transfer 分支使用前四项）
            maml_inner_lr=0.001,
            maml_meta_lr=0.001,
            maml_inner_steps=10,
            maml_meta_batch_size=8,
            maml_first_order=True,
            # CWRU condition 分支占位（generation 不走此分支）
            cwru_maml_inner_lr=0.1,
            cwru_maml_meta_lr=0.001,
            cwru_adapt_steps=5,
            cwru_meta_batch_size=32,
            cwru_train_task_num=200,
            # fault transfer 元训练轮数
            fault_transfer_train_task_num=200,
            fault_transfer_test_task_num=100,
        )

        self.result_dir: str = getattr(args, "result_dir", "./experiment_results")
        self._shot_configs: List[int] = list(getattr(args, "shot_configs", [1, 3, 5]))

    def get_shot_configs(self) -> List[int]:
        """返回当前实验的 shot 配置列表。"""
        return self._shot_configs
