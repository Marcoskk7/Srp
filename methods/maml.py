"""
MAML (Model-Agnostic Meta-Learning)
内置轻量级 MAML 实现，无需 learn2learn 依赖。
"""
import copy
import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from methods.base_trainer import BaseTrainer, EpisodeMetrics
from models.networks import CNN1dEncoder, LinearClassifier, init_weights
from data_loader import MetaTask, get_meta_loader


class _MAMLLearner(nn.Module):
    """单个内循环 learner，持有独立参数副本。"""

    def __init__(self, module: nn.Module, lr: float, first_order: bool):
        super().__init__()
        self.module = module
        self.lr = lr
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss: torch.Tensor) -> None:
        """执行一步内循环梯度更新。"""
        grads = torch.autograd.grad(
            loss,
            self.module.parameters(),
            create_graph=not self.first_order,
            allow_unused=True,
        )
        for param, grad in zip(self.module.parameters(), grads):
            if grad is not None:
                param.data = param.data - self.lr * grad


class _MAMLWrapper(nn.Module):
    """替代 l2l.algorithms.MAML 的轻量封装。"""

    def __init__(self, module: nn.Module, lr: float, first_order: bool = False):
        super().__init__()
        self.module = module
        self.lr = lr
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self) -> _MAMLLearner:
        """返回持有深拷贝参数的 learner，用于内循环。"""
        cloned = copy.deepcopy(self.module)
        return _MAMLLearner(cloned, self.lr, self.first_order)


class _L2LAlgorithms:
    @staticmethod
    def MAML(module: nn.Module, lr: float, first_order: bool = False) -> _MAMLWrapper:
        return _MAMLWrapper(module, lr=lr, first_order=first_order)


class _L2LCompat:
    algorithms = _L2LAlgorithms()


l2l = _L2LCompat()


class MAMLTrainer(BaseTrainer):
    """MAML 训练器——模型无关元学习。"""

    def __init__(self, config: Any):
        super().__init__("MAML", config)

        # fault transfer 分支（generation 固定走此路径）
        if self.dataset_type == "PU":
            self.inner_lr = config.training.maml_inner_lr
            self.meta_lr = config.training.maml_meta_lr
            self.inner_steps = config.training.maml_inner_steps
            self.meta_batch_size = config.training.maml_meta_batch_size
            self.train_episode = config.training.train_episode
        elif self.task_type == "condition":
            self.inner_lr = config.training.cwru_maml_inner_lr
            self.meta_lr = config.training.cwru_maml_meta_lr
            self.inner_steps = config.training.cwru_adapt_steps
            self.meta_batch_size = config.training.cwru_meta_batch_size
            self.train_episode = config.training.cwru_train_task_num
        else:  # fault transfer
            self.inner_lr = config.training.maml_inner_lr
            self.meta_lr = config.training.maml_meta_lr
            self.inner_steps = config.training.maml_inner_steps
            self.meta_batch_size = config.training.maml_meta_batch_size
            self.train_episode = config.training.fault_transfer_train_task_num

        self.first_order = config.training.maml_first_order
        self.test_episode = config.training.test_episode
        self.input_dim = self.feature_dim * self.adaptive_pool_size
        self.model_cache_path = f"{config.result_dir}/models/maml_metatrained.pkl"

        self.logger.info(
            f"MAML — inner_lr={self.inner_lr}, meta_lr={self.meta_lr}, "
            f"inner_steps={self.inner_steps}, meta_batch={self.meta_batch_size}, "
            f"train_episode={self.train_episode}"
        )

    def train(self, metatrain_data: list) -> Tuple[tuple, float]:
        """元训练：学习可快速适应的初始化参数。"""
        os.makedirs(os.path.dirname(self.model_cache_path), exist_ok=True)

        feature_encoder = CNN1dEncoder(
            feature_dim=self.feature_dim,
            flatten=True,
            adaptive_pool_size=self.adaptive_pool_size,
        ).to(self.device)
        classifier = LinearClassifier(
            input_dim=self.input_dim, num_classes=self.num_classes_train
        ).to(self.device)

        if (
            os.path.exists(self.model_cache_path)
            and not self.config.training.force_retrain
        ):
            self.logger.info(f"Loading cached MAML model: {self.model_cache_path}")
            ckpt = torch.load(self.model_cache_path, map_location=self.device, weights_only=True)
            feature_encoder.load_state_dict(ckpt["feature_encoder"])
            classifier.load_state_dict(ckpt["classifier"])
            maml = l2l.algorithms.MAML(
                nn.Sequential(feature_encoder, classifier),
                lr=self.inner_lr,
                first_order=self.first_order,
            )
            return (feature_encoder, classifier, maml), 0.0

        start_time = time.time()
        init_weights(feature_encoder)
        init_weights(classifier)

        maml = l2l.algorithms.MAML(
            nn.Sequential(feature_encoder, classifier),
            lr=self.inner_lr,
            first_order=self.first_order,
        )
        meta_optimizer = optim.Adam(maml.parameters(), lr=self.meta_lr)
        criterion = nn.CrossEntropyLoss()

        self.logger.info(f"Starting MAML meta-training ({self.train_episode} episodes)...")

        for episode in range(self.train_episode):
            meta_optimizer.zero_grad()
            meta_loss = torch.tensor(0.0, device=self.device)
            episode_accs = []

            for _ in range(self.meta_batch_size):
                learner = maml.clone()
                task = MetaTask(
                    metatrain_data,
                    num_classes=self.num_classes_train,
                    support_num=1,
                    query_num=10,
                    seed=self.config.training.random_seed + episode * 1000,
                )
                support_loader = get_meta_loader(
                    task,
                    num_per_class=1,
                    split="support",
                    shuffle=True,
                    data_type=self.data_type,
                    signal_length=self.signal_length,
                )
                query_loader = get_meta_loader(
                    task,
                    num_per_class=10,
                    split="query",
                    shuffle=True,
                    data_type=self.data_type,
                    signal_length=self.signal_length,
                )

                support_x, support_y = self._to_device(*next(iter(support_loader)))
                for _ in range(self.inner_steps):
                    learner.adapt(criterion(learner(support_x), support_y))

                query_x, query_y = self._to_device(*next(iter(query_loader)))
                query_logits = learner(query_x)
                meta_loss = meta_loss + criterion(query_logits, query_y)

                with torch.no_grad():
                    acc = (torch.argmax(query_logits, dim=1) == query_y).float().mean()
                    episode_accs.append(acc.item())

            (meta_loss / self.meta_batch_size).backward()
            meta_optimizer.step()

            if (episode + 1) % 20 == 0:
                avg_acc = sum(episode_accs) / len(episode_accs)
                self.logger.info(
                    f"Episode {episode + 1}/{self.train_episode} - "
                    f"Meta-Loss: {meta_loss.item() / self.meta_batch_size:.4f}, "
                    f"Query Acc: {avg_acc:.4f}"
                )

        train_time = time.time() - start_time
        torch.save(
            {
                "feature_encoder": feature_encoder.state_dict(),
                "classifier": classifier.state_dict(),
            },
            self.model_cache_path,
        )
        self.logger.info(f"MAML model cached: {self.model_cache_path}")
        return (feature_encoder, classifier, maml), train_time

    def test(self, model: tuple, metatest_data: list) -> Dict[str, Any]:
        _, _, maml = model
        results = {}
        for shot in self.config.get_shot_configs():
            self.logger.info(f"Testing {shot}-shot with MAML...")
            shot_acc = self._test_single_shot(maml, metatest_data, shot)
            results[f"{shot}shot"] = shot_acc
            self.logger.info(
                f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}"
            )
        return results

    def _test_single_shot(self, maml, metatest_data: list, shot: int) -> Dict[str, Any]:
        metrics = EpisodeMetrics()
        criterion = nn.CrossEntropyLoss()

        for episode in range(self.test_episode):
            learner = maml.clone()
            augment_num = self._get_augment_num(shot)
            task = MetaTask(
                metatest_data,
                num_classes=self.num_classes_test,
                support_num=shot,
                query_num=15,
                seed=self.config.training.random_seed + episode * 1000,
                aug_data=self.aug_data if augment_num > 0 else None,
                augment_num=augment_num,
            )
            if not task.support_files or not task.query_files:
                continue

            support_loader = get_meta_loader(
                task,
                num_per_class=shot + augment_num,
                split="support",
                shuffle=True,
                data_type=self.data_type,
                signal_length=self.signal_length,
            )
            query_loader = get_meta_loader(
                task,
                num_per_class=15,
                split="query",
                shuffle=False,
                data_type=self.data_type,
                signal_length=self.signal_length,
            )

            support_x, support_y = self._to_device(*next(iter(support_loader)))
            for _ in range(self.inner_steps):
                learner.adapt(criterion(learner(support_x), support_y))

            correct = total = 0
            for batch_x, batch_y in query_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)
                with torch.no_grad():
                    pred = torch.argmax(learner(batch_x), dim=1)
                    correct += (pred == batch_y).sum().item()
                    total += batch_y.size(0)

            if total > 0:
                metrics.update(correct / total)

        return metrics.compute()
