"""
DTN (Direct Training Network): 理论下界
每个 episode 独立随机初始化并训练，模拟无预训练的最坏情况。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Tuple

from methods.base_trainer import BaseTrainer, EpisodeMetrics
from models.networks import CNN1dEncoder, LinearClassifier, init_weights
from data_loader import FinetuneTask, get_finetune_loader


class DTNTrainer(BaseTrainer):
    """DTN: 无预训练的理论下界。"""

    def __init__(self, config: Any):
        super().__init__("DTN", config)

        if self.dataset_type == "PU":
            self.input_dim = self.feature_dim * 25
        else:
            self.input_dim = self.feature_dim * config.model.cwru_adaptive_pool_size

        self.learning_rate = config.training.learning_rate
        self.train_episode = config.training.finetune_episode
        self.test_episode = config.training.test_episode
        self.batch_size_test = config.training.batch_size_test
    def train(self, metatrain_data: list) -> Tuple[None, float]:
        """预训练阶段——完全跳过。"""
        self.logger.warning("DTN: No pre-training (theoretical lower bound)")
        return None, 0.0

    def test(self, model: None, metatest_data: list) -> Dict[str, Any]:
        """测试阶段：对每个 shot 配置运行 test_episode 个 episode。"""
        run_seed = self.config.training.random_seed + getattr(self, '_run_id', 0) * 100000
        self.logger.info(f"Using run_seed={run_seed}")

        results = {}
        for shot in self.config.get_shot_configs():
            self.logger.info(f"Testing {shot}-shot (training from scratch)...")
            shot_acc = self._test_single_shot(metatest_data, shot, run_seed)
            results[f"{shot}shot"] = shot_acc
            self.logger.info(
                f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}"
            )
        return results

    def _test_single_shot(
        self, metatest_data: list, shot: int, run_seed: int
    ) -> Dict[str, Any]:
        metrics = EpisodeMetrics()
        for episode in range(self.test_episode):
            episode_seed = run_seed + episode * 1000
            torch.manual_seed(episode_seed)
            np.random.seed(episode_seed)

            augment_num = self._get_augment_num(shot)
            task = FinetuneTask(
                metatest_data, support_num=shot, seed=episode_seed,
                aug_data=self.aug_data if augment_num > 0 else None,
                augment_num=augment_num,
            )

            support_loader = get_finetune_loader(
                task,
                batch_size=len(task.support_files),
                split="support",
                shuffle=True,
                data_type=self.data_type,
                signal_length=self.signal_length,
            )
            query_loader = get_finetune_loader(
                task,
                batch_size=self.batch_size_test,
                split="query",
                shuffle=False,
                data_type=self.data_type,
                signal_length=self.signal_length,
            )

            accuracy = self._train_and_evaluate(
                support_loader, query_loader, self.num_classes_test, episode_seed
            )
            metrics.update(accuracy)

            if (episode + 1) % 20 == 0:
                self.logger.info(
                    f"Episode {episode + 1}/{self.test_episode} - Acc: {accuracy:.4f}"
                )
        return metrics.compute()

    def _train_and_evaluate(
        self, support_loader, query_loader, num_classes: int, episode_seed: int
    ) -> float:
        torch.manual_seed(episode_seed)
        np.random.seed(episode_seed)

        feature_encoder = CNN1dEncoder(
            feature_dim=self.feature_dim,
            flatten=True,
            adaptive_pool_size=self.config.model.cwru_adaptive_pool_size,
        ).to(self.device)
        classifier = LinearClassifier(
            input_dim=self.input_dim, num_classes=num_classes
        ).to(self.device)
        init_weights(feature_encoder)
        init_weights(classifier)

        optimizer = optim.Adam(
            list(feature_encoder.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        feature_encoder.train()
        classifier.train()
        for _ in range(self.train_episode):
            for batch_x, batch_y in support_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)
                loss = criterion(classifier(feature_encoder(batch_x)), batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        feature_encoder.eval()
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in query_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)
                pred = torch.argmax(classifier(feature_encoder(batch_x)), dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        return correct / total
