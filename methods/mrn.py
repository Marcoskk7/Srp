"""
MRN (Meta Relation Network)
元学习关系网络：学习样本间的关系度量，使用 MSE 损失而非交叉熵。
"""
import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from methods.base_trainer import BaseTrainer, EpisodeMetrics
from models.networks import CNN1dEncoder, RelationNetwork1d, init_weights
from data_loader import MetaTask, get_meta_loader


class MRNTrainer(BaseTrainer):
    """MRN 训练器——Meta-learning with Relation Network。"""

    def __init__(self, config: Any):
        super().__init__("MRN", config)
        self.relation_dim = config.model.relation_dim
        self.learning_rate = config.training.learning_rate
        self.train_episode = config.training.train_episode
        self.test_episode = config.training.test_episode
        self.batch_num_per_class = config.training.batch_num_per_class
        self.model_cache_path = f"{config.result_dir}/models/mrn_metatrained.pkl"

    def train(self, metatrain_data: list) -> Tuple[tuple, float]:
        """元训练：学习关系度量空间。"""
        os.makedirs(os.path.dirname(self.model_cache_path), exist_ok=True)

        feature_encoder = CNN1dEncoder(
            feature_dim=self.feature_dim, flatten=False
        ).to(self.device)
        relation_network = RelationNetwork1d(
            input_dim=self.feature_dim, hidden_dim=self.relation_dim
        ).to(self.device)

        if (
            os.path.exists(self.model_cache_path)
            and not self.config.training.force_retrain
        ):
            self.logger.info(f"Loading cached model: {self.model_cache_path}")
            checkpoint = torch.load(self.model_cache_path, map_location=self.device, weights_only=True)
            feature_encoder.load_state_dict(checkpoint["feature_encoder"])
            relation_network.load_state_dict(checkpoint["relation_network"])
            return (feature_encoder, relation_network), 0.0

        start_time = time.time()
        init_weights(feature_encoder)
        init_weights(relation_network)

        optimizer = optim.Adam(
            list(feature_encoder.parameters()) + list(relation_network.parameters()),
            lr=self.learning_rate,
        )
        criterion = nn.MSELoss()

        feature_encoder.train()
        relation_network.train()

        for episode in range(self.train_episode):
            task = MetaTask(
                metatrain_data,
                num_classes=self.num_classes_train,
                support_num=1,
                query_num=10,
                seed=self.config.training.random_seed + episode,
            )
            support_loader = get_meta_loader(
                task,
                num_per_class=1,
                split="support",
                shuffle=False,
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

            support_features = None
            for batch_x, _ in support_loader:
                (batch_x,) = self._to_device(batch_x)
                support_features = feature_encoder(batch_x)
                break

            support_features = support_features.view(
                self.num_classes_train, 1, self.feature_dim, -1
            ).mean(dim=1)

            episode_loss = batch_count = 0
            for batch_x, batch_y in query_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)
                query_features = feature_encoder(batch_x)
                batch_size = query_features.size(0)

                query_expanded = query_features.unsqueeze(1).repeat(
                    1, self.num_classes_train, 1, 1
                )
                support_expanded = support_features.unsqueeze(0).repeat(
                    batch_size, 1, 1, 1
                )
                relation_pairs = torch.cat(
                    [query_expanded, support_expanded], dim=2
                ).view(-1, self.feature_dim * 2, query_features.size(-1))

                relations = relation_network(relation_pairs).view(
                    batch_size, self.num_classes_train
                )
                one_hot = (
                    torch.zeros(batch_size, self.num_classes_train)
                    .to(self.device)
                    .scatter_(1, batch_y.long().view(-1, 1), 1)
                )
                loss = criterion(relations, one_hot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
                batch_count += 1

            if (episode + 1) % 50 == 0:
                self.logger.info(
                    f"Episode {episode + 1}/{self.train_episode} - "
                    f"Loss: {episode_loss / batch_count:.4f}"
                )

        train_time = time.time() - start_time
        torch.save(
            {
                "feature_encoder": feature_encoder.state_dict(),
                "relation_network": relation_network.state_dict(),
            },
            self.model_cache_path,
        )
        self.logger.info(f"Model cached: {self.model_cache_path}")
        return (feature_encoder, relation_network), train_time

    def test(self, model: tuple, metatest_data: list) -> Dict[str, Any]:
        feature_encoder, relation_network = model
        results = {}
        for shot in self.config.get_shot_configs():
            self.logger.info(f"Testing {shot}-shot...")
            shot_acc = self._test_single_shot(
                feature_encoder, relation_network, metatest_data, shot
            )
            results[f"{shot}shot"] = shot_acc
            self.logger.info(
                f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}"
            )
        return results

    def _test_single_shot(
        self,
        feature_encoder: nn.Module,
        relation_network: nn.Module,
        metatest_data: list,
        shot: int,
    ) -> Dict[str, Any]:
        feature_encoder.eval()
        relation_network.eval()
        metrics = EpisodeMetrics()

        for episode in range(self.test_episode):
            available_classes = len(metatest_data)
            num_classes = min(self.num_classes_test, available_classes)

            min_samples = min(len(d) for d, _ in metatest_data)
            query_num = min(10, min_samples - shot)
            if query_num <= 0:
                self.logger.warning(
                    f"跳过 episode: shot={shot}, 最小样本数={min_samples}"
                )
                continue

            augment_num = self._get_augment_num(shot)
            task = MetaTask(
                metatest_data,
                num_classes=num_classes,
                support_num=shot,
                query_num=query_num,
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
                shuffle=False,
                data_type=self.data_type,
                signal_length=self.signal_length,
            )
            query_loader = get_meta_loader(
                task,
                num_per_class=query_num,
                split="query",
                shuffle=False,
                data_type=self.data_type,
                signal_length=self.signal_length,
            )

            try:
                with torch.no_grad():
                    support_features = None
                    for batch_x, _ in support_loader:
                        (batch_x,) = self._to_device(batch_x)
                        support_features = feature_encoder(batch_x)
                        break
                    if support_features is None:
                        continue

                    samples_per_class = shot + augment_num
                    support_features = support_features.view(
                        num_classes, samples_per_class, self.feature_dim, -1
                    ).mean(dim=1)

                    correct = total = 0
                    for batch_x, batch_y in query_loader:
                        batch_x, batch_y = self._to_device(batch_x, batch_y)
                        query_features = feature_encoder(batch_x)
                        batch_size = query_features.size(0)

                        query_expanded = query_features.unsqueeze(1).repeat(
                            1, num_classes, 1, 1
                        )
                        support_expanded = support_features.unsqueeze(0).repeat(
                            batch_size, 1, 1, 1
                        )
                        relation_pairs = torch.cat(
                            [query_expanded, support_expanded], dim=2
                        ).view(-1, self.feature_dim * 2, query_features.size(-1))

                        relations = relation_network(relation_pairs).view(
                            batch_size, num_classes
                        )
                        pred = torch.argmax(relations, dim=1)
                        correct += (pred == batch_y).sum().item()
                        total += batch_y.size(0)

                    if total > 0:
                        metrics.update(correct / total)
            except Exception as exc:
                self.logger.error(f"MRN 测试异常: {exc}")
                continue

        return metrics.compute()
