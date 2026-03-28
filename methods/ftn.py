"""
FTN (Fine-Tune Transfer Network)
源域预训练 → 分层冻结微调，支持 FTN_u0 ~ FTN_u4。
"""
import copy
import os
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from methods.base_trainer import BaseTrainer, EpisodeMetrics
from models.networks import CNN1dEncoder, LinearClassifier, freeze_layers, init_weights
from data_loader import DirectTask, FinetuneTask, get_direct_loader, get_finetune_loader


class FTNTrainer(BaseTrainer):
    """FTN 训练器，num_unfrozen_layers 控制微调时解冻的 CNN 层数。"""

    def __init__(self, config: Any, num_unfrozen_layers: int = 0):
        super().__init__(f"FTN_u{num_unfrozen_layers}", config)
        self.num_unfrozen_layers = num_unfrozen_layers
        self.learning_rate = config.training.learning_rate
        self.train_episode = config.training.train_episode
        self.finetune_episode = config.training.finetune_episode
        self.test_episode = config.training.test_episode
        self.batch_size_train = config.training.batch_size_train
        self.batch_size_test = config.training.batch_size_test
        self.model_cache_path = (
            f"{config.result_dir}/models/ftn_u{num_unfrozen_layers}_pretrained.pkl"
        )

    def train(self, metatrain_data: list) -> Tuple[nn.Module, float]:
        """源域预训练阶段；已有缓存且未强制重训时直接加载。"""
        os.makedirs(os.path.dirname(self.model_cache_path), exist_ok=True)

        feature_encoder = CNN1dEncoder(
            feature_dim=self.feature_dim,
            flatten=True,
            adaptive_pool_size=self.adaptive_pool_size,
        ).to(self.device)

        if (
            os.path.exists(self.model_cache_path)
            and not self.config.training.force_retrain
        ):
            self.logger.info(f"Loading cached model: {self.model_cache_path}")
            feature_encoder.load_state_dict(
                torch.load(self.model_cache_path, map_location=self.device, weights_only=True)
            )
            return feature_encoder, 0.0

        start_time = time.time()

        if self.dataset_type == "PU":
            input_dim = self.feature_dim * 25
        else:
            input_dim = self.feature_dim * self.adaptive_pool_size

        classifier = LinearClassifier(
            input_dim=input_dim, num_classes=self.num_classes_train
        ).to(self.device)
        init_weights(feature_encoder)
        init_weights(classifier)

        optimizer = optim.Adam(
            list(feature_encoder.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        task = DirectTask(
            metatrain_data, train_num=1000, seed=self.config.training.random_seed
        )
        train_loader = get_direct_loader(
            task,
            batch_size=self.batch_size_train,
            split="train",
            shuffle=True,
            data_type=self.data_type,
            signal_length=self.signal_length,
        )

        feature_encoder.train()
        classifier.train()
        for episode in range(self.train_episode):
            epoch_loss = batch_count = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = self._to_device(batch_x, batch_y)
                loss = criterion(classifier(feature_encoder(batch_x)), batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            if (episode + 1) % 50 == 0:
                self.logger.info(
                    f"Pretrain {episode + 1}/{self.train_episode} - "
                    f"Loss: {epoch_loss / batch_count:.4f}"
                )

        train_time = time.time() - start_time
        torch.save(feature_encoder.state_dict(), self.model_cache_path)
        self.logger.info(f"Model cached: {self.model_cache_path}")
        return feature_encoder, train_time

    def test(self, model: nn.Module, metatest_data: list) -> Dict[str, Any]:
        results = {}
        for shot in self.config.get_shot_configs():
            self.logger.info(f"Testing {shot}-shot with finetune...")
            shot_acc = self._test_single_shot(model, metatest_data, shot)
            results[f"{shot}shot"] = shot_acc
            self.logger.info(
                f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}"
            )
        return results

    def _test_single_shot(
        self, pretrained_encoder: nn.Module, metatest_data: list, shot: int
    ) -> Dict[str, Any]:
        metrics = EpisodeMetrics()
        if self.dataset_type == "PU":
            input_dim = self.feature_dim * 25
        else:
            input_dim = self.feature_dim * self.adaptive_pool_size

        for episode in range(self.test_episode):
            augment_num = self._get_augment_num(shot)
            task = FinetuneTask(
                metatest_data,
                support_num=shot,
                seed=self.config.training.random_seed + episode * 1000,
                aug_data=self.aug_data if augment_num > 0 else None,
                augment_num=augment_num,
            )
            feature_encoder = copy.deepcopy(pretrained_encoder)
            freeze_layers(feature_encoder, self.num_unfrozen_layers)

            classifier = LinearClassifier(
                input_dim=input_dim, num_classes=self.num_classes_test
            ).to(self.device)
            init_weights(classifier)

            accuracy = self._finetune_and_evaluate(feature_encoder, classifier, task)
            metrics.update(accuracy)
        return metrics.compute()

    def _finetune_and_evaluate(
        self, feature_encoder: nn.Module, classifier: nn.Module, task: FinetuneTask
    ) -> float:
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

        params = [p for p in feature_encoder.parameters() if p.requires_grad]
        params += list(classifier.parameters())
        optimizer = optim.Adam(params, lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        feature_encoder.train()
        classifier.train()
        for _ in range(self.finetune_episode):
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
