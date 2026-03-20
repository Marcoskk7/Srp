"""
神经网络模型定义
包含三种实验的网络结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
import numpy as np


def maml_init_(module):
    """MAML专用的初始化方法"""
    if isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    """专用卷积块 - 用于CWRU故障迁移任务"""

    def __init__(self, in_channels, out_channels, kernel_size, max_pool_factor=1.0):
        super().__init__()
        stride = int(2 * max_pool_factor)
        self.max_pool = nn.MaxPool1d(kernel_size=stride, stride=stride, ceil_mode=False)
        self.normalize = nn.BatchNorm1d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):
    """专用卷积基网络"""

    def __init__(self, hidden=64, channels=1, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, 3, max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, 3, max_pool_factor))
        super(ConvBase, self).__init__(*core)


class CNN4Backbone(ConvBase):
    """专用CNN骨干网络"""

    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class Net4CNN(torch.nn.Module):
    """专用CNN网络 - 用于CWRU故障迁移任务"""

    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers,
                                     max_pool_factor=4 // layers)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN1dEncoder(nn.Module):
    """1D卷积编码器 - 用于PU和CWRU工况迁移"""

    def __init__(self, feature_dim: int = 64, flatten: bool = False,
                 adaptive_pool_size: int = 25, conv_channels: Optional[List[int]] = None,
                 kernel_sizes: Optional[List[int]] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.flatten = flatten

        if conv_channels is None:
            conv_channels = [64, 64, 64, 64]
        if kernel_sizes is None:
            kernel_sizes = [10, 3, 3, 3]

        assert len(conv_channels) == len(kernel_sizes)

        self.conv_layers = nn.ModuleList()

        in_channels = 1
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                         stride=3 if i == 0 else 1, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels, momentum=1, affine=True),
                nn.ReLU(),
                nn.MaxPool1d(2) if i < 2 else nn.Identity()
            )
            self.conv_layers.append(conv_layer)
            in_channels = out_channels

        if conv_channels[-1] != feature_dim:
            self.final_conv = nn.Sequential(
                nn.Conv1d(conv_channels[-1], feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
                nn.ReLU()
            )
        else:
            self.final_conv = nn.Identity()

        self.adaptive_pool = nn.AdaptiveMaxPool1d(adaptive_pool_size)

    def forward(self, x):
        out = x
        for conv_layer in self.conv_layers:
            out = conv_layer(out)

        out = self.final_conv(out)
        out = self.adaptive_pool(out)

        if self.flatten:
            out = out.view(out.size(0), -1)

        return out

    def get_layer_groups(self) -> List[nn.Module]:
        """返回各层（用于选择性冻结）"""
        layer_groups = []
        for conv_layer in self.conv_layers:
            layer_groups.append(conv_layer)

        if hasattr(self, 'final_conv') and self.final_conv is not nn.Identity:
            layer_groups.append(self.final_conv)

        layer_groups.append(self.adaptive_pool)

        return layer_groups


class RelationNetwork1d(nn.Module):
    """1D关系网络 - 用于度量学习（MRN专用）"""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 8, adaptive_pool_size: int = 25):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim * 2, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_dim, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        pooled_size = adaptive_pool_size // 4
        fc_input_dim = input_dim * pooled_size

        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class LinearClassifier(nn.Module):
    """线性分类器"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        return self.fc(x)


def freeze_layers(model: nn.Module, num_unfrozen: int):
    """冻结模型层"""
    for param in model.parameters():
        param.requires_grad = False

    if num_unfrozen > 0:
        if hasattr(model, 'get_layer_groups'):
            layer_groups = model.get_layer_groups()
            for layer in layer_groups[-num_unfrozen:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            children = list(model.children())
            for layer in children[-num_unfrozen:]:
                for param in layer.parameters():
                    param.requires_grad = True


def init_weights(model: nn.Module):
    """权重初始化"""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(1)


class ModelFactory:
    """模型工厂 - 根据实验类型创建合适的模型"""

    @staticmethod
    def create_feature_encoder(config, flatten: bool = False) -> nn.Module:
        """创建特征编码器"""
        experiment_type = config.data.experiment_type

        if experiment_type == "PU_fault_transfer":
            return CNN1dEncoder(
                feature_dim=config.model.feature_dim,
                flatten=flatten,
                adaptive_pool_size=config.model.pu_adaptive_pool_size,
                conv_channels=config.model.pu_conv_channels,
                kernel_sizes=config.model.pu_kernel_sizes
            )

        elif experiment_type == "CWRU_domain_transfer":
            return CNN1dEncoder(
                feature_dim=config.model.feature_dim,
                flatten=flatten,
                adaptive_pool_size=config.model.cwru_domain_adaptive_pool_size,
                conv_channels=config.model.cwru_domain_conv_channels,
                kernel_sizes=config.model.cwru_domain_kernel_sizes
            )

        elif experiment_type == "CWRU_fault_transfer":
            # 使用专用网络结构
            if config.model.cwru_fault_network_type == "ConvBase":
                # 计算嵌入维度
                # 输入: [batch, 1, signal_length]
                # 经过4层ConvBase后，输出维度需要计算
                test_input = torch.randn(1, 1, config.data.cwru_fault_signal_length)
                test_encoder = CNN4Backbone(
                    hidden=config.model.cwru_fault_hidden_size,
                    channels=config.model.cwru_fault_channels,
                    layers=config.model.cwru_fault_layers,
                    max_pool_factor=4 // config.model.cwru_fault_layers
                )
                embedding_size = test_encoder(test_input).shape[1]
                config.model.cwru_fault_embedding_size = embedding_size

                return CNN4Backbone(
                    hidden=config.model.cwru_fault_hidden_size,
                    channels=config.model.cwru_fault_channels,
                    layers=config.model.cwru_fault_layers,
                    max_pool_factor=4 // config.model.cwru_fault_layers
                )
            else:
                # 使用标准CNN1d
                return CNN1dEncoder(
                    feature_dim=config.model.feature_dim,
                    flatten=flatten,
                    adaptive_pool_size=config.model.cwru_domain_adaptive_pool_size
                )

        else:
            raise ValueError(f"未知的实验类型: {experiment_type}")

    @staticmethod
    def create_classifier(config, input_dim: int, num_classes: int) -> nn.Module:
        """创建分类器"""
        experiment_type = config.data.experiment_type

        if experiment_type == "CWRU_fault_transfer" and config.model.cwru_fault_network_type == "ConvBase":
            # 使用专用分类器
            return nn.Linear(
                config.model.cwru_fault_embedding_size,
                num_classes,
                bias=True
            )
        else:
            # 使用标准线性分类器
            return LinearClassifier(input_dim, num_classes)

    @staticmethod
    def create_relation_network(config) -> nn.Module:
        """创建关系网络"""
        experiment_type = config.data.experiment_type

        if experiment_type == "PU_fault_transfer":
            adaptive_pool_size = config.model.pu_adaptive_pool_size
        elif experiment_type == "CWRU_domain_transfer":
            adaptive_pool_size = config.model.cwru_domain_adaptive_pool_size
        elif experiment_type == "CWRU_fault_transfer":
            # 对于专用网络，关系网络需要适配
            adaptive_pool_size = 25  # 默认值
        else:
            raise ValueError(f"未知的实验类型: {experiment_type}")

        return RelationNetwork1d(
            input_dim=config.model.feature_dim,
            hidden_dim=config.model.relation_dim,
            adaptive_pool_size=adaptive_pool_size
        )

    @staticmethod
    def create_full_model(config, num_classes: int) -> nn.Module:
        """创建完整模型（特征编码器+分类器）"""
        experiment_type = config.data.experiment_type

        if experiment_type == "CWRU_fault_transfer" and config.model.cwru_fault_network_type == "ConvBase":
            # 使用专用完整模型
            return Net4CNN(
                output_size=num_classes,
                hidden_size=config.model.cwru_fault_hidden_size,
                layers=config.model.cwru_fault_layers,
                channels=config.model.cwru_fault_channels,
                embedding_size=config.model.cwru_fault_embedding_size
            )
        else:
            # 使用标准组合
            feature_encoder = ModelFactory.create_feature_encoder(config, flatten=True)
            classifier = ModelFactory.create_classifier(config,
                input_dim=feature_encoder.feature_dim * getattr(config.model, f'{experiment_type.split("_")[0].lower()}_adaptive_pool_size'),
                num_classes=num_classes
            )
            return nn.Sequential(feature_encoder, classifier)