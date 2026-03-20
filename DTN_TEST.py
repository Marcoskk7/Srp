# # DTN_TEST.py
# """
# 在源域上进行小样本分类测试（类似 DTN 的测试逻辑）。
# 可指定增强方式：none / noise / gan，以及增强后的总样本数 augment_shot。
# 输出每个shot配置的准确率（均值±标准差）。
# """
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import os
# import argparse
# from collections import defaultdict
#
# # 导入项目模块
# from common import source_classes, unique_categories, minmax_scale_np
# from data_loader import SignalDataset, DataLoader
# from models.networks import CNN1dEncoder, LinearClassifier, init_weights
# from methods.base_trainer import EpisodeMetrics
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#
#
# def minmax_to_minus1_1(x):
#     """将 [0,1] 区间的数据线性映射到 [-1,1]"""
#     return x * 2 - 1
#
#
# class DTNTest:
#     """
#     源域小样本分类测试器，完全模拟 DTN 的测试流程。
#     """
#     def __init__(self, config, augment_type='none', augment_shot=0,
#                  target_gen_path=None, noise_level=0.05):
#         self.augment_type = augment_type
#         self.augment_shot = augment_shot          # 增强后的总样本数（若 > shot，则补充）
#         self.target_gen_path = target_gen_path
#         self.noise_level = noise_level
#
#         # 从 config 中获取参数（此处直接硬编码常用值，也可从配置对象传入）
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.signal_length = 2400                  # CWRU 预处理时使用的长度
#         self.data_type = 'time'                     # 使用时域信号（生成样本也是时域）
#         self.feature_dim = 64
#         self.adaptive_pool_size = 64                # CWRU 专用
#         self.input_dim = self.feature_dim * self.adaptive_pool_size
#         self.learning_rate = 0.001
#         self.train_episode = 50                      # 每个episode内的训练轮数
#         self.test_episode = 100                      # 测试episode数量
#         self.batch_size_test = 256
#
#         # 加载生成样本（如果使用 gan 增强）
#         self.gen_samples_by_class = {}
#         if self.augment_type == 'gan' and self.target_gen_path:
#             self._load_generated_samples()
#
#     def _load_generated_samples(self):
#         """加载 generate_source.npz 中的生成样本，按**全局类别索引**组织"""
#         if not os.path.exists(self.target_gen_path):
#             print(f"警告: 生成样本文件 {self.target_gen_path} 不存在，将退化为 none 增强。")
#             self.augment_type = 'none'
#             return
#         data = np.load(self.target_gen_path, allow_pickle=True)
#         X_gen = data['X']
#         y_gen = data['y']
#         # 生成样本的标签是全局类别索引（例如 3,7,4,8,5,9）
#         for cls in np.unique(y_gen):
#             mask = y_gen == cls
#             self.gen_samples_by_class[int(cls)] = X_gen[mask]
#         print(f"已加载生成样本，类别分布: {[len(v) for v in self.gen_samples_by_class.values()]}")
#
#     def _augment_with_noise(self, signal, num):
#         """对单个信号添加高斯噪声生成 num 个增强样本"""
#         signal_std = np.std(signal)
#         noise_std = self.noise_level * signal_std
#         augmented = []
#         for _ in range(num):
#             noise = np.random.normal(0, noise_std, size=signal.shape)
#             augmented.append(signal + noise)
#         return augmented
#
#     def run_test(self, X_source, y_source, source_class_indices, shot_configs):
#         """
#         执行测试。
#         X_source, y_source: 源域所有数据（已缩放至 [-1,1]）
#         source_class_indices: 源域类别在全局中的索引列表（应与生成样本的标签一致）
#         shot_configs: 要测试的shot数列表，如 [1,3,5]
#         """
#         # 检查每个类别的样本数是否足够进行所有shot测试（避免后续出错）
#         min_samples_per_class = min([np.sum(y_source == cls) for cls in source_class_indices])
#         max_shot = max(shot_configs)
#         if min_samples_per_class < max_shot + 1:   # 需要至少 max_shot+1 个样本（留一个查询）
#             print(f"警告：源域某些类别的样本数不足 {max_shot+1}，测试中可能使用有放回采样。")
#
#         results = {}
#         for shot in shot_configs:
#             # 计算需要补充的增强样本数
#             if self.augment_shot > 0:
#                 augment_num = max(0, self.augment_shot - shot)
#             else:
#                 augment_num = 0
#             print(f"\n测试 {shot}-shot (augment_type={self.augment_type}, augment_num={augment_num})...")
#             shot_acc = self._test_single_shot(
#                 X_source, y_source, source_class_indices,
#                 shot, augment_num, run_seed=42
#             )
#             results[f'{shot}shot'] = shot_acc
#             print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
#         return results
#
#     def _test_single_shot(self, X_source, y_source, class_indices,
#                           shot, augment_num, run_seed):
#         """单个shot配置的测试循环"""
#         metrics = EpisodeMetrics()
#         num_classes = len(class_indices)
#
#         for episode in range(self.test_episode):
#             episode_seed = run_seed + episode * 1000
#             set_seed(episode_seed)
#
#             # 构建支持集和查询集
#             support_signals = []
#             support_labels = []
#             query_signals = []
#             query_labels = []
#
#             for local_idx, global_cls in enumerate(class_indices):
#                 # 获取该类别的所有真实样本
#                 mask = y_source == global_cls
#                 data = X_source[mask]
#                 total = len(data)
#
#                 # --- 支持集抽取（真实样本） ---
#                 # 如果真实样本少于 shot，使用有放回采样
#                 if total < shot:
#                     real_indices = np.random.choice(total, shot, replace=True)
#                 else:
#                     real_indices = np.random.choice(total, shot, replace=False)
#                 support_class_signals = data[real_indices]  # 当前类的支持集信号
#                 support_signals.extend(support_class_signals)
#                 support_labels.extend([local_idx] * shot)
#
#                 # --- 查询集抽取（使用剩余样本，若不足则从真实样本中随机抽取） ---
#                 # 构建剩余样本的掩码（排除支持集选中的索引）
#                 mask_q = np.ones(total, dtype=bool)
#                 # 注意：当有放回采样时，支持集可能包含重复索引，需要按实际选中的索引移除（允许重复移除）
#                 # 这里使用简单的计数方法：对每个真实样本，统计它在支持集中出现的次数，然后剩余次数为总次数减去被抽中的次数
#                 from collections import Counter
#                 idx_counter = Counter(real_indices)
#                 # 剩余样本索引 = 每个真实样本重复其剩余次数
#                 remaining = [i for i in range(total) if idx_counter.get(i, 0) == 0]
#                 for i in range(total):
#                     count_in_support = idx_counter.get(i, 0)
#                     if count_in_support < 1:
#                         remaining.append(i)
#                     else:
#                         # 如果支持集中选取了该样本多次，我们还可以留一部分作为查询？通常查询集不应包含支持集的样本，
#                         # 所以这里严格保证查询集与支持集不重叠。对于有放回采样，支持集可能包含重复，查询集就不应该包含该样本的任何副本。
#                         # 所以这里只将完全没有被选中的样本加入查询集。如果支持集有放回且 shot > total，则所有样本都被选中至少一次，查询集为空。
#                         # 这种情况下，查询集会从所有样本中随机抽取（见后面处理）。
#                         pass
#                 if len(remaining) == 0:
#                     # 没有剩余样本，则从该类的所有样本中随机抽取（与支持集可能重叠）
#                     # 这种情况较少见，这里简单从总样本中抽取与支持集相同数量的样本作为查询集（可能重复）
#                     query_indices = np.random.choice(total, total, replace=False)  # 取所有（不重复）
#                     query_signals.extend(data[query_indices])
#                     query_labels.extend([local_idx] * len(query_indices))
#                 else:
#                     query_signals.extend(data[remaining])
#                     query_labels.extend([local_idx] * len(remaining))
#
#                 # 增强：补充额外样本
#                 if augment_num > 0:
#                     if self.augment_type == 'gan':
#                         # 使用 global_cls 查找生成样本
#                         gen_list = self.gen_samples_by_class.get(global_cls, [])
#                         if len(gen_list) >= augment_num:
#                             gen_indices = np.random.choice(len(gen_list), augment_num, replace=False)
#                             support_signals.extend(gen_list[gen_indices])
#                             support_labels.extend([local_idx] * augment_num)
#                         else:
#                             # 如果生成样本不足，使用所有并警告
#                             print(f"警告: 类别 {global_cls} 生成样本不足，实际使用 {len(gen_list)} 个")
#                             support_signals.extend(gen_list)
#                             support_labels.extend([local_idx] * len(gen_list))
#                     elif self.augment_type == 'noise':
#                         # 改进：从真实样本中随机挑选 augment_num 个基样本（允许重复），每个生成一个噪声副本
#                         base_choices = np.random.choice(shot, augment_num, replace=True)
#                         for idx_in_support in base_choices:
#                             base_signal = support_class_signals[idx_in_support]
#                             noisy = self._augment_with_noise(base_signal, 1)[0]  # 生成一个噪声副本
#                             support_signals.append(noisy)
#                             support_labels.append(local_idx)
#                     # 'none' 不增加
#
#             # 转换为numpy数组
#             support_signals = np.array(support_signals)
#             support_labels = np.array(support_labels)
#             query_signals = np.array(query_signals)
#             query_labels = np.array(query_labels)
#
#             # 如果查询集为空，则跳过该 episode（理论不会发生，但防御性编程）
#             if len(query_signals) == 0:
#                 print(f"警告: Episode {episode} 查询集为空，跳过")
#                 continue
#
#             # 创建 DataLoader
#             support_dataset = SignalDataset(support_signals, support_labels,
#                                             self.data_type, self.signal_length)
#             support_loader = DataLoader(support_dataset,
#                                         batch_size=len(support_signals),
#                                         shuffle=True)
#             query_dataset = SignalDataset(query_signals, query_labels,
#                                           self.data_type, self.signal_length)
#             query_loader = DataLoader(query_dataset,
#                                       batch_size=self.batch_size_test,
#                                       shuffle=False)
#
#             # 训练并评估
#             accuracy = self._train_and_evaluate(support_loader, query_loader,
#                                                 num_classes, episode_seed)
#             metrics.update(accuracy)
#
#             if (episode + 1) % 20 == 0:
#                 print(f"  Episode {episode+1}/{self.test_episode} - Acc: {accuracy:.4f}")
#
#         return metrics.compute()
#
#     def _train_and_evaluate(self, support_loader, query_loader,
#                             num_classes, episode_seed):
#         """单个 episode 的训练+评估（与 DTN 相同）"""
#         set_seed(episode_seed)
#
#         # 初始化模型
#         feature_encoder = CNN1dEncoder(
#             feature_dim=self.feature_dim,
#             flatten=True,
#             adaptive_pool_size=self.adaptive_pool_size
#         ).to(self.device)
#         classifier = LinearClassifier(
#             input_dim=self.input_dim,
#             num_classes=num_classes
#         ).to(self.device)
#         init_weights(feature_encoder)
#         init_weights(classifier)
#
#         optimizer = optim.Adam(
#             list(feature_encoder.parameters()) + list(classifier.parameters()),
#             lr=self.learning_rate
#         )
#         criterion = nn.CrossEntropyLoss()
#
#         # 训练
#         feature_encoder.train()
#         classifier.train()
#         for epoch in range(self.train_episode):
#             for batch_x, batch_y in support_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 features = feature_encoder(batch_x)
#                 logits = classifier(features)
#                 loss = criterion(logits, batch_y)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#         # 评估
#         feature_encoder.eval()
#         classifier.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_x, batch_y in query_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 features = feature_encoder(batch_x)
#                 logits = classifier(features)
#                 pred = torch.argmax(logits, dim=1)
#                 correct += (pred == batch_y).sum().item()
#                 total += batch_y.size(0)
#
#         return correct / total if total > 0 else 0.0
#
#
# # ============= 新增封装函数：运行DTN测试 =============
# def run_dtn_test(
#     X_source_scaled,
#     y_source,
#     source_indices,
#     augment_type,
#     augment_shot,
#     gen_path,
#     noise_level,
#     shot_configs,
#     seed=42
# ):
#     """
#     运行单次DTN测试，返回结果字典。
#     """
#     set_seed(seed)
#     tester = DTNTest(
#         config=None,
#         augment_type=augment_type,
#         augment_shot=augment_shot,
#         target_gen_path=gen_path if augment_type == 'gan' else None,
#         noise_level=noise_level
#     )
#     results = tester.run_test(X_source_scaled, y_source, source_indices, shot_configs)
#     return results
#
#
# # ============= 独立运行入口 =============
# def main():
#     parser = argparse.ArgumentParser(description="源域小样本分类测试")
#     parser.add_argument('--augment_type', type=str, default='gan',
#                         choices=['none', 'noise', 'gan'],
#                         help='增强方式')
#     parser.add_argument('--augment_shot', type=int, default=5,
#                         help='增强后的总shot数（若 > shot 则补充）')
#     parser.add_argument('--gen_path', type=str, default='generated_samples_cgan.npz',
#                         help='生成样本文件路径（仅当 augment_type=gan 时使用）')
#     parser.add_argument('--noise_level', type=float, default=0.05,
#                         help='噪声增强的标准差比例')
#     parser.add_argument('--shot_configs', type=int, nargs='+', default=[1,3,5],
#                         help='要测试的shot数列表')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='随机种子')
#     args = parser.parse_args()
#
#     # 设置随机种子
#     set_seed(args.seed)
#
#     # 加载源域数据
#     if not os.path.exists("source_data.npz"):
#         raise FileNotFoundError("请先运行 CWRU_preprocess.py 生成 source_data.npz")
#     data = np.load("source_data.npz")
#     X_source = data["X"]          # (N, T)
#     y_source = data["y"]          # (N,) 全局标签（0~9）
#
#     # 确定源域类别在全局中的索引
#     from common import source_classes, unique_categories
#     source_indices = [unique_categories.index(c) for c in source_classes]
#     print("source_data classes:", np.unique(y_source))
#
#     # 缩放信号至 [-1,1]（与生成样本一致）
#     X_source_scaled = minmax_scale_np(X_source)   # 假设 common.minmax_scale_np 返回 [0,1]
#     X_source_scaled = minmax_to_minus1_1(X_source_scaled)   # 转换为 [-1,1]
#
#     # 创建测试器
#     tester = DTNTest(
#         config=None,  # 此处未使用完整配置，参数已直接传入
#         augment_type=args.augment_type,
#         augment_shot=args.augment_shot,
#         target_gen_path=args.gen_path if args.augment_type == 'gan' else None,
#         noise_level=args.noise_level
#     )
#
#     # 运行测试
#     results = tester.run_test(X_source_scaled, y_source, source_indices, args.shot_configs)
#
#     # 打印汇总
#     print("\n========== 最终结果 ==========")
#     for shot, stats in results.items():
#         print(f"{shot}: {stats['mean']:.4f} ± {stats['std']:.4f}")
#
#     # 保存结果，文件名包含shot信息以免覆盖
#     import json
#     shot_str = '_'.join(str(s) for s in args.shot_configs)
#     out_file = f"dtn_test_{args.augment_type}_aug{args.augment_shot}_shots_{shot_str}.json"
#     with open(out_file, "w") as f:
#         json.dump(results, f, indent=2, default=str)
#     print(f"\n结果已保存至 {out_file}")
#
#
# if __name__ == "__main__":
#     main()
# DTN_TEST.py







#
# # DTN_TEST.py
# """
# 在源域或目标域上进行小样本分类测试（类似 DTN 的测试逻辑）。
# 可指定增强方式：none / noise / gan，以及增强后的总样本数 augment_shot。
# 输出每个shot配置的准确率（均值±标准差）。
# """
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import os
# import argparse
# from collections import Counter
#
# from common import source_classes, unique_categories, minmax_scale_np
# from data_loader import SignalDataset, DataLoader
# from models.networks import CNN1dEncoder, LinearClassifier, init_weights
# from methods.base_trainer import EpisodeMetrics
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#
#
# def minmax_to_minus1_1(x):
#     """将 [0,1] 区间的数据线性映射到 [-1,1]"""
#     return x * 2 - 1
#
#
# class DTNTest:
#     """
#     小样本分类测试器，支持源域测试和目标域零样本测试。
#     """
#     def __init__(self, config, augment_type='none', augment_shot=0,
#                  target_gen_path=None, noise_level=0.05):
#         self.augment_type = augment_type
#         self.augment_shot = augment_shot          # 增强后的总样本数（若 > shot，则补充）
#         self.target_gen_path = target_gen_path
#         self.noise_level = noise_level
#
#         # 从 config 中获取参数（此处直接硬编码常用值，也可从配置对象传入）
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.signal_length = 2400                  # CWRU 预处理时使用的长度
#         self.data_type = 'time'                     # 使用时域信号（生成样本也是时域）
#         self.feature_dim = 64
#         self.adaptive_pool_size = 64                # CWRU 专用
#         self.input_dim = self.feature_dim * self.adaptive_pool_size
#         self.learning_rate = 0.001
#         self.train_episode = 50                      # 每个episode内的训练轮数
#         self.test_episode = 100                      # 测试episode数量
#         self.batch_size_test = 256
#
#         # 加载生成样本（如果使用 gan 增强）
#         self.gen_samples_by_class = {}
#         if self.augment_type == 'gan' and self.target_gen_path:
#             self._load_generated_samples()
#
#     def _load_generated_samples(self):
#         """加载生成样本文件，按类别索引组织"""
#         if not os.path.exists(self.target_gen_path):
#             print(f"警告: 生成样本文件 {self.target_gen_path} 不存在，将退化为 none 增强。")
#             self.augment_type = 'none'
#             return
#         data = np.load(self.target_gen_path, allow_pickle=True)
#         X_gen = data['X']
#         y_gen = data['y']
#         for cls in np.unique(y_gen):
#             mask = y_gen == cls
#             self.gen_samples_by_class[int(cls)] = X_gen[mask]
#         print(f"已加载生成样本，类别分布: {[len(v) for v in self.gen_samples_by_class.values()]}")
#
#     def _augment_with_noise(self, signal, num):
#         """对单个信号添加高斯噪声生成 num 个增强样本"""
#         signal_std = np.std(signal)
#         noise_std = self.noise_level * signal_std
#         augmented = []
#         for _ in range(num):
#             noise = np.random.normal(0, noise_std, size=signal.shape)
#             augmented.append(signal + noise)
#         return augmented
#
#     def run_test(self, X_source, y_source, source_class_indices, shot_configs, base_seed=42):
#         """
#         源域测试：支持集和查询集均来自真实数据，可加入生成样本增强。
#         X_source, y_source: 源域所有数据（已缩放至 [-1,1]）
#         source_class_indices: 源域类别在全局中的索引列表
#         shot_configs: 要测试的shot数列表
#         base_seed: 基础随机种子，每个 episode 的种子为 base_seed + episode*1000
#         """
#         # 检查每个类别的样本数是否足够进行所有shot测试
#         min_samples_per_class = min([np.sum(y_source == cls) for cls in source_class_indices])
#         max_shot = max(shot_configs)
#         if min_samples_per_class < max_shot + 1:
#             print(f"警告：源域某些类别的样本数不足 {max_shot+1}，测试中可能使用有放回采样。")
#
#         results = {}
#         for shot_idx, shot in enumerate(shot_configs):
#             augment_num = max(0, self.augment_shot - shot) if self.augment_shot > 0 else 0
#             print(f"\n测试 {shot}-shot (augment_type={self.augment_type}, augment_num={augment_num})...")
#             # 每个 shot 使用不同的种子偏移，确保 shot 间的独立性
#             shot_acc = self._test_single_shot_source(
#                 X_source, y_source, source_class_indices,
#                 shot, augment_num, run_seed=base_seed + shot_idx * 10000
#             )
#             results[f'{shot}shot'] = shot_acc
#             print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
#         return results
#
#     def run_target_test(self, X_target, y_target, target_class_indices, shot_configs, base_seed=42):
#         """
#         目标域测试（零样本）：支持集全部来自生成样本，查询集来自真实目标域数据。
#         X_target, y_target: 目标域真实数据（已缩放至 [-1,1]）
#         target_class_indices: 目标域类别在全局中的索引列表
#         shot_configs: 要测试的shot数列表（每个类别从生成样本中抽取的数量）
#         base_seed: 基础随机种子
#         """
#         if self.augment_type != 'gan':
#             raise ValueError("目标域测试必须使用 gan 增强（需要生成样本）")
#         if not self.gen_samples_by_class:
#             raise ValueError("生成样本未加载，无法进行目标域测试")
#
#         # 检查生成样本是否包含目标域类别
#         missing = [c for c in target_class_indices if c not in self.gen_samples_by_class]
#         if missing:
#             raise ValueError(f"生成样本中缺少目标域类别: {missing}")
#
#         results = {}
#         for shot_idx, shot in enumerate(shot_configs):
#             print(f"\n目标域测试 {shot}-shot (支持集全为生成样本)...")
#             shot_acc = self._test_single_shot_target(
#                 X_target, y_target, target_class_indices,
#                 shot, run_seed=base_seed + shot_idx * 10000
#             )
#             results[f'{shot}shot'] = shot_acc
#             print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
#         return results
#
#     def _test_single_shot_source(self, X_source, y_source, class_indices,
#                                  shot, augment_num, run_seed):
#         """源域单次shot测试循环（支持集可混合真实和生成样本）"""
#         metrics = EpisodeMetrics()
#         num_classes = len(class_indices)
#
#         for episode in range(self.test_episode):
#             episode_seed = run_seed + episode * 1000
#             set_seed(episode_seed)
#
#             support_signals = []
#             support_labels = []
#             query_signals = []
#             query_labels = []
#
#             for local_idx, global_cls in enumerate(class_indices):
#                 mask = y_source == global_cls
#                 data = X_source[mask]
#                 total = len(data)
#
#                 # 支持集：从真实数据中抽取 shot 个
#                 if total < shot:
#                     real_indices = np.random.choice(total, shot, replace=True)
#                 else:
#                     real_indices = np.random.choice(total, shot, replace=False)
#                 support_class_signals = data[real_indices]
#                 support_signals.extend(support_class_signals)
#                 support_labels.extend([local_idx] * shot)
#
#                 # 查询集：从剩余真实数据中抽取（不重复）
#                 idx_counter = Counter(real_indices)
#                 remaining = [i for i in range(total) if idx_counter.get(i, 0) == 0]
#                 if len(remaining) == 0:
#                     # 无剩余样本，则从所有样本中随机抽取（可能重复）
#                     query_indices = np.random.choice(total, total, replace=False)
#                     query_signals.extend(data[query_indices])
#                     query_labels.extend([local_idx] * len(query_indices))
#                 else:
#                     query_signals.extend(data[remaining])
#                     query_labels.extend([local_idx] * len(remaining))
#
#                 # 增强：补充生成样本或噪声
#                 if augment_num > 0:
#                     if self.augment_type == 'gan':
#                         gen_list = self.gen_samples_by_class.get(global_cls, [])
#                         if len(gen_list) >= augment_num:
#                             gen_indices = np.random.choice(len(gen_list), augment_num, replace=False)
#                             support_signals.extend(gen_list[gen_indices])
#                             support_labels.extend([local_idx] * augment_num)
#                         else:
#                             print(f"警告: 类别 {global_cls} 生成样本不足，实际使用 {len(gen_list)} 个")
#                             support_signals.extend(gen_list)
#                             support_labels.extend([local_idx] * len(gen_list))
#                     elif self.augment_type == 'noise':
#                         base_choices = np.random.choice(shot, augment_num, replace=True)
#                         for idx_in_support in base_choices:
#                             base_signal = support_class_signals[idx_in_support]
#                             noisy = self._augment_with_noise(base_signal, 1)[0]
#                             support_signals.append(noisy)
#                             support_labels.append(local_idx)
#
#             support_signals = np.array(support_signals)
#             support_labels = np.array(support_labels)
#             query_signals = np.array(query_signals)
#             query_labels = np.array(query_labels)
#
#             if len(query_signals) == 0:
#                 print(f"警告: Episode {episode} 查询集为空，跳过")
#                 continue
#
#             accuracy = self._train_and_evaluate(
#                 support_signals, support_labels, query_signals, query_labels,
#                 num_classes, episode_seed
#             )
#             metrics.update(accuracy)
#
#             if (episode + 1) % 20 == 0:
#                 print(f"  Episode {episode+1}/{self.test_episode} - Acc: {accuracy:.4f}")
#
#         return metrics.compute()
#
#     def _test_single_shot_target(self, X_target, y_target, class_indices,
#                                  shot, run_seed):
#         """目标域单次shot测试循环：支持集全为生成样本，查询集为真实目标域数据"""
#         metrics = EpisodeMetrics()
#         num_classes = len(class_indices)
#
#         for episode in range(self.test_episode):
#             episode_seed = run_seed + episode * 1000
#             set_seed(episode_seed)
#
#             support_signals = []
#             support_labels = []
#             query_signals = []
#             query_labels = []
#
#             for local_idx, global_cls in enumerate(class_indices):
#                 # 支持集：从生成样本中抽取 shot 个
#                 gen_list = self.gen_samples_by_class[global_cls]
#                 if len(gen_list) < shot:
#                     print(f"警告: 类别 {global_cls} 生成样本不足 {shot}，使用有放回采样")
#                     gen_indices = np.random.choice(len(gen_list), shot, replace=True)
#                 else:
#                     gen_indices = np.random.choice(len(gen_list), shot, replace=False)
#                 support_signals.extend(gen_list[gen_indices])
#                 support_labels.extend([local_idx] * shot)
#
#                 # 查询集：从真实目标域数据中抽取（不重复，且不与支持集重叠，但支持集是生成的，所以无冲突）
#                 mask = y_target == global_cls
#                 data = X_target[mask]
#                 total = len(data)
#                 if total == 0:
#                     print(f"警告: 目标域类别 {global_cls} 无真实样本，跳过")
#                     continue
#                 # 抽取所有真实样本作为查询（可设置最大数量，这里全部使用）
#                 query_indices = np.arange(total)
#                 query_signals.extend(data[query_indices])
#                 query_labels.extend([local_idx] * total)
#
#             if len(query_signals) == 0:
#                 print(f"警告: Episode {episode} 查询集为空，跳过")
#                 continue
#
#             support_signals = np.array(support_signals)
#             support_labels = np.array(support_labels)
#             query_signals = np.array(query_signals)
#             query_labels = np.array(query_labels)
#
#             accuracy = self._train_and_evaluate(
#                 support_signals, support_labels, query_signals, query_labels,
#                 num_classes, episode_seed
#             )
#             metrics.update(accuracy)
#
#             if (episode + 1) % 20 == 0:
#                 print(f"  Episode {episode+1}/{self.test_episode} - Acc: {accuracy:.4f}")
#
#         return metrics.compute()
#
#     def _train_and_evaluate(self, support_signals, support_labels,
#                             query_signals, query_labels, num_classes, episode_seed):
#         """单个 episode 的训练+评估（与 DTN 相同）"""
#         set_seed(episode_seed)
#
#         # 初始化模型
#         feature_encoder = CNN1dEncoder(
#             feature_dim=self.feature_dim,
#             flatten=True,
#             adaptive_pool_size=self.adaptive_pool_size
#         ).to(self.device)
#         classifier = LinearClassifier(
#             input_dim=self.input_dim,
#             num_classes=num_classes
#         ).to(self.device)
#         init_weights(feature_encoder)
#         init_weights(classifier)
#
#         optimizer = optim.Adam(
#             list(feature_encoder.parameters()) + list(classifier.parameters()),
#             lr=self.learning_rate
#         )
#         criterion = nn.CrossEntropyLoss()
#
#         # 创建 DataLoader
#         support_dataset = SignalDataset(support_signals, support_labels,
#                                         self.data_type, self.signal_length)
#         support_loader = DataLoader(support_dataset,
#                                     batch_size=len(support_signals),
#                                     shuffle=True)
#         query_dataset = SignalDataset(query_signals, query_labels,
#                                       self.data_type, self.signal_length)
#         query_loader = DataLoader(query_dataset,
#                                   batch_size=self.batch_size_test,
#                                   shuffle=False)
#
#         # 训练
#         feature_encoder.train()
#         classifier.train()
#         for epoch in range(self.train_episode):
#             for batch_x, batch_y in support_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 features = feature_encoder(batch_x)
#                 logits = classifier(features)
#                 loss = criterion(logits, batch_y)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#         # 评估
#         feature_encoder.eval()
#         classifier.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_x, batch_y in query_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 features = feature_encoder(batch_x)
#                 logits = classifier(features)
#                 pred = torch.argmax(logits, dim=1)
#                 correct += (pred == batch_y).sum().item()
#                 total += batch_y.size(0)
#
#         return correct / total if total > 0 else 0.0
#
#
# # 封装函数，方便外部调用
# def run_dtn_test(X_source_scaled, y_source, source_indices,
#                  augment_type, augment_shot, gen_path, noise_level,
#                  shot_configs, seed=42):
#     set_seed(seed)
#     tester = DTNTest(
#         config=None,
#         augment_type=augment_type,
#         augment_shot=augment_shot,
#         target_gen_path=gen_path if augment_type == 'gan' else None,
#         noise_level=noise_level
#     )
#     return tester.run_test(X_source_scaled, y_source, source_indices, shot_configs, base_seed=seed)
#
#
# if __name__ == "__main__":
#     # 独立运行示例（略）
#     pass
"""
在源域或目标域上进行小样本分类测试（类似 DTN 的测试逻辑）。
可指定增强方式：none / noise / gan，以及增强后的总样本数 augment_shot。
输出每个shot配置的准确率（均值±标准差）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import argparse
from collections import Counter

from common import source_classes, unique_categories, minmax_scale_np
from data_loader import SignalDataset, DataLoader
from models.networks import CNN1dEncoder, LinearClassifier, init_weights
from methods.base_trainer import EpisodeMetrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class DTNTest:
    """
    小样本分类测试器，支持源域和目标域测试，支持三种增强方式。
    """
    def __init__(self, config, augment_type='none', augment_shot=0,
                 target_gen_path=None, noise_level=0.05):
        self.augment_type = augment_type
        self.augment_shot = augment_shot          # 增强后的总样本数（若 > shot，则补充）
        self.target_gen_path = target_gen_path
        self.noise_level = noise_level

        # 固定参数（与第一个项目一致）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.signal_length = 2400                  # CWRU 预处理时使用的长度
        self.data_type = 'none'                     # MODIFIED: 数据已预处理，不进行额外变换
        self.feature_dim = 64
        self.adaptive_pool_size = 64                # CWRU 专用
        self.input_dim = self.feature_dim * self.adaptive_pool_size
        self.learning_rate = 0.001
        self.train_episode = 50                      # 每个episode内的训练轮数
        self.test_episode = 100                      # 测试episode数量
        self.batch_size_test = 256

        # 加载生成样本（如果使用 gan 增强）
        self.gen_samples_by_class = {}
        if self.augment_type == 'gan' and self.target_gen_path:
            self._load_generated_samples()

    def _load_generated_samples(self):
        """加载生成样本文件，按类别索引组织"""
        if not os.path.exists(self.target_gen_path):
            print(f"警告: 生成样本文件 {self.target_gen_path} 不存在，将退化为 none 增强。")
            self.augment_type = 'none'
            return
        data = np.load(self.target_gen_path, allow_pickle=True)
        X_gen = data['X']
        y_gen = data['y']
        for cls in np.unique(y_gen):
            mask = y_gen == cls
            self.gen_samples_by_class[int(cls)] = X_gen[mask]
        print(f"已加载生成样本，类别分布: {[len(v) for v in self.gen_samples_by_class.values()]}")

    def _augment_with_noise(self, signal, num):
        """对单个信号添加高斯噪声生成 num 个增强样本"""
        signal_std = np.std(signal)
        noise_std = self.noise_level * signal_std
        augmented = []
        for _ in range(num):
            noise = np.random.normal(0, noise_std, size=signal.shape)
            augmented.append(signal + noise)
        return augmented

    def run_test(self, X_source, y_source, source_class_indices, shot_configs, base_seed=42):
        """
        源域测试：支持集和查询集均来自真实数据，可加入生成样本增强。
        X_source, y_source: 源域所有数据（已缩放至 [-1,1]）
        source_class_indices: 源域类别在全局中的索引列表
        shot_configs: 要测试的shot数列表
        base_seed: 基础随机种子
        """
        min_samples_per_class = min([np.sum(y_source == cls) for cls in source_class_indices])
        max_shot = max(shot_configs)
        if min_samples_per_class < max_shot + 1:
            print(f"警告：源域某些类别的样本数不足 {max_shot+1}，测试中可能使用有放回采样。")

        results = {}
        for shot_idx, shot in enumerate(shot_configs):
            augment_num = max(0, self.augment_shot - shot) if self.augment_shot > 0 else 0
            print(f"\n测试 {shot}-shot (augment_type={self.augment_type}, augment_num={augment_num})...")
            shot_acc = self._test_single_shot_generic(
                X_data=X_source,
                y_data=y_source,
                class_indices=source_class_indices,
                shot=shot,
                augment_num=augment_num,
                run_seed=base_seed + shot_idx * 10000
            )
            results[f'{shot}shot'] = shot_acc
            print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
        return results

    def run_target_test(self, X_target, y_target, target_class_indices, shot_configs, base_seed=42):
        """
        目标域测试（小样本域适应）：支持集和查询集均来自真实目标域数据，
        可加入生成样本增强（生成样本由源域知识引导生成）。
        X_target, y_target: 目标域真实数据（已缩放至 [-1,1]）
        target_class_indices: 目标域类别在全局中的索引列表
        shot_configs: 要测试的shot数列表
        base_seed: 基础随机种子
        """
        min_samples_per_class = min([np.sum(y_target == cls) for cls in target_class_indices])
        max_shot = max(shot_configs)
        if min_samples_per_class < max_shot + 1:
            print(f"警告：目标域某些类别的样本数不足 {max_shot+1}，测试中可能使用有放回采样。")

        results = {}
        for shot_idx, shot in enumerate(shot_configs):
            augment_num = max(0, self.augment_shot - shot) if self.augment_shot > 0 else 0
            print(f"\n目标域测试 {shot}-shot (augment_type={self.augment_type}, augment_num={augment_num})...")
            shot_acc = self._test_single_shot_generic(
                X_data=X_target,
                y_data=y_target,
                class_indices=target_class_indices,
                shot=shot,
                augment_num=augment_num,
                run_seed=base_seed + shot_idx * 10000
            )
            results[f'{shot}shot'] = shot_acc
            print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
        return results

    def _test_single_shot_generic(self, X_data, y_data, class_indices,
                                  shot, augment_num, run_seed):
        """
        通用的单次shot测试循环，可指定数据源（源域或目标域）。
        """
        metrics = EpisodeMetrics()
        num_classes = len(class_indices)

        for episode in range(self.test_episode):
            episode_seed = run_seed + episode * 1000
            set_seed(episode_seed)

            support_signals = []
            support_labels = []
            query_signals = []
            query_labels = []

            for local_idx, global_cls in enumerate(class_indices):
                mask = y_data == global_cls
                data = X_data[mask]
                total = len(data)

                # 支持集：从真实数据中抽取 shot 个
                if total < shot:
                    real_indices = np.random.choice(total, shot, replace=True)
                else:
                    real_indices = np.random.choice(total, shot, replace=False)
                support_class_signals = data[real_indices]
                support_signals.extend(support_class_signals)
                support_labels.extend([local_idx] * shot)

                # 查询集：从剩余真实数据中抽取（不重复）
                idx_counter = Counter(real_indices)
                remaining = [i for i in range(total) if idx_counter.get(i, 0) == 0]
                if len(remaining) == 0:
                    # 无剩余样本，则从所有样本中随机抽取（不重复）
                    query_indices = np.random.choice(total, total, replace=False)
                    query_signals.extend(data[query_indices])
                    query_labels.extend([local_idx] * len(query_indices))
                else:
                    query_signals.extend(data[remaining])
                    query_labels.extend([local_idx] * len(remaining))

                # 增强：补充生成样本或噪声
                if augment_num > 0:
                    if self.augment_type == 'gan':
                        gen_list = self.gen_samples_by_class.get(global_cls, [])
                        if len(gen_list) >= augment_num:
                            gen_indices = np.random.choice(len(gen_list), augment_num, replace=False)
                            support_signals.extend(gen_list[gen_indices])
                            support_labels.extend([local_idx] * augment_num)
                        else:
                            print(f"警告: 类别 {global_cls} 生成样本不足，实际使用 {len(gen_list)} 个")
                            support_signals.extend(gen_list)
                            support_labels.extend([local_idx] * len(gen_list))
                    elif self.augment_type == 'noise':
                        base_choices = np.random.choice(shot, augment_num, replace=True)
                        for idx_in_support in base_choices:
                            base_signal = support_class_signals[idx_in_support]
                            noisy = self._augment_with_noise(base_signal, 1)[0]
                            support_signals.append(noisy)
                            support_labels.append(local_idx)

            support_signals = np.array(support_signals)
            support_labels = np.array(support_labels)
            query_signals = np.array(query_signals)
            query_labels = np.array(query_labels)

            if len(query_signals) == 0:
                print(f"警告: Episode {episode} 查询集为空，跳过")
                continue

            accuracy = self._train_and_evaluate(
                support_signals, support_labels, query_signals, query_labels,
                num_classes, episode_seed
            )
            metrics.update(accuracy)

            if (episode + 1) % 20 == 0:
                print(f"  Episode {episode+1}/{self.test_episode} - Acc: {accuracy:.4f}")

        return metrics.compute()

    def _train_and_evaluate(self, support_signals, support_labels,
                            query_signals, query_labels, num_classes, episode_seed):
        """单个 episode 的训练+评估（与 DTN 相同）"""
        set_seed(episode_seed)

        feature_encoder = CNN1dEncoder(
            feature_dim=self.feature_dim,
            flatten=True,
            adaptive_pool_size=self.adaptive_pool_size
        ).to(self.device)
        classifier = LinearClassifier(
            input_dim=self.input_dim,
            num_classes=num_classes
        ).to(self.device)
        init_weights(feature_encoder)
        init_weights(classifier)

        optimizer = optim.Adam(
            list(feature_encoder.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        # 创建 DataLoader
        support_dataset = SignalDataset(support_signals, support_labels,
                                        self.data_type, self.signal_length)
        support_loader = DataLoader(support_dataset,
                                    batch_size=len(support_signals),
                                    shuffle=True)
        query_dataset = SignalDataset(query_signals, query_labels,
                                      self.data_type, self.signal_length)
        query_loader = DataLoader(query_dataset,
                                  batch_size=self.batch_size_test,
                                  shuffle=False)

        # 训练
        feature_encoder.train()
        classifier.train()
        for epoch in range(self.train_episode):
            for batch_x, batch_y in support_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                features = feature_encoder(batch_x)
                logits = classifier(features)
                loss = criterion(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 评估
        feature_encoder.eval()
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in query_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                features = feature_encoder(batch_x)
                logits = classifier(features)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total if total > 0 else 0.0


# 封装函数，方便外部调用
def run_dtn_test(X_source_scaled, y_source, source_indices,
                 augment_type, augment_shot, gen_path, noise_level,
                 shot_configs, seed=42):
    set_seed(seed)
    tester = DTNTest(
        config=None,
        augment_type=augment_type,
        augment_shot=augment_shot,
        target_gen_path=gen_path if augment_type == 'gan' else None,
        noise_level=noise_level
    )
    return tester.run_test(X_source_scaled, y_source, source_indices, shot_configs, base_seed=seed)


if __name__ == "__main__":
    pass
# # DTN_TEST.py
# """
# 在源域或目标域上进行小样本分类测试（类似 DTN 的测试逻辑）。
# 可指定增强方式：none / noise / gan，以及增强后的总样本数 augment_shot。
# 输出每个shot配置的准确率（均值±标准差）。
# """
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import os
# import argparse
# from collections import Counter
#
# from common import source_classes, unique_categories, minmax_scale_np
# from data_loader import SignalDataset, DataLoader
# from models.networks import CNN1dEncoder, LinearClassifier, init_weights
# from methods.base_trainer import EpisodeMetrics
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#
#
# def minmax_to_minus1_1(x):
#     """将 [0,1] 区间的数据线性映射到 [-1,1]"""
#     return x * 2 - 1
#
#
# class DTNTest:
#     """
#     小样本分类测试器，支持源域和目标域测试，支持三种增强方式。
#     """
#     def __init__(self, config, augment_type='none', augment_shot=0,
#                  target_gen_path=None, noise_level=0.05):
#         self.augment_type = augment_type
#         self.augment_shot = augment_shot          # 增强后的总样本数（若 > shot，则补充）
#         self.target_gen_path = target_gen_path
#         self.noise_level = noise_level
#
#         # 从 config 中获取参数（此处直接硬编码常用值，也可从配置对象传入）
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.signal_length = 2400                  # CWRU 预处理时使用的长度
#         self.data_type = 'time'                     # 使用时域信号（生成样本也是时域）
#         self.feature_dim = 64
#         self.adaptive_pool_size = 64                # CWRU 专用
#         self.input_dim = self.feature_dim * self.adaptive_pool_size
#         self.learning_rate = 0.001
#         self.train_episode = 50                      # 每个episode内的训练轮数
#         self.test_episode = 100                      # 测试episode数量
#         self.batch_size_test = 256
#
#         # 加载生成样本（如果使用 gan 增强）
#         self.gen_samples_by_class = {}
#         if self.augment_type == 'gan' and self.target_gen_path:
#             self._load_generated_samples()
#
#     def _load_generated_samples(self):
#         """加载生成样本文件，按类别索引组织"""
#         if not os.path.exists(self.target_gen_path):
#             print(f"警告: 生成样本文件 {self.target_gen_path} 不存在，将退化为 none 增强。")
#             self.augment_type = 'none'
#             return
#         data = np.load(self.target_gen_path, allow_pickle=True)
#         X_gen = data['X']
#         y_gen = data['y']
#         for cls in np.unique(y_gen):
#             mask = y_gen == cls
#             self.gen_samples_by_class[int(cls)] = X_gen[mask]
#         print(f"已加载生成样本，类别分布: {[len(v) for v in self.gen_samples_by_class.values()]}")
#
#     def _augment_with_noise(self, signal, num):
#         """对单个信号添加高斯噪声生成 num 个增强样本"""
#         signal_std = np.std(signal)
#         noise_std = self.noise_level * signal_std
#         augmented = []
#         for _ in range(num):
#             noise = np.random.normal(0, noise_std, size=signal.shape)
#             augmented.append(signal + noise)
#         return augmented
#
#     def run_test(self, X_source, y_source, source_class_indices, shot_configs, base_seed=42):
#         """
#         源域测试：支持集和查询集均来自真实数据，可加入生成样本增强。
#         X_source, y_source: 源域所有数据（已缩放至 [-1,1]）
#         source_class_indices: 源域类别在全局中的索引列表
#         shot_configs: 要测试的shot数列表
#         base_seed: 基础随机种子，每个 episode 的种子为 base_seed + episode*1000
#         """
#         min_samples_per_class = min([np.sum(y_source == cls) for cls in source_class_indices])
#         max_shot = max(shot_configs)
#         if min_samples_per_class < max_shot + 1:
#             print(f"警告：源域某些类别的样本数不足 {max_shot+1}，测试中可能使用有放回采样。")
#
#         results = {}
#         for shot_idx, shot in enumerate(shot_configs):
#             augment_num = max(0, self.augment_shot - shot) if self.augment_shot > 0 else 0
#             print(f"\n测试 {shot}-shot (augment_type={self.augment_type}, augment_num={augment_num})...")
#             shot_acc = self._test_single_shot_generic(
#                 X_data=X_source,
#                 y_data=y_source,
#                 class_indices=source_class_indices,
#                 shot=shot,
#                 augment_num=augment_num,
#                 run_seed=base_seed + shot_idx * 10000
#             )
#             results[f'{shot}shot'] = shot_acc
#             print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
#         return results
#
#     def run_target_test(self, X_target, y_target, target_class_indices, shot_configs, base_seed=42):
#         """
#         目标域测试（小样本域适应）：支持集和查询集均来自真实目标域数据，
#         可加入生成样本增强（生成样本由源域知识引导生成）。
#         X_target, y_target: 目标域真实数据（已缩放至 [-1,1]）
#         target_class_indices: 目标域类别在全局中的索引列表
#         shot_configs: 要测试的shot数列表
#         base_seed: 基础随机种子
#         """
#         min_samples_per_class = min([np.sum(y_target == cls) for cls in target_class_indices])
#         max_shot = max(shot_configs)
#         if min_samples_per_class < max_shot + 1:
#             print(f"警告：目标域某些类别的样本数不足 {max_shot+1}，测试中可能使用有放回采样。")
#
#         results = {}
#         for shot_idx, shot in enumerate(shot_configs):
#             augment_num = max(0, self.augment_shot - shot) if self.augment_shot > 0 else 0
#             print(f"\n目标域测试 {shot}-shot (augment_type={self.augment_type}, augment_num={augment_num})...")
#             shot_acc = self._test_single_shot_generic(
#                 X_data=X_target,
#                 y_data=y_target,
#                 class_indices=target_class_indices,
#                 shot=shot,
#                 augment_num=augment_num,
#                 run_seed=base_seed + shot_idx * 10000
#             )
#             results[f'{shot}shot'] = shot_acc
#             print(f"{shot}-shot: Mean={shot_acc['mean']:.4f} ± {shot_acc['std']:.4f}")
#         return results
#
#     def _test_single_shot_generic(self, X_data, y_data, class_indices,
#                                   shot, augment_num, run_seed):
#         """
#         通用的单次shot测试循环，可指定数据源（源域或目标域）。
#         X_data, y_data: 真实数据及标签（已缩放至 [-1,1]）
#         class_indices: 当前测试的全局类别索引列表
#         shot: 支持集中真实样本数
#         augment_num: 需要补充的增强样本数
#         run_seed: 当前shot的基础随机种子
#         """
#         metrics = EpisodeMetrics()
#         num_classes = len(class_indices)
#
#         for episode in range(self.test_episode):
#             episode_seed = run_seed + episode * 1000
#             set_seed(episode_seed)
#
#             support_signals = []
#             support_labels = []
#             query_signals = []
#             query_labels = []
#
#             for local_idx, global_cls in enumerate(class_indices):
#                 mask = y_data == global_cls
#                 data = X_data[mask]
#                 total = len(data)
#
#                 # 支持集：从真实数据中抽取 shot 个
#                 if total < shot:
#                     real_indices = np.random.choice(total, shot, replace=True)
#                 else:
#                     real_indices = np.random.choice(total, shot, replace=False)
#                 support_class_signals = data[real_indices]
#                 support_signals.extend(support_class_signals)
#                 support_labels.extend([local_idx] * shot)
#
#                 # 查询集：从剩余真实数据中抽取（不重复）
#                 idx_counter = Counter(real_indices)
#                 remaining = [i for i in range(total) if idx_counter.get(i, 0) == 0]
#                 if len(remaining) == 0:
#                     # 无剩余样本，则从所有样本中随机抽取（可能重复）
#                     query_indices = np.random.choice(total, total, replace=False)
#                     query_signals.extend(data[query_indices])
#                     query_labels.extend([local_idx] * len(query_indices))
#                 else:
#                     query_signals.extend(data[remaining])
#                     query_labels.extend([local_idx] * len(remaining))
#
#                 # 增强：补充生成样本或噪声
#                 if augment_num > 0:
#                     if self.augment_type == 'gan':
#                         gen_list = self.gen_samples_by_class.get(global_cls, [])
#                         if len(gen_list) >= augment_num:
#                             gen_indices = np.random.choice(len(gen_list), augment_num, replace=False)
#                             support_signals.extend(gen_list[gen_indices])
#                             support_labels.extend([local_idx] * augment_num)
#                         else:
#                             print(f"警告: 类别 {global_cls} 生成样本不足，实际使用 {len(gen_list)} 个")
#                             support_signals.extend(gen_list)
#                             support_labels.extend([local_idx] * len(gen_list))
#                     elif self.augment_type == 'noise':
#                         base_choices = np.random.choice(shot, augment_num, replace=True)
#                         for idx_in_support in base_choices:
#                             base_signal = support_class_signals[idx_in_support]
#                             noisy = self._augment_with_noise(base_signal, 1)[0]
#                             support_signals.append(noisy)
#                             support_labels.append(local_idx)
#                     # 'none' 不增加
#
#             support_signals = np.array(support_signals)
#             support_labels = np.array(support_labels)
#             query_signals = np.array(query_signals)
#             query_labels = np.array(query_labels)
#
#             if len(query_signals) == 0:
#                 print(f"警告: Episode {episode} 查询集为空，跳过")
#                 continue
#
#             accuracy = self._train_and_evaluate(
#                 support_signals, support_labels, query_signals, query_labels,
#                 num_classes, episode_seed
#             )
#             metrics.update(accuracy)
#
#             if (episode + 1) % 20 == 0:
#                 print(f"  Episode {episode+1}/{self.test_episode} - Acc: {accuracy:.4f}")
#
#         return metrics.compute()
#
#     def _train_and_evaluate(self, support_signals, support_labels,
#                             query_signals, query_labels, num_classes, episode_seed):
#         """单个 episode 的训练+评估（与 DTN 相同）"""
#         set_seed(episode_seed)
#
#         # 初始化模型
#         feature_encoder = CNN1dEncoder(
#             feature_dim=self.feature_dim,
#             flatten=True,
#             adaptive_pool_size=self.adaptive_pool_size
#         ).to(self.device)
#         classifier = LinearClassifier(
#             input_dim=self.input_dim,
#             num_classes=num_classes
#         ).to(self.device)
#         init_weights(feature_encoder)
#         init_weights(classifier)
#
#         optimizer = optim.Adam(
#             list(feature_encoder.parameters()) + list(classifier.parameters()),
#             lr=self.learning_rate
#         )
#         criterion = nn.CrossEntropyLoss()
#
#         # 创建 DataLoader
#         support_dataset = SignalDataset(support_signals, support_labels,
#                                         self.data_type, self.signal_length)
#         support_loader = DataLoader(support_dataset,
#                                     batch_size=len(support_signals),
#                                     shuffle=True)
#         query_dataset = SignalDataset(query_signals, query_labels,
#                                       self.data_type, self.signal_length)
#         query_loader = DataLoader(query_dataset,
#                                   batch_size=self.batch_size_test,
#                                   shuffle=False)
#
#         # 训练
#         feature_encoder.train()
#         classifier.train()
#         for epoch in range(self.train_episode):
#             for batch_x, batch_y in support_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 features = feature_encoder(batch_x)
#                 logits = classifier(features)
#                 loss = criterion(logits, batch_y)
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#         # 评估
#         feature_encoder.eval()
#         classifier.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch_x, batch_y in query_loader:
#                 batch_x = batch_x.to(self.device)
#                 batch_y = batch_y.to(self.device)
#                 features = feature_encoder(batch_x)
#                 logits = classifier(features)
#                 pred = torch.argmax(logits, dim=1)
#                 correct += (pred == batch_y).sum().item()
#                 total += batch_y.size(0)
#
#         return correct / total if total > 0 else 0.0
#
#
# # 封装函数，方便外部调用
# def run_dtn_test(X_source_scaled, y_source, source_indices,
#                  augment_type, augment_shot, gen_path, noise_level,
#                  shot_configs, seed=42):
#     set_seed(seed)
#     tester = DTNTest(
#         config=None,
#         augment_type=augment_type,
#         augment_shot=augment_shot,
#         target_gen_path=gen_path if augment_type == 'gan' else None,
#         noise_level=noise_level
#     )
#     return tester.run_test(X_source_scaled, y_source, source_indices, shot_configs, base_seed=seed)
#
#
# if __name__ == "__main__":
#     # 独立运行示例（略）
#     pass