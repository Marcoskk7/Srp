"""
项目主控脚本（增强版）
功能：
1. 检查预处理和知识图谱缓存，若缺失则运行 CWRU_preprocess.py 和 KG.py
2. 根据参数选择 cGAN 版本（eval/cond/constraint/pc）并训练/加载生成样本
3. 执行源域或目标域的 DTN 测试，支持重复运行取平均
4. 对 eval 版本额外进行生成质量评估（FID, MMD, KID, t-SNE）
"""

import os
import sys
import argparse
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import source_classes, target_classes, unique_categories, minmax_scale_np
from cGAN_evaluation import cGAN_Trainer, train_cgan_and_generate as train_eval
from cGAN_evaluation import evaluate_generation_simple  # 导入评估函数
from cGAN_condition import cGAN_Condition_Trainer
from cGAN_constraint import cGAN_Constraint_Trainer
from PCGAN import PcGAN_Trainer  # 新增导入
from DTN_TEST import DTNTest, set_seed


def check_preprocessing():
    """检查预处理和KG缓存是否存在，若缺失则运行相应脚本"""
    required_files = [
        "source_data.npz",
        "target_data.npz",
        os.path.join("knowledge_graphs", "kg_step2_w_v_sigma.npz"),
        os.path.join("knowledge_graphs", "kg_step3_P_transition.npy"),
        os.path.join("knowledge_graphs", "Ec.npy")
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("缺失以下缓存文件，将运行预处理和KG构建：")
        for f in missing:
            print(f"  - {f}")
        print("\n运行 CWRU_preprocess.py ...")
        os.system(f"{sys.executable} CWRU_preprocess.py")
        print("\n运行 KG.py ...")
        os.system(f"{sys.executable} KG.py")
        print("预处理和KG构建完成。")
    else:
        print("所有预处理/KG缓存文件已存在，跳过预处理步骤。")


def load_kg_data():
    """加载知识图谱中的 w, v, E_c（全10类）"""
    kg_path = os.path.join("knowledge_graphs", "kg_step2_w_v_sigma.npz")
    ec_path = os.path.join("knowledge_graphs", "Ec.npy")
    kg = np.load(kg_path, allow_pickle=True)
    w_full = kg["w"]          # (10, D)
    v_full = kg["v"]          # (10, D)
    E_c_full = np.load(ec_path)   # (10, 4)
    return w_full, v_full, E_c_full


def aggregate_results(results_list, shot_configs):
    """对多次运行的结果进行平均"""
    aggregated = {}
    for shot in shot_configs:
        key = f"{shot}shot"
        means = [r[key]['mean'] for r in results_list]
        stds = [r[key]['std'] for r in results_list]
        aggregated[key] = {
            'mean_mean': float(np.mean(means)),
            'mean_std': float(np.std(means)),
            'std_mean': float(np.mean(stds)),
            'all_means': means,
            'all_stds': stds
        }
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="项目主控脚本")

    # cGAN 选择与超参数
    parser.add_argument('--cgan_version', type=str, default='pc',
                        choices=['eval', 'cond', 'constraint', 'pc'],
                        help='cGAN 版本: eval=标准, cond=条件嵌入, constraint=物理约束, pc=融合版')
    parser.add_argument('--cgan_epochs', type=int, default=50, help='cGAN训练轮数')
    parser.add_argument('--cgan_batch_size', type=int, default=64, help='cGAN batch size')
    parser.add_argument('--cgan_z_dim', type=int, default=128, help='生成器噪声维度')
    parser.add_argument('--cgan_lr_g', type=float, default=2e-4, help='生成器学习率')
    parser.add_argument('--cgan_lr_d', type=float, default=1e-4, help='判别器学习率')
    parser.add_argument('--cgan_n_critic', type=int, default=2, help='判别器训练次数 per generator step')
    parser.add_argument('--cgan_log_dir', type=str, default='./runs/cgan', help='TensorBoard日志目录')
    parser.add_argument('--num_per_class', type=int, default=50, help='每类生成样本数')
    parser.add_argument('--lambda_phys', type=float, default=0.75,
                        help='物理约束损失权重（仅 constraint/pc 版本有效）')
    parser.add_argument('--alpha_E', type=float, default=1.0,
                        help='频带能量损失权重（仅 constraint/pc 版本有效）')

    # DTN 测试参数
    parser.add_argument('--augment_type', type=str, default='gan',
                        choices=['none', 'noise', 'gan'],
                        help='增强方式（测试时）')
    parser.add_argument('--augment_shot', type=int, default=5,
                        help='增强后的总shot数（若 > shot 则补充）')
    parser.add_argument('--noise_level', type=float, default=0.05,
                        help='噪声增强标准差比例')
    parser.add_argument('--shot_configs', type=int, nargs='+', default=[1, 3, 5],
                        help='要测试的shot数列表')

    # 运行控制
    parser.add_argument('--target_test', action='store_true',
                        help='是否进行目标域测试（否则为源域测试）')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='重复运行次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='基础随机种子（每次运行会偏移）')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='强制重新生成样本（即使已有缓存）')
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='使用缓存生成样本（如果存在）')
    parser.add_argument('--skip_preproc', action='store_true',
                        help='强制跳过预处理检查')
    parser.add_argument('--run_eval', action='store_true', default=True,
                        help='对 eval 版本执行生成质量评估（默认开启）')

    args = parser.parse_args()

    # 预处理检查
    if not args.skip_preproc:
        check_preprocessing()
    else:
        print("已跳过预处理检查。")

    # 加载源域和目标域数据（已全局归一化到 [-1,1]）
    source_data = np.load("source_data.npz")
    X_source = source_data["X"]
    y_source = source_data["y"]
    full_class_names = source_data["class_names"].tolist()

    target_data = np.load("target_data.npz")
    X_target = target_data["X"]
    y_target = target_data["y"]   # 标签为全局索引（6-9）

    # 确定测试域
    if args.target_test:
        print("执行目标域测试（零样本增强）")
        test_classes = target_classes
        test_indices = [unique_categories.index(c) for c in test_classes]
        X_test_real = X_target
        y_test_real = y_target
        # 目标域测试时，eval 版本无法生成未见类，自动切换为 cond
        if args.cgan_version == 'eval':
            print("警告：目标域测试需要零样本生成，自动切换为 cond 版本。")
            args.cgan_version = 'cond'
    else:
        print("执行源域测试")
        test_classes = source_classes
        test_indices = [unique_categories.index(c) for c in test_classes]
        X_test_real = X_source
        y_test_real = y_source

    # 加载全量 KG 数据
    w_full, v_full, E_c_full = load_kg_data()

    # 准备生成样本文件名（根据版本和测试域区分）
    gen_file = f"generated_samples_{args.cgan_version}_{'target' if args.target_test else 'source'}.npz"

    # 根据增强类型决定是否使用生成样本
    if args.augment_type != 'gan':
        print(f"增强类型为 {args.augment_type}，跳过 cGAN 生成。")
        gen_path = None
        Xg, yg = None, None   # 未生成
    else:
        # 检查缓存
        if args.use_cache and os.path.exists(gen_file) and not args.force_regenerate:
            print(f"找到缓存生成样本 {gen_file}，跳过训练。")
            data = np.load(gen_file, allow_pickle=True)
            Xg = data['X']
            yg = data['y']
            gen_path = gen_file
        else:
            print(f"训练 cGAN ({args.cgan_version} 版本) 并生成样本...")

            # 根据版本创建训练器
            if args.cgan_version == 'eval':
                # 标准 cGAN 只能生成源域类别
                if args.target_test:
                    raise ValueError("标准 cGAN (eval) 无法生成未见类样本，请使用 cond 或 constraint 版本。")
                trainer = cGAN_Trainer(
                    X_signals=X_source,
                    y=y_source,
                    class_names=[full_class_names[i] for i in test_indices],
                    batch_size=args.cgan_batch_size,
                    z_dim=args.cgan_z_dim,
                    lr_g=args.cgan_lr_g,
                    lr_d=args.cgan_lr_d,
                    n_critic=args.cgan_n_critic,
                    log_dir=os.path.join(args.cgan_log_dir, "eval"),
                    use_tensorboard=False,
                    do_minmax=False,          # 数据已全局归一化
                )
                trainer.fit(epochs=args.cgan_epochs, log_every=50)
                Xg = trainer.synthesize(y=np.arange(len(test_indices)), num_per_class=args.num_per_class)
                yg = np.repeat(np.arange(len(test_indices)), args.num_per_class)

            elif args.cgan_version == 'cond':
                trainer = cGAN_Condition_Trainer(
                    X_signals=X_source,
                    y=y_source,
                    class_names=full_class_names,
                    w_real=w_full,
                    E_c=E_c_full,
                    batch_size=args.cgan_batch_size,
                    z_dim=args.cgan_z_dim,
                    lr_g=args.cgan_lr_g,
                    lr_d=args.cgan_lr_d,
                    n_critic=args.cgan_n_critic,
                    log_dir=os.path.join(args.cgan_log_dir, "cond"),
                    use_tensorboard=False,
                    do_minmax=False,
                )
                trainer.fit(epochs=args.cgan_epochs, log_every=50)
                Xg = trainer.synthesize(y=np.array(test_indices), num_per_class=args.num_per_class)
                yg = np.repeat(test_indices, args.num_per_class)

            elif args.cgan_version == 'constraint':
                trainer = cGAN_Constraint_Trainer(
                    X_signals=X_source,
                    y=y_source,
                    class_names=full_class_names,
                    v_real=v_full,
                    w_real=w_full,
                    E_c=E_c_full,
                    batch_size=args.cgan_batch_size,
                    z_dim=args.cgan_z_dim,
                    lr_g=args.cgan_lr_g,
                    lr_d=args.cgan_lr_d,
                    lambda_phys=args.lambda_phys,
                    alpha_E=args.alpha_E,
                    n_critic=args.cgan_n_critic,
                    log_dir=os.path.join(args.cgan_log_dir, "constraint"),
                    use_tensorboard=False,
                    do_minmax=False,
                )
                trainer.fit(epochs=args.cgan_epochs, log_every=50)
                Xg = trainer.synthesize(y=np.array(test_indices), num_per_class=args.num_per_class)
                yg = np.repeat(test_indices, args.num_per_class)

            elif args.cgan_version == 'pc':
                trainer = PcGAN_Trainer(
                    X_signals=X_source,
                    y=y_source,
                    class_names=full_class_names,
                    w_real=w_full,
                    E_c=E_c_full,
                    v_real=v_full,
                    batch_size=args.cgan_batch_size,
                    z_dim=args.cgan_z_dim,
                    lr_g=args.cgan_lr_g,
                    lr_d=args.cgan_lr_d,
                    lambda_phys=args.lambda_phys,
                    alpha_E=args.alpha_E,
                    n_critic=args.cgan_n_critic,
                    log_dir=os.path.join(args.cgan_log_dir, "pc"),
                    use_tensorboard=False,
                    do_minmax=False,
                )
                trainer.fit(epochs=args.cgan_epochs, log_every=50)
                Xg = trainer.synthesize(y=np.array(test_indices), num_per_class=args.num_per_class)
                yg = np.repeat(test_indices, args.num_per_class)

            else:
                raise ValueError(f"未知 cGAN 版本: {args.cgan_version}")

            # 保存生成样本
            np.savez(gen_file, X=Xg, y=yg, class_names=np.array(full_class_names))
            print(f"生成样本已保存至 {gen_file}")
            gen_path = gen_file

        # ---------- 对 eval 版本额外进行生成质量评估 ----------
        if args.cgan_version == 'eval' and args.run_eval and Xg is not None:
            print("\n=== 对 eval 版本进行生成质量评估 ===")
            # 从真实测试集中抽取与生成样本数量相当的真实样本（每类相同数量）
            real_subset_signals = []
            real_subset_labels = []
            for local_idx, global_cls in enumerate(test_indices):
                mask = y_test_real == global_cls
                data = X_test_real[mask]
                if len(data) >= args.num_per_class:
                    sel = np.random.choice(len(data), args.num_per_class, replace=False)
                else:
                    sel = np.arange(len(data))  # 如果不足则全部使用
                real_subset_signals.append(data[sel])
                real_subset_labels.append(np.full(len(sel), local_idx))
            real_subset_signals = np.concatenate(real_subset_signals, axis=0)
            real_subset_labels = np.concatenate(real_subset_labels, axis=0)

            # 创建保存目录
            eval_save_dir = os.path.join(args.cgan_log_dir, "eval_metrics")
            os.makedirs(eval_save_dir, exist_ok=True)

            # 调用评估函数（注意：真实信号已全局归一化，评估函数内部会再次缩放，可能导致偏差，但作为近似参考）
            metrics = evaluate_generation_simple(
                real_signals=real_subset_signals,
                real_labels=real_subset_labels,
                fake_signals=Xg,
                fake_labels=yg,
                class_names=[full_class_names[i] for i in test_indices],
                tag=f"cgan_eval_{'target' if args.target_test else 'source'}",
                save_dir=eval_save_dir,
                selected_features=[4, 6, 15],  # 示例特征索引（峰值因子、峭度、重心频率）
                compute_mmd=True,
                mmd_kernel='rbf',
                compute_kid=True,
                kid_subsample=100,
                mmd_subsample=None
            )
            # 保存评估指标
            metrics_file = os.path.join(eval_save_dir, "generation_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"评估指标已保存至 {metrics_file}")

    # ---------- DTN 测试 ----------
    # 准备测试数据：直接使用已全局归一化的信号，不再重复归一化
    X_test_scaled = X_test_real   # 已经是 [-1,1]

    # 存储每次运行的结果
    all_results = []

    for run in range(args.num_runs):
        print(f"\n========== 第 {run+1}/{args.num_runs} 次运行 ==========")
        run_seed = args.seed + run * 100
        set_seed(run_seed)

        tester = DTNTest(
            config=None,
            augment_type=args.augment_type,
            augment_shot=args.augment_shot,
            target_gen_path=gen_path if args.augment_type == 'gan' else None,
            noise_level=args.noise_level
        )

        if args.target_test:
            results = tester.run_target_test(
                X_target=X_test_scaled,
                y_target=y_test_real,
                target_class_indices=test_indices,
                shot_configs=args.shot_configs,
                base_seed=run_seed
            )
        else:
            results = tester.run_test(
                X_source=X_test_scaled,
                y_source=y_test_real,
                source_class_indices=test_indices,
                shot_configs=args.shot_configs,
                base_seed=run_seed
            )

        all_results.append(results)

        run_file = f"dtn_results_{'target' if args.target_test else 'source'}_run{run+1}.json"
        with open(run_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"第 {run+1} 次结果已保存至 {run_file}")

    # 聚合结果
    aggregated = aggregate_results(all_results, args.shot_configs)
    print("\n========== 最终聚合结果（三次平均） ==========")
    for shot, stats in aggregated.items():
        print(f"{shot}: mean = {stats['mean_mean']:.4f} ± {stats['mean_std']:.4f} "
              f"(原始std平均 = {stats['std_mean']:.4f})")

    agg_file = f"dtn_results_{'target' if args.target_test else 'source'}_aggregated.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"\n聚合结果已保存至 {agg_file}")


if __name__ == "__main__":
    main()


# """
# 项目主控脚本
# 功能：
# 1. 检查预处理和知识图谱缓存，若缺失则运行 CWRU_preprocess.py 和 KG.py
# 2. 根据参数选择 cGAN 版本（eval/cond/constraint）并训练/加载生成样本
# 3. 执行源域或目标域的 DTN 测试，支持重复运行取平均
# """
#
# import os
# import sys
# import argparse
# import numpy as np
# import json
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# from common import source_classes, target_classes, unique_categories, minmax_scale_np
# from cGAN_evaluation import cGAN_Trainer, train_cgan_and_generate as train_eval
# from cGAN_condition import cGAN_Condition_Trainer
# from cGAN_constraint import cGAN_Constraint_Trainer
# from DTN_TEST import DTNTest, run_dtn_test, set_seed
#
#
# def check_preprocessing():
#     """检查预处理和KG缓存是否存在，若缺失则运行相应脚本"""
#     required_files = [
#         "source_data.npz",
#         "target_data.npz",
#         os.path.join("knowledge_graphs", "kg_step2_w_v_sigma.npz"),
#         os.path.join("knowledge_graphs", "kg_step3_P_transition.npy"),
#         os.path.join("knowledge_graphs", "Ec.npy")
#     ]
#     missing = [f for f in required_files if not os.path.exists(f)]
#     if missing:
#         print("缺失以下缓存文件，将运行预处理和KG构建：")
#         for f in missing:
#             print(f"  - {f}")
#         print("\n运行 CWRU_preprocess.py ...")
#         os.system(f"{sys.executable} CWRU_preprocess.py")
#         print("\n运行 KG.py ...")
#         os.system(f"{sys.executable} KG.py")
#         print("预处理和KG构建完成。")
#     else:
#         print("所有预处理/KG缓存文件已存在，跳过预处理步骤。")
#
#
# def load_kg_data():
#     """加载知识图谱中的 w, v, E_c（全10类）"""
#     kg_path = os.path.join("knowledge_graphs", "kg_step2_w_v_sigma.npz")
#     ec_path = os.path.join("knowledge_graphs", "Ec.npy")
#     kg = np.load(kg_path, allow_pickle=True)
#     w_full = kg["w"]          # (10, D)
#     v_full = kg["v"]          # (10, D)
#     E_c_full = np.load(ec_path)   # (10, 4)
#     return w_full, v_full, E_c_full
#
#
# def aggregate_results(results_list, shot_configs):
#     """对多次运行的结果进行平均"""
#     aggregated = {}
#     for shot in shot_configs:
#         key = f"{shot}shot"
#         means = [r[key]['mean'] for r in results_list]
#         stds = [r[key]['std'] for r in results_list]
#         aggregated[key] = {
#             'mean_mean': float(np.mean(means)),
#             'mean_std': float(np.std(means)),
#             'std_mean': float(np.mean(stds)),
#             'all_means': means,
#             'all_stds': stds
#         }
#     return aggregated
#
#
# def main():
#     parser = argparse.ArgumentParser(description="项目主控脚本")
#
#     # cGAN 选择与超参数
#     parser.add_argument('--cgan_version', type=str, default='cond',
#                         choices=['eval', 'cond', 'constraint'],
#                         help='cGAN 版本: eval=标准, cond=条件嵌入, constraint=物理约束')
#     parser.add_argument('--cgan_epochs', type=int, default=50, help='cGAN训练轮数')
#     parser.add_argument('--cgan_batch_size', type=int, default=64, help='cGAN batch size')
#     parser.add_argument('--cgan_z_dim', type=int, default=128, help='生成器噪声维度')
#     parser.add_argument('--cgan_lr_g', type=float, default=2e-4, help='生成器学习率')
#     parser.add_argument('--cgan_lr_d', type=float, default=1e-4, help='判别器学习率')
#     parser.add_argument('--cgan_n_critic', type=int, default=2, help='判别器训练次数 per generator step')
#     parser.add_argument('--cgan_log_dir', type=str, default='./runs/cgan', help='TensorBoard日志目录')
#     parser.add_argument('--num_per_class', type=int, default=50, help='每类生成样本数')
#     parser.add_argument('--lambda_phys', type=float, default=0.75,
#                         help='物理约束损失权重（仅 constraint 版本有效）')
#     parser.add_argument('--alpha_E', type=float, default=1.0,
#                         help='频带能量损失权重（仅 constraint 版本有效）')
#
#     # DTN 测试参数
#     parser.add_argument('--augment_type', type=str, default='gan',
#                         choices=['none', 'noise', 'gan'],
#                         help='增强方式（测试时）')
#     parser.add_argument('--augment_shot', type=int, default=5,
#                         help='增强后的总shot数（若 > shot 则补充）')
#     parser.add_argument('--noise_level', type=float, default=0.05,
#                         help='噪声增强标准差比例')
#     parser.add_argument('--shot_configs', type=int, nargs='+', default=[1, 3, 5],
#                         help='要测试的shot数列表')
#
#     # 运行控制
#     parser.add_argument('--target_test', action='store_true',
#                         help='是否进行目标域测试（否则为源域测试）')
#     parser.add_argument('--num_runs', type=int, default=3,
#                         help='重复运行次数')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='基础随机种子（每次运行会偏移）')
#     parser.add_argument('--force_regenerate', action='store_true',
#                         help='强制重新生成样本（即使已有缓存）')
#     parser.add_argument('--use_cache', action='store_true', default=True,
#                         help='使用缓存生成样本（如果存在）')
#     parser.add_argument('--skip_preproc', action='store_true',
#                         help='强制跳过预处理检查')
#
#     args = parser.parse_args()
#
#     # 预处理检查
#     if not args.skip_preproc:
#         check_preprocessing()
#     else:
#         print("已跳过预处理检查。")
#
#     # 加载源域和目标域数据（已全局归一化到 [-1,1]）
#     source_data = np.load("source_data.npz")
#     X_source = source_data["X"]
#     y_source = source_data["y"]
#     full_class_names = source_data["class_names"].tolist()
#
#     target_data = np.load("target_data.npz")
#     X_target = target_data["X"]
#     y_target = target_data["y"]   # 标签为全局索引（6-9）
#
#     # 确定测试域
#     if args.target_test:
#         print("执行目标域测试（零样本增强）")
#         test_classes = target_classes
#         test_indices = [unique_categories.index(c) for c in test_classes]
#         X_test_real = X_target
#         y_test_real = y_target
#         # 目标域测试时，必须使用 cond 或 constraint 版本（因为 eval 无法生成未见类）
#         if args.cgan_version == 'eval':
#             print("警告：目标域测试需要零样本生成，自动切换为 cond 版本。")
#             args.cgan_version = 'cond'
#     else:
#         print("执行源域测试")
#         test_classes = source_classes
#         test_indices = [unique_categories.index(c) for c in test_classes]
#         X_test_real = X_source
#         y_test_real = y_source
#
#     # 加载全量 KG 数据
#     w_full, v_full, E_c_full = load_kg_data()
#
#     # 准备生成样本文件名（根据版本和测试域区分）
#     gen_file = f"generated_samples_{args.cgan_version}_{'target' if args.target_test else 'source'}.npz"
#
#     # 根据增强类型决定是否使用生成样本
#     if args.augment_type != 'gan':
#         print(f"增强类型为 {args.augment_type}，跳过 cGAN 生成。")
#         gen_path = None
#     else:
#         # 检查缓存
#         if args.use_cache and os.path.exists(gen_file) and not args.force_regenerate:
#             print(f"找到缓存生成样本 {gen_file}，跳过训练。")
#             gen_path = gen_file
#         else:
#             print(f"训练 cGAN ({args.cgan_version} 版本) 并生成样本...")
#
#             if args.cgan_version == 'eval':
#                 if args.target_test:
#                     raise ValueError("标准 cGAN (eval) 无法生成未见类样本，请使用 cond 或 constraint 版本。")
#                 trainer = cGAN_Trainer(
#                     X_signals=X_source,
#                     y=y_source,
#                     class_names=[full_class_names[i] for i in test_indices],
#                     batch_size=args.cgan_batch_size,
#                     z_dim=args.cgan_z_dim,
#                     lr_g=args.cgan_lr_g,
#                     lr_d=args.cgan_lr_d,
#                     n_critic=args.cgan_n_critic,
#                     log_dir=os.path.join(args.cgan_log_dir, "eval"),
#                     use_tensorboard=False,
#                     do_minmax=False,          # MODIFIED: 数据已全局归一化
#                 )
#                 trainer.fit(epochs=args.cgan_epochs, log_every=50)
#                 Xg = trainer.synthesize(y=np.arange(len(test_indices)), num_per_class=args.num_per_class)
#                 yg = np.repeat(np.arange(len(test_indices)), args.num_per_class)
#
#             elif args.cgan_version == 'cond':
#                 trainer = cGAN_Condition_Trainer(
#                     X_signals=X_source,
#                     y=y_source,
#                     class_names=full_class_names,
#                     w_real=w_full,
#                     E_c=E_c_full,
#                     batch_size=args.cgan_batch_size,
#                     z_dim=args.cgan_z_dim,
#                     lr_g=args.cgan_lr_g,
#                     lr_d=args.cgan_lr_d,
#                     n_critic=args.cgan_n_critic,
#                     log_dir=os.path.join(args.cgan_log_dir, "cond"),
#                     use_tensorboard=False,
#                     do_minmax=False,          # MODIFIED
#                 )
#                 trainer.fit(epochs=args.cgan_epochs, log_every=50)
#                 Xg = trainer.synthesize(y=np.array(test_indices), num_per_class=args.num_per_class)
#                 yg = np.repeat(test_indices, args.num_per_class)
#
#             elif args.cgan_version == 'constraint':
#                 trainer = cGAN_Constraint_Trainer(
#                     X_signals=X_source,
#                     y=y_source,
#                     class_names=full_class_names,
#                     v_real=v_full,
#                     w_real=w_full,
#                     E_c=E_c_full,
#                     batch_size=args.cgan_batch_size,
#                     z_dim=args.cgan_z_dim,
#                     lr_g=args.cgan_lr_g,
#                     lr_d=args.cgan_lr_d,
#                     lambda_phys=args.lambda_phys,
#                     alpha_E=args.alpha_E,
#                     n_critic=args.cgan_n_critic,
#                     log_dir=os.path.join(args.cgan_log_dir, "constraint"),
#                     use_tensorboard=False,
#                     do_minmax=False,          # MODIFIED
#                 )
#                 trainer.fit(epochs=args.cgan_epochs, log_every=50)
#                 Xg = trainer.synthesize(y=np.array(test_indices), num_per_class=args.num_per_class)
#                 yg = np.repeat(test_indices, args.num_per_class)
#
#             else:
#                 raise ValueError(f"未知 cGAN 版本: {args.cgan_version}")
#
#             np.savez(gen_file, X=Xg, y=yg, class_names=np.array(full_class_names))
#             print(f"生成样本已保存至 {gen_file}")
#             gen_path = gen_file
#
#     # 准备测试数据：直接使用已归一化的信号
#     X_test_scaled = X_test_real   # 已经是 [-1,1]
#
#     # 存储每次运行的结果
#     all_results = []
#
#     for run in range(args.num_runs):
#         print(f"\n========== 第 {run+1}/{args.num_runs} 次运行 ==========")
#         run_seed = args.seed + run * 100
#         set_seed(run_seed)
#
#         tester = DTNTest(
#             config=None,
#             augment_type=args.augment_type,
#             augment_shot=args.augment_shot,
#             target_gen_path=gen_path if args.augment_type == 'gan' else None,
#             noise_level=args.noise_level
#         )
#
#         if args.target_test:
#             results = tester.run_target_test(
#                 X_target=X_test_scaled,
#                 y_target=y_test_real,
#                 target_class_indices=test_indices,
#                 shot_configs=args.shot_configs,
#                 base_seed=run_seed
#             )
#         else:
#             results = tester.run_test(
#                 X_source=X_test_scaled,
#                 y_source=y_test_real,
#                 source_class_indices=test_indices,
#                 shot_configs=args.shot_configs,
#                 base_seed=run_seed
#             )
#
#         all_results.append(results)
#
#         run_file = f"dtn_results_{'target' if args.target_test else 'source'}_run{run+1}.json"
#         with open(run_file, "w") as f:
#             json.dump(results, f, indent=2, default=str)
#         print(f"第 {run+1} 次结果已保存至 {run_file}")
#
#     aggregated = aggregate_results(all_results, args.shot_configs)
#     print("\n========== 最终聚合结果（三次平均） ==========")
#     for shot, stats in aggregated.items():
#         print(f"{shot}: mean = {stats['mean_mean']:.4f} ± {stats['mean_std']:.4f} "
#               f"(原始std平均 = {stats['std_mean']:.4f})")
#
#     agg_file = f"dtn_results_{'target' if args.target_test else 'source'}_aggregated.json"
#     with open(agg_file, "w") as f:
#         json.dump(aggregated, f, indent=2, default=str)
#     print(f"\n聚合结果已保存至 {agg_file}")
#
#
# if __name__ == "__main__":
#     main()
