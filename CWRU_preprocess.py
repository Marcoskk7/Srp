# import json
# import os
# import random
# import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt
#
# from common import (
#     DATA_DIR, file_mapping, unique_categories,
#     SIGNAL_FILE, FEATURE_FILE, KG_SAVE_DIR,
#     source_classes, target_classes, minmax_scale_np
# )
#
# def load_cwru_data_fixed(
#     file_mapping: dict,
#     data_dir: str,
#     sample_length: int = 2400,
#     num_samples_per_class: int = 500,
#     use_record: bool = False,
#     selected_classes: list = None   # 如果为None，则加载所有类
# ):
#     """
#     从 CWRU 数据集中抽样，并支持读取/写入抽样记录。
#     selected_classes: 指定要加载的类别名称列表，例如 ['IR_007', 'Normal']。
#                       如果为None，则加载所有类别。
#     """
#     record_path = "cwru_samples_record.json"
#
#     if use_record:
#         if not os.path.exists(record_path):
#             print(f"[警告] use_record=True 但记录文件 {record_path} 不存在，将退化为随机抽样。")
#             record = {}
#         else:
#             with open(record_path, "r", encoding="utf-8") as f:
#                 record = json.load(f)
#             print(f"[信息] 已从 {record_path} 读取抽样记录。")
#     else:
#         record = {}
#         new_record = {}
#
#     X, y = [], []
#     class_to_idx = {cls: idx for idx, cls in enumerate(unique_categories)}
#
#     for label_name, file_name in file_mapping.items():
#         # 提取类别名称（如 'IR_007'）
#         parts = label_name.split('_')
#         if parts[0] == 'Normal':
#             class_name = 'Normal'
#         else:
#             class_name = parts[0] + '_' + parts[1]
#
#         if selected_classes is not None and class_name not in selected_classes:
#             continue   # 跳过不需要的类别
#
#         category_idx = class_to_idx[class_name]
#
#         file_path = os.path.join(data_dir, file_name)
#         if not os.path.exists(file_path):
#             print(f"[文件缺失] {file_path}")
#             continue
#
#         data = sio.loadmat(file_path)
#         signal_key = [k for k in data.keys() if "_DE_time" in k]
#         if not signal_key:
#             print(f"[未找到 DE_time 信号] {file_path}")
#             continue
#
#         signal = data[signal_key[0]].flatten()
#         if len(signal) < sample_length:
#             print(f"[信号过短] {file_path}, 长度: {len(signal)}")
#             continue
#
#         record_key = f"{label_name}|||{file_name}"
#         if use_record:
#             recorded_starts = record.get(record_key, [])
#         else:
#             recorded_starts = []
#             new_record[record_key] = []
#
#         valid_max_start = len(signal) - sample_length
#
#         for i in range(num_samples_per_class):
#             if use_record and i < len(recorded_starts):
#                 start = int(recorded_starts[i])
#                 if start < 0 or start > valid_max_start:
#                     print(f"[警告] 记录起点无效: {start}，改为随机抽样")
#                     start = random.randint(0, valid_max_start)
#             else:
#                 start = random.randint(0, valid_max_start)
#                 if not use_record:
#                     new_record[record_key].append(int(start))
#
#             segment = signal[start: start + sample_length]
#             X.append(segment)
#             y.append(category_idx)
#
#         print(f"[完成] {label_name} -> {class_name} 从 {file_name} 采样 {num_samples_per_class} 个片段"
#               + ("（使用记录）" if use_record and recorded_starts else ""))
#
#     X = np.array(X)
#     y = np.array(y)
#
#     if not use_record:
#         try:
#             with open(record_path, "w", encoding="utf-8") as f:
#                 json.dump(new_record, f, ensure_ascii=False, indent=2)
#             print(f"[信息] 抽样记录已写入 {record_path}")
#         except Exception as e:
#             print(f"[错误] 写记录文件失败: {e}")
#
#     return X, y, unique_categories
#
# def plot_original_signals(X, y, class_names, num_samples_per_class=5, save_path="original_signals.png"):
#     C = len(class_names)
#     fig, axes = plt.subplots(C, num_samples_per_class, figsize=(15, 2*C))
#     if C == 1:
#         axes = axes[np.newaxis, :]
#     for i in range(C):
#         idxs = np.where(y == i)[0][:num_samples_per_class]
#         for j, idx in enumerate(idxs):
#             ax = axes[i, j] if C > 1 else axes[j]
#             ax.plot(X[idx])
#             ax.set_xticks([])
#             ax.set_yticks([])
#             if j == 0:
#                 ax.set_ylabel(class_names[i], fontsize=10)
#     plt.suptitle("Original Vibration Signals (each column is a sample)")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.show()
#     print(f"原始信号图已保存至 {save_path}")
#
# if __name__ == "__main__":
#     # 参数设置
#     SAMPLE_LENGTH = 2400
#     NUM_SAMPLES_PER_CLASS = 500   # 每类采样数
#     USE_RECORD = False
#
#     # 加载所有类别的数据（用于知识图谱构建）
#     X_all, y_all, class_names_all = load_cwru_data_fixed(
#         file_mapping, DATA_DIR,
#         sample_length=SAMPLE_LENGTH,
#         num_samples_per_class=NUM_SAMPLES_PER_CLASS,
#         use_record=USE_RECORD,
#         selected_classes=None   # 加载所有类
#     )
#     print(f"总样本数: {X_all.shape}, 类别数: {len(np.unique(y_all))}")
#
#     # 保存全量信号
#     np.save(SIGNAL_FILE, X_all)
#     np.savez(FEATURE_FILE.replace(".npz", "_labels_only.npz"),
#              y=y_all, class_names=np.array(class_names_all))
#     print(f"原始信号已保存至 {SIGNAL_FILE}")
#
#     # 绘制原始信号示例图
#     plot_original_signals(X_all, y_all, class_names_all,
#                           save_path=os.path.join(KG_SAVE_DIR, "original_signals_10class.png"))
#
#     # 可选：单独保存源域和目标域的信号（用于后续训练验证）
#     source_idx = [i for i, cls in enumerate(class_names_all) if cls in source_classes]
#     target_idx = [i for i, cls in enumerate(class_names_all) if cls in target_classes]
#
#     mask_source = np.isin(y_all, source_idx)
#     X_source = X_all[mask_source]
#     y_source = y_all[mask_source]
#     np.savez("source_data.npz", X=X_source, y=y_source, class_names=np.array(class_names_all))
#
#     mask_target = np.isin(y_all, target_idx)
#     X_target = X_all[mask_target]
#     y_target = y_all[mask_target]
#     np.savez("target_data.npz", X=X_target, y=y_target, class_names=np.array(class_names_all))
#
#     print("源域和目标域数据已单独保存。")




import json
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from common import (
    DATA_DIR, file_mapping, unique_categories,
    SIGNAL_FILE, FEATURE_FILE, KG_SAVE_DIR,
    source_classes, target_classes
)

def load_cwru_data_fixed(
    file_mapping: dict,
    data_dir: str,
    sample_length: int = 2400,
    overlap_ratio: float = 0.5,
    max_samples_per_file: int = 500
):
    """
    加载 CWRU 数据，使用滑动窗口采样，并进行全局归一化到 [-1,1]。
    同时返回归一化后的信号片段和原始信号片段。
    """
    X_norm = []      # 归一化后的信号（用于训练/测试）
    X_raw = []       # 原始信号（用于知识图谱）
    y = []
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_categories)}
    stride = int(sample_length * (1 - overlap_ratio))

    for label_name, file_name in file_mapping.items():
        # 提取类别名
        parts = label_name.split('_')
        if parts[0] == 'Normal':
            class_name = 'Normal'
        else:
            class_name = parts[0] + '_' + parts[1]
        class_idx = class_to_idx[class_name]

        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"[文件缺失] {file_path}")
            continue

        data = sio.loadmat(file_path)
        signal_key = [k for k in data.keys() if "_DE_time" in k]
        if not signal_key:
            print(f"[未找到 DE_time 信号] {file_path}")
            continue
        raw_signal = data[signal_key[0]].flatten().astype(np.float64)
        if len(raw_signal) < sample_length:
            print(f"[信号过短] {file_path}")
            continue

        # 全局归一化到 [-1,1]
        s_min, s_max = raw_signal.min(), raw_signal.max()
        norm_signal = (raw_signal - s_min) / (s_max - s_min + 1e-10) * 2 - 1

        # 滑动窗口采样
        num_samples = (len(raw_signal) - sample_length) // stride + 1
        num_samples = min(num_samples, max_samples_per_file)

        for i in range(num_samples):
            start = i * stride
            end = start + sample_length
            X_norm.append(norm_signal[start:end])
            X_raw.append(raw_signal[start:end])
            y.append(class_idx)

        print(f"[完成] {label_name} -> {class_name}, 采样 {num_samples} 个片段")

    X_norm = np.array(X_norm, dtype=np.float32)
    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X_norm, X_raw, y, unique_categories

if __name__ == "__main__":
    SAMPLE_LENGTH = 2400
    OVERLAP = 0.5
    MAX_SAMPLES = 500   # 每个文件最多采样数

    # 加载所有类别的数据（同时得到归一化和原始信号）
    X_norm, X_raw, y, class_names_all = load_cwru_data_fixed(
        file_mapping, DATA_DIR,
        sample_length=SAMPLE_LENGTH,
        overlap_ratio=OVERLAP,
        max_samples_per_file=MAX_SAMPLES
    )
    print(f"总样本数: {X_norm.shape}, 类别数: {len(np.unique(y))}")

    # 保存原始信号（未归一化）供 KG 使用
    np.save(SIGNAL_FILE, X_raw)
    # 保存标签和类名（供 KG 加载）
    np.savez(FEATURE_FILE.replace(".npz", "_labels_only.npz"),
             y=y, class_names=np.array(class_names_all))
    print(f"原始信号已保存至 {SIGNAL_FILE}")

    # 根据源域和目标域划分保存归一化后的信号
    source_idx = [class_names_all.index(c) for c in source_classes]
    target_idx = [class_names_all.index(c) for c in target_classes]

    mask_source = np.isin(y, source_idx)
    X_source = X_norm[mask_source]
    y_source = y[mask_source]
    np.savez("source_data.npz", X=X_source, y=y_source,
             class_names=np.array(class_names_all))

    mask_target = np.isin(y, target_idx)
    X_target = X_norm[mask_target]
    y_target = y[mask_target]
    np.savez("target_data.npz", X=X_target, y=y_target,
             class_names=np.array(class_names_all))

    print("源域和目标域归一化数据已保存。")
