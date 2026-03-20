import os
import numpy as np

# ============= 路径配置 =============
DATA_DIR = r"D:\CWRU"                     # CWRU 原始数据目录
FEATURE_FILE = "cwru_features_labels_10class.npz"   # 10类特征缓存
SIGNAL_FILE = "cwru_signals_10class.npy"            # 10类原始信号缓存
KG_SAVE_DIR = "./knowledge_graphs"                  # 知识图谱保存目录
os.makedirs(KG_SAVE_DIR, exist_ok=True)

# ============= 文件映射（含具体故障尺寸） =============
file_mapping = {
    "Normal_0": "97.mat", "Normal_1": "98.mat", "Normal_2": "99.mat", "Normal_3": "100.mat",
    "OR_007_0": "130.mat", "OR_007_1": "131.mat", "OR_007_2": "132.mat", "OR_007_3": "133.mat",
    "OR_014_0": "197.mat", "OR_014_1": "198.mat", "OR_014_2": "199.mat", "OR_014_3": "200.mat",
    "OR_021_0": "234.mat", "OR_021_1": "235.mat", "OR_021_2": "236.mat", "OR_021_3": "237.mat",
    "IR_007_0": "105.mat", "IR_007_1": "106.mat", "IR_007_2": "107.mat", "IR_007_3": "108.mat",
    "IR_014_0": "169.mat", "IR_014_1": "170.mat", "IR_014_2": "171.mat", "IR_014_3": "172.mat",
    "IR_021_0": "209.mat", "IR_021_1": "210.mat", "IR_021_2": "211.mat", "IR_021_3": "212.mat",
    "B_007_0": "118.mat", "B_007_1": "119.mat", "B_007_2": "120.mat", "B_007_3": "121.mat",
    "B_014_0": "185.mat", "B_014_1": "186.mat", "B_014_2": "187.mat", "B_014_3": "188.mat",
    "B_021_0": "222.mat", "B_021_1": "223.mat", "B_021_2": "224.mat", "B_021_3": "225.mat",
}

# ============= 源域与目标域划分（固定顺序） =============
source_classes = ['IR_007', 'OR_007', 'IR_014', 'OR_014', 'IR_021', 'OR_021']
target_classes = ['Normal', 'B_007', 'B_014', 'B_021']

# ============= 完整的类别列表（源域在前0-5，目标域在后6-9） =============
FULL_CLASS_LIST = source_classes + target_classes   # 总共10类
unique_categories = FULL_CLASS_LIST                  # 用于标签编码的类别顺序

# 验证所有类别是否都出现在文件映射中
_all_keys = set()
for key in file_mapping.keys():
    parts = key.split('_')
    if parts[0] == 'Normal':
        cls = 'Normal'
    else:
        cls = parts[0] + '_' + parts[1]
    _all_keys.add(cls)
assert set(unique_categories) == _all_keys, "类别列表与文件映射不一致！"

# ============= 特征名称定义 =============
TIME_FEATURE_NAMES = [
    "Mean", "Std", "Sq-Mean-of-Sqrt", "RMS", "Peak",
    "Skewness", "Kurtosis", "Waveform-Factor", "Peak-Factor",
    "Impulse-Factor", "Crest-Factor"
]
FREQ_FEATURE_NAMES = [
    "Freq-Mean", "Freq-Var", "Freq-Skewness", "Freq-Kurtosis",
    "Freq-Center", "Freq-RMS", "Freq-Std", "Freq-4th-RMS",
    "Freq-Shape-Factor", "Freq-Skew-Factor", "Freq-Skewness-2", "Freq-Kurtosis-2"
]
VMD_FEATURE_NAMES = [
    "VMD-Energy-1", "VMD-Energy-2", "VMD-Energy-3", "VMD-Energy-4",
    "VMD-SVD-1", "VMD-SVD-2", "VMD-SVD-3", "VMD-SVD-4"
]
FULL_FEATURE_NAMES = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES + VMD_FEATURE_NAMES

# ============= 通用工具函数 =============
def minmax_scale_np(x, eps=1e-8):
    """
    对一批信号进行逐样本缩放至 [-1, 1]。
    x : (N, T) 或 (N, 1, T)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        axis = (1, 2)
    elif x.ndim == 2:
        axis = 1
    else:
        raise ValueError("输入应为 2D 或 3D 数组")
    max_abs = np.max(np.abs(x), axis=axis, keepdims=True) + eps
    return x / max_abs