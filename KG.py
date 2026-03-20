import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.fft import fft
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

from common import (
    SIGNAL_FILE, FEATURE_FILE, KG_SAVE_DIR,
    TIME_FEATURE_NAMES, FREQ_FEATURE_NAMES, VMD_FEATURE_NAMES, FULL_FEATURE_NAMES,
    unique_categories, minmax_scale_np
)

# ============= 特征提取函数（与之前相同） =============
def extract_time_domain_features(signal):
    N = len(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal, ddof=1)
    abs_signal = np.abs(signal)
    sqrt_abs = np.sqrt(abs_signal)
    mean_abs = np.mean(abs_signal)
    mean_sqrt_abs = np.mean(sqrt_abs)

    p1 = mean_val
    p2 = std_val
    p3 = np.square(np.mean(sqrt_abs))
    p4 = np.sqrt(np.mean(signal**2))
    p5 = np.max(abs_signal)
    p6 = skew(signal)
    p7 = kurtosis(signal)
    p8 = p5 / p4
    p9 = p5 / np.square(mean_sqrt_abs)
    p10 = p4 / mean_abs
    p11 = p5 / mean_abs
    return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]

def extract_frequency_domain_features(signal, fs=12000):
    n = len(signal)
    fft_vals = fft(signal)
    amplitude_spectrum = np.abs(fft_vals[: n // 2]) / n
    freqs = np.fft.fftfreq(n, 1 / fs)[: n // 2]
    if len(freqs) == 0:
        return np.zeros(12)

    s = amplitude_spectrum
    s_sum = np.sum(s) + 1e-12
    L = len(s)

    p12 = np.sum(s) / L
    p13 = np.sum((s - p12) ** 2) / (L - 1)
    p14 = np.sum((s - p12) ** 3) / (L * (np.sqrt(p13)) ** 3)
    p15 = np.sum((s - p12) ** 4) / (L * p13**2)
    p16 = np.sum(freqs * s) / s_sum
    p17 = np.sqrt(np.sum(freqs**2 * s) / s_sum)
    p18 = np.sqrt(np.sum((freqs - p16) ** 2 * s) / L)
    p19 = np.sqrt(np.sum(freqs**4 * s) / np.sum(freqs**2 * s))
    numerator = np.sum(freqs**2 * s)
    denominator = s_sum * np.sum(freqs**4 * s)
    p20 = np.sqrt(numerator / denominator)
    p21 = p18 / p16
    p22 = np.sum((freqs - p16) ** 3 * s) / (L * p18**3)
    p23 = np.sum((freqs - p16) ** 4 * s) / (L * p18**4)

    return [p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23]

def extract_vmd_features(signal, alpha=1000, tau=0, K=4, DC=0, init=1, tol=1e-7):
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    if u.shape[0] != 4:
        raise ValueError(f"VMD分解应得到4个模态，但实际得到{u.shape[0]}个")
    energy_features = [np.sum(np.square(mode)) for mode in u]
    mode_matrix = np.array(u)
    U, S, Vh = np.linalg.svd(mode_matrix, full_matrices=False)
    svd_features = S.tolist()
    while len(svd_features) < 4:
        svd_features.append(0.0)
    return energy_features + svd_features

def extract_all_features(signal, fs=12000, vmd_params=None):
    if vmd_params is None:
        vmd_params = {}
    time_feat = extract_time_domain_features(signal)
    freq_feat = extract_frequency_domain_features(signal, fs=fs)
    vmd_feat = extract_vmd_features(signal, **vmd_params)
    return time_feat + freq_feat + vmd_feat

def batch_extract_features(signals, fs=12000, vmd_params=None, scaler=None,
                           fit_scaler=True, n_jobs=-1):
    if vmd_params is None:
        vmd_params = {}
    if n_jobs is None or n_jobs == 1:
        all_features = [extract_all_features(sig, fs=fs, vmd_params=vmd_params)
                        for sig in signals]
    else:
        def _extract_one(sig):
            return extract_all_features(sig, fs=fs, vmd_params=vmd_params)
        all_features = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_extract_one)(sig) for sig in signals
        )
    features = np.asarray(all_features, dtype=float)
    if scaler is None:
        scaler = MinMaxScaler()
    if fit_scaler:
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    return features_scaled, scaler

def signals_to_features(X_signals, fs=12000, fit_scaler=True, n_jobs=-1):
    X_signals = np.asarray(X_signals, dtype=np.float64)
    feats_scaled, _ = batch_extract_features(
        signals=X_signals, fs=fs, vmd_params=None,
        scaler=None, fit_scaler=fit_scaler, n_jobs=n_jobs
    )
    assert feats_scaled.shape[1] == len(FULL_FEATURE_NAMES)
    return feats_scaled

# ============= 知识图谱构建函数 =============
def compute_feature_fault_weights(X, y, class_names, feature_names, membership_soft=None):
    N, D = X.shape
    C = len(class_names)

    if membership_soft is not None:
        U = membership_soft.astype(float)
        U = np.clip(U, 0.0, 1.0)
    else:
        U = np.zeros((N, C), dtype=float)
        U[np.arange(N), y] = 1.0

    denom = U.sum(axis=0, keepdims=True).T
    denom[denom == 0] = 1e-12
    v = (U.T @ X) / denom

    sigma = np.zeros((C, D), dtype=float)
    for i in range(C):
        diff = X - v[i]
        sigma[i] = (U[:, i][:, None] * (diff**2)).sum(axis=0)
    sigma = np.maximum(sigma, 1e-12)

    inv_sqrt = 1.0 / np.sqrt(sigma)
    w = inv_sqrt / inv_sqrt.sum(axis=1, keepdims=True)

    # 保存
    np.savez(os.path.join(KG_SAVE_DIR, "kg_step2_w_v_sigma.npz"),
             w=w, v=v, sigma=sigma,
             feature_names=np.array(feature_names),
             class_names=np.array(class_names))
    return v, sigma, w

def build_fault_transition_matrix(y, num_classes, group_ids=None, smoothing=1e-3):
    C = num_classes
    P = np.zeros((C, C), dtype=float)

    if group_ids is not None:
        groups = defaultdict(list)
        for idx, g in enumerate(group_ids):
            groups[g].append(idx)
        for g, idxs in groups.items():
            idxs = sorted(idxs)
            for a, b in zip(idxs[:-1], idxs[1:]):
                P[y[a], y[b]] += 1.0
        P = P + smoothing
        P = P / P.sum(axis=1, keepdims=True)
    else:
        P[:] = smoothing
        np.fill_diagonal(P, 1.0)
        P = P / P.sum(axis=1, keepdims=True)

    np.save(os.path.join(KG_SAVE_DIR, "kg_step3_P_transition.npy"), P)
    return P

def estimate_Ec_from_real(X_signals, y, fs=12000):
    from scipy.signal import stft
    C = len(np.unique(y))
    Ec = np.zeros((C, 4), dtype=np.float32)
    for cls in range(C):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        signals = X_signals[idx]
        signals = minmax_scale_np(signals)
        all_E = []
        for sig in signals:
            f, t, Zxx = stft(sig, fs=fs, nperseg=256, noverlap=128)
            mag = np.abs(Zxx)
            energy_per_band = []
            bands = [(0,600), (600,1800), (1800,3600), (3600,6000)]
            freqs = f
            for (low, high) in bands:
                mask = (freqs >= low) & (freqs < high)
                if mask.sum() == 0:
                    band_E = 0.0
                else:
                    band_E = np.sum(mag[mask, :])
                energy_per_band.append(band_E)
            E = np.array(energy_per_band)
            E_ratio = E / (E.sum() + 1e-8)
            all_E.append(E_ratio)
        Ec[cls] = np.mean(all_E, axis=0)
    return Ec

def assemble_kg_graph(w, P, class_names, feature_names, graph_name="bearing_KG"):
    G = nx.DiGraph(name=graph_name)
    for k, fn in enumerate(feature_names):
        G.add_node(f"F_{k}", kind="feature", name=fn)
    for i, cn in enumerate(class_names):
        G.add_node(f"C_{i}", kind="fault", name=cn)

    C, D = w.shape
    for i in range(C):
        for k in range(D):
            weight = float(w[i, k])
            if weight > 1e-8:
                G.add_edge(f"F_{k}", f"C_{i}", etype="feature_fault", weight=weight)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            prob = float(P[i, j])
            if prob > 1e-8:
                G.add_edge(f"C_{i}", f"C_{j}", etype="fault_transition", prob=prob)

    nx.write_graphml(G, os.path.join(KG_SAVE_DIR, "bearing_KG.graphml"))
    data = nx.node_link_data(G, link="links")
    with open(os.path.join(KG_SAVE_DIR, "bearing_KG.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return G

def plot_bipartite_feature_fault(
    w, class_names, feature_names,
    topk=None, cmap_name="magma",
    min_width=0.8, max_width=5.0,
    node_size_feature=160, node_size_fault=600,
    figsize=(16, 9),
    title="Feature→Fault KG (bipartite)",
    draw_transitions=False, P=None,
    trans_thr=0.0, trans_min_width=1.0, trans_max_width=4.0,
    save_path=None
):
    C, D = w.shape
    w_min, w_max = float(np.min(w)), float(np.max(w))
    norm = Normalize(vmin=w_min, vmax=w_max, clip=True)
    cmap = plt.get_cmap(cmap_name)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    pos = {}
    xs_feat = np.linspace(0.02, 0.98, D)
    for k, x in enumerate(xs_feat):
        pos[f"F_{k}"] = (x, 0.1)
    xs_cls = np.linspace(0.1, 0.9, C)
    for i, x in enumerate(xs_cls):
        pos[f"C_{i}"] = (x, 0.9)

    edges = []
    if topk is None:
        for i in range(C):
            for k in range(D):
                edges.append((f"F_{k}", f"C_{i}", float(w[i, k])))
    else:
        for i in range(C):
            idx = np.argsort(w[i])[::-1][:topk]
            for k in idx:
                edges.append((f"F_{k}", f"C_{i}", float(w[i, k])))

    fig, ax = plt.subplots(figsize=figsize)

    feature_nodes = [f"F_{k}" for k in range(D)]
    fault_nodes = [f"C_{i}" for i in range(C)]
    _G = nx.DiGraph()
    _G.add_nodes_from(feature_nodes + fault_nodes)

    nx.draw_networkx_nodes(
        _G, pos, nodelist=feature_nodes,
        node_color="#bcdff5", node_size=node_size_feature,
        edgecolors="#5b8fb9", linewidths=0.6, alpha=0.95, ax=ax
    )
    nx.draw_networkx_nodes(
        _G, pos, nodelist=fault_nodes,
        node_color="#f7b3ab", node_size=node_size_fault,
        edgecolors="#b85e57", linewidths=1.2, alpha=0.95, ax=ax
    )

    fault_labels = {f"C_{i}": class_names[i] for i in range(C)}
    nx.draw_networkx_labels(_G, pos, labels=fault_labels, font_size=11, font_weight="bold", ax=ax)

    for k in range(D):
        x, y = pos[f"F_{k}"]
        ax.text(x, y - 0.02, feature_names[k], fontsize=7, rotation=45,
                ha="right", va="top")

    for u, v, wt in edges:
        if u not in pos or v not in pos:
            continue
        color = cmap(norm(wt))
        width = min_width + (max_width - min_width) * norm(wt)
        nx.draw_networkx_edges(
            _G, pos, edgelist=[(u, v)],
            width=width, edge_color=[color], alpha=0.85,
            arrows=False, ax=ax
        )

    if draw_transitions and (P is not None):
        for i in range(C):
            for j in range(C):
                if i != j and P[i, j] > trans_thr:
                    val = float(P[i, j])
                    color = cmap(norm(val))
                    width = trans_min_width + (trans_max_width - trans_min_width) * norm(val)
                    nx.draw_networkx_edges(
                        _G, pos, edgelist=[(f"C_{i}", f"C_{j}")],
                        width=width, edge_color=[color], alpha=0.85,
                        arrows=True, arrowstyle="-|>", arrowsize=15,
                        connectionstyle="arc3,rad=0.2", ax=ax
                    )

    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Feature→Fault weight $w_{ik}$", rotation=90)

    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    # 1. 加载全量信号
    if not os.path.exists(SIGNAL_FILE):
        raise FileNotFoundError(f"请先运行 CWRU_preprocess.py 生成 {SIGNAL_FILE}")
    X_signals = np.load(SIGNAL_FILE)
    # 加载标签
    label_file = FEATURE_FILE.replace(".npz", "_labels_only.npz")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"请先运行 CWRU_preprocess.py 生成 {label_file}")
    ld = np.load(label_file, allow_pickle=True)
    y = ld["y"]
    class_names = ld["class_names"].tolist()

    # 2. 提取特征
    print("提取特征...")
    X = signals_to_features(X_signals, fs=12000, n_jobs=-1)
    np.savez(FEATURE_FILE, X=X, y=y, class_names=np.array(class_names),
             feature_names=np.array(FULL_FEATURE_NAMES))
    print(f"特征已保存至 {FEATURE_FILE}")

    # 3. 构建知识图谱
    v, sigma, w = compute_feature_fault_weights(
        X=X, y=y, class_names=class_names, feature_names=FULL_FEATURE_NAMES,
        membership_soft=None
    )
    P = build_fault_transition_matrix(y, num_classes=len(class_names),
                                      group_ids=None, smoothing=1e-3)
    G = assemble_kg_graph(w, P, class_names=class_names,
                          feature_names=FULL_FEATURE_NAMES)

    # 4. 估计并保存 E_c
    print("估计四频带能量比例 E_c...")
    E_c = estimate_Ec_from_real(X_signals, y, fs=12000)
    np.save(os.path.join(KG_SAVE_DIR, "Ec.npy"), E_c)

    print("KG 构建完成：")
    print(f"  - w shape: {w.shape}")
    print(f"  - P shape: {P.shape}")
    print(f"  - E_c shape: {E_c.shape}")
    print(f"  - Graph nodes/edges: {G.number_of_nodes()}, {G.number_of_edges()}")

    # 5. 可视化并保存图片
    plot_bipartite_feature_fault(
        w=w, class_names=class_names, feature_names=FULL_FEATURE_NAMES,
        topk=None, cmap_name="magma",
        min_width=0.8, max_width=5.0,
        figsize=(15, 8),
        title="Feature→Fault KG (10 classes)",
        draw_transitions=True, P=P, trans_thr=0,
        trans_min_width=1.5, trans_max_width=4.0,
        save_path=os.path.join(KG_SAVE_DIR, "kg_bipartite_10class.png")
    )