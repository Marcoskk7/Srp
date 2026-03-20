# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from common import minmax_scale_np
#
# # ============= 新增评估所需导入 =============
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# from scipy.linalg import sqrtm
# import json
# from KG import extract_all_features          # 用于特征提取
# from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel   # 用于 MMD / KID
#
# from typing import Optional, List, Union
#
# # ============= 辅助函数 =============
# def to_tensor(x, device):
#     return torch.as_tensor(x, dtype=torch.float32, device=device)
#
# # ============= 数据集包装 =============
# class SignalsByClass(Dataset):
#     def __init__(self, X_signals: np.ndarray, y: np.ndarray, do_minmax=True):
#         X = np.asarray(X_signals, dtype=np.float32)
#         if do_minmax:
#             X = minmax_scale_np(X)
#         if X.ndim == 2:
#             X = X[:, None, :]  # (B, 1, T)
#         self.X = X
#         self.y = np.asarray(y, dtype=np.int64)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)
#
# # ============= 条件生成器（仅使用标签嵌入） =============
# class CondGenerator1D(nn.Module):
#     def __init__(self, z_dim: int = 128, out_len: int = 2400,
#                  base_ch: int = 128, emb_dim: int = 16, num_classes: int = 10):
#         super().__init__()
#         self.out_len = out_len
#         self.z_dim = z_dim
#         self.emb = nn.Embedding(num_classes, emb_dim)
#         # 将标签嵌入映射到与噪声拼接的维度
#         self.cond_proj = nn.Linear(emb_dim, z_dim, bias=True)
#         self.init_len = 75
#         self.fc = nn.Linear(z_dim * 2, base_ch * self.init_len)
#         self.net = nn.Sequential(
#             nn.ConvTranspose1d(base_ch, base_ch//2, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose1d(base_ch//2, base_ch//4, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose1d(base_ch//4, base_ch//8, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose1d(base_ch//8, base_ch//16, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.ConvTranspose1d(base_ch//16, 1, 4, 2, 1),
#             nn.Tanh(),
#         )
#
#     def forward(self, z, y, return_h=False):
#         # z: (B, z_dim), y: (B,)
#         emb = self.emb(y)                     # (B, emb_dim)
#         c = self.cond_proj(emb)                # (B, z_dim)
#         h = torch.cat([z, c], dim=1)           # (B, 2*z_dim)
#         h = self.fc(h)                         # (B, base_ch * init_len)
#         B = h.size(0)
#         C = h.numel() // (B * self.init_len)
#         h = h.view(B, C, self.init_len)        # (B, C, init_len)
#         x = self.net(h)                         # (B, 1, out_len)
#         if return_h:
#             return x, h
#         else:
#             return x
#
# # ============= 条件判别器（仅使用标签嵌入） =============
# class SimpleDiscriminator1D(nn.Module):
#     def __init__(self, base_ch: int = 64, emb_dim: int = 16, num_classes: int = 10):
#         super().__init__()
#         self.emb = nn.Embedding(num_classes, emb_dim)
#         self.cond_proj = nn.Linear(emb_dim, base_ch * 4)
#
#         self.conv1 = nn.utils.spectral_norm(nn.Conv1d(1, base_ch, 9, 2, 4))
#         self.act1 = nn.LeakyReLU(0.2, inplace=True)
#
#         self.conv2 = nn.utils.spectral_norm(nn.Conv1d(base_ch, base_ch*2, 9, 2, 4))
#         self.act2 = nn.LeakyReLU(0.2, inplace=True)
#
#         self.conv3 = nn.utils.spectral_norm(nn.Conv1d(base_ch*2, base_ch*4, 9, 2, 4))
#         self.act3 = nn.LeakyReLU(0.2, inplace=True)
#
#         self.uncond_head = nn.utils.spectral_norm(nn.Conv1d(base_ch*4, 1, 7, padding=3))
#
#     def forward(self, x, y, noise_std=0.02):
#         if noise_std > 0:
#             x = x + noise_std * torch.randn_like(x)
#         h = self.conv1(x)
#         h = self.act1(h)
#         h = self.conv2(h)
#         h = self.act2(h)
#         h = self.conv3(h)
#         h = self.act3(h)
#
#         g = h.mean(dim=-1)                     # (B, base_ch*4)
#         logits_uncond = self.uncond_head(h).mean(dim=-1).squeeze(1)  # (B,)
#
#         emb = self.emb(y)                       # (B, emb_dim)
#         e = self.cond_proj(emb)                 # (B, base_ch*4)
#         logits = logits_uncond + (g * e).sum(dim=1)
#         return logits
#
# # ============= 简化训练器（无物理约束） =============
# class cGAN_Trainer:
#     def __init__(
#         self,
#         X_signals,
#         y,
#         class_names,
#         batch_size=64,
#         z_dim=128,
#         lr_g=2e-4,
#         lr_d=1e-4,
#         device=None,
#         n_critic=1,
#         log_dir: Optional[str] = None,
#         use_tensorboard: bool = True,
#     ):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.n_critic = n_critic
#         self.step = 0
#
#         self.dataset = SignalsByClass(X_signals, y, do_minmax=True)
#         self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#         self.num_classes = len(class_names)
#         self.class_names = class_names
#
#         self.use_tensorboard = use_tensorboard
#         self.log_dir = log_dir
#         self.writer = None
#         if self.use_tensorboard:
#             if self.log_dir is None:
#                 self.log_dir = os.path.join("runs", "cgan")
#             self.writer = SummaryWriter(log_dir=self.log_dir)
#
#         self.history = {
#             "step": [], "epoch": [], "d_loss": [], "g_loss": []
#         }
#
#         T = self.dataset.X.shape[-1]
#         self.G = CondGenerator1D(
#             z_dim=z_dim, out_len=T, num_classes=self.num_classes
#         ).to(self.device)
#
#         self.D = SimpleDiscriminator1D(num_classes=self.num_classes).to(self.device)
#
#         self.optG = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
#         self.optD = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))
#
#     def sample_noise(self, B, z_dim):
#         return torch.randn(B, z_dim, device=self.device)
#
#     def d_hinge_loss(self, real_logits, fake_logits):
#         return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
#
#     def g_hinge_loss(self, fake_logits):
#         return -fake_logits.mean()
#
#     def fit(self, epochs=50, log_every=50, d_noise_std=0.02):
#         for ep in range(epochs):
#             for xb, yb in self.loader:
#                 xb, yb = xb.to(self.device), yb.to(self.device)
#                 B = xb.size(0)
#
#                 # 训练判别器
#                 for _ in range(self.n_critic):
#                     self.optD.zero_grad()
#                     real_logits = self.D(xb, yb, noise_std=d_noise_std)
#                     z = self.sample_noise(B, self.G.z_dim)
#                     x_fake = self.G(z, yb).detach()
#                     fake_logits = self.D(x_fake, yb, noise_std=d_noise_std)
#                     d_loss = self.d_hinge_loss(real_logits, fake_logits)
#                     d_loss.backward()
#                     self.optD.step()
#
#                 # 训练生成器
#                 self.optG.zero_grad()
#                 z = self.sample_noise(B, self.G.z_dim)
#                 x_fake = self.G(z, yb)
#                 fake_logits = self.D(x_fake, yb, noise_std=0.0)
#                 gan_loss = self.g_hinge_loss(fake_logits)
#                 g_loss = gan_loss
#                 g_loss.backward()
#                 self.optG.step()
#
#                 # 记录日志
#                 self.history["step"].append(self.step)
#                 self.history["epoch"].append(ep)
#                 self.history["d_loss"].append(d_loss.item())
#                 self.history["g_loss"].append(g_loss.item())
#
#                 if self.writer is not None:
#                     self.writer.add_scalar("loss/D", d_loss.item(), self.step)
#                     self.writer.add_scalar("loss/G", g_loss.item(), self.step)
#
#                 if self.step % log_every == 0:
#                     print(
#                         f"[ep {ep:03d} step {self.step:06d}] "
#                         f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
#                     )
#
#                 self.step += 1
#
#     @torch.no_grad()
#     def synthesize(self, y: np.ndarray, num_per_class=10):
#         y = np.asarray(y, dtype=np.int64)
#         all_out = []
#         for yi in y:
#             z = self.sample_noise(num_per_class, self.G.z_dim)
#             y_tensor = torch.full((num_per_class,), yi, dtype=torch.long, device=self.device)
#             xg = self.G(z, y_tensor)
#             all_out.append(xg.squeeze(1).cpu().numpy())
#         return np.concatenate(all_out, axis=0)
#
# # ============= 修正后的 MMD / KID 计算函数 =============
# def _mmd(X, Y, kernel='rbf', gamma=None, degree=3, coef0=1, subsample=None):
#     """
#     计算最大均值差异 (MMD^2) 的无偏估计。
#     X, Y : array-like, shape (n_samples, n_features)
#     kernel : 'rbf' 或 'poly'
#     subsample : int or None, 如果样本数过多，随机子采样以加速计算
#     """
#     if subsample is not None and subsample < len(X):
#         idx = np.random.choice(len(X), subsample, replace=False)
#         X = X[idx]
#     if subsample is not None and subsample < len(Y):
#         idx = np.random.choice(len(Y), subsample, replace=False)
#         Y = Y[idx]
#
#     if kernel == 'rbf':
#         if gamma is None:
#             gamma = 1.0 / X.shape[1]  # 默认使用 inverse of feature dimension
#         K_xx = rbf_kernel(X, X, gamma=gamma)
#         K_yy = rbf_kernel(Y, Y, gamma=gamma)
#         K_xy = rbf_kernel(X, Y, gamma=gamma)
#     elif kernel == 'poly':
#         K_xx = polynomial_kernel(X, X, degree=degree, gamma=None, coef0=coef0)
#         K_yy = polynomial_kernel(Y, Y, degree=degree, gamma=None, coef0=coef0)
#         K_xy = polynomial_kernel(X, Y, degree=degree, gamma=None, coef0=coef0)
#     else:
#         raise ValueError("kernel must be 'rbf' or 'poly'")
#
#     m = X.shape[0]
#     n = Y.shape[0]
#     # 无偏估计：除去对角线的项
#     mmd = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1)) + \
#           (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1)) - \
#           2 * K_xy.mean()
#     return mmd
#
# def _kid(X, Y, degree=3, coef0=1, subsample=100):
#     """
#     计算核 Inception 距离 (KID) 的无偏估计，使用多项式核。
#     默认子采样到最多 100 个样本（因 KID 通常用于小批量）。
#     """
#     return _mmd(X, Y, kernel='poly', degree=degree, coef0=coef0, subsample=subsample)
#
# # ============= 评估函数（增强版） =============
# def plot_training_curves(history, save_dir=None):
#     """绘制训练过程中的损失曲线"""
#     steps = history["step"]
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#
#     axes[0].plot(steps, history["d_loss"], label="D_loss", color='blue')
#     axes[0].set_xlabel("Step")
#     axes[0].set_ylabel("Loss")
#     axes[0].set_title("Discriminator Loss")
#     axes[0].grid(True)
#
#     axes[1].plot(steps, history["g_loss"], label="G_loss", color='red')
#     axes[1].set_xlabel("Step")
#     axes[1].set_ylabel("Loss")
#     axes[1].set_title("Generator Loss")
#     axes[1].grid(True)
#
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(os.path.join(save_dir, "cgan_training_curves.png"), dpi=150)
#     plt.show()
#
# def plot_waveform_comparison(real_signals, real_labels, fake_signals, fake_labels,
#                              class_names, num_samples=3, save_dir=None, tag=""):
#     """随机选取每类几个样本，绘制真实与生成信号的时域波形对比"""
#     unique_cls = np.unique(real_labels)
#     fig, axes = plt.subplots(len(unique_cls), num_samples, figsize=(num_samples * 4, len(unique_cls) * 2))
#     if len(unique_cls) == 1:
#         axes = axes[np.newaxis, :]
#     for i, cls in enumerate(unique_cls):
#         real_idx = np.where(real_labels == cls)[0]
#         fake_idx = np.where(fake_labels == cls)[0]
#         # 随机选择 num_samples 个
#         real_sel = np.random.choice(real_idx, min(len(real_idx), num_samples), replace=False)
#         fake_sel = np.random.choice(fake_idx, min(len(fake_idx), num_samples), replace=False)
#
#         for j in range(num_samples):
#             ax = axes[i, j]
#             if j < len(real_sel):
#                 ax.plot(real_signals[real_sel[j]], color='blue', alpha=0.7, label='real' if j == 0 else "")
#             if j < len(fake_sel):
#                 ax.plot(fake_signals[fake_sel[j]], color='red', alpha=0.7, label='fake' if j == 0 else "")
#             ax.set_title(f"Class {class_names[cls]} sample {j}")
#             if i == 0 and j == 0:
#                 ax.legend()
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(os.path.join(save_dir, f"cgan_waveform_comparison_{tag}.png"), dpi=150)
#     plt.show()
#
# def evaluate_generation_simple(
#     real_signals, real_labels, fake_signals, fake_labels,
#     class_names,
#     tag="cgan",
#     save_dir=None,
#     selected_features: Optional[List[int]] = None,   # 用于 FID 的特征索引列表
#     compute_mmd: bool = False,
#     mmd_kernel: str = 'rbf',
#     compute_kid: bool = False,
#     kid_subsample: int = 100,
#     mmd_subsample: Optional[int] = None
# ):
#     """
#     评估生成质量：
#     - 计算完整31维特征的 FID
#     - 若 selected_features 不为空，计算基于所选特征的 FID
#     - 若 compute_mmd 为真，计算 MMD
#     - 若 compute_kid 为真，计算 KID
#     - t-SNE 可视化
#     - 时域波形对比
#     返回包含所有指标的字典。
#     """
#     print(f"\n========== 评估生成质量 ({tag}) ==========")
#
#     # 缩放真实信号（生成信号假设已在 [-1,1]）
#     real_scaled = minmax_scale_np(real_signals)
#
#     # 提取31维特征
#     def extract_feats_batch(X):
#         feats = []
#         for sig in X:
#             sig = np.asarray(sig, dtype=np.float64)
#             feats.append(extract_all_features(sig, fs=12000))
#         return np.asarray(feats, dtype=np.float32)
#
#     real_feats = extract_feats_batch(real_scaled)
#     fake_feats = extract_feats_batch(fake_signals)
#
#     metrics = {}
#
#     # ----- 完整特征 FID -----
#     mu_r = np.mean(real_feats, axis=0)
#     sigma_r = np.cov(real_feats, rowvar=False)
#     mu_g = np.mean(fake_feats, axis=0)
#     sigma_g = np.cov(fake_feats, rowvar=False)
#
#     diff = mu_r - mu_g
#     covmean = sqrtm(sigma_r @ sigma_g)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid_full = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
#     metrics['FID_full'] = float(fid_full)
#     print(f"FID (31D feature space): {fid_full:.4f}")
#
#     # ----- 可选：基于选定特征的 FID -----
#     if selected_features is not None:
#         real_feats_sel = real_feats[:, selected_features]
#         fake_feats_sel = fake_feats[:, selected_features]
#         mu_r_sel = np.mean(real_feats_sel, axis=0)
#         sigma_r_sel = np.cov(real_feats_sel, rowvar=False)
#         mu_g_sel = np.mean(fake_feats_sel, axis=0)
#         sigma_g_sel = np.cov(fake_feats_sel, rowvar=False)
#
#         diff_sel = mu_r_sel - mu_g_sel
#         covmean_sel = sqrtm(sigma_r_sel @ sigma_g_sel)
#         if np.iscomplexobj(covmean_sel):
#             covmean_sel = covmean_sel.real
#         fid_sel = diff_sel @ diff_sel + np.trace(sigma_r_sel + sigma_g_sel - 2 * covmean_sel)
#         metrics['FID_selected'] = float(fid_sel)
#         print(f"FID (selected {len(selected_features)} features): {fid_sel:.4f}")
#
#     # ----- MMD -----
#     if compute_mmd:
#         mmd_val = _mmd(real_feats, fake_feats, kernel=mmd_kernel, subsample=mmd_subsample)
#         metrics['MMD'] = float(mmd_val)
#         print(f"MMD ({mmd_kernel} kernel): {mmd_val:.6f}")
#
#     # ----- KID -----
#     if compute_kid:
#         kid_val = _kid(real_feats, fake_feats, subsample=kid_subsample)
#         metrics['KID'] = float(kid_val)
#         print(f"KID (poly kernel, subsample={kid_subsample}): {kid_val:.6f}")
#
#     # ----- t-SNE 可视化 -----
#     X_all = np.vstack([real_feats, fake_feats])
#     domain_labels = np.concatenate([np.zeros(len(real_feats)), np.ones(len(fake_feats))])
#     class_labels = np.concatenate([real_labels, fake_labels])
#
#     scaler = StandardScaler()
#     X_std = scaler.fit_transform(X_all)
#
#     tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
#                 init="random", random_state=42)
#     X_emb = tsne.fit_transform(X_std)
#
#     plt.figure(figsize=(10, 8))
#     unique_cls = np.unique(class_labels)
#     colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cls)))
#     for i, c in enumerate(unique_cls):
#         mask_real = (class_labels == c) & (domain_labels == 0)
#         mask_fake = (class_labels == c) & (domain_labels == 1)
#         plt.scatter(X_emb[mask_real, 0], X_emb[mask_real, 1],
#                     s=30, c=colors[i].reshape(1, -1), marker='o',
#                     label=f"{class_names[c]} (real)")
#         plt.scatter(X_emb[mask_fake, 0], X_emb[mask_fake, 1],
#                     s=30, facecolors='none', edgecolors=colors[i], marker='o',
#                     label=f"{class_names[c]} (fake)")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.title(f"t-SNE: Real vs Generated ({tag})")
#     plt.tight_layout()
#     if save_dir:
#         plt.savefig(os.path.join(save_dir, f"cgan_tsne_{tag}.png"), dpi=150)
#     plt.show()
#     print(f"t-SNE 图已保存至 {save_dir}")
#
#     # 绘制时域波形对比图
#     plot_waveform_comparison(real_signals, real_labels, fake_signals, fake_labels,
#                              class_names, num_samples=3, save_dir=save_dir, tag=tag)
#
#     return metrics
#
# # ============= 新增封装函数：训练cGAN并生成样本 =============
# def train_cgan_and_generate(
#     X_source,
#     y_source,
#     class_names,
#     cgan_params: dict,
#     num_per_class=50,
#     save_path="generated_samples_cgan.npz"
# ):
#     """
#     训练cGAN并生成样本，返回生成样本和标签，同时保存。
#     cgan_params: dict包含 'batch_size', 'z_dim', 'lr_g', 'lr_d', 'n_critic', 'epochs', 'log_dir' 等。
#     """
#     # 创建训练器
#     trainer = cGAN_Trainer(
#         X_signals=X_source,
#         y=y_source,
#         class_names=class_names,
#         batch_size=cgan_params.get('batch_size', 64),
#         z_dim=cgan_params.get('z_dim', 128),
#         lr_g=cgan_params.get('lr_g', 2e-4),
#         lr_d=cgan_params.get('lr_d', 1e-4),
#         n_critic=cgan_params.get('n_critic', 1),
#         log_dir=cgan_params.get('log_dir', None),
#         use_tensorboard=cgan_params.get('use_tensorboard', False)
#     )
#     # 训练
#     trainer.fit(epochs=cgan_params.get('epochs', 50), log_every=50)
#     # 生成
#     unique_classes = np.unique(y_source)
#     Xg = trainer.synthesize(y=unique_classes, num_per_class=num_per_class)
#     yg = np.repeat(unique_classes, num_per_class)
#     # 保存
#     np.savez(save_path, X=Xg, y=yg, class_names=np.array(class_names))
#     print(f"生成样本已保存至 {save_path}")
#     return Xg, yg, trainer.history
#
# # ============= 独立运行示例 =============
# if __name__ == "__main__":
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#     # 1. 加载源域信号和类名（假设只包含类别0-5）
#     source_data = np.load("source_data.npz")
#     X_source = source_data["X"]
#     y_source = source_data["y"]
#     full_class_names = source_data["class_names"].tolist()  # 可能包含10个类名
#
#     # 获取源域中实际出现的类别（假设为0-5）
#     unique_classes = np.unique(y_source)
#     print(f"源域实际类别: {unique_classes}")
#
#     # 根据实际类别筛选 class_names 子集
#     class_names_sub = [full_class_names[i] for i in unique_classes]
#     num_classes = len(class_names_sub)
#
#     print(f"源域样本数: {X_source.shape}, 类别分布: {np.unique(y_source, return_counts=True)}")
#     print(f"使用的类别名称: {class_names_sub}")
#
#     # 2. 创建训练器（示例参数）
#     trainer = cGAN_Trainer(
#         X_signals=X_source,
#         y=y_source,
#         class_names=class_names_sub,
#         lr_g=2e-4,
#         lr_d=1e-4,
#         n_critic=2,
#         log_dir="./runs/cgan",
#         use_tensorboard=True,
#     )
#
#     # 3. 训练
#     trainer.fit(epochs=50, log_every=50)
#
#     # 4. 生成所有类别的样本（每类50个）
#     Xg = trainer.synthesize(y=np.arange(num_classes), num_per_class=50)
#     yg = np.repeat(np.arange(num_classes), 50)
#     print("Synthesized shape:", Xg.shape)
#
#     # 5. 保存生成样本
#     np.savez("generated_samples_cgan.npz", X=Xg, y=yg, class_names=np.array(class_names_sub))
#     print("生成样本已保存至 generated_samples_cgan.npz")
#
#     # ============= 新增评估部分 =============
#     RUN_EVAL = True  # 可设为 True 执行评估
#     if RUN_EVAL:
#         print("\n=== 开始评估生成质量 ===")
#         # 从源域中抽取与生成样本数量相当的真实样本（每类50个）
#         num_per_class_eval = 50
#         real_subset_signals = []
#         real_subset_labels = []
#         for cls in range(num_classes):
#             idx = np.where(y_source == cls)[0]
#             if len(idx) >= num_per_class_eval:
#                 sel = np.random.choice(idx, num_per_class_eval, replace=False)
#             else:
#                 sel = idx
#             real_subset_signals.append(X_source[sel])
#             real_subset_labels.append(np.full(len(sel), cls))
#         real_subset_signals = np.concatenate(real_subset_signals, axis=0)
#         real_subset_labels = np.concatenate(real_subset_labels, axis=0)
#
#         # 调用增强版评估函数
#         selected_indices = [4, 6, 15]  # 例如峰值因子、峭度、重心频率（需对应 FULL_FEATURE_NAMES 索引）
#
#         metrics = evaluate_generation_simple(
#             real_signals=real_subset_signals,
#             real_labels=real_subset_labels,
#             fake_signals=Xg,
#             fake_labels=yg,
#             class_names=class_names_sub,
#             tag="cgan_eval",
#             save_dir="./runs/cgan",
#             selected_features=selected_indices,   # 启用特征选择 FID
#             compute_mmd=True,                      # 启用 MMD（RBF核）
#             mmd_kernel='rbf',
#             compute_kid=True,                      # 启用 KID
#             kid_subsample=100,
#             mmd_subsample=None                      # 不额外子采样
#         )
#         print("评估指标:", metrics)
#         with open(os.path.join("./runs/cgan", "generation_metrics_cgan.json"), "w") as f:
#             json.dump(metrics, f, indent=2)
#
#         # 绘制训练曲线
#         plot_training_curves(trainer.history, save_dir="./runs/cgan")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from common import minmax_scale_np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
import json
from KG import extract_all_features          # 用于特征提取
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel   # 用于 MMD / KID

from typing import Optional, List, Union

# ============= 辅助函数 =============
def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ============= 数据集包装（MODIFIED：添加 do_minmax 参数，默认 False）=============
class SignalsByClass(Dataset):
    def __init__(self, X_signals: np.ndarray, y: np.ndarray, do_minmax=False):
        X = np.asarray(X_signals, dtype=np.float32)
        if do_minmax:
            # 缩放到 [-1, 1]（如果信号不是全局归一化的，则启用）
            X = (X - X.min(axis=-1, keepdims=True)) / (X.max(axis=-1, keepdims=True) - X.min(axis=-1, keepdims=True) + 1e-8) * 2 - 1
        if X.ndim == 2:
            X = X[:, None, :]  # (B, 1, T)
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

# ============= 条件生成器（仅使用标签嵌入） =============
class CondGenerator1D(nn.Module):
    def __init__(self, z_dim: int = 128, out_len: int = 2400,
                 base_ch: int = 128, emb_dim: int = 16, num_classes: int = 10):
        super().__init__()
        self.out_len = out_len
        self.z_dim = z_dim
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.cond_proj = nn.Linear(emb_dim, z_dim, bias=True)
        self.init_len = 75
        self.fc = nn.Linear(z_dim * 2, base_ch * self.init_len)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base_ch, base_ch//2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(base_ch//2, base_ch//4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(base_ch//4, base_ch//8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(base_ch//8, base_ch//16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(base_ch//16, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, y, return_h=False):
        emb = self.emb(y)
        c = self.cond_proj(emb)
        h = torch.cat([z, c], dim=1)
        h = self.fc(h)
        B = h.size(0)
        C = h.numel() // (B * self.init_len)
        h = h.view(B, C, self.init_len)
        x = self.net(h)
        if return_h:
            return x, h
        else:
            return x

# ============= 条件判别器（仅使用标签嵌入） =============
class SimpleDiscriminator1D(nn.Module):
    def __init__(self, base_ch: int = 64, emb_dim: int = 16, num_classes: int = 10):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.cond_proj = nn.Linear(emb_dim, base_ch * 4)

        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(1, base_ch, 9, 2, 4))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.utils.spectral_norm(nn.Conv1d(base_ch, base_ch*2, 9, 2, 4))
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(base_ch*2, base_ch*4, 9, 2, 4))
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.uncond_head = nn.utils.spectral_norm(nn.Conv1d(base_ch*4, 1, 7, padding=3))

    def forward(self, x, y, noise_std=0.02):
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.conv3(h)
        h = self.act3(h)

        g = h.mean(dim=-1)
        logits_uncond = self.uncond_head(h).mean(dim=-1).squeeze(1)

        emb = self.emb(y)
        e = self.cond_proj(emb)
        logits = logits_uncond + (g * e).sum(dim=1)
        return logits

# ============= 简化训练器（无物理约束）（MODIFIED：添加 do_minmax 参数）=============
class cGAN_Trainer:
    def __init__(
        self,
        X_signals,
        y,
        class_names,
        batch_size=64,
        z_dim=128,
        lr_g=2e-4,
        lr_d=1e-4,
        device=None,
        n_critic=1,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = True,
        do_minmax=False,            # ADDED
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_critic = n_critic
        self.step = 0

        self.dataset = SignalsByClass(X_signals, y, do_minmax=do_minmax)   # MODIFIED
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.num_classes = len(class_names)
        self.class_names = class_names

        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.writer = None
        if self.use_tensorboard:
            if self.log_dir is None:
                self.log_dir = os.path.join("runs", "cgan")
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.history = {
            "step": [], "epoch": [], "d_loss": [], "g_loss": []
        }

        T = self.dataset.X.shape[-1]
        self.G = CondGenerator1D(
            z_dim=z_dim, out_len=T, num_classes=self.num_classes
        ).to(self.device)

        self.D = SimpleDiscriminator1D(num_classes=self.num_classes).to(self.device)

        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    def sample_noise(self, B, z_dim):
        return torch.randn(B, z_dim, device=self.device)

    def d_hinge_loss(self, real_logits, fake_logits):
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()

    def g_hinge_loss(self, fake_logits):
        return -fake_logits.mean()

    def fit(self, epochs=50, log_every=50, d_noise_std=0.02):
        for ep in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B = xb.size(0)

                # 训练判别器
                for _ in range(self.n_critic):
                    self.optD.zero_grad()
                    real_logits = self.D(xb, yb, noise_std=d_noise_std)
                    z = self.sample_noise(B, self.G.z_dim)
                    x_fake = self.G(z, yb).detach()
                    fake_logits = self.D(x_fake, yb, noise_std=d_noise_std)
                    d_loss = self.d_hinge_loss(real_logits, fake_logits)
                    d_loss.backward()
                    self.optD.step()

                # 训练生成器
                self.optG.zero_grad()
                z = self.sample_noise(B, self.G.z_dim)
                x_fake = self.G(z, yb)
                fake_logits = self.D(x_fake, yb, noise_std=0.0)
                gan_loss = self.g_hinge_loss(fake_logits)
                g_loss = gan_loss
                g_loss.backward()
                self.optG.step()

                self.history["step"].append(self.step)
                self.history["epoch"].append(ep)
                self.history["d_loss"].append(d_loss.item())
                self.history["g_loss"].append(g_loss.item())

                if self.writer is not None:
                    self.writer.add_scalar("loss/D", d_loss.item(), self.step)
                    self.writer.add_scalar("loss/G", g_loss.item(), self.step)

                if self.step % log_every == 0:
                    print(
                        f"[ep {ep:03d} step {self.step:06d}] "
                        f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}"
                    )

                self.step += 1

    @torch.no_grad()
    def synthesize(self, y: np.ndarray, num_per_class=10):
        y = np.asarray(y, dtype=np.int64)
        all_out = []
        for yi in y:
            z = self.sample_noise(num_per_class, self.G.z_dim)
            y_tensor = torch.full((num_per_class,), yi, dtype=torch.long, device=self.device)
            xg = self.G(z, y_tensor)
            all_out.append(xg.squeeze(1).cpu().numpy())
        return np.concatenate(all_out, axis=0)

# ============= MMD/KID 计算函数（保持不变） =============
def _mmd(X, Y, kernel='rbf', gamma=None, degree=3, coef0=1, subsample=None):
    if subsample is not None and subsample < len(X):
        idx = np.random.choice(len(X), subsample, replace=False)
        X = X[idx]
    if subsample is not None and subsample < len(Y):
        idx = np.random.choice(len(Y), subsample, replace=False)
        Y = Y[idx]

    if kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        K_xx = rbf_kernel(X, X, gamma=gamma)
        K_yy = rbf_kernel(Y, Y, gamma=gamma)
        K_xy = rbf_kernel(X, Y, gamma=gamma)
    elif kernel == 'poly':
        K_xx = polynomial_kernel(X, X, degree=degree, gamma=None, coef0=coef0)
        K_yy = polynomial_kernel(Y, Y, degree=degree, gamma=None, coef0=coef0)
        K_xy = polynomial_kernel(X, Y, degree=degree, gamma=None, coef0=coef0)
    else:
        raise ValueError("kernel must be 'rbf' or 'poly'")

    m = X.shape[0]
    n = Y.shape[0]
    mmd = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1)) + \
          (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1)) - \
          2 * K_xy.mean()
    return mmd

def _kid(X, Y, degree=3, coef0=1, subsample=100):
    return _mmd(X, Y, kernel='poly', degree=degree, coef0=coef0, subsample=subsample)

# ============= 评估函数（保持不变） =============
def plot_training_curves(history, save_dir=None):
    steps = history["step"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, history["d_loss"], label="D_loss", color='blue')
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Discriminator Loss")
    axes[0].grid(True)
    axes[1].plot(steps, history["g_loss"], label="G_loss", color='red')
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Generator Loss")
    axes[1].grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "cgan_training_curves.png"), dpi=150)
    plt.show()

def plot_waveform_comparison(real_signals, real_labels, fake_signals, fake_labels,
                             class_names, num_samples=3, save_dir=None, tag=""):
    unique_cls = np.unique(real_labels)
    fig, axes = plt.subplots(len(unique_cls), num_samples, figsize=(num_samples * 4, len(unique_cls) * 2))
    if len(unique_cls) == 1:
        axes = axes[np.newaxis, :]
    for i, cls in enumerate(unique_cls):
        real_idx = np.where(real_labels == cls)[0]
        fake_idx = np.where(fake_labels == cls)[0]
        real_sel = np.random.choice(real_idx, min(len(real_idx), num_samples), replace=False)
        fake_sel = np.random.choice(fake_idx, min(len(fake_idx), num_samples), replace=False)
        for j in range(num_samples):
            ax = axes[i, j]
            if j < len(real_sel):
                ax.plot(real_signals[real_sel[j]], color='blue', alpha=0.7, label='real' if j == 0 else "")
            if j < len(fake_sel):
                ax.plot(fake_signals[fake_sel[j]], color='red', alpha=0.7, label='fake' if j == 0 else "")
            ax.set_title(f"Class {class_names[cls]} sample {j}")
            if i == 0 and j == 0:
                ax.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"cgan_waveform_comparison_{tag}.png"), dpi=150)
    plt.show()

def evaluate_generation_simple(
    real_signals, real_labels, fake_signals, fake_labels,
    class_names,
    tag="cgan",
    save_dir=None,
    selected_features: Optional[List[int]] = None,
    compute_mmd: bool = False,
    mmd_kernel: str = 'rbf',
    compute_kid: bool = False,
    kid_subsample: int = 100,
    mmd_subsample: Optional[int] = None
):
    print(f"\n========== 评估生成质量 ({tag}) ==========")
    real_scaled = minmax_scale_np(real_signals)

    def extract_feats_batch(X):
        feats = []
        for sig in X:
            sig = np.asarray(sig, dtype=np.float64)
            feats.append(extract_all_features(sig, fs=12000))
        return np.asarray(feats, dtype=np.float32)

    real_feats = extract_feats_batch(real_scaled)
    fake_feats = extract_feats_batch(fake_signals)

    metrics = {}

    mu_r = np.mean(real_feats, axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    mu_g = np.mean(fake_feats, axis=0)
    sigma_g = np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_full = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    metrics['FID_full'] = float(fid_full)
    print(f"FID (31D feature space): {fid_full:.4f}")

    if selected_features is not None:
        real_feats_sel = real_feats[:, selected_features]
        fake_feats_sel = fake_feats[:, selected_features]
        mu_r_sel = np.mean(real_feats_sel, axis=0)
        sigma_r_sel = np.cov(real_feats_sel, rowvar=False)
        mu_g_sel = np.mean(fake_feats_sel, axis=0)
        sigma_g_sel = np.cov(fake_feats_sel, rowvar=False)
        diff_sel = mu_r_sel - mu_g_sel
        covmean_sel = sqrtm(sigma_r_sel @ sigma_g_sel)
        if np.iscomplexobj(covmean_sel):
            covmean_sel = covmean_sel.real
        fid_sel = diff_sel @ diff_sel + np.trace(sigma_r_sel + sigma_g_sel - 2 * covmean_sel)
        metrics['FID_selected'] = float(fid_sel)
        print(f"FID (selected {len(selected_features)} features): {fid_sel:.4f}")

    if compute_mmd:
        mmd_val = _mmd(real_feats, fake_feats, kernel=mmd_kernel, subsample=mmd_subsample)
        metrics['MMD'] = float(mmd_val)
        print(f"MMD ({mmd_kernel} kernel): {mmd_val:.6f}")

    if compute_kid:
        kid_val = _kid(real_feats, fake_feats, subsample=kid_subsample)
        metrics['KID'] = float(kid_val)
        print(f"KID (poly kernel, subsample={kid_subsample}): {kid_val:.6f}")

    X_all = np.vstack([real_feats, fake_feats])
    domain_labels = np.concatenate([np.zeros(len(real_feats)), np.ones(len(fake_feats))])
    class_labels = np.concatenate([real_labels, fake_labels])
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_all)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                init="random", random_state=42)
    X_emb = tsne.fit_transform(X_std)
    plt.figure(figsize=(10, 8))
    unique_cls = np.unique(class_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cls)))
    for i, c in enumerate(unique_cls):
        mask_real = (class_labels == c) & (domain_labels == 0)
        mask_fake = (class_labels == c) & (domain_labels == 1)
        plt.scatter(X_emb[mask_real, 0], X_emb[mask_real, 1],
                    s=30, c=colors[i].reshape(1, -1), marker='o',
                    label=f"{class_names[c]} (real)")
        plt.scatter(X_emb[mask_fake, 0], X_emb[mask_fake, 1],
                    s=30, facecolors='none', edgecolors=colors[i], marker='o',
                    label=f"{class_names[c]} (fake)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE: Real vs Generated ({tag})")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"cgan_tsne_{tag}.png"), dpi=150)
    plt.show()
    plot_waveform_comparison(real_signals, real_labels, fake_signals, fake_labels,
                             class_names, num_samples=3, save_dir=save_dir, tag=tag)
    return metrics

# ============= 封装函数（修改 do_minmax 参数） =============
def train_cgan_and_generate(
    X_source,
    y_source,
    class_names,
    cgan_params: dict,
    num_per_class=50,
    save_path="generated_samples_cgan.npz"
):
    trainer = cGAN_Trainer(
        X_signals=X_source,
        y=y_source,
        class_names=class_names,
        batch_size=cgan_params.get('batch_size', 64),
        z_dim=cgan_params.get('z_dim', 128),
        lr_g=cgan_params.get('lr_g', 2e-4),
        lr_d=cgan_params.get('lr_d', 1e-4),
        n_critic=cgan_params.get('n_critic', 1),
        log_dir=cgan_params.get('log_dir', None),
        use_tensorboard=cgan_params.get('use_tensorboard', False),
        do_minmax=False,          # ADDED: 数据已全局归一化，不再内部缩放
    )
    trainer.fit(epochs=cgan_params.get('epochs', 50), log_every=50)
    unique_classes = np.unique(y_source)
    Xg = trainer.synthesize(y=unique_classes, num_per_class=num_per_class)
    yg = np.repeat(unique_classes, num_per_class)
    np.savez(save_path, X=Xg, y=yg, class_names=np.array(class_names))
    print(f"生成样本已保存至 {save_path}")
    return Xg, yg, trainer.history

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    source_data = np.load("source_data.npz")
    X_source = source_data["X"]
    y_source = source_data["y"]
    full_class_names = source_data["class_names"].tolist()
    unique_classes = np.unique(y_source)
    class_names_sub = [full_class_names[i] for i in unique_classes]
    print(f"源域实际类别: {unique_classes}")
    print(f"源域样本数: {X_source.shape}, 类别分布: {np.unique(y_source, return_counts=True)}")
    print(f"使用的类别名称: {class_names_sub}")

    trainer = cGAN_Trainer(
        X_signals=X_source,
        y=y_source,
        class_names=class_names_sub,
        lr_g=2e-4,
        lr_d=1e-4,
        n_critic=2,
        log_dir="./runs/cgan",
        use_tensorboard=True,
        do_minmax=False,          # MODIFIED
    )
    trainer.fit(epochs=50, log_every=50)
    Xg = trainer.synthesize(y=np.arange(len(class_names_sub)), num_per_class=50)
    yg = np.repeat(np.arange(len(class_names_sub)), 50)
    print("Synthesized shape:", Xg.shape)
    np.savez("generated_samples_cgan.npz", X=Xg, y=yg, class_names=np.array(class_names_sub))
    print("生成样本已保存至 generated_samples_cgan.npz")

    RUN_EVAL = True
    if RUN_EVAL:
        print("\n=== 开始评估生成质量 ===")
        num_per_class_eval = 50
        real_subset_signals = []
        real_subset_labels = []
        for cls in range(len(class_names_sub)):
            idx = np.where(y_source == cls)[0]
            if len(idx) >= num_per_class_eval:
                sel = np.random.choice(idx, num_per_class_eval, replace=False)
            else:
                sel = idx
            real_subset_signals.append(X_source[sel])
            real_subset_labels.append(np.full(len(sel), cls))
        real_subset_signals = np.concatenate(real_subset_signals, axis=0)
        real_subset_labels = np.concatenate(real_subset_labels, axis=0)
        selected_indices = [4, 6, 15]
        metrics = evaluate_generation_simple(
            real_signals=real_subset_signals,
            real_labels=real_subset_labels,
            fake_signals=Xg,
            fake_labels=yg,
            class_names=class_names_sub,
            tag="cgan_eval",
            save_dir="./runs/cgan",
            selected_features=selected_indices,
            compute_mmd=True,
            mmd_kernel='rbf',
            compute_kid=True,
            kid_subsample=100,
            mmd_subsample=None
        )
        print("评估指标:", metrics)
        with open(os.path.join("./runs/cgan", "generation_metrics_cgan.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        plot_training_curves(trainer.history, save_dir="./runs/cgan")