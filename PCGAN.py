"""
PcGAN: Physics-constrained Conditional GAN
融合 cGAN_condition（条件嵌入 + FiLM）与 cGAN_constraint（物理损失）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

# 从 cGAN_condition 导入共用组件
from cGAN_condition import ConditionProvider, SignalsByClass, to_tensor
from cGAN_condition import CondGenerator1D as CondGenerator1D_film
from cGAN_condition import SimpleDiscriminator1D

# 从 cGAN_constraint 导入物理损失相关组件
from cGAN_constraint import PhysicsConstraintLoss, DifferentiableFeatures, FourBandEnergy


class PcGAN_Trainer:
    """
    训练器：结合条件嵌入和物理约束
    - 生成器：接收 w 和 E 条件，使用 FiLM 调制
    - 判别器：接收拼接的 [w, E] 条件
    - 物理损失：特征权重损失 + 频带能量比例损失
    """
    def __init__(self, X_signals, y, class_names, w_real, E_c, v_real,
                 batch_size=64, z_dim=128, lr_g=2e-4, lr_d=1e-4,
                 lambda_phys=0.75, alpha_E=1.0, device=None, fs=12000,
                 n_critic=3, ratio_metric="l1", ema_momentum=0.1,
                 lambda_warmup_steps=1000, log_dir=None, use_tensorboard=True,
                 do_minmax=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_critic = n_critic
        self.step = 0
        self.lambda_phys_base = float(lambda_phys)
        self.lambda_phys = 0.0
        self.lambda_warmup_steps = int(lambda_warmup_steps)

        # 数据集（输入信号已全局归一化）
        self.dataset = SignalsByClass(X_signals, y, do_minmax=do_minmax)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.num_classes = len(class_names)
        self.class_names = class_names

        # 条件提供器
        self.cond = ConditionProvider(class_names, w_real, E_c)

        # 获取 w 和 E 的维度
        dummy_w = self.cond.get_w(np.array([0]))
        dummy_E = self.cond.get_E(np.array([0]))
        w_dim = dummy_w.shape[1]
        E_dim = dummy_E.shape[1]

        T = self.dataset.X.shape[-1]
        # 生成器（带 FiLM）
        self.G = CondGenerator1D_film(
            w_dim=w_dim, E_dim=E_dim, z_dim=z_dim, out_len=T, num_classes=self.num_classes
        ).to(self.device)

        # 判别器条件维度
        dummy_cond = np.concatenate([dummy_w, dummy_E], axis=1)
        cond_dim = dummy_cond.shape[1]
        self.D = SimpleDiscriminator1D(cond_dim=cond_dim).to(self.device)

        # 物理损失模块
        self.phi = DifferentiableFeatures(sample_len=T, d_feat=w_dim).to(self.device)
        self.band_energy = FourBandEnergy(T=T, fs=fs, learnable_bands=False).to(self.device)
        self.phys_loss = PhysicsConstraintLoss(
            v_real=v_real, w_real=w_real, E_c=E_c,
            feat_func=self.phi, band_energy=self.band_energy,
            alpha_E=alpha_E, ratio_metric=ratio_metric, ema_momentum=ema_momentum
        ).to(self.device)

        # 优化器
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        # 日志
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.writer = None
        if self.use_tensorboard:
            if self.log_dir is None:
                self.log_dir = os.path.join("runs", "pcgan")
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.history = {
            "step": [], "epoch": [], "d_loss": [], "g_loss": [], "gan_loss": [],
            "phy_loss": [], "loss_w": [], "loss_E": [], "lambda_phys": []
        }

    def _lambda_update(self):
        """物理损失权重 warmup"""
        if self.lambda_warmup_steps <= 0:
            self.lambda_phys = self.lambda_phys_base
        else:
            t = min(self.step / self.lambda_warmup_steps, 1.0)
            self.lambda_phys = (0.1 + 0.9 * t) * self.lambda_phys_base

    def sample_noise(self, B, z_dim):
        return torch.randn(B, z_dim, device=self.device)

    def make_cond(self, y: torch.Tensor):
        """返回拼接后的条件向量（用于判别器）"""
        w_np = self.cond.get_w(y.cpu().numpy())
        E_np = self.cond.get_E(y.cpu().numpy())
        cond_np = np.concatenate([w_np, E_np], axis=1)
        return to_tensor(cond_np, self.device)

    def get_w_E(self, y):
        w_np = self.cond.get_w(y.cpu().numpy())
        E_np = self.cond.get_E(y.cpu().numpy())
        return to_tensor(w_np, self.device), to_tensor(E_np, self.device)

    def d_hinge_loss(self, real_logits, fake_logits):
        return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()

    def g_hinge_loss(self, fake_logits):
        return -fake_logits.mean()

    def fit(self, epochs=50, log_every=50, d_noise_std=0.02):
        for ep in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B = xb.size(0)

                # 获取当前 batch 的 w 和 E
                w, E = self.get_w_E(yb)

                # ---------- 训练判别器 ----------
                for _ in range(self.n_critic):
                    self.optD.zero_grad()
                    cond = self.make_cond(yb)
                    real_logits = self.D(xb, cond, noise_std=d_noise_std)

                    z = self.sample_noise(B, self.G.z_dim)
                    x_fake = self.G(z, w, E).detach()
                    fake_logits = self.D(x_fake, cond, noise_std=d_noise_std)

                    d_loss = self.d_hinge_loss(real_logits, fake_logits)
                    d_loss.backward()
                    self.optD.step()

                # ---------- 训练生成器 ----------
                self.optG.zero_grad()
                z = self.sample_noise(B, self.G.z_dim)
                x_fake = self.G(z, w, E)
                cond = self.make_cond(yb)
                fake_logits = self.D(x_fake, cond, noise_std=0.0)
                gan_loss = self.g_hinge_loss(fake_logits)

                # 物理损失
                phy, loss_w_det, loss_E_det = self.phys_loss(x_fake, yb)

                self._lambda_update()
                g_loss = gan_loss + self.lambda_phys * phy
                g_loss.backward()
                self.optG.step()

                # 记录
                self.history["step"].append(self.step)
                self.history["epoch"].append(ep)
                self.history["d_loss"].append(d_loss.item())
                self.history["g_loss"].append(g_loss.item())
                self.history["gan_loss"].append(gan_loss.item())
                self.history["phy_loss"].append(phy.item())
                self.history["loss_w"].append(loss_w_det.mean().item())
                self.history["loss_E"].append(loss_E_det.mean().item())
                self.history["lambda_phys"].append(self.lambda_phys)

                if self.writer is not None:
                    self.writer.add_scalar("loss/D", d_loss.item(), self.step)
                    self.writer.add_scalar("loss/G_total", g_loss.item(), self.step)
                    self.writer.add_scalar("loss/G_gan", gan_loss.item(), self.step)
                    self.writer.add_scalar("loss/L_phy", phy.item(), self.step)
                    self.writer.add_scalar("loss/L_w", loss_w_det.mean().item(), self.step)
                    self.writer.add_scalar("loss/L_E", loss_E_det.mean().item(), self.step)
                    self.writer.add_scalar("lambda/phys", self.lambda_phys, self.step)

                if self.step % log_every == 0:
                    print(
                        f"[ep {ep:03d} step {self.step:06d}] "
                        f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | GAN: {gan_loss.item():.4f} "
                        f"| λ_phy:{self.lambda_phys:.3f} | L_phy:{phy.item():.4f} (w:{loss_w_det.mean().item():.4f}, E:{loss_E_det.mean().item():.4f})"
                    )

                self.step += 1

    @torch.no_grad()
    def synthesize(self, y: np.ndarray, num_per_class=10):
        """生成指定类别的样本（y 为全局类别索引）"""
        y = np.asarray(y, dtype=np.int64)
        all_out = []
        for yi in y:
            w_np = self.cond.get_w(np.array([yi]))
            E_np = self.cond.get_E(np.array([yi]))
            w = to_tensor(w_np, self.device).repeat(num_per_class, 1)
            E = to_tensor(E_np, self.device).repeat(num_per_class, 1)
            z = self.sample_noise(num_per_class, self.G.z_dim)
            xg = self.G(z, w, E)
            all_out.append(xg.squeeze(1).cpu().numpy())
        return np.concatenate(all_out, axis=0)