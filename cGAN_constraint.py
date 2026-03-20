# cGAN_constraint.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

# ============= 辅助函数 =============
def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# ============= 数据集包装 =============
class SignalsByClass(Dataset):
    # def __init__(self, X_signals: np.ndarray, y: np.ndarray, do_minmax=True):
    #     X = np.asarray(X_signals, dtype=np.float32)
    #     if do_minmax:
    #         # 缩放到 [-1, 1] 以匹配生成器输出
    #         X = (X - X.min(axis=-1, keepdims=True)) / (X.max(axis=-1, keepdims=True) - X.min(axis=-1, keepdims=True) + 1e-8) * 2 - 1
    #     if X.ndim == 2:
    #         X = X[:, None, :]  # (B, 1, T)
    #     self.X = X
    #     self.y = np.asarray(y, dtype=np.int64)
    def __init__(self, X_signals: np.ndarray, y: np.ndarray, do_minmax=False):
        X = np.asarray(X_signals, dtype=np.float32)
        if do_minmax:
            X = (X - X.min(axis=-1, keepdims=True)) / (
                        X.max(axis=-1, keepdims=True) - X.min(axis=-1, keepdims=True) + 1e-8) * 2 - 1
        if X.ndim == 2:
            X = X[:, None, :]
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

# ============= 生成器（仅使用标签嵌入，同标准cGAN） =============
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

# ============= 判别器（仅使用标签嵌入，同标准cGAN） =============
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

# ============= 可微特征提取器（用于物理损失） =============
class DifferentiableFeatures(nn.Module):
    def __init__(self, sample_len: int = 2400, d_feat: int = 31):
        super().__init__()
        self.sample_len = sample_len
        self.d_feat = d_feat

    def forward(self, x):
        X = torch.fft.rfft(x.squeeze(1), dim=-1)
        mag = torch.abs(X)
        B, F = mag.shape
        if F < self.d_feat:
            pad = self.d_feat - F
            mag = F.pad(mag, (0, pad))
            F = mag.shape[1]
        idx = torch.linspace(0, F-1, self.d_feat, device=mag.device)
        idx0 = idx.floor().long().clamp(0, F-2)
        alpha = (idx - idx0.float()).unsqueeze(0)
        feats = mag[:, idx0] * (1-alpha) + mag[:, idx0+1] * alpha
        return feats

# ============= 频带能量提取器 =============
class FourBandEnergy(nn.Module):
    def __init__(self, T: int, fs: int = 12000, nfft: int = 512,
                 win_len: int = 256, hop: int = 128,
                 bands=None, learnable_bands: bool = False):
        super().__init__()
        self.fs = fs
        self.nfft = nfft
        self.hop = hop
        self.win_len = win_len
        self.register_buffer("window", torch.hann_window(win_len))
        if bands is None:
            bands = [(0,600), (600,1800), (1800,3600), (3600,6000)]
        self.learnable_bands = learnable_bands
        if learnable_bands:
            init = torch.tensor([b[1] for b in bands], dtype=torch.float32)
            self.b_raw = nn.Parameter(torch.log(torch.exp(init/(fs/2-1e-3))-1.0))
        else:
            self.bands = bands

    def _get_edges(self, device):
        if not self.learnable_bands:
            edges = torch.tensor([b[1] for b in self.bands], dtype=torch.float32, device=device)
        else:
            sp = F.softplus(self.b_raw)
            cum = torch.cumsum(sp, dim=0)
            edges = (self.fs/2) * (cum/(cum[-1]+1e-8))
        e0 = torch.tensor([0.0], device=device)
        edges = torch.cat([e0, edges], dim=0)
        return edges

    def forward(self, x):
        X = torch.stft(
            x.squeeze(1),
            n_fft=self.nfft,
            hop_length=self.hop,
            win_length=self.win_len,
            window=self.window.to(x.device),
            return_complex=True,
            center=True,
            pad_mode="reflect"
        )
        mag2 = X.real**2 + X.imag**2
        Fbins = mag2.size(1)
        freqs = torch.linspace(0, self.fs/2, Fbins, device=x.device)

        edges = self._get_edges(x.device)
        Es = []
        for k in range(4):
            f0, f1 = edges[k], edges[k+1]
            mask = (freqs >= f0) & (freqs < f1)
            if mask.sum() == 0:
                band_E = mag2.new_zeros((x.size(0),))
            else:
                band_E = mag2[:, mask, :].sum(dim=(1,2))
            Es.append(band_E)
        E = torch.stack(Es, dim=1)
        E_ratio = E / (E.sum(dim=1, keepdim=True) + 1e-8)
        return E_ratio

# ============= 物理约束损失 =============
class PhysicsConstraintLoss(nn.Module):
    def __init__(self, v_real: np.ndarray, w_real: np.ndarray, E_c: np.ndarray,
                 feat_func: nn.Module, band_energy: nn.Module,
                 eps=1e-8, alpha_E=1.0, ratio_metric: str = "l1",
                 ema_momentum: float = 0.1):
        super().__init__()
        v_real = np.asarray(v_real, dtype=np.float32)
        w_real = np.asarray(w_real, dtype=np.float32)
        E_c = np.asarray(E_c, dtype=np.float32)
        C, D = v_real.shape
        assert w_real.shape == (C, D)
        self.C, self.D = C, D
        self.register_buffer("v", torch.tensor(v_real))
        self.register_buffer("w_target", torch.tensor(w_real))
        self.register_buffer("E_target", torch.tensor(E_c))
        self.phi = feat_func
        self.band_energy = band_energy
        self.eps = eps
        self.alpha_E = alpha_E
        self.ratio_metric = ratio_metric
        self.m = ema_momentum

        self.register_buffer("ema_var", torch.ones(C, D))
        self.register_buffer("ema_ready", torch.zeros(C))

    def _ratio_distance(self, p, q):
        if self.ratio_metric == "l1":
            return F.l1_loss(p, q)
        elif self.ratio_metric == "kl":
            p = p + 1e-8
            q = q + 1e-8
            return 0.5 * ((p * (p/q).log()).sum(dim=1).mean() + (q * (q/p).log()).sum(dim=1).mean())
        elif self.ratio_metric == "logcosh":
            return torch.log(torch.cosh(p - q)).mean()
        else:
            return F.l1_loss(p, q)

    def forward(self, x_gen: torch.Tensor, y: torch.Tensor):
        feats = self.phi(x_gen)
        if feats.shape[1] != self.D:
            if feats.shape[1] > self.D:
                feats = feats[:, :self.D]
            else:
                feats = F.pad(feats, (0, self.D - feats.shape[1]))

        loss_w = 0.0
        classes = y.unique()
        for i in classes:
            i = int(i.item())
            mask = y == i
            Fi = feats[mask]
            if Fi.numel() == 0:
                continue
            vi = self.v[i]
            cur_var = ((Fi - vi.unsqueeze(0))**2).mean(dim=0)
            if self.ema_ready[i] < 0.5:
                self.ema_var[i] = cur_var.detach()
                self.ema_ready[i] = 1.0
            else:
                self.ema_var[i] = (1 - self.m) * self.ema_var[i] + self.m * cur_var.detach()
            invsqrt = torch.rsqrt(self.ema_var[i] + self.eps)
            wi_gen = invsqrt / (invsqrt.sum() + self.eps)
            wi_tgt = self.w_target[i]
            loss_w = loss_w + F.mse_loss(wi_gen, wi_tgt)
        loss_w = loss_w / max(len(classes), 1)

        Ec_gen = self.band_energy(x_gen)
        Ec_tgt = self.E_target[y]
        loss_E = self._ratio_distance(Ec_gen, Ec_tgt)

        return loss_w + self.alpha_E * loss_E, loss_w.detach(), loss_E.detach()

# ============= 训练器（带物理约束） =============
class cGAN_Constraint_Trainer:
    # def __init__(
    #     self,
    #     X_signals,
    #     y,
    #     class_names,
    #     v_real,
    #     w_real,
    #     E_c,
    #     batch_size=64,
    #     z_dim=128,
    #     lr_g=2e-4,
    #     lr_d=1e-4,
    #     lambda_phys=0.75,
    #     alpha_E=1.0,
    #     device=None,
    #     fs=12000,
    #     n_critic=3,
    #     ratio_metric="l1",
    #     ema_momentum=0.1,
    #     lambda_warmup_steps: int = 1000,
    #     log_dir: Optional[str] = None,
    #     use_tensorboard: bool = True,
    # ):
    def __init__(self, X_signals, y, class_names, v_real, w_real, E_c,
                 batch_size=64, z_dim=128, lr_g=2e-4, lr_d=1e-4,
                 lambda_phys=0.75, alpha_E=1.0, device=None, fs=12000,
                 n_critic=3, ratio_metric="l1", ema_momentum=0.1,
                 lambda_warmup_steps=1000, log_dir=None, use_tensorboard=True,
                 do_minmax=False):  # ADDED
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_phys_base = float(lambda_phys)
        self.lambda_phys = 0.0
        self.lambda_warmup_steps = int(lambda_warmup_steps)
        self.n_critic = n_critic
        self.step = 0

        # self.dataset = SignalsByClass(X_signals, y, do_minmax=True)
        self.dataset = SignalsByClass(X_signals, y, do_minmax=do_minmax)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.num_classes = len(class_names)
        self.class_names = class_names

        T = self.dataset.X.shape[-1]
        self.G = CondGenerator1D(z_dim=z_dim, out_len=T, num_classes=self.num_classes).to(self.device)
        self.D = SimpleDiscriminator1D(num_classes=self.num_classes).to(self.device)

        # 可微特征提取与物理损失
        self.phi = DifferentiableFeatures(sample_len=T, d_feat=w_real.shape[1]).to(self.device)
        self.band_energy = FourBandEnergy(T=T, fs=fs, learnable_bands=False).to(self.device)
        self.phys_loss = PhysicsConstraintLoss(
            v_real=v_real, w_real=w_real, E_c=E_c,
            feat_func=self.phi, band_energy=self.band_energy,
            alpha_E=alpha_E, ratio_metric=ratio_metric, ema_momentum=ema_momentum
        ).to(self.device)

        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.writer = None
        if self.use_tensorboard:
            if self.log_dir is None:
                self.log_dir = os.path.join("runs", "cgan_constraint")
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.history = {
            "step": [], "epoch": [], "d_loss": [], "g_loss": [], "gan_loss": [],
            "phy_loss": [], "loss_w": [], "loss_E": [], "lambda_phys": []
        }

    def _lambda_update(self):
        if self.lambda_warmup_steps <= 0:
            self.lambda_phys = self.lambda_phys_base
        else:
            t = min(self.step / self.lambda_warmup_steps, 1.0)
            self.lambda_phys = (0.1 + 0.9 * t) * self.lambda_phys_base

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
        y = np.asarray(y, dtype=np.int64)
        all_out = []
        for yi in y:
            z = self.sample_noise(num_per_class, self.G.z_dim)
            y_tensor = torch.full((num_per_class,), yi, dtype=torch.long, device=self.device)
            xg = self.G(z, y_tensor)
            all_out.append(xg.squeeze(1).cpu().numpy())
        return np.concatenate(all_out, axis=0)

# ============= 使用示例 =============
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    KG_SAVE_DIR = "knowledge_graphs"
    SOURCE_DATA_FILE = "source_data.npz"

    # 1. 加载源域数据
    source_data = np.load(SOURCE_DATA_FILE)
    X_source = source_data["X"]
    y_source = source_data["y"]
    full_class_names = source_data["class_names"].tolist()

    unique_classes = np.unique(y_source)
    print(f"源域实际类别: {unique_classes}")

    class_names_sub = [full_class_names[i] for i in unique_classes]
    num_classes = len(class_names_sub)

    # 2. 加载知识图谱中的 v, w, E_c
    kg_path = os.path.join(KG_SAVE_DIR, "kg_step2_w_v_sigma.npz")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"请先运行 KG.py 生成 {kg_path}")
    kg = np.load(kg_path, allow_pickle=True)
    v_full = kg["v"]          # (10, D)
    w_full = kg["w"]          # (10, D)

    ec_path = os.path.join(KG_SAVE_DIR, "Ec.npy")
    if not os.path.exists(ec_path):
        raise FileNotFoundError(f"请先运行 KG.py 生成 {ec_path}")
    E_c_full = np.load(ec_path)   # (10, 4)

    # 根据源域实际类别取出子集
    v_sub = v_full[unique_classes]
    w_sub = w_full[unique_classes]
    E_c_sub = E_c_full[unique_classes]

    print(f"源域样本数: {X_source.shape}, 类别分布: {np.unique(y_source, return_counts=True)}")
    print(f"使用的类别名称: {class_names_sub}")

    # 3. 创建训练器
    trainer = cGAN_Constraint_Trainer(
        X_signals=X_source,
        y=y_source,
        class_names=class_names_sub,
        v_real=v_sub,
        w_real=w_sub,
        E_c=E_c_sub,
        lr_g=2e-4,
        lr_d=1e-4,
        lambda_phys=0.75,
        alpha_E=1.0,
        n_critic=2,
        lambda_warmup_steps=1000,
        log_dir="./runs/cgan_constraint",
        use_tensorboard=True,
    )

    # 4. 训练
    trainer.fit(epochs=50, log_every=50)

    # 5. 生成所有类别的样本（每类50个）
    Xg = trainer.synthesize(y=np.arange(num_classes), num_per_class=50)
    yg = np.repeat(np.arange(num_classes), 50)
    print("Synthesized shape:", Xg.shape)

    # 6. 保存生成样本
    np.savez("generated_samples_cgan_constraint.npz", X=Xg, y=yg, class_names=np.array(class_names_sub))
    print("生成样本已保存至 generated_samples_cgan_constraint.npz")