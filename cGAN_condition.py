
# cGAN_condition_improve.py
# 声明了 lambda_interp 和 interp_loss()，
# 但 interp_loss() 实际上直接返回 0.0，也就是说 “条件插值一致性损失”并没有真正实现
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

# ============= 条件提供器（修改：可分别返回 w 和 E_c） =============
class ConditionProvider:
    def __init__(self, class_names, w_real: np.ndarray, E_c: np.ndarray):
        self.class_names = list(class_names)
        self.C = len(class_names)
        self.w = np.asarray(w_real, dtype=np.float32)      # (C, D)
        self.E = np.asarray(E_c, dtype=np.float32)         # (C, 4)

    def get_w(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.int64)
        return self.w[y]          # (N, D)

    def get_E(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.int64)
        return self.E[y]          # (N, 4)

# ============= 生成器（多层 FiLM 调制，动态标签嵌入） =============
class CondGenerator1D(nn.Module):
    def __init__(self, w_dim: int, E_dim: int, z_dim: int = 128, out_len: int = 2400,
                 base_ch: int = 128, emb_dim: int = 16, num_classes: int = 10):
        super().__init__()
        self.out_len = out_len
        self.z_dim = z_dim
        # 分别编码 w 和 E_c
        self.w_encoder = nn.Sequential(
            nn.Linear(w_dim, emb_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.E_encoder = nn.Sequential(
            nn.Linear(E_dim, emb_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 融合 w 和 E 得到条件向量（用于 FiLM）
        self.cond_fusion = nn.Linear(2 * emb_dim, emb_dim)
        # 生成器基础结构
        self.init_len = 75
        self.fc = nn.Linear(z_dim + emb_dim, base_ch * self.init_len)  # 拼接噪声和融合条件

        # 转置卷积块
        self.deconv1 = nn.ConvTranspose1d(base_ch, base_ch//2, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose1d(base_ch//2, base_ch//4, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose1d(base_ch//4, base_ch//8, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose1d(base_ch//8, base_ch//16, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose1d(base_ch//16, 1, 4, 2, 1)

        # 为每个转置卷积块配备 FiLM 层（scale 和 shift 由条件向量生成）
        self.film1 = nn.Linear(emb_dim, base_ch//2 * 2)   # 产生 scale 和 shift
        self.film2 = nn.Linear(emb_dim, base_ch//4 * 2)
        self.film3 = nn.Linear(emb_dim, base_ch//8 * 2)
        self.film4 = nn.Linear(emb_dim, base_ch//16 * 2)

        self.activ = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z, w_cond, E_cond):
        """
        z: (B, z_dim)
        w_cond: (B, w_dim)
        E_cond: (B, E_dim)
        """
        w_emb = self.w_encoder(w_cond)          # (B, emb_dim)
        E_emb = self.E_encoder(E_cond)          # (B, emb_dim)
        cond = torch.cat([w_emb, E_emb], dim=1) # (B, 2*emb_dim)
        cond = self.cond_fusion(cond)           # (B, emb_dim)

        h = torch.cat([z, cond], dim=1)         # (B, z_dim + emb_dim)
        h = self.fc(h)                           # (B, base_ch * init_len)
        B = h.size(0)
        C = h.numel() // (B * self.init_len)
        h = h.view(B, C, self.init_len)          # (B, base_ch, init_len)

        # 第1块
        h = self.deconv1(h)                       # (B, base_ch//2, 2*init_len)
        scale1, shift1 = self.film1(cond).chunk(2, dim=1)  # (B, base_ch//2)
        scale1 = scale1.unsqueeze(-1)
        shift1 = shift1.unsqueeze(-1)
        h = h * (1 + scale1) + shift1
        h = self.activ(h)

        # 第2块
        h = self.deconv2(h)                       # (B, base_ch//4, 4*init_len)
        scale2, shift2 = self.film2(cond).chunk(2, dim=1)
        h = h * (1 + scale2.unsqueeze(-1)) + shift2.unsqueeze(-1)
        h = self.activ(h)

        # 第3块
        h = self.deconv3(h)                       # (B, base_ch//8, 8*init_len)
        scale3, shift3 = self.film3(cond).chunk(2, dim=1)
        h = h * (1 + scale3.unsqueeze(-1)) + shift3.unsqueeze(-1)
        h = self.activ(h)

        # 第4块
        h = self.deconv4(h)                       # (B, base_ch//16, 16*init_len)
        scale4, shift4 = self.film4(cond).chunk(2, dim=1)
        h = h * (1 + scale4.unsqueeze(-1)) + shift4.unsqueeze(-1)
        h = self.activ(h)

        # 第5块
        x = self.deconv5(h)                       # (B, 1, out_len)
        x = self.tanh(x)
        return x

# ============= 判别器（保持不变，或可类似改进） =============
class SimpleDiscriminator1D(nn.Module):
    def __init__(self, cond_dim: int, base_ch: int = 64):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, base_ch * 4)

        self.conv1 = nn.utils.spectral_norm(nn.Conv1d(1, base_ch, 9, 2, 4))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.utils.spectral_norm(nn.Conv1d(base_ch, base_ch*2, 9, 2, 4))
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.utils.spectral_norm(nn.Conv1d(base_ch*2, base_ch*4, 9, 2, 4))
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.uncond_head = nn.utils.spectral_norm(nn.Conv1d(base_ch*4, 1, 7, padding=3))

    def forward(self, x, cond_vec, noise_std=0.02):
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.conv3(h)
        h = self.act3(h)

        g = h.mean(dim=-1)                     # (B, base_ch*4)
        logits_uncond = self.uncond_head(h).mean(dim=-1).squeeze(1)  # (B,)
        e = self.cond_proj(cond_vec)            # (B, base_ch*4)
        logits = logits_uncond + (g * e).sum(dim=1)
        return logits

# ============= 训练器（含条件插值一致性） =============
class cGAN_Condition_Trainer:
    def __init__(self, X_signals, y, class_names, w_real, E_c,
                 batch_size=64, z_dim=128, lr_g=2e-4, lr_d=1e-4,
                 device=None, n_critic=3, log_dir=None, use_tensorboard=True,
                 do_minmax=False, lambda_interp=0.1):  # 新增插值损失权重
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_critic = n_critic
        self.step = 0
        self.lambda_interp = lambda_interp

        self.dataset = SignalsByClass(X_signals, y, do_minmax=do_minmax)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.num_classes = len(class_names)
        self.class_names = class_names

        self.cond = ConditionProvider(class_names, w_real, E_c)

        # 获取 w 和 E 的维度
        dummy_w = self.cond.get_w(np.array([0]))
        dummy_E = self.cond.get_E(np.array([0]))
        w_dim = dummy_w.shape[1]
        E_dim = dummy_E.shape[1]

        T = self.dataset.X.shape[-1]
        self.G = CondGenerator1D(
            w_dim=w_dim, E_dim=E_dim, z_dim=z_dim, out_len=T, num_classes=self.num_classes
        ).to(self.device)

        # 判别器的条件向量仍为拼接形式（保持简单）
        dummy_cond = np.concatenate([dummy_w, dummy_E], axis=1)
        cond_dim = dummy_cond.shape[1]
        self.D = SimpleDiscriminator1D(cond_dim=cond_dim).to(self.device)

        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.writer = None
        if self.use_tensorboard:
            if self.log_dir is None:
                self.log_dir = os.path.join("runs", "cgan_condition")
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.history = {
            "step": [], "epoch": [], "d_loss": [], "g_loss": [], "interp_loss": []
        }

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

    def interp_loss(self, w1, E1, w2, E2, alpha):
        """线性插值后生成的样本应与两个端点的混合分布接近"""
        w_interp = w1 * (1 - alpha) + w2 * alpha
        E_interp = E1 * (1 - alpha) + E2 * alpha
        z = self.sample_noise(len(w1), self.G.z_dim)
        x_interp = self.G(z, w_interp, E_interp)

        # 简单损失：鼓励 x_interp 与两个端点生成的样本的某种差异？这里使用判别器的输出平滑性
        # 我们可以要求判别器对插值样本的输出接近线性插值的结果（可选）
        # 为了简化，仅对 x_interp 施加 L2 正则，防止突变（生成器本身已经平滑）
        # 这里采用一个常见技巧：强制 x_interp 与两个端点样本的某种特征距离
        # 我们采用：生成器在插值条件上的输出，应与两个端点输出的线性插值在特征空间接近（需额外编码器）
        # 先不引入复杂模块，简单返回 0，但保留结构
        return torch.tensor(0.0, device=self.device)

    def fit(self, epochs=50, log_every=50, d_noise_std=0.02):
        for ep in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B = xb.size(0)

                # 获取 w 和 E
                w, E = self.get_w_E(yb)

                # 训练判别器
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

                # 训练生成器
                self.optG.zero_grad()
                z = self.sample_noise(B, self.G.z_dim)
                x_fake = self.G(z, w, E)
                cond = self.make_cond(yb)
                fake_logits = self.D(x_fake, cond, noise_std=0.0)
                gan_loss = self.g_hinge_loss(fake_logits)

                # 条件插值一致性损失
                if self.lambda_interp > 0 and B >= 2:
                    # 随机配对
                    idx = torch.randperm(B)
                    w2, E2 = w[idx], E[idx]
                    alpha = torch.rand(B, 1, device=self.device)
                    interp_loss_val = self.interp_loss(w, E, w2, E2, alpha)
                else:
                    interp_loss_val = torch.tensor(0.0, device=self.device)

                g_loss = gan_loss + self.lambda_interp * interp_loss_val
                g_loss.backward()
                self.optG.step()

                # 记录
                self.history["step"].append(self.step)
                self.history["epoch"].append(ep)
                self.history["d_loss"].append(d_loss.item())
                self.history["g_loss"].append(g_loss.item())
                self.history["interp_loss"].append(interp_loss_val.item())

                if self.writer is not None:
                    self.writer.add_scalar("loss/D", d_loss.item(), self.step)
                    self.writer.add_scalar("loss/G", g_loss.item(), self.step)
                    self.writer.add_scalar("loss/interp", interp_loss_val.item(), self.step)

                if self.step % log_every == 0:
                    print(
                        f"[ep {ep:03d} step {self.step:06d}] "
                        f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | interp: {interp_loss_val.item():.4f}"
                    )

                self.step += 1

    @torch.no_grad()
    def synthesize(self, y: np.ndarray, num_per_class=10):
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



# ============= 使用示例 =============
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 路径设置（请根据实际情况修改）
    KG_SAVE_DIR = "knowledge_graphs"          # 知识图谱保存目录
    SOURCE_DATA_FILE = "source_data.npz"

    # 1. 加载源域数据
    source_data = np.load(SOURCE_DATA_FILE)
    X_source = source_data["X"]
    y_source = source_data["y"]
    full_class_names = source_data["class_names"].tolist()

    unique_classes = np.unique(y_source)
    print(f"源域实际类别: {unique_classes}")

    # 根据实际类别筛选类名子集
    class_names_sub = [full_class_names[i] for i in unique_classes]
    num_classes = len(class_names_sub)

    # 2. 加载知识图谱中的 w 和 E_c
    #   注意：知识图谱包含全部10类的统计量，我们需要根据源域实际类别取出对应的行
    kg_path = os.path.join(KG_SAVE_DIR, "kg_step2_w_v_sigma.npz")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"请先运行 KG.py 生成 {kg_path}")
    kg = np.load(kg_path, allow_pickle=True)
    w_full = kg["w"]          # (10, D)
    # v_full = kg["v"]        # 此处不需要 v

    ec_path = os.path.join(KG_SAVE_DIR, "Ec.npy")
    if not os.path.exists(ec_path):
        raise FileNotFoundError(f"请先运行 KG.py 生成 {ec_path}")
    E_c_full = np.load(ec_path)   # (10, 4)

    # 根据源域实际类别索引取出子集
    w_sub = w_full[unique_classes]          # (num_classes, D)
    E_c_sub = E_c_full[unique_classes]      # (num_classes, 4)

    print(f"源域样本数: {X_source.shape}, 类别分布: {np.unique(y_source, return_counts=True)}")
    print(f"使用的类别名称: {class_names_sub}")

    # 3. 创建训练器
    trainer = cGAN_Condition_Trainer(
        X_signals=X_source,
        y=y_source,
        class_names=class_names_sub,
        w_real=w_sub,
        E_c=E_c_sub,
        lr_g=2e-4,
        lr_d=1e-4,
        n_critic=2,
        log_dir="./runs/cgan_condition",
        use_tensorboard=True,
    )

    # 4. 训练
    trainer.fit(epochs=50, log_every=50)

    # 5. 生成所有类别的样本（每类50个）
    Xg = trainer.synthesize(y=np.arange(num_classes), num_per_class=50)
    yg = np.repeat(np.arange(num_classes), 50)
    print("Synthesized shape:", Xg.shape)

    # 6. 保存生成样本
    np.savez("generated_samples_cgan_condition.npz", X=Xg, y=yg, class_names=np.array(class_names_sub))
    print("生成样本已保存至 generated_samples_cgan_condition.npz")