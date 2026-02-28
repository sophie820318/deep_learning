"""
变分自编码器 VAE (Variational Autoencoder) - PyTorch
基于《写给新手的深度学习 2》 我妻幸长

知识点：
- 编码器（Encoder）：将输入压缩为潜在空间的均值 μ 和方差 σ
- 重参数化技巧（Reparameterization Trick）：z = μ + ε * σ，ε ~ N(0,1)
- 解码器（Decoder）：从潜在向量 z 重建输入
- ELBO 损失 = 重建损失 + KL 散度
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────
# VAE 模型
# ──────────────────────────────────────────────

class VAE(nn.Module):
    """
    变分自编码器
    输入维度：input_dim
    潜在维度：latent_dim
    """

    def __init__(self, input_dim=16, hidden_dim=32, latent_dim=4):
        super().__init__()

        # ── 编码器 ──────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)   # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # log 方差

        # ── 解码器 ──────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   # 输出范围 [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：
        z = μ + ε * exp(0.5 * log_var)，ε ~ N(0, 1)
        使梯度可以通过随机变量反向传播
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ──────────────────────────────────────────────
# ELBO 损失
# ──────────────────────────────────────────────

def vae_loss(x_recon, x, mu, logvar):
    """
    ELBO 损失 = 重建损失 + KL 散度

    重建损失：BCE（适合 [0,1] 范围的输入）
    KL 散度：-0.5 * sum(1 + log_var - μ² - σ²)
    """
    # 重建损失（二元交叉熵，逐元素求和后平均）
    recon_loss = nn.functional.binary_cross_entropy(
        x_recon, x, reduction="sum"
    ) / x.size(0)

    # KL 散度：让潜在分布接近标准正态 N(0,1)
    kl_div = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    return recon_loss + kl_div


# ──────────────────────────────────────────────
# 合成数据生成
# ──────────────────────────────────────────────

def make_binary_patterns(num_samples=500, input_dim=16, seed=42):
    """
    生成两种二值模式（模拟简单图像）：
    - 模式 A：前半段为 1，后半段为 0
    - 模式 B：前半段为 0，后半段为 1
    加入少量噪声以增加多样性
    """
    rng = np.random.default_rng(seed)
    data = []
    half = input_dim // 2
    for _ in range(num_samples // 2):
        x = np.zeros(input_dim, dtype=np.float32)
        x[:half] = 1.0
        x += rng.normal(0, 0.05, input_dim).astype(np.float32)
        data.append(np.clip(x, 0, 1))
    for _ in range(num_samples // 2):
        x = np.zeros(input_dim, dtype=np.float32)
        x[half:] = 1.0
        x += rng.normal(0, 0.05, input_dim).astype(np.float32)
        data.append(np.clip(x, 0, 1))
    return np.stack(data)


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("变分自编码器 (VAE) 演示")
    print("=" * 50)

    torch.manual_seed(42)

    INPUT_DIM  = 16
    LATENT_DIM = 4
    EPOCHS     = 30
    BATCH      = 64

    # 数据
    data = make_binary_patterns(num_samples=500, input_dim=INPUT_DIM)
    x_tensor = torch.tensor(data)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=BATCH, shuffle=True)

    # 模型
    model     = VAE(input_dim=INPUT_DIM, hidden_dim=32, latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"输入维度: {INPUT_DIM}, 潜在维度: {LATENT_DIM}")
    print(f"训练样本: {len(data)}\n")
    print(f"{'Epoch':>6}  {'Loss':>10}")
    print("-" * 22)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for (xb,) in loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = model(xb)
            loss = vae_loss(x_recon, xb, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(data)
        if epoch % 10 == 0:
            print(f"{epoch:>6}  {avg_loss:>10.4f}")

    # 演示：从潜在空间采样生成新样本
    print("\n── 从潜在空间采样生成新样本 ──")
    model.eval()
    with torch.no_grad():
        z = torch.randn(4, LATENT_DIM)
        generated = model.decode(z).numpy()
    for i, g in enumerate(generated):
        bar = "".join("█" if v > 0.5 else "░" for v in g)
        print(f"  样本 {i+1}: {bar}")

    # 演示：重建已有样本
    print("\n── 重建演示 ──")
    model.eval()
    with torch.no_grad():
        sample = x_tensor[:2]
        recon, mu, _ = model(sample)
    for i in range(2):
        orig = "".join("█" if v > 0.5 else "░" for v in sample[i].numpy())
        recs = "".join("█" if v > 0.5 else "░" for v in recon[i].numpy())
        print(f"  原始  {i+1}: {orig}")
        print(f"  重建  {i+1}: {recs}")
        print(f"  潜在均值 μ: {np.round(mu[i].numpy(), 2)}")
        print()

    print("✅ VAE 演示完成！")


if __name__ == "__main__":
    main()
