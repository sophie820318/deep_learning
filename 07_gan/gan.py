"""
生成对抗网络 GAN (Generative Adversarial Network) - PyTorch
基于《写给新手的深度学习 2》 我妻幸长

知识点：
- 生成器 (Generator)：从随机噪声生成假样本
- 判别器 (Discriminator)：区分真实样本和生成样本
- 对抗训练：生成器和判别器相互博弈
- 训练目标：G 最小化，D 最大化（minimax 博弈）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────
# 生成器 (Generator)
# ──────────────────────────────────────────────

class Generator(nn.Module):
    """
    生成器：将噪声向量 z ~ N(0,1) 映射到数据空间
    latent_dim → hidden → output_dim
    """

    def __init__(self, latent_dim=8, hidden_dim=32, output_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# ──────────────────────────────────────────────
# 判别器 (Discriminator)
# ──────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    判别器：判断输入样本是真实(1)还是生成的(0)
    input_dim → hidden → 1
    """

    def __init__(self, input_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 合成数据生成（两个高斯分布的混合）
# ──────────────────────────────────────────────

def make_real_data(num_samples=500, data_dim=16, seed=42):
    """
    真实数据：两个高斯分布混合，归一化到 [-1, 1]
    - 模式 A：均值向量均为 +0.5
    - 模式 B：均值向量均为 -0.5
    """
    rng = np.random.default_rng(seed)
    half = num_samples // 2
    a = rng.normal(+0.5, 0.2, (half, data_dim)).astype(np.float32)
    b = rng.normal(-0.5, 0.2, (half, data_dim)).astype(np.float32)
    data = np.vstack([a, b])
    # 归一化到 [-1, 1]
    data = np.clip(data, -1, 1)
    return data


# ──────────────────────────────────────────────
# 训练步骤
# ──────────────────────────────────────────────

def train_discriminator(D, G, real_batch, latent_dim, optimizer_D, criterion):
    """
    训练判别器：
    - 真实样本 → 判别器输出接近 1
    - 生成样本 → 判别器输出接近 0
    """
    batch_size = real_batch.size(0)

    # 真实样本标签为 1，平滑处理 (0.9) 以提高训练稳定性
    real_labels = torch.full((batch_size, 1), 0.9)
    fake_labels = torch.zeros(batch_size, 1)

    # 真实样本损失
    D.zero_grad()
    real_output = D(real_batch)
    loss_real = criterion(real_output, real_labels)

    # 生成样本损失
    z = torch.randn(batch_size, latent_dim)
    fake_batch = G(z).detach()  # detach 防止梯度流向 G
    fake_output = D(fake_batch)
    loss_fake = criterion(fake_output, fake_labels)

    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()

    return loss_D.item()


def train_generator(D, G, batch_size, latent_dim, optimizer_G, criterion):
    """
    训练生成器：
    - 生成样本经过判别器 → 希望输出接近 1（欺骗判别器）
    """
    real_labels = torch.ones(batch_size, 1)  # 欺骗判别器

    G.zero_grad()
    z = torch.randn(batch_size, latent_dim)
    fake_batch = G(z)
    fake_output = D(fake_batch)

    loss_G = criterion(fake_output, real_labels)
    loss_G.backward()
    optimizer_G.step()

    return loss_G.item()


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("生成对抗网络 (GAN) 演示")
    print("=" * 50)

    torch.manual_seed(42)

    DATA_DIM   = 16
    LATENT_DIM = 8
    EPOCHS     = 50
    BATCH      = 64

    # 真实数据
    real_data = make_real_data(num_samples=500, data_dim=DATA_DIM)
    x_tensor = torch.tensor(real_data)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=BATCH, shuffle=True)

    # 模型
    G = Generator(latent_dim=LATENT_DIM, hidden_dim=32, output_dim=DATA_DIM)
    D = Discriminator(input_dim=DATA_DIM, hidden_dim=32)

    optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion   = nn.BCELoss()

    print(f"潜在维度: {LATENT_DIM}, 数据维度: {DATA_DIM}")
    print(f"训练样本: {len(real_data)}\n")
    print(f"{'Epoch':>6}  {'D Loss':>8}  {'G Loss':>8}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        total_d, total_g, count = 0.0, 0.0, 0
        for (real_batch,) in loader:
            loss_D = train_discriminator(
                D, G, real_batch, LATENT_DIM, optimizer_D, criterion
            )
            loss_G = train_generator(
                D, G, real_batch.size(0), LATENT_DIM, optimizer_G, criterion
            )
            total_d += loss_D
            total_g += loss_G
            count   += 1

        if epoch % 10 == 0:
            print(f"{epoch:>6}  {total_d/count:>8.4f}  {total_g/count:>8.4f}")

    # 验证：生成样本统计
    print("\n── 生成样本统计 ──")
    G.eval()
    with torch.no_grad():
        z = torch.randn(200, LATENT_DIM)
        gen_samples = G(z).numpy()

    print(f"  真实数据 — 均值: {real_data.mean():.4f}, 标准差: {real_data.std():.4f}")
    print(f"  生成样本 — 均值: {gen_samples.mean():.4f}, 标准差: {gen_samples.std():.4f}")

    # 判别器对生成样本的评分（越接近 0.5 说明越难区分）
    D.eval()
    with torch.no_grad():
        score = D(torch.tensor(gen_samples)).mean().item()
    print(f"  判别器对生成样本评分（接近 0.5 说明混淆成功）: {score:.4f}")

    print("\n✅ GAN 演示完成！")


if __name__ == "__main__":
    main()
