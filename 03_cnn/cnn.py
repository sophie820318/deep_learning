"""
卷积神经网络 CNN (Convolutional Neural Network) - PyTorch
基于《新手的深度学习》 我妻幸长

知识点：
- 卷积层（Conv2d）：提取局部特征
- 池化层（MaxPool2d）：降维，保留主要特征
- 全连接层：分类输出
- 使用合成数据演示图像分类
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────
# CNN 模型定义
# ──────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    简单 CNN 结构（适合小图像分类）
    输入：(batch, 1, 16, 16) 灰度图
    输出：(batch, num_classes) 分类概率
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # 特征提取部分
        self.features = nn.Sequential(
            # 第一卷积块：1→8 通道，3×3 卷积，ReLU，2×2 MaxPool
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # → (8, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (8, 8, 8)

            # 第二卷积块：8→16 通道
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # → (16, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # → (16, 4, 4)
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),          # → (16*4*4) = 256
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# 合成数据集生成
# ──────────────────────────────────────────────

def make_synthetic_dataset(num_samples=300, num_classes=3, img_size=16, seed=42):
    """
    生成合成图像数据集：
    - 类别 0：随机噪声图像
    - 类别 1：中心亮点图像
    - 类别 2：边缘亮点图像
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []

    per_class = num_samples // num_classes
    for cls in range(num_classes):
        for _ in range(per_class):
            img = rng.random((img_size, img_size)).astype(np.float32) * 0.1
            if cls == 0:
                pass  # 只有噪声
            elif cls == 1:
                # 中心亮点
                cx, cy = img_size // 2, img_size // 2
                img[cx-2:cx+2, cy-2:cy+2] = 1.0
            else:
                # 边缘亮点
                img[0:3, :] = 1.0
                img[-3:, :] = 1.0
            images.append(img)
            labels.append(cls)

    images = np.stack(images)[:, np.newaxis, :, :]  # (N, 1, H, W)
    labels = np.array(labels)

    # 打乱顺序
    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


# ──────────────────────────────────────────────
# 训练与评估
# ──────────────────────────────────────────────

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    return correct / len(loader.dataset)


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("卷积神经网络 (CNN) 演示")
    print("=" * 50)

    torch.manual_seed(42)

    # 生成数据
    images, labels = make_synthetic_dataset(num_samples=300, num_classes=3)
    split = int(0.8 * len(labels))
    x_train = torch.tensor(images[:split])
    y_train = torch.tensor(labels[:split], dtype=torch.long)
    x_test  = torch.tensor(images[split:])
    y_test  = torch.tensor(labels[split:], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(x_test, y_test),  batch_size=32)

    # 模型、优化器、损失函数
    model     = SimpleCNN(num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"模型结构：\n{model}\n")
    print(f"训练集：{len(y_train)} 样本，测试集：{len(y_test)} 样本\n")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Test Acc':>9}")
    print("-" * 30)

    for epoch in range(1, 11):
        loss = train(model, train_loader, optimizer, criterion)
        acc  = evaluate(model, test_loader)
        print(f"{epoch:>6}  {loss:>8.4f}  {acc:>8.1%}")

    print(f"\n最终测试准确率: {evaluate(model, test_loader):.1%}")
    print("\n✅ CNN 演示完成！")


if __name__ == "__main__":
    main()
