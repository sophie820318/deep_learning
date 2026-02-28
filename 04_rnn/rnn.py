"""
循环神经网络 RNN (Recurrent Neural Network) - PyTorch
基于《写给新手的深度学习 2》 我妻幸长

知识点：
- RNN 的循环结构：隐藏状态随时间步传递
- 序列数据处理
- 使用 RNN 学习简单的数值序列规律
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ──────────────────────────────────────────────
# RNN 模型定义
# ──────────────────────────────────────────────

class SimpleRNN(nn.Module):
    """
    单层 RNN，用于序列回归预测
    输入：(batch, seq_len, input_size)
    输出：(batch, output_size)
    """

    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,    # 输入形状为 (batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out: (batch, seq_len, hidden_size)
        # h_n: (1, batch, hidden_size)
        out, h_n = self.rnn(x)
        # 取最后一个时间步的输出用于预测
        last_out = out[:, -1, :]   # (batch, hidden_size)
        return self.fc(last_out)   # (batch, output_size)


# ──────────────────────────────────────────────
# 数据生成：正弦波序列预测
# ──────────────────────────────────────────────

def make_sin_dataset(num_samples=500, seq_len=20, seed=42):
    """
    生成正弦波数据集
    输入：长度为 seq_len 的序列片段
    目标：序列之后的下一个点
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, num_samples + seq_len)
    # 加入不同频率的正弦信号
    signal = np.sin(t) + 0.5 * np.sin(2 * t)
    signal = signal.astype(np.float32)

    X, y = [], []
    for i in range(num_samples):
        X.append(signal[i: i + seq_len])
        y.append(signal[i + seq_len])

    X = np.stack(X)[:, :, np.newaxis]  # (N, seq_len, 1)
    y = np.array(y)[:, np.newaxis]     # (N, 1)
    return X, y


# ──────────────────────────────────────────────
# 训练与评估
# ──────────────────────────────────────────────

def train_epoch(model, x_tensor, y_tensor, optimizer, criterion, batch_size=64):
    model.train()
    n = len(x_tensor)
    idx = torch.randperm(n)
    total_loss = 0.0
    for start in range(0, n, batch_size):
        batch_idx = idx[start: start + batch_size]
        xb, yb = x_tensor[batch_idx], y_tensor[batch_idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)
    return total_loss / n


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("循环神经网络 (RNN) 演示 — 正弦波序列预测")
    print("=" * 50)

    torch.manual_seed(42)

    SEQ_LEN = 20
    X, y = make_sin_dataset(num_samples=500, seq_len=SEQ_LEN)

    split = int(0.8 * len(y))
    x_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    x_test  = torch.tensor(X[split:])
    y_test  = torch.tensor(y[split:])

    model     = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"序列长度: {SEQ_LEN}")
    print(f"训练集: {len(y_train)} 样本, 测试集: {len(y_test)} 样本\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Loss':>10}")
    print("-" * 35)

    for epoch in range(1, 21):
        train_loss = train_epoch(model, x_train, y_train, optimizer, criterion)
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(x_test), y_test).item()
        if epoch % 5 == 0:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {test_loss:>10.6f}")

    # 简单预测演示
    model.eval()
    with torch.no_grad():
        sample_pred = model(x_test[:3]).numpy().flatten()
        sample_true = y_test[:3].numpy().flatten()

    print("\n预测示例（前 3 个测试样本）：")
    for true, pred in zip(sample_true, sample_pred):
        print(f"  真实值: {true:+.4f}  预测值: {pred:+.4f}")

    print("\n✅ RNN 演示完成！")


if __name__ == "__main__":
    main()
