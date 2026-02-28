"""
长短期记忆网络 LSTM (Long Short-Term Memory) - PyTorch
基于《写给新手的深度学习 2》 我妻幸长

知识点：
- LSTM 的门控机制（输入门、遗忘门、输出门）
- 相比普通 RNN，LSTM 能处理更长的依赖关系
- 使用 LSTM 进行字符级语言模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ──────────────────────────────────────────────
# LSTM 语言模型
# ──────────────────────────────────────────────

class CharLSTM(nn.Module):
    """
    字符级 LSTM 语言模型
    给定一段字符序列，预测下一个字符
    """

    def __init__(self, vocab_size, embed_dim=16, hidden_size=64, num_layers=1):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim)
        self.lstm   = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                              batch_first=True)
        self.fc     = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) — 字符索引
        emb = self.embed(x)                   # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)  # out: (batch, seq_len, hidden)
        logits = self.fc(out)                 # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, hidden_size, num_layers, device):
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (h0, c0)


# ──────────────────────────────────────────────
# 数据准备：字符级编码
# ──────────────────────────────────────────────

# 使用一段简单重复的文本作为训练数据
TEXT = (
    "deep learning is fun. "
    "neural networks learn features from data. "
    "lstm remembers long sequences better than rnn. "
    "deep learning is powerful. "
) * 6  # 重复 6 次以获得足够训练数据


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return chars, char2idx, idx2char


def text_to_sequences(text, char2idx, seq_len=30):
    """将文本切成 (输入序列, 目标序列) 对"""
    encoded = [char2idx[c] for c in text]
    X, y = [], []
    for i in range(0, len(encoded) - seq_len):
        X.append(encoded[i: i + seq_len])
        y.append(encoded[i + 1: i + seq_len + 1])  # 目标是向右移一位
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# 文本生成
# ──────────────────────────────────────────────

def generate_text(model, char2idx, idx2char, seed_text, length=50, temperature=1.0):
    """
    使用训练好的模型生成文本

    temperature：控制随机性
    - 低温 (<1)：更确定性，更保守
    - 高温 (>1)：更随机，更有创意
    """
    model.eval()
    device = next(model.parameters()).device

    generated = list(seed_text)
    input_ids = torch.tensor(
        [[char2idx.get(c, 0) for c in seed_text]], device=device
    )
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_ids, hidden)
            # 取最后一个时间步
            last_logits = logits[:, -1, :] / temperature    # (1, vocab)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(idx2char[next_id])
            input_ids = torch.tensor([[next_id]], device=device)

    return "".join(generated)


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    print("=" * 50)
    print("LSTM 字符级语言模型演示")
    print("=" * 50)

    torch.manual_seed(42)
    device = torch.device("cpu")

    SEQ_LEN = 30
    HIDDEN   = 64
    EPOCHS   = 20
    BATCH    = 64

    chars, char2idx, idx2char = build_vocab(TEXT)
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size} 个字符")

    X, y = text_to_sequences(TEXT, char2idx, seq_len=SEQ_LEN)
    print(f"序列数量: {len(X)}\n")

    x_tensor = torch.tensor(X, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    model     = CharLSTM(vocab_size, embed_dim=16, hidden_size=HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"{'Epoch':>6}  {'Loss':>8}")
    print("-" * 20)

    n = len(x_tensor)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, BATCH):
            batch_idx = idx[start: start + BATCH]
            xb = x_tensor[batch_idx]
            yb = y_tensor[batch_idx]

            optimizer.zero_grad()
            logits, _ = model(xb)
            # logits: (batch, seq_len, vocab) → reshape for CrossEntropy
            loss = criterion(
                logits.reshape(-1, vocab_size),
                yb.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        avg_loss = total_loss / n
        if epoch % 5 == 0:
            print(f"{epoch:>6}  {avg_loss:>8.4f}")

    # 生成文本示例
    print("\n── 生成文本示例 ──")
    seed = "deep learning"
    generated = generate_text(model, char2idx, idx2char, seed, length=60)
    print(f"种子文本: '{seed}'")
    print(f"生成结果: '{generated}'")

    print("\n✅ LSTM 演示完成！")


if __name__ == "__main__":
    main()
