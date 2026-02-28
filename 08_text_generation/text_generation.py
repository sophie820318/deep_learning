"""
文本生成 / 生成语言 (Text Generation) - PyTorch
基于《写给新手的深度学习 2》 我妻幸长

知识点：
- Transformer 自注意力机制（Attention）
- 语言模型：预测序列中的下一个词（或字符）
- 贪心解码（Greedy Decoding）
- 温度采样（Temperature Sampling）控制生成多样性

注：本示例为教学目的使用极小的模型和数据集，
    工业级文生图/大语言模型需要更大的模型和算力。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 自注意力 (Self-Attention) 语言模型
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    因果（单向）多头自注意力：每个位置只能看到之前的位置
    """

    def __init__(self, embed_dim, num_heads, max_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 因果掩码：下三角矩阵，确保不看到未来
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        # 计算 Q、K、V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]   # (B, heads, T, head_dim)

        # 缩放点积注意力
        scale = math.sqrt(self.head_dim)
        attn  = (q @ k.transpose(-2, -1)) / scale        # (B, heads, T, T)
        attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """单个 Transformer 块：自注意力 + 前馈网络 + 残差连接"""

    def __init__(self, embed_dim, num_heads, max_len, ff_mult=4):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_len)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim),
            nn.GELU(),
            nn.Linear(ff_mult * embed_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniLM(nn.Module):
    """
    迷你语言模型：字符级 Transformer（GPT-like）
    给定前缀字符序列，预测下一个字符
    """

    def __init__(self, vocab_size, embed_dim=64, num_heads=2,
                 num_layers=2, max_len=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed   = nn.Embedding(max_len, embed_dim)
        self.blocks      = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, max_len) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_embed(idx) + self.pos_embed(positions)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)   # (B, T, vocab_size)


# ──────────────────────────────────────────────
# 训练数据（简单英文短句，重复以增加样本量）
# ──────────────────────────────────────────────

CORPUS = (
    "the cat sat on the mat. "
    "a dog ran in the park. "
    "the sun is bright today. "
    "birds fly high in the sky. "
    "deep learning models learn from data. "
    "the cat sat on the mat. "
    "a dog ran in the park. "
    "deep learning is a powerful technique. "
    "neural networks mimic the human brain. "
    "the cat sat on the mat. "
) * 8


def build_vocab(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c


def make_batches(text, c2i, seq_len, batch_size, seed=42):
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    # 随机窗口采样
    rng = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size * 50,),
                           generator=rng)
    X = torch.stack([data[s: s + seq_len] for s in starts])
    y = torch.stack([data[s + 1: s + seq_len + 1] for s in starts])
    return X, y


# ──────────────────────────────────────────────
# 文本生成（温度采样）
# ──────────────────────────────────────────────

@torch.no_grad()
def generate(model, c2i, i2c, seed_text, max_new_tokens=80, temperature=0.8):
    """
    自回归文本生成：
    - 用 seed_text 作为初始上下文
    - 逐个预测下一个字符，追加到序列后继续生成
    """
    model.eval()
    ctx = torch.tensor([[c2i.get(c, 0) for c in seed_text]], dtype=torch.long)
    generated = list(seed_text)

    for _ in range(max_new_tokens):
        # 截断至最大长度
        ctx_crop = ctx[:, -model.max_len:]
        logits   = model(ctx_crop)[:, -1, :]  # (1, vocab)
        probs    = F.softmax(logits / temperature, dim=-1)
        next_id  = torch.multinomial(probs, 1).item()
        generated.append(i2c[next_id])
        ctx = torch.cat([ctx, torch.tensor([[next_id]])], dim=1)

    return "".join(generated)


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    print("=" * 55)
    print("文本生成 — 迷你 Transformer 语言模型演示")
    print("=" * 55)

    torch.manual_seed(42)

    SEQ_LEN   = 40
    BATCH     = 64
    EPOCHS    = 20
    EMBED_DIM = 64

    c2i, i2c = build_vocab(CORPUS)
    vocab_size = len(c2i)
    print(f"语料长度: {len(CORPUS)} 字符, 词表大小: {vocab_size}\n")

    X, y = make_batches(CORPUS, c2i, seq_len=SEQ_LEN, batch_size=BATCH)
    print(f"训练序列数: {len(X)}\n")

    model     = MiniLM(vocab_size, embed_dim=EMBED_DIM, num_heads=2,
                       num_layers=2, max_len=SEQ_LEN)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}\n")
    print(f"{'Epoch':>6}  {'Loss':>8}")
    print("-" * 20)

    dataset_size = len(X)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(dataset_size)
        total_loss = 0.0
        for start in range(0, dataset_size, BATCH):
            idx = perm[start: start + BATCH]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            logits = model(xb)                                # (B, T, V)
            loss   = criterion(logits.reshape(-1, vocab_size),
                               yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / dataset_size
        if epoch % 5 == 0:
            print(f"{epoch:>6}  {avg_loss:>8.4f}")

    # 文本生成示例
    print("\n── 文本生成示例 ──\n")
    seeds = ["the cat", "deep learning", "a dog"]
    for seed in seeds:
        result = generate(model, c2i, i2c, seed, max_new_tokens=60,
                          temperature=0.7)
        print(f"  种子: '{seed}'")
        print(f"  生成: '{result}'")
        print()

    print("✅ 文本生成演示完成！")


if __name__ == "__main__":
    main()
