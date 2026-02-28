"""
反向传播 (Backpropagation) - NumPy 从零实现
基于《新手的深度学习》 我妻幸长

知识点：
- 损失函数（均方误差、交叉熵）
- 链式法则求梯度
- 梯度下降更新权重
- 完整的训练循环
"""

import numpy as np


# ──────────────────────────────────────────────
# 激活函数及其导数
# ──────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    """Sigmoid 的导数：σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU 的导数：x>0 时为 1，否则为 0"""
    return (x > 0).astype(float)


# ──────────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────────

def mean_squared_error(y_pred, y_true):
    """均方误差 MSE = mean((y_pred - y_true)^2)"""
    return np.mean((y_pred - y_true) ** 2)


def cross_entropy_loss(y_pred, y_true):
    """
    交叉熵损失（配合 Sigmoid 输出的二分类）
    L = -mean(y * log(p) + (1-y) * log(1-p))
    """
    eps = 1e-8  # 防止 log(0)
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )


# ──────────────────────────────────────────────
# 简单两层网络（反向传播手工推导）
# ──────────────────────────────────────────────

class TwoLayerNet:
    """
    两层全连接网络，含完整的前向 + 反向传播
    结构：输入 → 隐藏层（Sigmoid）→ 输出层（Sigmoid）
    """

    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        np.random.seed(0)
        # Xavier 初始化：权重标准差 = sqrt(1 / 输入维度)，帮助 Sigmoid 网络收敛
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.lr = lr  # 学习率

        # 缓存中间变量，供反向传播使用
        self._cache = {}

    def forward(self, x):
        """前向传播，缓存中间值"""
        z1 = np.dot(x, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)

        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2}
        return a2

    def backward(self, y_pred, y_true):
        """
        反向传播（链式法则求各参数梯度）

        损失对输出的梯度（MSE）:
            dL/da2 = 2*(a2 - y) / n
        """
        x  = self._cache["x"]
        z1 = self._cache["z1"]
        a1 = self._cache["a1"]
        z2 = self._cache["z2"]
        n  = x.shape[0]  # batch size

        # ── 输出层 ──────────────────────────────
        # dL/da2
        d_a2 = 2.0 * (y_pred - y_true) / n          # (n, output)
        # dL/dz2 = dL/da2 * sigmoid'(z2)
        d_z2 = d_a2 * sigmoid_grad(z2)              # (n, output)
        # dL/dW2 = a1^T · dL/dz2
        d_W2 = np.dot(a1.T, d_z2)                   # (hidden, output)
        # dL/db2 = sum(dL/dz2)
        d_b2 = np.sum(d_z2, axis=0)                 # (output,)

        # ── 隐藏层 ──────────────────────────────
        # dL/da1 = dL/dz2 · W2^T
        d_a1 = np.dot(d_z2, self.W2.T)              # (n, hidden)
        # dL/dz1 = dL/da1 * sigmoid'(z1)
        d_z1 = d_a1 * sigmoid_grad(z1)              # (n, hidden)
        # dL/dW1 = x^T · dL/dz1
        d_W1 = np.dot(x.T, d_z1)                    # (input, hidden)
        # dL/db1
        d_b1 = np.sum(d_z1, axis=0)                 # (hidden,)

        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}

    def update(self, grads):
        """梯度下降更新参数"""
        self.W1 -= self.lr * grads["W1"]
        self.b1 -= self.lr * grads["b1"]
        self.W2 -= self.lr * grads["W2"]
        self.b2 -= self.lr * grads["b2"]

    def train_step(self, x, y):
        """单步训练：前向 → 计算损失 → 反向 → 更新"""
        y_pred = self.forward(x)
        loss = mean_squared_error(y_pred, y)
        grads = self.backward(y_pred, y)
        self.update(grads)
        return loss


# ──────────────────────────────────────────────
# 演示：学习 XOR 问题
# ──────────────────────────────────────────────

def demo_xor():
    """
    XOR 问题：线性模型无法解决，需要隐藏层
      x1  x2 | y
      0   0  | 0
      0   1  | 1
      1   0  | 1
      1   1  | 0
    """
    print("=" * 50)
    print("XOR 问题：反向传播训练演示")
    print("=" * 50)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = TwoLayerNet(input_size=2, hidden_size=4, output_size=1, lr=1.0)

    print(f"{'Epoch':>6}  {'Loss':>10}")
    print("-" * 20)
    for epoch in range(1, 5001):
        loss = net.train_step(X, y)
        if epoch % 1000 == 0:
            print(f"{epoch:>6}  {loss:>10.6f}")

    # 最终预测
    print("\n最终预测：")
    preds = net.forward(X)
    for xi, yi, pi in zip(X, y, preds):
        print(f"  输入 {xi.astype(int)} → 目标 {int(yi[0])}, 预测 {pi[0]:.4f}")
    print()


def demo_gradient_check():
    """
    数值梯度验证：用数值微分检验解析梯度是否正确
    （验证反向传播实现的正确性）
    """
    print("=" * 50)
    print("梯度验证（数值梯度 vs 解析梯度）")
    print("=" * 50)

    np.random.seed(1)
    X = np.random.randn(4, 2)
    y = np.random.rand(4, 1)
    net = TwoLayerNet(input_size=2, hidden_size=3, output_size=1, lr=0.1)

    # 解析梯度
    y_pred = net.forward(X)
    grads_analytic = net.backward(y_pred, y)

    # 数值梯度（用于验证）
    eps = 1e-5

    def loss_fn():
        p = net.forward(X)
        return mean_squared_error(p, y)

    for param_name in ["W1", "b1"]:
        param = getattr(net, param_name)
        grad_numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            loss_plus = loss_fn()
            param[idx] = original - eps
            loss_minus = loss_fn()
            param[idx] = original
            grad_numeric[idx] = (loss_plus - loss_minus) / (2 * eps)

        diff = np.max(np.abs(grad_numeric - grads_analytic[param_name]))
        print(f"  {param_name}: 最大差异 = {diff:.2e}  {'✅ 通过' if diff < 1e-5 else '❌ 不通过'}")

    print()


if __name__ == "__main__":
    demo_gradient_check()
    demo_xor()
    print("✅ 反向传播演示完成！")
