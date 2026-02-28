"""
前向传播 (Forward Propagation) - NumPy 从零实现
基于《新手的深度学习》 我妻幸长

知识点：
- 激活函数（Sigmoid、ReLU）
- 神经元计算
- 多层神经网络的前向传播
"""

import numpy as np


# ──────────────────────────────────────────────
# 激活函数
# ──────────────────────────────────────────────

def sigmoid(x):
    """Sigmoid 激活函数：将任意实数映射到 (0, 1)"""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """ReLU 激活函数：负数变为 0，正数保持不变"""
    return np.maximum(0, x)


def softmax(x):
    """Softmax 激活函数：输出概率分布（用于多分类输出层）"""
    # 减去最大值以提高数值稳定性
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# ──────────────────────────────────────────────
# 神经网络类
# ──────────────────────────────────────────────

class NeuralNetwork:
    """
    简单三层全连接神经网络
    结构：输入层 → 隐藏层1 → 隐藏层2 → 输出层
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        np.random.seed(42)
        # 用随机小数初始化权重，用零初始化偏置
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros(hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros(hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros(output_size)

    def forward(self, x):
        """
        前向传播：逐层计算 z = Wx + b，再应用激活函数

        参数：
            x: 输入数据，形状 (batch_size, input_size)

        返回：
            output: 输出概率，形状 (batch_size, output_size)
        """
        # 第一层：线性变换 + Sigmoid 激活
        z1 = np.dot(x, self.W1) + self.b1   # (batch, hidden1)
        a1 = sigmoid(z1)

        # 第二层：线性变换 + ReLU 激活
        z2 = np.dot(a1, self.W2) + self.b2  # (batch, hidden2)
        a2 = relu(z2)

        # 输出层：线性变换 + Softmax（输出概率）
        z3 = np.dot(a2, self.W3) + self.b3  # (batch, output)
        output = softmax(z3)

        return output


# ──────────────────────────────────────────────
# 演示
# ──────────────────────────────────────────────

def demo_activation_functions():
    """演示各激活函数的输出"""
    print("=" * 50)
    print("激活函数演示")
    print("=" * 50)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"输入 x       : {x}")
    print(f"Sigmoid(x)  : {np.round(sigmoid(x), 4)}")
    print(f"ReLU(x)     : {relu(x)}")
    print(f"Softmax([1,2,3]): {np.round(softmax(np.array([1.0, 2.0, 3.0])), 4)}")
    print()


def demo_single_neuron():
    """演示单个神经元的计算"""
    print("=" * 50)
    print("单个神经元演示")
    print("=" * 50)
    # 输入：2 个特征
    x = np.array([0.5, 0.8])
    # 权重和偏置
    w = np.array([0.3, -0.2])
    b = 0.1
    # 线性组合
    z = np.dot(w, x) + b
    # 激活
    a = sigmoid(z)
    print(f"输入 x = {x}")
    print(f"权重 w = {w}, 偏置 b = {b}")
    print(f"线性组合 z = w·x + b = {z:.4f}")
    print(f"激活输出 a = sigmoid(z) = {a:.4f}")
    print()


def demo_forward_propagation():
    """演示多层网络前向传播"""
    print("=" * 50)
    print("多层网络前向传播演示")
    print("=" * 50)

    # 网络结构：3 输入 → 4 隐藏1 → 3 隐藏2 → 2 输出
    net = NeuralNetwork(input_size=3, hidden1_size=4, hidden2_size=3, output_size=2)

    # 批量输入：5 个样本，每个样本 3 个特征
    x = np.random.randn(5, 3)
    output = net.forward(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出（概率分布，每行之和为 1）:")
    print(np.round(output, 4))
    print(f"每行概率之和: {np.round(output.sum(axis=1), 4)}")
    print()


if __name__ == "__main__":
    demo_activation_functions()
    demo_single_neuron()
    demo_forward_propagation()
    print("✅ 前向传播演示完成！")
