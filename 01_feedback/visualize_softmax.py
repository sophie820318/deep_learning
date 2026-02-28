import numpy as np
import matplotlib.pyplot as plt

# Softmax 函数可视化
def softmax(u):
    exp_u = np.exp(u)
    return exp_u / np.sum(exp_u, keepdims=True)

# 示例1: 二分类情况（代码中的情况）
print("=== 二分类 Softmax (n_out=2) ===")
u1 = np.array([2.0, 1.0])  # 类别0得分高
s1 = softmax(u1)
print(f"输入 u = {u1}")
print(f"Softmax输出 = {s1}")
print(f"概率之和 = {np.sum(s1)}")
print(f"预测: 类别0概率={s1[0]:.4f}, 类别1概率={s1[1]:.4f}")

u2 = np.array([0.5, 2.5])  # 类别1得分高
s2 = softmax(u2)
print(f"\n输入 u = {u2}")
print(f"Softmax输出 = {s2}")
print(f"预测: 类别0概率={s2[0]:.4f}, 类别1概率={s2[1]:.4f}")

# 示例2: 不同输入值的对比
print("\n=== 不同输入值的 Softmax 输出 ===")
test_cases = [
    ([-5, 0, 5], "极端差异"),
    ([-1, 0, 1], "中等差异"),
    ([0, 0, 0], "相同输入"),
    ([10, 5, 0], "大数值"),
]

for u, desc in test_cases:
    u_arr = np.array(u)
    s = softmax(u_arr)
    print(f"{desc}: 输入{u} -> Softmax{s} (和={np.sum(s):.6f})")

# 可视化：Softmax 对输入差异的响应
print("\n=== Softmax 特性 ===")
print("1. 所有输出概率之和始终为 1")
print("2. 输出值在 (0, 1) 范围内")
print("3. 输入值越大，对应输出概率越高")
print("4. 输入值差异越大，概率分布越极端（接近 one-hot）")
print("5. 所有输入相同时，输出均匀分布")

