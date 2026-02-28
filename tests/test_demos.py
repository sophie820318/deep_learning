"""
测试所有深度学习示例能正常运行
Run: python3 tests/test_demos.py
"""

import sys
import os
import importlib.util
import numpy as np

# 将各模块目录加入路径
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def load_module(rel_path):
    abs_path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location("mod", abs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────
# 01 前向传播
# ──────────────────────────────────────────────

def test_activation_functions():
    mod = load_module("01_forward_propagation/forward_propagation.py")
    x = np.array([-1.0, 0.0, 1.0])
    s = mod.sigmoid(x)
    assert s.shape == (3,)
    assert abs(s[1] - 0.5) < 1e-9, "sigmoid(0) 应为 0.5"
    r = mod.relu(x)
    assert r[0] == 0.0 and r[2] == 1.0
    sm = mod.softmax(x)
    assert abs(sm.sum() - 1.0) < 1e-9, "softmax 之和应为 1"


def test_forward_propagation():
    mod = load_module("01_forward_propagation/forward_propagation.py")
    net = mod.NeuralNetwork(input_size=3, hidden1_size=4, hidden2_size=3, output_size=2)
    x = np.random.randn(5, 3)
    out = net.forward(x)
    assert out.shape == (5, 2), "输出形状错误"
    assert np.allclose(out.sum(axis=1), 1.0), "输出应为概率分布（行和为 1）"


# ──────────────────────────────────────────────
# 02 反向传播
# ──────────────────────────────────────────────

def test_gradient_check():
    mod = load_module("02_backpropagation/backpropagation.py")
    np.random.seed(1)
    X = np.random.randn(4, 2)
    y = np.random.rand(4, 1)
    net = mod.TwoLayerNet(input_size=2, hidden_size=3, output_size=1, lr=0.1)

    y_pred = net.forward(X)
    grads = net.backward(y_pred, y)

    eps = 1e-5
    for param_name in ["W1", "b1"]:
        param = getattr(net, param_name)
        for idx in np.ndindex(param.shape):
            orig = param[idx]
            param[idx] = orig + eps
            loss_p = mod.mean_squared_error(net.forward(X), y)
            param[idx] = orig - eps
            loss_m = mod.mean_squared_error(net.forward(X), y)
            param[idx] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            diff = abs(num_grad - grads[param_name][idx])
            assert diff < 1e-5, f"{param_name}[{idx}] 梯度差异过大: {diff}"


def test_xor_learning():
    mod = load_module("02_backpropagation/backpropagation.py")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    net = mod.TwoLayerNet(input_size=2, hidden_size=4, output_size=1, lr=1.0)
    for _ in range(3000):
        net.train_step(X, y)
    preds = net.forward(X)
    # 检查 XOR 的四个输出方向是否正确
    assert preds[0, 0] < 0.5, "XOR(0,0) 应接近 0"
    assert preds[1, 0] > 0.5, "XOR(0,1) 应接近 1"
    assert preds[2, 0] > 0.5, "XOR(1,0) 应接近 1"
    assert preds[3, 0] < 0.5, "XOR(1,1) 应接近 0"


# ──────────────────────────────────────────────
# 03 CNN
# ──────────────────────────────────────────────

def test_cnn_forward():
    import torch
    mod = load_module("03_cnn/cnn.py")
    net = mod.SimpleCNN(num_classes=3)
    x = torch.randn(4, 1, 16, 16)
    out = net(x)
    assert out.shape == (4, 3), "CNN 输出形状错误"


def test_cnn_dataset():
    mod = load_module("03_cnn/cnn.py")
    images, labels = mod.make_synthetic_dataset(num_samples=30, num_classes=3)
    assert images.shape == (30, 1, 16, 16)
    assert labels.shape == (30,)
    assert set(labels).issubset({0, 1, 2})


# ──────────────────────────────────────────────
# 04 RNN
# ──────────────────────────────────────────────

def test_rnn_forward():
    import torch
    mod = load_module("04_rnn/rnn.py")
    net = mod.SimpleRNN(input_size=1, hidden_size=16, output_size=1)
    x = torch.randn(8, 20, 1)
    out = net(x)
    assert out.shape == (8, 1), "RNN 输出形状错误"


def test_rnn_dataset():
    mod = load_module("04_rnn/rnn.py")
    X, y = mod.make_sin_dataset(num_samples=50, seq_len=10)
    assert X.shape == (50, 10, 1)
    assert y.shape == (50, 1)


# ──────────────────────────────────────────────
# 05 LSTM
# ──────────────────────────────────────────────

def test_lstm_forward():
    import torch
    mod = load_module("05_lstm/lstm.py")
    vocab_size = 20
    net = mod.CharLSTM(vocab_size, embed_dim=8, hidden_size=16)
    x = torch.randint(0, vocab_size, (4, 10))
    logits, hidden = net(x)
    assert logits.shape == (4, 10, vocab_size), "LSTM logits 形状错误"


def test_lstm_vocab():
    mod = load_module("05_lstm/lstm.py")
    text = "hello world"
    chars, c2i, i2c = mod.build_vocab(text)
    assert len(chars) == len(set(text))
    assert all(i2c[c2i[c]] == c for c in text)


# ──────────────────────────────────────────────
# 06 VAE
# ──────────────────────────────────────────────

def test_vae_forward():
    import torch
    mod = load_module("06_vae/vae.py")
    vae = mod.VAE(input_dim=16, hidden_dim=32, latent_dim=4)
    x = torch.rand(8, 16)
    x_recon, mu, logvar = vae(x)
    assert x_recon.shape == (8, 16), "VAE 重建形状错误"
    assert mu.shape == (8, 4), "VAE mu 形状错误"


def test_vae_loss():
    import torch
    mod = load_module("06_vae/vae.py")
    x = torch.rand(8, 16)
    x_recon = torch.rand(8, 16)
    mu = torch.zeros(8, 4)
    logvar = torch.zeros(8, 4)
    loss = mod.vae_loss(x_recon, x, mu, logvar)
    assert loss.item() > 0, "VAE 损失应大于 0"


# ──────────────────────────────────────────────
# 07 GAN
# ──────────────────────────────────────────────

def test_gan_forward():
    import torch
    mod = load_module("07_gan/gan.py")
    G = mod.Generator(latent_dim=8, hidden_dim=16, output_dim=16)
    D = mod.Discriminator(input_dim=16, hidden_dim=16)
    z = torch.randn(4, 8)
    fake = G(z)
    assert fake.shape == (4, 16), "Generator 输出形状错误"
    score = D(fake)
    assert score.shape == (4, 1), "Discriminator 输出形状错误"
    assert (score >= 0).all() and (score <= 1).all(), "Discriminator 输出应在 [0,1]"


# ──────────────────────────────────────────────
# 08 文本生成
# ──────────────────────────────────────────────

def test_text_generation_forward():
    import torch
    mod = load_module("08_text_generation/text_generation.py")
    vocab_size = 30
    model = mod.MiniLM(vocab_size, embed_dim=16, num_heads=2, num_layers=1, max_len=20)
    x = torch.randint(0, vocab_size, (2, 10))
    out = model(x)
    assert out.shape == (2, 10, vocab_size), "MiniLM 输出形状错误"


def test_text_generation_generate():
    import torch
    mod = load_module("08_text_generation/text_generation.py")
    c2i, i2c = mod.build_vocab(mod.CORPUS)
    vocab_size = len(c2i)
    model = mod.MiniLM(vocab_size, embed_dim=16, num_heads=2, num_layers=1, max_len=20)
    result = mod.generate(model, c2i, i2c, "the", max_new_tokens=10)
    assert result.startswith("the"), "生成文本应以种子文本开头"
    assert len(result) == len("the") + 10, "生成长度错误"


# ──────────────────────────────────────────────
# 运行所有测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("01_前向传播 - 激活函数", test_activation_functions),
        ("01_前向传播 - 网络前向传播", test_forward_propagation),
        ("02_反向传播 - 梯度验证", test_gradient_check),
        ("02_反向传播 - XOR 学习", test_xor_learning),
        ("03_CNN - 前向传播", test_cnn_forward),
        ("03_CNN - 数据集生成", test_cnn_dataset),
        ("04_RNN - 前向传播", test_rnn_forward),
        ("04_RNN - 数据集生成", test_rnn_dataset),
        ("05_LSTM - 前向传播", test_lstm_forward),
        ("05_LSTM - 词表构建", test_lstm_vocab),
        ("06_VAE - 前向传播", test_vae_forward),
        ("06_VAE - 损失函数", test_vae_loss),
        ("07_GAN - 前向传播", test_gan_forward),
        ("08_文本生成 - 前向传播", test_text_generation_forward),
        ("08_文本生成 - 生成函数", test_text_generation_generate),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"测试结果: {passed} 通过, {failed} 失败")
    if failed > 0:
        sys.exit(1)
    else:
        print("所有测试通过！✅")
