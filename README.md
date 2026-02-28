# 深度学习入门 / Deep Learning for Beginners

基于我妻幸长三部曲整理的深度学习入门示例代码，适合初学者学习。

参考书目：
- 《深度学习数学知识》
- 《新手的深度学习》
- 《写给新手的深度学习 2》

## 目录结构

| 目录 | 内容 |
|------|------|
| [01_forward_propagation](01_forward_propagation/) | 前向传播（NumPy 从零实现） |
| [02_backpropagation](02_backpropagation/) | 反向传播（NumPy 从零实现） |
| [03_cnn](03_cnn/) | 卷积神经网络 CNN（PyTorch） |
| [04_rnn](04_rnn/) | 循环神经网络 RNN（PyTorch） |
| [05_lstm](05_lstm/) | 长短期记忆网络 LSTM（PyTorch） |
| [06_vae](06_vae/) | 变分自编码器 VAE（PyTorch） |
| [07_gan](07_gan/) | 生成对抗网络 GAN（PyTorch） |
| [08_text_generation](08_text_generation/) | 文本生成 / 生成语言（PyTorch） |

## 快速开始

```bash
pip install -r requirements.txt
```

运行任意示例：

```bash
python 01_forward_propagation/forward_propagation.py
python 02_backpropagation/backpropagation.py
python 03_cnn/cnn.py
python 04_rnn/rnn.py
python 05_lstm/lstm.py
python 06_vae/vae.py
python 07_gan/gan.py
python 08_text_generation/text_generation.py
```

## 环境要求

- Python 3.8+
- NumPy
- Matplotlib
- PyTorch（CPU 版本即可）
