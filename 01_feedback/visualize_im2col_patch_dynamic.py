import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# 构造一个简单的单通道8x8图像
img = np.arange(1, 65).reshape(1, 1, 8, 8)

# 卷积核参数
flt_h, flt_w = 3, 3
stride = 1
pad = 1
n_bt, n_ch, img_h, img_w = img.shape
out_h = (img_h + 2 * pad - flt_h) // stride + 1
out_w = (img_w + 2 * pad - flt_w) // stride + 1

# padding
img_pad = np.pad(img, [(0,0),(0,0),(pad, pad),(pad, pad)], "constant")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img_pad[0, 0], cmap='gray', vmin=0, vmax=64)
ax.set_title('im2col动态提取patch演示')
ax.axis('off')

# 动态演示每个(h, w)下所有patch的采样
for h in range(flt_h):
    for w in range(flt_w):
        # 清除旧的矩形
        [p.remove() for p in reversed(ax.patches)]
        # 画所有滑动窗口下的当前(h, w)格子采样位置
        for i in range(out_h):
            for j in range(out_w):
                rect = patches.Rectangle((w + j*stride - 0.5, h + i*stride - 0.5), 1, 1, edgecolor='red', facecolor='none', lw=2)
                ax.add_patch(rect)
        ax.set_title(f'im2col: 卷积核格子(h={h}, w={w})采样位置')
        plt.pause(0.8)

plt.show()

# 详细注释：
# 1. 图像img_pad[0,0]为原始8x8图像padding后10x10的效果。
# 2. 外层循环(h, w)遍历卷积核的每个格子。
# 3. 内层(i, j)遍历所有滑动窗口，红框显示当前(h, w)格子在所有窗口下的采样点。
# 4. 每次pause后，红框会动态切换，帮助理解im2col如何展开所有patch。
# 5. 你可以调整pause时间或参数，观察不同卷积核/步幅效果。
