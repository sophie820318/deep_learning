import numpy as np
import matplotlib.pyplot as plt

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

fig, axes = plt.subplots(flt_h, flt_w, figsize=(8, 8))

for h in range(flt_h):
    h_lim = h + stride * out_h
    for w in range(flt_w):
        w_lim = w + stride * out_w
        patch = img_pad[0, 0, h:h_lim:stride, w:w_lim:stride]
        ax = axes[h, w]
        ax.imshow(img_pad[0, 0], cmap='gray', vmin=0, vmax=64)
        # 用红框标出当前patch
        for i in range(out_h):
            for j in range(out_w):
                rect = plt.Rectangle((w + j*stride - 0.5, h + i*stride - 0.5), 1, 1, edgecolor='red', facecolor='none', lw=2)
                ax.add_patch(rect)
        ax.set_title(f'h={h}, w={w}')
        ax.axis('off')

plt.suptitle('im2col每个(h,w)提取的所有patch位置')
plt.tight_layout()
plt.show()
