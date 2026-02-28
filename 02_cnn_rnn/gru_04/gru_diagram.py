import matplotlib.pyplot as plt


def plot_gru_diagram(save_path="gru_architecture.jpg", show=True):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.0)

    # 输入层
    ax.add_patch(plt.Rectangle((0.05, 0.55), 0.18, 0.25, fill=False, linewidth=2))
    ax.text(0.14, 0.675, "Input\n(x_t)", ha="center", va="center")

    # 上一时刻隐状态
    ax.text(0.05, 0.9, "y_{t-1}", ha="left", va="center")

    # GRU 单元外框
    ax.add_patch(plt.Rectangle((0.30, 0.12), 0.46, 0.78, fill=False, linewidth=2))
    ax.text(0.53, 0.86, "GRU Cell", ha="center", va="center")

    # a0/a1/a2 计算框（同一列对齐）
    box_w, box_h = 0.24, 0.12
    box_x = 0.41
    ax.add_patch(plt.Rectangle((box_x, 0.70), box_w, box_h, fill=False, linewidth=1.5))
    ax.text(box_x + box_w / 2, 0.76, "a0 = σ(·)\nUpdate", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((box_x, 0.52), box_w, box_h, fill=False, linewidth=1.5))
    ax.text(box_x + box_w / 2, 0.58, "a1 = σ(·)\nReset", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((box_x, 0.34), box_w, box_h, fill=False, linewidth=1.5))
    ax.text(box_x + box_w / 2, 0.40, "a2 = tanh(·)\nCandidate", ha="center", va="center", fontsize=9)

    # 输出叠加计算框（同一列）
    ax.add_patch(plt.Rectangle((box_x, 0.16), box_w, box_h, fill=False, linewidth=1.5))
    ax.text(box_x + box_w / 2, 0.22, "y_t = (1-a0)⊙y_{t-1}\n+ a0⊙a2", ha="center", va="center", fontsize=8)

    # 输出层
    ax.add_patch(plt.Rectangle((0.82, 0.55), 0.15, 0.25, fill=False, linewidth=2))
    ax.text(0.895, 0.675, "Output\n(y_t)", ha="center", va="center")

    # 反向传播区块（更形象）
    back_y = -0.15
    back_h = 0.18
    ax.add_patch(plt.Rectangle((0.30, back_y), 0.46, back_h, fill=False, linewidth=1.5, linestyle="--"))
    ax.text(0.53, back_y + back_h + 0.005, "Backward Path", ha="center", va="bottom",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    # 反向传播节点
    b_w, b_h = 0.12, 0.06
    b_y = back_y + 0.05
    b_xs = [0.32, 0.50, 0.68]
    labels = ["δa2", "δa0", "δa1"]
    for bx, label in zip(b_xs, labels):
        ax.add_patch(plt.Rectangle((bx, b_y), b_w, b_h, fill=False, linewidth=1.2, linestyle="--"))
        ax.text(bx + b_w / 2, b_y + b_h / 2, label, ha="center", va="center", fontsize=9)

    # 梯度汇总输出
    ax.add_patch(plt.Rectangle((0.40, back_y + 0.13), 0.42, 0.06, fill=False, linewidth=1.3, linestyle="--"))
    ax.text(0.61, back_y + 0.160, "grad_w / grad_v\n grad_x / grad_y_prev",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))

    # 连接箭头：输入与隐状态进入 GRU 单元
    ax.annotate("", xy=(0.30, 0.675), xytext=(0.23, 0.675),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.annotate("", xy=(0.30, 0.86), xytext=(0.12, 0.9),
                arrowprops=dict(arrowstyle="->", linewidth=2))

    # 输入到 a0/a1/a2（水平线）
    ax.annotate("", xy=(box_x, 0.76), xytext=(0.30, 0.76),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(box_x, 0.58), xytext=(0.30, 0.58),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(box_x, 0.40), xytext=(0.30, 0.40),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))

    # a0/a2 到输出叠加（垂直线）
    ax.annotate("", xy=(box_x + box_w / 2, 0.28), xytext=(box_x + box_w / 2, 0.34),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(box_x + box_w / 2, 0.28), xytext=(box_x + box_w / 2, 0.70),
                arrowprops=dict(arrowstyle="->", linewidth=1.5))

    # 连接到输出（水平箭头）
    ax.annotate("", xy=(0.82, 0.22), xytext=(box_x + box_w, 0.22),
                arrowprops=dict(arrowstyle="->", linewidth=2))

    # 反向传播箭头（虚线）
    ax.annotate("", xy=(0.76, 0.56), xytext=(0.76, back_y + back_h),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, linestyle="--"))
    ax.annotate("", xy=(0.30, back_y + back_h), xytext=(0.30, 0.56),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, linestyle="--"))

    # δa2/δa0/δa1 到梯度汇总
    for bx in b_xs:
        ax.annotate("", xy=(0.61, back_y + 0.13), xytext=(bx + b_w / 2, b_y + b_h),
                    arrowprops=dict(arrowstyle="->", linewidth=1.2, linestyle="--"))

    # 反向传播闭合：梯度回到输入 x_t / y_{t-1}
    ax.annotate("", xy=(0.23, 0.675), xytext=(0.61, back_y + 0.19),
                arrowprops=dict(arrowstyle="->", linewidth=1.2, linestyle="--"))
    ax.annotate("", xy=(0.12, 0.9), xytext=(0.61, back_y + 0.19),
                arrowprops=dict(arrowstyle="->", linewidth=1.2, linestyle="--"))

    plt.title("GRU 前向传播结构示意图（a0/a1/a2 与输出叠加）")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_gru_diagram()
