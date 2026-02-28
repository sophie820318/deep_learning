import matplotlib.pyplot as plt


def plot_lstm_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    # 输入层
    ax.add_patch(plt.Rectangle((0.05, 0.55), 0.18, 0.25, fill=False, linewidth=2))
    ax.text(0.14, 0.675, "Input\n(x_t)", ha="center", va="center")

    # 上一时刻隐状态与记忆单元
    ax.text(0.05, 0.9, "h_{t-1}", ha="left", va="center")
    ax.text(0.05, 0.1, "c_{t-1}", ha="left", va="center")

    # LSTM 单元外框
    ax.add_patch(plt.Rectangle((0.28, 0.2), 0.44, 0.7, fill=False, linewidth=2))
    ax.text(0.5, 0.86, "LSTM Cell", ha="center", va="center")

    # 四个门
    gate_w, gate_h = 0.16, 0.12
    ax.add_patch(plt.Rectangle((0.32, 0.65), gate_w, gate_h, fill=False, linewidth=1.5))
    ax.text(0.4, 0.71, "Forget\nGate", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((0.52, 0.65), gate_w, gate_h, fill=False, linewidth=1.5))
    ax.text(0.6, 0.71, "Input\nGate", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((0.32, 0.45), gate_w, gate_h, fill=False, linewidth=1.5))
    ax.text(0.4, 0.51, "Candidate\nGate", ha="center", va="center", fontsize=9)

    ax.add_patch(plt.Rectangle((0.52, 0.45), gate_w, gate_h, fill=False, linewidth=1.5))
    ax.text(0.6, 0.51, "Output\nGate", ha="center", va="center", fontsize=9)

    # 状态输出
    ax.text(0.78, 0.9, "h_t", ha="left", va="center")
    ax.text(0.78, 0.1, "c_t", ha="left", va="center")

    # 输出层
    ax.add_patch(plt.Rectangle((0.78, 0.55), 0.18, 0.25, fill=False, linewidth=2))
    ax.text(0.87, 0.675, "Output\n(y_t)", ha="center", va="center")

    # 连接箭头：输入与隐状态进入 LSTM 单元
    ax.annotate("", xy=(0.28, 0.675), xytext=(0.23, 0.675),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.annotate("", xy=(0.36, 0.86), xytext=(0.12, 0.9),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.annotate("", xy=(0.36, 0.24), xytext=(0.12, 0.1),
                arrowprops=dict(arrowstyle="->", linewidth=2))

    # 连接箭头：LSTM 单元到输出与状态
    ax.annotate("", xy=(0.78, 0.675), xytext=(0.72, 0.675),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.annotate("", xy=(0.78, 0.9), xytext=(0.72, 0.86),
                arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.annotate("", xy=(0.78, 0.1), xytext=(0.72, 0.24),
                arrowprops=dict(arrowstyle="->", linewidth=2))

    plt.title("LSTM 模型结构示意图（含4个门）")
    plt.show()


if __name__ == "__main__":
    plot_lstm_diagram()
