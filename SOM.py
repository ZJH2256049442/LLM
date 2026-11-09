import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Windows 常见中文字体；如果本机没有，可改为已有中文字体名称。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class SOM:
    """二维网格 Self-Organizing Map，自组织映射。"""

    def __init__(
        self,
        grid_rows=6,
        grid_cols=6,
        n_epochs=100,
        learning_rate=0.5,
        radius=None,
        random_state=42,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_epochs = n_epochs
        self.learning_rate0 = learning_rate
        self.radius0 = radius
        self.random_state = random_state
        self.weights = None
        self.quantization_error_ = []

    def _grid_coordinates(self):
        rows, cols = np.indices((self.grid_rows, self.grid_cols))
        return np.stack([rows, cols], axis=-1).astype(float)

    def _find_bmu(self, x):
        """BMU = Best Matching Unit，最佳匹配单元。"""
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")
        if len(X) == 0:
            raise ValueError("X 不能是空数组")

        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        grid = self._grid_coordinates()

        # 每个神经元的权重是一个二维向量：[特征1，特征2]
        low = X.min(axis=0)
        high = X.max(axis=0)
        self.weights = rng.uniform(
            low,
            high,
            size=(self.grid_rows, self.grid_cols, n_features),
        )

        radius0 = (
            self.radius0
            if self.radius0 is not None
            else max(self.grid_rows, self.grid_cols) / 2
        )

        for epoch in range(self.n_epochs):
            # 学习率和邻域半径逐渐减小
            decay = np.exp(-epoch / max(self.n_epochs / 2, 1))
            learning_rate = self.learning_rate0 * decay
            radius = max(0.5, radius0 * decay)

            # 每一轮随机输入样本
            order = rng.permutation(len(X))
            for index in order:
                x = X[index]
                bmu = self._find_bmu(x)

                # 计算每个 SOM 神经元到 BMU 的网格距离平方
                bmu_position = np.array(bmu, dtype=float)
                grid_distance2 = np.sum(
                    (grid - bmu_position) ** 2,
                    axis=2,
                )

                # 高斯邻域函数：BMU 影响最大，越远影响越小
                neighborhood = np.exp(
                    -grid_distance2 / (2 * radius**2)
                )[..., None]

                # SOM 核心更新公式
                self.weights += (
                    learning_rate
                    * neighborhood
                    * (x - self.weights)
                )

            self.quantization_error_.append(
                self.quantization_error(X)
            )

        return self

    def winner(self, X):
        """返回每个样本对应的 BMU 网格坐标，结果为 [行号，列号]。"""
        X = np.asarray(X, dtype=float)
        return np.array([self._find_bmu(x) for x in X])

    def quantization_error(self, X):
        X = np.asarray(X, dtype=float)
        errors = []
        for x in X:
            bmu = self._find_bmu(x)
            errors.append(np.linalg.norm(x - self.weights[bmu]))
        return float(np.mean(errors))

    def u_matrix(self):
        """计算每个神经元与上下左右邻居的平均权重距离。"""
        u = np.zeros((self.grid_rows, self.grid_cols))

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        neighbors.append(
                            np.linalg.norm(
                                self.weights[row, col]
                                - self.weights[nr, nc]
                            )
                        )
                u[row, col] = np.mean(neighbors)
        return u


def make_data(random_state=7):
    """生成四组带标签的二维演示数据。

    特征 1 可以理解为第一项测量值，特征 2 可以理解为第二项测量值。
    如果换成真实数据，应把下面两个特征名改成实际含义，例如：
    "平均负荷" 和 "峰值负荷"。
    """
    rng = np.random.default_rng(random_state)

    centers = np.array([
        [-2.5, 2.5],
        [2.5, 2.5],
        [-2.5, -2.5],
        [2.5, -2.5],
    ])

    data_parts = []
    label_parts = []
    for group_id, center in enumerate(centers):
        points = rng.normal(center, 0.45, size=(40, 2))
        data_parts.append(points)
        label_parts.append(np.full(len(points), group_id, dtype=int))

    X = np.vstack(data_parts)
    labels = np.concatenate(label_parts)
    return X, labels


def add_legend(ax, colors, names):
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=7,
            label=name,
        )
        for color, name in zip(colors, names)
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)


def draw_bmu_grid(ax, grid_rows, grid_cols):
    """在右图上画出 SOM 网格和四个投影区域边界。"""
    ax.set_xticks(range(grid_cols))
    ax.set_yticks(range(grid_rows))
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(-0.5, grid_rows - 0.5)
    ax.grid(alpha=0.28, linewidth=0.8)

    # 分界线放在左侧最后一个单元与右侧第一个单元之间，
    # 不穿过任何 BMU 的中心。例如 7×7 网格的边界是 3.5。
    col_boundary = (grid_cols - 1) // 2 + 0.5
    row_boundary = (grid_rows - 1) // 2 + 0.5
    ax.axvline(col_boundary, color="black", linewidth=1.2, alpha=0.75)
    ax.axhline(row_boundary, color="black", linewidth=1.2, alpha=0.75)


def group_by_bmu_position(bmus, grid_rows, grid_cols):
    """按照绘图坐标中的 BMU 位置划分四个投影区域。

    这里的 bmus[:, 0] 是绘图用的纵坐标：0 在底部，数值越大越靠上。
    因此返回顺序为：左下、右下、左上、右上。左右图共用这组标签。
    """
    bmus = np.asarray(bmus, dtype=int)
    row_split = grid_rows / 2
    col_split = grid_cols / 2

    is_top = bmus[:, 0] >= row_split
    is_right = bmus[:, 1] >= col_split
    # 编号顺序：左下、右下、左上、右上
    return is_top.astype(int) * 2 + is_right.astype(int)


def main():
    # 在本演示中：
    # 特征 1 = 横坐标 x
    # 特征 2 = 纵坐标 y
    # 真实项目中请改成你的实际特征名称。
    feature_names = ("特征 1：测量值 X", "特征 2：测量值 Y")

    X, _true_labels = make_data()

    som = SOM(
        grid_rows=7,
        grid_cols=7,
        n_epochs=120,
        learning_rate=0.5,
        radius=3.5,
        random_state=12,
    )
    som.fit(X)

    # SOM 内部通常把第 0 行放在矩阵最上方；绘图时改成数学坐标习惯：
    # 第 0 行在下方，行号越大越靠上。左右图和中图都使用这个方向。
    bmus_internal = som.winner(X)
    bmus = bmus_internal.copy()
    bmus[:, 0] = som.grid_rows - 1 - bmus_internal[:, 0]

    projection_labels = group_by_bmu_position(
        bmus,
        grid_rows=som.grid_rows,
        grid_cols=som.grid_cols,
    )
    u_matrix = som.u_matrix()
    u_matrix_display = np.flipud(u_matrix)

    # 左图和右图共用这一组颜色：同一个样本在两张图中的颜色完全一致。
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    projection_counts = np.bincount(projection_labels, minlength=4)
    group_names = [
        f"左下区（n={projection_counts[0]}）",
        f"右下区（n={projection_counts[1]}）",
        f"左上区（n={projection_counts[2]}）",
        f"右上区（n={projection_counts[3]}）",
    ]

    # 画布加宽，并使用 constrained_layout 自动给标题、颜色条和图例留空间。
    # 原来的画布太窄，导致右图图例被裁切，颜色条文字挤到右图附近。
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6.5),
        dpi=120,
        constrained_layout=True,
    )

    # ==================== 左图 ====================
    # 灰色：原始样本；红色：SOM 神经元；蓝线：SOM 网格连接
    # 左图：原始样本也按照它们实际 BMU 所在的投影区域着色。
    # 因此左图和右图中，同一个样本始终使用同一种颜色。
    for group_id, (color, group_name) in enumerate(zip(colors, group_names)):
        mask = projection_labels == group_id
        axes[0].scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            s=25,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.3,
            label=group_name,
        )

    # 用与右图相同的纵坐标方向显示 SOM 神经元权重：行 0 在下方。
    weights_display = np.flipud(som.weights)

    # 左图中的 49 个红点就是 7×7 个 SOM 神经元。
    weight_points = weights_display.reshape(-1, 2)
    axes[0].scatter(
        weight_points[:, 0],
        weight_points[:, 1],
        c="red",
        edgecolors="black",
        linewidths=0.3,
        s=45,
        label="SOM 神经元权重",
        zorder=4,
    )

    for row in range(som.grid_rows):
        axes[0].plot(
            weights_display[row, :, 0],
            weights_display[row, :, 1],
            color="steelblue",
            linewidth=0.8,
            zorder=2,
        )
    for col in range(som.grid_cols):
        axes[0].plot(
            weights_display[:, col, 0],
            weights_display[:, col, 1],
            color="steelblue",
            linewidth=0.8,
            zorder=2,
        )

    axes[0].set_title("左图：特征空间中的 SOM 神经元（49 个）")
    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].legend(fontsize=7, loc="best", title="按 BMU 投影区域")
    axes[0].grid(alpha=0.25)
    axes[0].set_aspect("equal", adjustable="box")

    # ==================== 中图 ====================
    image = axes[1].imshow(
        u_matrix_display,
        cmap="viridis",
        origin="lower",
    )
    axes[1].set_title("中图：U-Matrix（邻居差异，行 0 在下）")
    axes[1].set_xlabel("SOM 网格列号")
    axes[1].set_ylabel("SOM 网格行号")
    axes[1].set_xticks(range(som.grid_cols))
    axes[1].set_yticks(range(som.grid_rows))
    # 颜色条单独绑定到中图，避免颜色条标签和右图的 y 轴标签互相挤压。
    fig.colorbar(
        image,
        ax=axes[1],
        fraction=0.046,
        pad=0.08,
        label="相邻神经元权重距离",
    )

    # ==================== 右图 ====================
    # 先画出全部 49 个 BMU 神经元位置，即使某些神经元没有样本投影，
    # 右图仍然能看到完整的 7×7 神经元网格。
    neuron_cols, neuron_rows = np.meshgrid(
        np.arange(som.grid_cols),
        np.arange(som.grid_rows),
    )
    axes[2].scatter(
        neuron_cols.ravel(),
        neuron_rows.ravel(),
        s=105,
        facecolors="none",
        edgecolors="dimgray",
        linewidths=0.8,
        marker="s",
        label="SOM 神经元（49 个）",
        zorder=2,
    )

    # 右图每个圆点都对应左图的一个样本；坐标严格使用 BMU 的整数位置，
    # 不添加随机抖动。多个样本落在同一 BMU 时，它们会自然重叠，
    # 这正表示“它们被 SOM 映射到同一个神经元”。
    for group_id, (color, group_name) in enumerate(zip(colors, group_names)):
        mask = projection_labels == group_id
        axes[2].scatter(
            bmus[mask, 1],
            bmus[mask, 0],
            c=color,
            s=44,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.5,
            label=group_name,
            zorder=3,
        )

    draw_bmu_grid(axes[2], som.grid_rows, som.grid_cols)
    axes[2].set_title("右图：每个样本实际投影到的 BMU（49 个神经元）")
    axes[2].set_xlabel("SOM 网格列号（左 → 右）")
    axes[2].set_ylabel("SOM 网格行号（下 → 上）")
    axes[2].legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        borderaxespad=0,
        frameon=True,
        title="颜色 = 左右图同一批样本",
        title_fontsize=8,
    )

    # 让总标题和底部说明位于布局区域内，不再使用超出画布的坐标。
    fig.suptitle(
        "SOM 自组织映射：特征空间、邻居差异与样本网格位置",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.015,
        "读图方法：左图任取一个样本 → 沿同颜色找到右图；右图坐标就是它实际投影到的 BMU。左图和右图均为同一套 7×7=49 个 SOM 神经元，颜色样本数量完全相同",
        ha="center",
        fontsize=9,
    )

    output_path = Path(__file__).with_name("som_result_labeled.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"最终量化误差：{som.quantization_error_[-1]:.4f}")
    print(f"图片已保存到：{output_path}")
    plt.show()


if __name__ == "__main__":
    main()