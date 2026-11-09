import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

# Windows 常见中文字体；如果电脑没有这些字体，删除 Microsoft YaHei 和 SimHei 即可。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class KMeans:
    """不依赖 sklearn 的简化版 K-means。"""

    def __init__(self, n_clusters=4, max_iter=100, random_state=12):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = None
        self.labels = None
        self.n_iter_ = 0

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)

        # 使用 k-means++ 选择初始质心，比完全随机初始化更稳定。
        centers = [X[rng.integers(len(X))].copy()]
        for _ in range(1, self.n_clusters):
            distances_to_centers = np.min(
                np.linalg.norm(
                    X[:, None, :] - np.array(centers)[None, :, :],
                    axis=2,
                ) ** 2,
                axis=1,
            )
            probabilities = distances_to_centers / distances_to_centers.sum()
            next_index = rng.choice(len(X), p=probabilities)
            centers.append(X[next_index].copy())
        centers = np.array(centers)

        for iteration in range(1, self.max_iter + 1):
            # distances[i, j] 表示第 i 个样本到第 j 个质心的距离
            distances = np.linalg.norm(
                X[:, None, :] - centers[None, :, :], axis=2
            )
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for cluster_id in range(self.n_clusters):
                members = X[labels == cluster_id]
                if len(members) > 0:
                    new_centers[cluster_id] = members.mean(axis=0)

            if np.allclose(centers, new_centers):
                centers = new_centers
                self.n_iter_ = iteration
                break

            centers = new_centers
        else:
            self.n_iter_ = self.max_iter

        # 收敛后重新分配一次，保证 labels 与最终 centers 对应
        distances = np.linalg.norm(
            X[:, None, :] - centers[None, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

        self.centers = centers
        self.labels = labels
        return self

    def predict(self, X):
        if self.centers is None:
            raise RuntimeError("请先调用 fit(X)，再调用 predict(X)。")

        distances = np.linalg.norm(
            X[:, None, :] - self.centers[None, :, :], axis=2
        )
        return np.argmin(distances, axis=1)


def make_data(random_state=7, points_per_cluster=18):
    """生成四组彼此分离的二维数据，并返回真实分组标签。"""
    rng = np.random.default_rng(random_state)

    # 固定的真实中心：左上、右上、左下、右下
    true_centers = np.array([
        [-2.4,  2.8],  # A：圆形
        [ 2.2,  3.3],  # B：菱形
        [-2.2, -2.4],  # C：三角形
        [ 2.4, -2.3],  # D：方形
    ])

    covariances = [
        [[0.42, 0.00], [0.00, 0.30]],
        [[0.36, 0.00], [0.00, 0.42]],
        [[0.42, 0.00], [0.00, 0.30]],
        [[0.38, 0.00], [0.00, 0.36]],
    ]

    data_parts = []
    true_labels_parts = []

    for group_id, (center, covariance) in enumerate(
        zip(true_centers, covariances)
    ):
        points = rng.multivariate_normal(
            mean=center,
            cov=covariance,
            size=points_per_cluster,
        )
        data_parts.append(points)
        true_labels_parts.append(
            np.full(points_per_cluster, group_id, dtype=int)
        )

    return (
        np.vstack(data_parts),
        np.concatenate(true_labels_parts),
        true_centers,
    )


def add_cluster_ellipse(ax, center, width, height, angle, color):
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=2.0,
        alpha=0.85,
        zorder=1,
    )
    ax.add_patch(ellipse)


def main():
    # 点的形状表示该点所在的原始数据组；颜色表示 K-means 的聚类结果。
    # 这样可以直观看出 K-means 是否把不同形状的数据混在一起。
    X, true_labels, _ = make_data()

    # 运行 K-means
    model = KMeans(n_clusters=4, max_iter=100, random_state=12)
    model.fit(X)

    # K-means 的簇编号本身没有固定含义，所以按空间位置重编号。
    top = sorted(
        [i for i in range(4) if model.centers[i, 1] >= 0],
        key=lambda i: model.centers[i, 0],
    )
    bottom = sorted(
        [i for i in range(4) if model.centers[i, 1] < 0],
        key=lambda i: model.centers[i, 0],
    )
    ordered_old_ids = top + bottom
    old_to_new = {
        old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)
    }

    # 这是 K-means 真正得到的类别标签，只是把编号改成 A/B/C/D 的顺序。
    cluster_labels = np.array(
        [old_to_new[old_id] for old_id in model.labels]
    )
    cluster_centers = np.array(
        [model.centers[old_id] for old_id in ordered_old_ids]
    )

    colors = ["red", "limegreen", "blue", "magenta"]
    markers = ["o", "D", "^", "s"]
    cluster_names = ["簇 A", "簇 B", "簇 C", "簇 D"]
    center_names = ["质心 a", "质心 b", "质心 c", "质心 d"]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)

    # 点的形状表示原始数据组；颜色表示 K-means 的聚类结果。
    # 若同一种形状出现多种颜色，说明 K-means 把它拆分了。
    for shape_id in range(4):
        for cluster_id in range(4):
            mask = (true_labels == shape_id) & (cluster_labels == cluster_id)
            points = X[mask]
            if len(points) == 0:
                continue

            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=95,
                marker=markers[shape_id],
                facecolors="white",
                edgecolors=colors[cluster_id],
                linewidths=1.5,
                zorder=3,
                label=(
                    f"{cluster_names[shape_id]}（K-means {cluster_names[cluster_id]}）"
                    if shape_id == cluster_id else None
                ),
            )

    # 椭圆只用于展示每个簇的大致范围，不是 K-means 的数学边界。
    ellipse_sizes = [
        (3.9, 2.5, 0),
        (4.0, 2.8, 0),
        (4.2, 2.4, 0),
        (4.1, 2.6, 0),
    ]
    for cluster_id, (width, height, angle) in enumerate(ellipse_sizes):
        add_cluster_ellipse(
            ax,
            cluster_centers[cluster_id],
            width,
            height,
            angle,
            colors[cluster_id],
        )

    # 绘制 K-means 计算出的质心。
    for cluster_id, (center, color) in enumerate(
        zip(cluster_centers, colors)
    ):
        ax.scatter(
            center[0],
            center[1],
            marker="+",
            s=850,
            c=color,
            linewidths=3.2,
            zorder=5,
        )
        ax.text(
            center[0] + 0.22,
            center[1] + 0.42,
            center_names[cluster_id],
            color=color,
            fontsize=14,
            weight="bold",
            zorder=6,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5,
            ),
        )

    label_positions = [
        (-2.4, 0.35),
        (2.2, 0.55),
        (-2.2, -4.25),
        (2.4, -4.15),
    ]
    for name, (x, y), color in zip(
        cluster_names, label_positions, colors
    ):
        ax.text(
            x, y, name,
            color=color,
            fontsize=14,
            weight="bold",
            ha="center",
        )

    ax.set_title("K-means 聚类结果示意图", fontsize=17, pad=12)
    ax.set_xlabel("特征 1", fontsize=12)
    ax.set_ylabel("特征 2", fontsize=12)
    ax.set_xlim(-4.7, 4.7)
    ax.set_ylim(-5.0, 5.2)
    ax.set_xticks(np.arange(-4, 5, 2))
    ax.set_yticks(np.arange(-4, 6, 2))
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    # 图例放到坐标轴外侧，避免遮挡左上方的簇 A 和质心 a。
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        title="样本形状",
    )

    # 为右侧图例预留空间。
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    output_path = Path(__file__).with_name("kmeans_cluster_result.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"K-means 在 {model.n_iter_} 次迭代后收敛")
    print("最终质心：")
    print(np.round(cluster_centers, 2))
    print(f"图片已保存到：{output_path}")
    plt.show()


if __name__ == "__main__":
    main()