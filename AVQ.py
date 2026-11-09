import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class AVQ:
    """Kohonen Adaptive Vector Quantisation (论文中的无监督 AVQ).

    论文算法对应关系：
    - 公式(8)：选择最小距离的获胜神经元，可选 bias 项
    - 公式(9)：只更新获胜神经元
    - 公式(10)：学习率指数衰减，并设最小学习率
    - 公式(11)：可用误差改善情况作为停止条件
    """

    def __init__(
        self,
        n_clusters=4,
        n_epochs=100,
        alpha0=0.5,
        alpha_min=1e-5,
        tau_alpha=30.0,
        bias=0.0,
        init="random",
        random_state=42,
        tol=1e-5,
        patience=5,
    ):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.alpha0 = alpha0
        self.alpha_min = alpha_min
        self.tau_alpha = tau_alpha
        self.bias = bias
        self.init = init
        self.random_state = random_state
        self.tol = tol
        self.patience = patience

        self.centers = None
        self.labels_ = None
        self.inertia_history_ = []
        self.n_iter_ = 0

    def _initialize_centers(self, X, rng):
        """初始化码字/质心。

        paper_exact：严格按照论文，将所有权重初始化为 0.5。
        random：从样本中随机选择初始权重，通常更适合一般数据。
        """
        if self.init == "paper_exact":
            return np.full((self.n_clusters, X.shape[1]), 0.5, dtype=float)

        if self.init == "random":
            indices = rng.choice(len(X), self.n_clusters, replace=False)
            return X[indices].astype(float).copy()

        raise ValueError("init 只能是 'paper_exact' 或 'random'")

    def _winner(self, x, centers, population):
        """根据论文公式(8)寻找获胜神经元。"""
        distances = np.linalg.norm(centers - x, axis=1)

        # 论文中的 bias：distance + nu * (N_j / N)。
        # 无偏置时 bias=0，退化为普通最近质心规则。
        scores = distances + self.bias * population / max(population.sum(), 1)
        return int(np.argmin(scores))

    def _assign_labels(self, X, centers):
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def fit(self, X):
        """训练 AVQ；X 的形状必须是 (样本数, 特征数)。"""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")
        if len(X) < self.n_clusters:
            raise ValueError("样本数不能少于聚类数")
        if self.alpha0 <= 0 or self.alpha_min <= 0:
            raise ValueError("alpha0 和 alpha_min 必须大于 0")
        if self.tau_alpha <= 0:
            raise ValueError("tau_alpha 必须大于 0")

        rng = np.random.default_rng(self.random_state)
        centers = self._initialize_centers(X, rng)
        previous_error = np.inf
        no_improvement_epochs = 0
        self.inertia_history_ = []

        for epoch in range(self.n_epochs):
            # 论文要求：每个 epoch 随机呈现训练向量。
            order = rng.permutation(len(X))
            population = np.zeros(self.n_clusters, dtype=int)
            alpha = max(
                self.alpha_min,
                self.alpha0 * np.exp(-epoch / self.tau_alpha),
            )

            for index in order:
                x = X[index]
                winner = self._winner(x, centers, population)

                # 论文公式(9)：只移动获胜中心，其余中心保持不变。
                centers[winner] += alpha * (x - centers[winner])
                population[winner] += 1

            labels = self._assign_labels(X, centers)
            squared_distances = np.sum((X - centers[labels]) ** 2, axis=1)
            error = float(np.mean(squared_distances))
            self.inertia_history_.append(error)

            if np.isfinite(previous_error):
                improvement = abs(previous_error - error) / max(abs(previous_error), 1e-12)
                if improvement < self.tol:
                    no_improvement_epochs += 1
                else:
                    no_improvement_epochs = 0

                if no_improvement_epochs >= self.patience:
                    self.n_iter_ = epoch + 1
                    break

            previous_error = error
            self.n_iter_ = epoch + 1
        else:
            self.n_iter_ = self.n_epochs

        self.centers = centers
        self.labels_ = self._assign_labels(X, centers)
        return self

    def predict(self, X):
        if self.centers is None:
            raise RuntimeError("请先调用 fit(X)")
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.centers)


def make_demo_data(random_state=7):
    """生成四组二维演示数据。"""
    rng = np.random.default_rng(random_state)
    true_centers = np.array([
        [-2.4, 2.8],
        [2.2, 3.3],
        [-2.2, -2.4],
        [2.4, -2.3],
    ])
    covariances = [
        [[0.35, 0.0], [0.0, 0.30]],
        [[0.35, 0.0], [0.0, 0.40]],
        [[0.40, 0.0], [0.0, 0.30]],
        [[0.35, 0.0], [0.0, 0.35]],
    ]
    groups = [
        rng.multivariate_normal(center, cov, size=20)
        for center, cov in zip(true_centers, covariances)
    ]
    return np.vstack(groups)


def plot_result(X, model, output_name="avq_result.png"):
    labels = model.labels_
    colors = ["red", "limegreen", "blue", "magenta"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="Set1", s=45)
    axes[0].scatter(
        model.centers[:, 0], model.centers[:, 1],
        marker="+", c="black", s=500, linewidths=2.5,
    )
    axes[0].set_title("AVQ 聚类结果")
    axes[0].set_xlabel("特征 1")
    axes[0].set_ylabel("特征 2")
    axes[0].grid(alpha=0.25)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].plot(
        np.arange(1, len(model.inertia_history_) + 1),
        model.inertia_history_,
        color="darkorange",
    )
    axes[1].set_title("每个 epoch 的量化误差")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("J")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    output_path = Path(__file__).with_name(output_name)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"AVQ 在 {model.n_iter_} 个 epoch 后停止")
    print("最终码字/质心：")
    print(np.round(model.centers, 3))
    print(f"图片已保存到：{output_path}")
    plt.show()


def main():
    X = make_demo_data()

    model = AVQ(
        n_clusters=4,
        n_epochs=150,
        alpha0=0.5,
        alpha_min=1e-5,
        tau_alpha=35,
        bias=0.0,
        # 使用 random 更容易在二维演示数据上快速得到四个分离簇。
        # 若要严格使用论文公式(初始权重全为 0.5)，改成 paper_exact。
        init="random",
        random_state=12,
    )
    model.fit(X)
    plot_result(X, model)


if __name__ == "__main__":
    main()