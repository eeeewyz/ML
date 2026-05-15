import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import numpy as np


def test():
    """
    使用 scikit-learn 的 GaussianMixture 演示 GMM 的两个基本用途：
    1. 聚类：根据数据点的分布，将样本划分到不同的高斯成分中；
    2. 数据增强：从训练好的混合高斯分布中随机采样，生成新的模拟数据。

    这里使用的是二维模拟数据，方便在散点图中直观看到聚类效果。
    """

    # 固定随机种子，保证每次运行时生成的数据和模型初始化结果尽量一致，
    # 这样便于调试、复现实验结果，也方便之后和手写 GMM-EM 算法进行比较。
    np.random.seed(0)

    # 使用 make_blobs 生成人工数据集。
    # n_samples=1000 表示生成 1000 个样本点；
    # n_features=2 表示每个样本有两个特征，也就是二维坐标；
    # centers 指定两个真实簇中心，分别位于 (0, 1.5) 和 (1, 0.5) 附近；
    # cluster_std 控制每个簇的离散程度，数值越大，数据点越分散；
    # random_state 固定 make_blobs 自己的随机状态，进一步保证实验可复现。
    x, y = make_blobs(n_samples=1000,
                      n_features=2,
                      centers=[(0, 1.5), [1, 0.5]],
                      cluster_std=[0.4, 0.5],
                      random_state=42)

    # 创建 GMM 模型。
    # n_components=3 表示模型假设数据由 3 个高斯分布混合而成。
    # 注意：这里真实数据是由 2 个中心生成的，但模型设为 3 个成分，
    # 可以观察 GMM 是否会把某个真实簇进一步拆分成更细的高斯成分。
    estimator = GaussianMixture(n_components=3, random_state=42)

    # 训练模型。
    # fit(x) 会使用 EM 算法估计每个高斯成分的均值、协方差和混合权重。
    # 训练结束后，estimator 内部就保存了一个拟合当前数据分布的混合高斯模型。
    estimator.fit(x)

    # 1. 聚类用途：
    # predict(x) 会计算每个样本点最可能属于哪一个高斯成分，
    # 返回值 y_pred 是每个样本的聚类标签，例如 0、1、2。
    # 这些标签不代表真实类别名称，只代表 GMM 学到的第几个高斯成分。
    y_pred = estimator.predict(x)
    print(y_pred)

    # 2. 数据增强用途：
    # sample(3) 会从训练好的 GMM 分布中随机生成 3 个新样本。
    # 返回结果是一个二元组：
    # data[0] 是生成的新样本坐标；
    # data[1] 是这些新样本分别来自哪个高斯成分的标签。
    # 这体现了 GMM 不仅能做聚类，还能根据已有数据分布生成相似的新数据。
    data = estimator.sample(3)
    print(data)

    # 可视化聚类结果。
    # x[:, 0] 取所有样本的第一个特征作为横坐标；
    # x[:, 1] 取所有样本的第二个特征作为纵坐标；
    # c=y_pred 表示用 GMM 预测出的聚类标签给点着色，
    # 因此图中不同颜色代表模型认为的不同高斯成分。
    plt.grid()
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()


if __name__ == '__main__':
    # 当这个文件被直接运行时，执行 test() 函数；
    # 如果这个文件被其他 Python 文件 import，则不会自动运行。
    test()
