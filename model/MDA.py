import torch
import numpy as np

np.random.seed(42)


class MDAProcess:

    def __init__(self):
        pass

    def k_mode_unfolding(self, tensor, mode):
        """实现张量的 k-mode 展开"""
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

    def compute_scatter_matrices(self, tensors, labels):
        # 计算类间散布 S_b 和类内散布 S_w
        pass

    def optimize_subspaces(self, tensors, labels, num_subspaces, max_iter=10):
        # 初始化投影矩阵
        projections = [np.eye(tensors.shape[i]) for i in range(len(tensors.shape))]
        for _ in range(max_iter):
            for mode in range(len(tensors.shape)):
                unfolded = self.k_mode_unfolding(tensors, mode)
                # 优化 mode 方向的投影矩阵
                pass
        return projections
