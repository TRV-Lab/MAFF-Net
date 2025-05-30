# ！/usr/bin/python3
# _*_coding: utf-8 _*_
#
# Copyright (C) 2024 - 2024 Caien Weng, Inc. All Rights Reserved
#
# @Time   : 2024/4/18 下午7:53
# @Author : Caien Weng
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class CQCA_cfa(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dbscan_map_w = self.model_cfg.DBSCAN_MAP_W
        self.dbscan_map_h = self.model_cfg.DBSCAN_MAP_H
        self.dbscan_feature = self.model_cfg.DBSCAN_FEATURE
        self.dbscan_v = self.model_cfg.DBSCAN_V
        self.dbscan_y = self.model_cfg.DBSCAN_Y
        self.resolution = self.model_cfg.RESOLUTION
        self.dbscan_eps = self.model_cfg.DBSCAN_EPS
        self.dbscan_sample = self.model_cfg.DBSCAN_SAMPLE
        self.point_x = self.model_cfg.POINTX
        self.point_y = self.model_cfg.POINTY
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def generate_map(self, points_xyv, eps, samples):
        points_xyv = points_xyv.cpu()
        dbscan = DBSCAN(eps=eps, min_samples=samples)
        labels = torch.tensor(dbscan.fit_predict(points_xyv))
        density_values = torch.zeros_like(labels, dtype=torch.float)
        for i, label in enumerate(labels):
            if label != -1:
                neighborhood_indices = torch.where(labels == label)[0]
                density_values[i] = len(neighborhood_indices)

        for i, density in enumerate(density_values):
            if density > 100 and labels[i] != -1:
                labels[i] = -1  # 标记为噪声
        points_density = torch.cat([points_xyv, density_values.reshape(-1, 1), labels.reshape(-1, 1)], dim=1)
        dbscan_points = points_density[torch.where(labels > 0)]
        # 初始化 BEV 地图
        resolution = self.resolution
        bev_map = torch.zeros((self.dbscan_map_w, self.dbscan_map_h, self.dbscan_feature),
                              dtype=torch.float32).to(points_xyv.device)
        # 计算点云在 BEV 地图中的位置
        bev_x = ((dbscan_points[:, 0]) / resolution).long() - 1
        bev_y = ((dbscan_points[:, 1] + self.dbscan_y) / resolution).long() - 1
        bev_map[bev_y, bev_x] = dbscan_points[:, 2:]
        return bev_map.permute(2, 0, 1).cuda(), labels

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']
        batch_size = batch_dict['batch_size']
        dbscan_map2 = []
        cluster_points = []
        for batch_idx in range(batch_size):
            batch_mask = points[:, 0] == batch_idx
            point = points[batch_mask]
            point_xyv = torch.cat([point[:, 1:3], point[:, self.dbscan_v].reshape(-1, 1)], dim=1)
            db_map, db_labels = self.generate_map(point_xyv, self.dbscan_eps, self.dbscan_sample)
            dbscan_map2.append(db_map)
            final_points = point[db_labels != -1]
            cluster_points.append(final_points)

        dbscan_map2 = torch.stack(dbscan_map2, 0)
        cluster_points = torch.cat(cluster_points, 0)
        dbscan_map2 = dbscan_map2.view(batch_size, 3, self.dbscan_map_w, self.dbscan_map_h)
        out = self.conv(dbscan_map2)
        batch_dict['spatial_features_img'] = out
        batch_dict['cluster_points'] = cluster_points
        return batch_dict


def show_result(kde_map_feature):
    import matplotlib.pyplot as plt
    import numpy as np
    intermediate_image = kde_map_feature[0].cpu().detach().numpy()
    # 将多个通道相加
    summed_image = np.sum(intermediate_image, axis=0)  # 沿着通道维度求和
    plt.figure(figsize=(10, 5))
    plt.imshow(summed_image, cmap='gray')
    plt.title('Density Image')
    plt.show()


class AdaptiveDBSCAN:
    def __init__(self, min_samples=5):
        self.min_samples = min_samples

    def fit(self, points):
        self.points = points
        self.X = points[:, 0:2]
        X = points[:, 0:2]
        self.labels = np.zeros(len(X))
        self.cluster_idx = 0
        self.kd_tree = KDTree(X)
        self.alte = 2.5
        for i in range(len(X)):
            if self.labels[i] == 0:  # unvisited point
                if self.expand_cluster(i):
                    self.cluster_idx += 1

    def expand_cluster(self, idx):
        neighbors = self.query_neighbors(idx)
        if len(neighbors) < self.min_samples:
            self.labels[idx] = -1  # noise point
            return False
        else:
            self.labels[idx] = self.cluster_idx
            for neighbor_idx in neighbors:
                if self.labels[neighbor_idx] == 0:  # unvisited point
                    self.labels[neighbor_idx] = self.cluster_idx
                    neighbor_neighbors = self.query_neighbors(neighbor_idx)
                    if len(neighbor_neighbors) >= self.min_samples:
                        neighbors = np.append(neighbors, neighbor_neighbors)
            return True

    def query_neighbors(self, idx):
        eps = self.alte * np.linalg.norm(self.X[idx]) * np.tan(np.pi / 180 * 0.75)
        if eps < self.alte * 0.2:
            eps = self.alte * 0.2
        neighbors = self.kd_tree.query_radius([self.X[idx]], r=eps)[0]
        return neighbors
