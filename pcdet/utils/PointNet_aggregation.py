# ！/usr/bin/python3
# _*_coding: utf-8 _*_
#
# Copyright (C) 2024 - 2024 Caien Weng, Inc. All Rights Reserved
#
# @Time   : 2024/5/31 下午2:50
# @Author : Caien Weng

import torch
import torch.nn as nn


class PointNetPPFeatureAggregation(nn.Module):
    def __init__(self, num_features, num_neighbors):
        super(PointNetPPFeatureAggregation, self).__init__()
        self.num_features = num_features
        self.num_neighbors = num_neighbors

    def forward(self, x, points):
        """
        x: 输入特征张量，形状为 [batch_size, num_points, num_features]
        points: 输入点云数据张量，形状为 [batch_size, num_points, 3]
        """
        batch_size, num_points, _ = points.size()

        # 首先计算每个点之间的距离
        dists = torch.norm(points.unsqueeze(2) - points.unsqueeze(1), dim=-1)  # [batch_size, num_points, num_points]

        # 找到每个点的最近邻
        _, indices = torch.topk(-dists, self.num_neighbors, dim=-1)  # 负号表示取最小值

        # 使用索引从输入特征中获取相应的邻居特征
        neighbor_features = torch.gather(x.unsqueeze(2).expand(batch_size, num_points, num_points, self.num_features),
                                         2, indices.unsqueeze(3).expand(batch_size, num_points, self.num_neighbors,
                                                                        self.num_features))

        # 对邻居特征进行聚合操作
        aggregated_features, _ = torch.max(neighbor_features, dim=2)  # 在邻居维度上取最大值

        return aggregated_features


if __name__ == '__main__':
    # 使用示例
    batch_size = 32
    num_points = 1024
    num_features = 64
    num_neighbors = 32

    # 创建 PointNet++ 特征聚合模块
    feature_aggregation = PointNetPPFeatureAggregation(num_features, num_neighbors)

    # 生成示例数据
    points = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, num_features)

    # 调用特征聚合模块
    aggregated_features = feature_aggregation(features, points)

    print(aggregated_features.shape)  # 输出形状为 [batch_size, num_points, num_features]
