import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        self.gnn_feature = 0
        if 'GNN_FEATURE' in config:
        # if config.GNN_FEATURE is not None:
            self.gnn_feature = config.GNN_FEATURE
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None) + self.gnn_feature

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        if 'points' in data_dict:
            data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['points']
            )
        else:
            data_dict['lidar_points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['lidar_points']
            )
            data_dict['radar_points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
                data_dict['radar_points']
            )
        data_dict['use_lead_xyz'] = use_lead_xyz

        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            max_sweeps = self.point_encoding_config.max_sweeps
            idx = self.src_feature_list.index('timestamp')
            dt = np.round(data_dict['points'][:, idx], 2)
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt)) - 1, max_sweeps - 1)]
            data_dict['points'] = data_dict['points'][dt <= max_dt]

        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        if isinstance(points, list):
            point_features = []
            for point in points:
                point_feature_list = [point[:, 0:3]]
                for x in self.used_feature_list:
                    if x in ['x', 'y', 'z']:
                        continue
                    idx = self.src_feature_list.index(x)
                    point_feature_list.append(point[:, idx:idx + 1])
                point_features.append(np.concatenate(point_feature_list, axis=1))
        else:
            point_feature_list = [points[:, 0:3]]
            for x in self.used_feature_list:
                if x in ['x', 'y', 'z']:
                    continue
                idx = self.src_feature_list.index(x)
                point_feature_list.append(points[:, idx:idx + 1])
            point_features = np.concatenate(point_feature_list, axis=1)

        return point_features, True
