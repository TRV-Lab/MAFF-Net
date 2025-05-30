import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from ...model_utils.se_module import SELayer_vfe


class interRAL(nn.Module):
    def __init__(self, channels):
        super(interRAL, self).__init__()
        self.linear = nn.Linear(10, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1)  # b, n, c
        y_k = self.k_conv(y).permute(2, 1, 0)  # b, c, n
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0)  # b, c, n
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x


class RadarPillarVFE_TJ(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_VR", "USE_RANGE", "USE_SNR", "USE_ALPHA", "USE_BETA"]
        # hasattr()函数用于判断对象是否包含对应的属性  all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False
        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_vr = self.model_cfg.USE_VR
            self.use_range = self.model_cfg.USE_RANGE
            self.use_SNR = self.model_cfg.USE_SNR
            self.use_alpha = self.model_cfg.USE_ALPHA
            self.use_beta = self.model_cfg.USE_BETA
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'v_r', 'range', 'SNR', 'alpha', 'beta']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.vr_ind = self.available_features.index('v_r')
        self.range_ind = self.available_features.index('range')
        self.SNR_ind = self.available_features.index('SNR')
        self.alpha_ind = self.available_features.index('alpha')
        self.beta_ind = self.available_features.index('beta')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_SNR:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.SNR_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_range:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.range_ind)

        if self.use_alpha:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.alpha_ind)  # adding  time channel to the indexes

        if self.use_beta:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.beta_ind)

        ## LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        self.se = SELayer_vfe(num_point_features)
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)
        # features = self.se(features.permute(0, 2, 1)).permute(0, 2, 1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


# 完善voxel内点的数据并使用简化版的pointnet网络对每个pillar中的数据进行特征提取
class RadarPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        num_point_features = 0
        self.use_norm = self.model_cfg.USE_NORM  # whether to use batchnorm in the PFNLayer
        self.use_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.selected_indexes = []

        ## check if config has the correct params, if not, throw exception
        radar_config_params = ["USE_RCS", "USE_VR", "USE_VR_COMP", "USE_TIME", "USE_ELEVATION"]
        # hasattr()函数用于判断对象是否包含对应的属性  all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False
        if all(hasattr(self.model_cfg, attr) for attr in radar_config_params):
            self.use_RCS = self.model_cfg.USE_RCS
            self.use_vr = self.model_cfg.USE_VR
            self.use_vr_comp = self.model_cfg.USE_VR_COMP
            self.use_time = self.model_cfg.USE_TIME
            self.use_elevation = self.model_cfg.USE_ELEVATION

        else:
            raise Exception("config does not have the right parameters, please use a radar config")

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

        num_point_features += 6  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes

        ## LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, :, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        points_mean = orig_xyz.sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, :, 0] = voxel_features[:, :, self.x_ind] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, self.y_ind] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, self.z_ind] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, :, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_cluster, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 2, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            # 根据论文中，这是是简化版pointnet网络层的初始化
            # 论文中使用的是 1x1 的卷积层完成这里的升维操作（理论上使用卷积的计算速度会更快）
            # 输入的通道数是刚刚经过数据增强过后的点云特征，每个点云有13个特征，
            # 输出的通道数是64
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            # 一维BN层
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000
        # self.catt = PointCloudTransformer()

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            # x的维度由（M, 32, 10）升维成了（M, 32, 64）
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        # BatchNorm1d层:(M, 64, 32) --> (M, 32, 64)
        # （pillars,num_point,channel）->(pillars,channel,num_points)
        # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # 完成pointnet的最大池化操作，找出每个pillar中最能代表该pillar的点
        # x_max shape ：（M, 1, 64）
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # x_max = self.catt(x)

        if self.last_vfe:
            # 返回经过简化版pointnet处理pillar的结果
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated