from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils
from sklearn.cluster import DBSCAN

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            # print(" points ",points.shape)
            # print(" voxel_output ", voxel_output)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            # voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            if isinstance(points, list):
                voxels = []
                coordinates = []
                num_points = []
                for point in points:
                    voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(point))
                    tv_voxels, tv_coordinates, tv_num_points = voxel_output
                    # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
                    voxels.append(tv_voxels.numpy())
                    coordinates.append(tv_coordinates.numpy())
                    num_points.append(tv_num_points.numpy())
            else:
                voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
                tv_voxels, tv_coordinates, tv_num_points = voxel_output
                # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
                voxels = tv_voxels.numpy()
                coordinates = tv_coordinates.numpy()
                num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        ls = []
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        # print(" Mask 前的 points ", data_dict['points'].shape)
        if data_dict.get('points', None) is not None:
            if isinstance(data_dict['points'], list):
                for i in range(len(data_dict['points'])):
                    mask = common_utils.mask_points_by_range(data_dict['points'][i], self.point_cloud_range)
                    data_dict['points'][i] = data_dict['points'][i][mask]
            else:
                mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
                data_dict['points'] = data_dict['points'][mask]
        else:
            lidar_mask = common_utils.mask_points_by_range(data_dict['lidar_points'], self.point_cloud_range)
            radar_mask = common_utils.mask_points_by_range(data_dict['radar_points'], self.point_cloud_range)
            data_dict['lidar_points'] = data_dict['lidar_points'][lidar_mask]
            data_dict['radar_points'] = data_dict['radar_points'][radar_mask]
        # print(" Mask 后的 points ", data_dict['points'].shape)
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            # if len(data_dict['gt_boxes']) == 0:
            #     ls.append(data_dict['frame_id'])
            # print("gt_boxes:",data_dict['gt_boxes'],"  ",data_dict['frame_id'])
            # print("frame_id:",data_dict['frame_id'])
            # print(ls)
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            if 'points' in data_dict:
                points = data_dict['points']
                if isinstance(points, list):
                    for i in range(len(points)):
                        shuffle_idx = np.random.permutation(points[i].shape[0])
                        points[i] = points[i][shuffle_idx]
                        data_dict['points'][i] = points[i]
                else:
                    shuffle_idx = np.random.permutation(points.shape[0])
                    points = points[shuffle_idx]
                    data_dict['points'] = points
            else:
                lidar_points = data_dict['lidar_points']
                radar_points = data_dict['radar_points']
                lidar_shuffle_idx = np.random.permutation(lidar_points.shape[0])
                radar_shuffle_idx = np.random.permutation(radar_points.shape[0])
                lidar_points = lidar_points[lidar_shuffle_idx]
                radar_points = radar_points[radar_shuffle_idx]
                data_dict['lidar_points'] = lidar_points
                data_dict['radar_points'] = radar_points
        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)
        # print("self.point_cloud_range ",self.point_cloud_range)
        # print("config.VOXEL_SIZE ",config.VOXEL_SIZE)
        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
        if 'points' in data_dict:
            points = data_dict['points']
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output

            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        if 'points' in data_dict:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates  # ***********
            # print("points ",points.shape)
            # print("voxels ", voxels.shape)
            # print("coordinates ", coordinates[:,0])
            data_dict['voxel_num_points'] = num_points
            # print("data_dict ",data_dict)
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        if isinstance(data_dict["camera_imgs"], list):
            data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        else:
            data_dict["camera_imgs"] = [compose(data_dict["camera_imgs"])]
        return data_dict

    def image_calibrate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        imgs = data_dict["camera_imgs"]
        transforms = []
        if isinstance(imgs, list):
            for img_process_info in img_process_infos:
                resize, crop, flip, rotate = img_process_info

                rotation = torch.eye(2)
                translation = torch.zeros(2)
                # post-homography transformation
                rotation *= resize
                translation -= torch.Tensor(crop[:2])
                if flip:
                    A = torch.Tensor([[-1, 0], [0, 1]])
                    b = torch.Tensor([crop[2] - crop[0], 0])
                    rotation = A.matmul(rotation)
                    translation = A.matmul(translation) + b
                theta = rotate / 180 * np.pi
                A = torch.Tensor(
                    [
                        [np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)],
                    ]
                )
                b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
                b = A.matmul(-b) + b
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
                transform = torch.eye(4)
                transform[:2, :2] = rotation
                transform[:2, 3] = translation
                transforms.append(transform.numpy())
        else:
            resize, crop, flip, rotate = img_process_infos

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms = [transform.numpy()]
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
