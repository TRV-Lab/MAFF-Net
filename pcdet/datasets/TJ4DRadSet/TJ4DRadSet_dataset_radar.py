import copy
import pickle

import numpy as np
import torch
import tqdm
from skimage import io

from pcdet.datasets.TJ4DRadSet import kitti_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import DatasetTemplate
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import os
import io as sysio
from sklearn.cluster import DBSCAN

envpath = '/home/tongji/anaconda3/envs/casa/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


class TJDatasetRadar(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        self.img_size = (484, 304)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading TJ4DRadSet dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for TJ4DRadSet dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        # print("idx",idx)
        # print(lidar_file)
        number_of_channels = 8  # [X, Y, Z, V_r, Range, Power, Alpha, Beta]
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)

        # replace the list values with statistical values; for x, y, z and time, use 0 and 1 as means and std to avoid normalization
        means = [0, 0, 0, 0, 0, 0, 0, 0]  # [X, Y, Z, V_r, Range, Power, Alpha, Beta]
        stds = [1, 1, 1, 1, 1, 1, 1, 1]  # [X, Y, Z, V_r, Range, Power, Alpha, Beta]

        # we then norm the channels
        points = (points - means) / stds
        # points[:,2] = 0
        # print("00000000000000")
        # print(points[0])
        return points
        # return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 7)

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        img = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        if self.training == True:
            fH, fW = self.camera_image_config.FINAL_DIM
            resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
            resize = np.random.uniform(*resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = newH - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            fH, fW = self.camera_image_config.FINAL_DIM
            resize_lim = self.camera_image_config.RESIZE_LIM_TEST
            resize = np.mean(resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = newH - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # reisze and crop image
        img = img.resize(resize_dims)
        crop_images = img.crop(crop)
        img_process_infos = [resize, crop, False, 0]

        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_camera_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = Image.open(str(img_file))
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def show_image(self, window_name, image, size_wh=None, location_xy=None):

        if size_wh is not None:
            cv2.namedWindow(window_name,
                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(window_name, *size_wh)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        if location_xy is not None:
            cv2.moveWindow(window_name, *location_xy)
        cv2.imshow(window_name, image)

    def show_result(self, process_dict, main_image):

        x_offset = self.img_size[0]
        y_offset = self.img_size[1]
        x_padding = 0
        y_padding = 28
        x_start = 0
        y_start = 100
        img_x = x_start
        img_y = y_start
        max_x = 1500
        row_idx = 0
        for key, value in process_dict.items():
            if key == 'main_image':
                image_jet = main_image
                self.show_image(
                    key, image_jet,
                    self.img_size, (img_x, img_y))
                img_x += x_offset + x_padding
                if (img_x + x_offset + x_padding) > max_x:
                    img_x = x_start
                    row_idx += 1
                img_y = y_start + row_idx * (y_offset + y_padding)
            else:
                image_jet = cv2.applyColorMap(
                    np.uint8(value / np.amax(value) * 255),
                    cv2.COLORMAP_JET)
                self.show_image(
                    key, image_jet,
                    self.img_size, (img_x, img_y))
                img_x += x_offset + x_padding
                if (img_x + x_offset + x_padding) > max_x:
                    img_x = x_start
                    row_idx += 1
                img_y = y_start + row_idx * (y_offset + y_padding)
        cv2.waitKey()

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        # print(calib_file)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def points_to_image(self, points, lidar2image):
        points_xyz1 = np.hstack([points, np.ones([points.shape[0], 1])])
        img_points_xyz = np.matmul(points_xyz1, np.transpose(lidar2image))
        img_points = img_points_xyz / img_points_xyz[:, [2]]
        return img_points

    def image_to_points(self, img_points, image2lidar):
        img_points_xyz1 = np.hstack([img_points, np.ones([img_points.shape[0], 1])])
        points_xy = np.matmul(img_points_xyz1, np.transpose(image2lidar))
        points_xyz = np.hstack([points_xy[:, 0:2], img_points])
        return points_xyz

    def get_points_in_image_with_rgb(self, input_dict):
        points = input_dict["points"]
        lidar2image = input_dict["lidar2image"][0]
        image = input_dict["images"]
        img_points = self.points_to_image(points[:, 0:3], lidar2image)
        filter = np.where(
            (img_points[:, 0] < 1936) & (img_points[:, 1] < 1216) & (img_points[:, 0] >= 0) & (img_points[:, 1] >= 0))
        img_points = img_points[filter]
        points = points[filter]
        rgb = image[np.int32(img_points[:, 1]), np.int32(img_points[:, 0]), ::-1].astype(np.float32)
        points_rgb = np.hstack([points, rgb])
        return points_rgb

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 5, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points  # points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / (
            'gt_database_TJ' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('TJ_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def print_str(value, *arg, sstream=None):

        if sstream is None:
            sstream = sysio.StringIO()
        sstream.truncate(0)
        sstream.seek(0)
        print(value, *arg, file=sstream)
        return sstream.getvalue()

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}
        # if current_class is None:
        current_class = [0, 1, 2, 5]
        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        # ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        evaluation_result = {}
        evaluation_result.update(kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, current_class))
        results = evaluation_result
        ap_result_str = self.print_str("Results: \n"
                                       f"Car: mAP3d: {results['entire_area']['Car_3d_all']}, mAPbev:{results['entire_area']['Car_bev_all']} \n"
                                       f"Pedestrian: mAP3d: {results['entire_area']['Pedestrian_3d_all']}, mAPbev: {results['entire_area']['Pedestrian_bev_all']} \n"
                                       f"Cyclist: mAP3d: {results['entire_area']['Cyclist_3d_all']}, mAPbev: {results['entire_area']['Cyclist_bev_all']} \n"
                                       f"Truck: mAP3d: {results['entire_area']['Truck_3d_all']} ,mAPbev: {results['entire_area']['Truck_bev_all']}\n"
                                       f"mAP3d: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all'] + results['entire_area']['Truck_3d_all']) / 4} \n"
                                       f"mAPbev: {(results['entire_area']['Car_bev_all'] + results['entire_area']['Pedestrian_bev_all'] + results['entire_area']['Cyclist_bev_all'] + results['entire_area']['Truck_bev_all']) / 4} \n"
                                       f"mAOS: {(results['entire_area']['Car_aos_all'] + results['entire_area']['Pedestrian_aos_all'] + results['entire_area']['Cyclist_aos_all'] + results['entire_area']['Truck_aos_all']) / 4} \n"
                                       )

        return ap_result_str, results['entire_area']

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        # print("111111111111111111")
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST',
                                             ['points'])  # 与kitti_dataset.yaml中GET_ITEM_LIST: ["points"]一样
        # if self.dataset_cfg.FOV_POINTS_ONLY:
        #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
        #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        #     points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            # if self.dataset_cfg.FOV_POINTS_ONLY:
            #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
            #     fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            #     points = points[fov_flag]
            input_dict['points'] = points

        if "calib_matricies" in get_item_list:
            # input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)
            input_dict["lidar2camera"], input_dict["camera_intrinsics"], input_dict["camera2lidar"], input_dict[
                "lidar2image"] = kitti_utils.calib_to_matricies(calib)

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)
            input_dict['camera_imgs'] = self.get_camera_image(sample_idx)
            input_dict["ori_shape"] = input_dict['camera_imgs'].size
            input_dict = self.crop_image(input_dict)

        if "points_rgb" in get_item_list:
            points_rgb = self.get_points_in_image_with_rgb(input_dict)
            input_dict["points_rgb"] = points_rgb

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape

        return data_dict


def show_point(point1, point2):
    import mayavi.mlab
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(640, 500))
    mayavi.mlab.points3d(point1[:, 0], point1[:, 1], point1[:, 2],
                         # z,  # Values used for Color
                         # mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'spectral
                         color=(0, 0, 1),  # Used a fixed (r,g,b) instead
                         scale_factor=0.2,
                         figure=fig,
                         )
    mayavi.mlab.points3d(point2[:, 0], point2[:, 1], point2[:, 2],
                         # z,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'spectral
                         color=(1, 0, 0),  # Used a fixed (r,g,b) instead
                         scale_factor=0.2,
                         figure=fig,
                         )
    mayavi.mlab.show()


def create_TJ_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = TJDatasetRadar(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('TJ_infos_%s.pkl' % train_split)
    val_filename = save_path / ('TJ_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'TJ_infos_trainval.pkl'
    test_filename = save_path / 'TJ_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    print(dataset)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('TJ4DRadSet info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('TJ4DRadSet info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('TJ4DRadSet info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('TJ4DRadSet info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_TJ_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_TJ_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck'],
            data_path=ROOT_DIR / 'data' / 'TJ4DRadSet',
            save_path=ROOT_DIR / 'data' / 'TJ4DRadSet'
        )

# python -m pcdet.datasets.TJ4DRadSet.TJ4DRadSet_dataset_radar create_TJ_infos tools/cfgs/dataset_configs/TJ4DRadSet_dataset_radar.yaml
