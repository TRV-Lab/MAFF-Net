import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from .anchor_head_template import AnchorHeadTemplate


class FeatureDistillationLoss(nn.Module):
    def __init__(self, sigma=0.1, tau=0.5, lambda1=5, lambda2=1):
        super(FeatureDistillationLoss, self).__init__()
        self.sigma = sigma
        self.tau = tau
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def channelwise_softmax(self, F):
        exp_F = torch.exp(F / self.tau)
        sum_exp_F = exp_F.sum(dim=1, keepdim=True)
        return exp_F / sum_exp_F

    def normalize(self, tensor):
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        normalized_tensor = (tensor - mean) / (std + 1e-6)
        return normalized_tensor

    def forward(self, F_rdr, F_ldr):
        # 生成预测热力图和真实热力图
        Hcls_rdr = F_rdr.sum(dim=1)
        Hcls_GT = F_ldr.sum(dim=1)
        # 正则化处理（L2正则化）
        Hcls_rdr_norm = self.normalize(Hcls_rdr)
        Hcls_GT_norm = self.normalize(Hcls_GT)
        # 定义 TP, FP, FN 区域
        TP = (Hcls_GT_norm > self.sigma) & (Hcls_rdr_norm > self.sigma)
        FP = (Hcls_GT_norm < self.sigma) & (Hcls_rdr_norm > self.sigma)
        FN = (Hcls_GT_norm > self.sigma) & (Hcls_rdr_norm < self.sigma)
        # 各区域的像素数量
        N_TP_FN = TP.sum().float() + FN.sum().float()
        N_FP = FP.sum().float()
        # 缩放掩码
        Mscale = torch.zeros_like(Hcls_rdr_norm)
        Mscale[TP | FN] = self.lambda1 / (N_TP_FN + 1e-6)  # 避免除零
        Mscale[FP] = self.lambda2 / (N_FP + 1e-6)  # 避免除零
        # 应用通道级别的softmax到高层BEV特征
        S_ldr = self.channelwise_softmax(F_ldr)
        S_rdr = self.channelwise_softmax(F_rdr)
        # 计算高层蒸馏损失
        Mscale = Mscale.unsqueeze(1)  # 扩展Mscale的维度以匹配特征图
        L_high = torch.sum(Mscale * torch.abs(S_ldr - S_rdr))
        # 多次PFD损失的平均值
        LPFD = 0.5 * L_high
        return LPFD


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.sensor_type = self.model_cfg.get('TYPE', None)
        if self.sensor_type == 'radar_lidar':
            self.alpha = self.model_cfg.get('ALPHA', 0.0003)
            self.beta = self.model_cfg.get('BETA', 0.00005)
            # self.norm = nn.BatchNorm2d(8)
            self.labeda = self.model_cfg.get('LABEDA', 0.05)
            self.pfd = FeatureDistillationLoss()

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def afd_rcs_loss(self):
        F_l_dr = self.forward_ret_dict['lidar_spatial_features_int']
        F_l1_rdr = self.forward_ret_dict['radar_spatial_features_rcs_1']
        F_l2_rdr = self.forward_ret_dict['radar_spatial_features_rcs_2']
        # lidar_int = F_l_dr[0, ...].cpu().detach().numpy()
        # radar_rcs = F_l1_rdr[0, ...].cpu().detach().numpy()
        # fig, axs = plt.subplots(2, figsize=(8, 8))
        # fig.patch.set_facecolor('lightgray')
        # lidar_int = np.sum(lidar_int, axis=0)
        # axs[0].imshow(lidar_int, cmap='gray')  # 可能需要调整 cmap 根据特征图的类型
        # axs[0].set_title(f'Feature Map lidar_int')
        # axs[0].axis('off')  # 关闭坐标轴
        # radar_rcs = np.sum(radar_rcs, axis=0)
        # axs[1].imshow(radar_rcs, cmap='gray')  # 可能需要调整 cmap 根据特征图的类型
        # axs[1].set_title(f'Feature Map after_cma_rcs')
        # axs[1].axis('off')  # 关闭坐标轴
        # plt.tight_layout()
        # plt.show()
        # Calculate the active masks
        M_l_dr = (F_l_dr.sum(dim=1) > 0).float()
        M_l1_rdr = (F_l1_rdr.sum(dim=1) > 0).float()
        M_l2_rdr = (F_l2_rdr.sum(dim=1) > 0).float()

        # Calculate Active Regions (AR) and Inactive Regions (IR)
        AR_l1 = (M_l1_rdr == 1) & (M_l_dr == 1)
        IR_l1 = (M_l1_rdr == 1) & (M_l_dr == 0)

        AR_l2 = (M_l2_rdr == 1) & (M_l_dr == 1)
        IR_l2 = (M_l2_rdr == 1) & (M_l_dr == 0)

        # Calculate the relative importance of IR over AR
        rho_l1 = AR_l1.sum() / (IR_l1.sum() + 0.001)
        rho_l2 = AR_l2.sum() / (IR_l2.sum() + 0.001)

        # Calculate adaptive loss weights
        W_sep_l1 = self.alpha * AR_l1.float() + rho_l1 * self.beta * IR_l1.float()
        W_sep_l2 = self.alpha * AR_l2.float() + rho_l2 * self.beta * IR_l2.float()

        # Calculate distillation losses
        L_low_l1 = (W_sep_l1.unsqueeze(1) * (F_l_dr - F_l1_rdr).pow(2)).sum()
        L_low_l2 = (W_sep_l2.unsqueeze(1) * (F_l_dr - F_l2_rdr).pow(2)).sum()

        # Final AFD loss
        L_AFD = 0.5 * (L_low_l1 + L_low_l2)

        return L_AFD

    def afd_z_loss(self):
        F_l_dr = self.forward_ret_dict['lidar_spatial_features_z']
        F_l1_rdr = self.forward_ret_dict['radar_spatial_features_z_1']
        F_l2_rdr = self.forward_ret_dict['radar_spatial_features_z_2']
        # lidar_int = F_l_dr[0, ...].cpu().detach().numpy()
        # radar_rcs = F_l1_rdr[0, ...].cpu().detach().numpy()
        # fig, axs = plt.subplots(2, figsize=(8, 8))
        # fig.patch.set_facecolor('lightgray')
        # lidar_int = np.sum(lidar_int, axis=0)
        # axs[0].imshow(lidar_int, cmap='gray')  # 可能需要调整 cmap 根据特征图的类型
        # axs[0].set_title(f'Feature Map lidar_int')
        # axs[0].axis('off')  # 关闭坐标轴
        # radar_rcs = np.sum(radar_rcs, axis=0)
        # axs[1].imshow(radar_rcs, cmap='gray')  # 可能需要调整 cmap 根据特征图的类型
        # axs[1].set_title(f'Feature Map after_cma_rcs')
        # axs[1].axis('off')  # 关闭坐标轴
        # plt.tight_layout()
        # plt.show()
        # Calculate the active masks
        M_l_dr = (F_l_dr.sum(dim=1) > 0).float()
        M_l1_rdr = (F_l1_rdr.sum(dim=1) > 0).float()
        M_l2_rdr = (F_l2_rdr.sum(dim=1) > 0).float()

        # Calculate Active Regions (AR) and Inactive Regions (IR)
        AR_l1 = (M_l1_rdr == 1) & (M_l_dr == 1)
        IR_l1 = (M_l1_rdr == 1) & (M_l_dr == 0)

        AR_l2 = (M_l2_rdr == 1) & (M_l_dr == 1)
        IR_l2 = (M_l2_rdr == 1) & (M_l_dr == 0)

        # Calculate the relative importance of IR over AR
        rho_l1 = AR_l1.sum() / (IR_l1.sum() + 0.001)
        rho_l2 = AR_l2.sum() / (IR_l2.sum() + 0.001)

        # Calculate adaptive loss weights
        W_sep_l1 = self.alpha * AR_l1.float() + rho_l1 * self.beta * IR_l1.float()
        W_sep_l2 = self.alpha * AR_l2.float() + rho_l2 * self.beta * IR_l2.float()

        # Calculate distillation losses
        L_low_l1 = (W_sep_l1.unsqueeze(1) * (F_l_dr - F_l1_rdr).pow(2)).sum()
        L_low_l2 = (W_sep_l2.unsqueeze(1) * (F_l_dr - F_l2_rdr).pow(2)).sum()

        # Final AFD loss
        L_AFD = 0.5 * (L_low_l1 + L_low_l2)

        return L_AFD

    def get_loss(self):
        """
            总损失函数最终形式如下：
                L^total = β^1 L^cls + ​β^2 (L^reg-θ + L^reg-other) + ​β^3 L^dir
                其中L^cls是分类损失,L^reg-other是位置和维度的回归损失,L^reg-θ是新的角度损失,L^dir是方向分类损失
                β^1 = 1.0、 β^2 = 2.0和 β^3 = 0.2 是损失公式的常数系数，使用相对较小的 β^3 值来避免网络难以识别物体方向的情况。
        """
        # 计算classifiction layer的loss，tb_dict内容和cls_loss相同，形式不同，一个是torch.tensor一个是字典值
        cls_loss, tb_dict = self.get_cls_layer_loss()
        """
            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            tb_dict = {
                'rpn_loss_cls': cls_loss.item()
            }
            return cls_loss, tb_dict
        """
        # 计算regression layer的loss
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        # 在tb_dict中添加tb_dict_box，在python的字典中添加值，如果添加的也是字典，用updae方法，如果是键值对则采用赋值的方式
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        if self.sensor_type == 'radar_lidar':
            afd_rcs_loss = self.afd_rcs_loss()
            afd_z_loss = self.afd_z_loss()
            # pfd_loss = self.pfd(self.forward_ret_dict['lidar_spatial_features_2d'],
            #                     self.forward_ret_dict['radar_spatial_features_2d'])
            rpn_loss = rpn_loss + self.labeda * torch.log(afd_rcs_loss + afd_z_loss)   # + 5 * pfd_loss
        # 在tb_dict中添加rpn_loss，此时tb_dict中包含cls_loss,reg_loss和rpn_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        if self.sensor_type == 'lidar':
            spatial_features_2d = data_dict['lidar_spatial_features_2d']
            self.forward_ret_dict['lidar_spatial_features_2d'] = data_dict['lidar_spatial_features_2d']
            self.forward_ret_dict['lidar_spatial_features_int'] = data_dict['lidar_spatial_features_int']
            self.forward_ret_dict['lidar_spatial_features_z'] = data_dict['lidar_spatial_features_z']
        elif self.sensor_type == 'radar':
            spatial_features_2d = data_dict['radar_spatial_features_2d']
        elif self.sensor_type == 'radar_lidar':
            spatial_features_2d = data_dict['radar_spatial_features_2d']
            self.forward_ret_dict['lidar_spatial_features_2d'] = data_dict['lidar_spatial_features_2d']
            self.forward_ret_dict['radar_spatial_features_2d'] = data_dict['radar_spatial_features_2d']
            self.forward_ret_dict['lidar_spatial_features_int'] = data_dict['lidar_spatial_features_int']
            self.forward_ret_dict['lidar_spatial_features_z'] = data_dict['lidar_spatial_features_z']
            self.forward_ret_dict['radar_spatial_features_rcs_1'] = data_dict['radar_spatial_features_rcs_1']
            self.forward_ret_dict['radar_spatial_features_z_1'] = data_dict['radar_spatial_features_z_1']
            self.forward_ret_dict['radar_spatial_features_rcs_2'] = data_dict['radar_spatial_features_rcs_2']
            self.forward_ret_dict['radar_spatial_features_z_2'] = data_dict['radar_spatial_features_z_2']
        else:
            spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # print("!!!!!!!!!!!!!cls_preds  ",cls_preds.shape[0])

        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

