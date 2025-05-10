# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F
import os
import piexif
# from datetime import datetime
import torch
import torch.nn.functional as F
import piexif
import datetime

def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss




class ComputeLoss:
    def __init__(self, model, paths=None, autobalance=False, margin=0.3, alpha=0.1,
                 proto_weight=0.4, contrast_weight=0.6, time_factor=0.01, momentum=0.9):
        self.device = next(model.parameters()).device
        self.margin = margin
        self.alpha = alpha
        self.proto_weight = proto_weight
        self.contrast_weight = contrast_weight
        self.time_factor = time_factor
        self.momentum = momentum

    def extract_date_from_exif(self, image_path):
        """提取图像 EXIF 日期信息为 'YYYY-MM-DD'"""
        exif_dict = piexif.load(image_path)
        time_str = exif_dict['Exif'].get(piexif.ExifIFD.DateTimeOriginal, None)
        if time_str is None:
            raise ValueError(f"图像 {image_path} 缺少 DateTimeOriginal EXIF 信息")
        if isinstance(time_str, bytes):
            time_str = time_str.decode("utf-8")
        date_part = time_str.split(" ")[0].replace(":", "-")
        return date_part

    def date_diff_in_days(self, date1_str, date2_str):
        """返回两个日期字符串之间的天数差"""
        date_format = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(date1_str, date_format)
        d2 = datetime.datetime.strptime(date2_str, date_format)
        return abs((d1 - d2).days)

    def compute_prototypes(self, p_list, paths):
        """按图像日期计算每层的类别原型"""
        date_tags = [self.extract_date_from_exif(path) for path in paths]
        proto_dicts = [{} for _ in range(len(p_list))]
        count_dicts = [{} for _ in range(len(p_list))]

        for i, p in enumerate(p_list):
            p_flat = p.view(p.size(0), -1)
            for j in range(p.size(0)):
                category = date_tags[j]
                if category not in proto_dicts[i]:
                    proto_dicts[i][category] = torch.zeros_like(p_flat[j]).to(self.device)
                    count_dicts[i][category] = 0
                proto_dicts[i][category] += p_flat[j]
                count_dicts[i][category] += 1

        for i in range(len(proto_dicts)):
            for key in proto_dicts[i]:
                proto_dicts[i][key] /= count_dicts[i][key]

        return proto_dicts

    def triplet_loss(self, p_list, paths, proto_dicts):
        """按图像日期计算 Triplet Loss（支持时间差动态 margin）"""
        date_tags = [self.extract_date_from_exif(path) for path in paths]
        total_triplet_loss = 0

        for i, p in enumerate(p_list):
            p_flat = p.view(p.size(0), -1)
            triplet_losses = []
            for j in range(p.size(0)):
                anchor = p_flat[j]
                category = date_tags[j]
                pos_proto = proto_dicts[i].get(category, None)
                neg_proto = None
                neg_category = None
                for key in proto_dicts[i]:
                    if key != category:
                        neg_proto = proto_dicts[i][key]
                        neg_category = key
                        break

                if pos_proto is not None and neg_proto is not None:
                    d_pos = F.pairwise_distance(anchor.unsqueeze(0), pos_proto.unsqueeze(0))
                    d_neg = F.pairwise_distance(anchor.unsqueeze(0), neg_proto.unsqueeze(0))
                    time_diff = self.date_diff_in_days(category, neg_category)
                    time_margin = self.time_factor * time_diff
                    margin = self.alpha * torch.mean(d_pos).detach() + time_margin
                    triplet_loss_sample = F.relu(d_pos - d_neg + margin)
                    triplet_losses.append(triplet_loss_sample)

            if triplet_losses:
                total_triplet_loss += torch.mean(torch.stack(triplet_losses))
        return total_triplet_loss

    def contrastive_loss(self, p_list, paths):
        """按图像日期计算 InfoNCE 对比损失"""
        date_tags = [self.extract_date_from_exif(path) for path in paths]
        total_contrast_loss = 0
        for i, p in enumerate(p_list):
            p_flat = p.view(p.size(0), -1)
            losses = []
            for j in range(p.size(0)):
                for k in range(j + 1, p.size(0)):
                    cosine_sim = F.cosine_similarity(p_flat[j].unsqueeze(0), p_flat[k].unsqueeze(0), dim=-1)
                    date1 = date_tags[j]
                    date2 = date_tags[k]
                    time_diff = self.date_diff_in_days(date1, date2)
                    time_weight = 1 + self.time_factor * time_diff
                    if date1 == date2:
                        loss_val = F.relu((1 - cosine_sim) / time_weight)
                    else:
                        loss_val = F.relu(cosine_sim * time_weight)
                    losses.append(loss_val)
            if losses:
                total_contrast_loss += torch.mean(torch.stack(losses))
        return total_contrast_loss

    def __call__(self, p_list, paths):
        """融合 Triplet 与 Contrastive 多层损失"""
        proto_dicts = self.compute_prototypes(p_list, paths)
        t_loss = self.triplet_loss(p_list, paths, proto_dicts)
        c_loss = self.contrastive_loss(p_list, paths)
        total_loss = self.proto_weight * t_loss + self.contrast_weight * c_loss
        return total_loss




# class ComputeLoss:
#     def __init__(self, model, paths=None, autobalance=False, margin=0.3, alpha=0.1,
#                  proto_weight=0.4, contrast_weight=0.6, time_factor=0.01, momentum=0.9):
#         """
#         time_factor: 用于调节时间差对损失的影响，值越大时间差的影响越明显。
#         """
#         self.device = next(model.parameters()).device  
#         self.margin = margin  
#         self.alpha = alpha  
#         self.proto_weight = proto_weight  
#         self.contrast_weight = contrast_weight
#         self.time_factor = time_factor
#         self.momentum = momentum

#         # # 注册每一层的原型字典作为 buffer
#         # self.num_layers = len(model.backbone)  # 假设这是你的 backbone 层数
#         # for i in range(3):
#         #     # 注册原型字典
#         #     self.register_buffer(f'proto_dict_layer{i}', {})

#     def compute_prototypes(self, p_list, paths):
#         """
#         对 p_list 中每一层，按照图片的父目录（假设为日期字符串，例如 "2024-09-09"）计算类别原型。
#         返回一个列表，每一项为对应层的类别原型字典。
#         """
#         parent_dirs = [os.path.basename(os.path.dirname(path)) for path in paths]
#         proto_dicts = [{} for _ in range(len(p_list))]
#         count_dicts = [{} for _ in range(len(p_list))]

#         for i, p in enumerate(p_list):
#             # 将每个特征展平为 (batch_size, feature_dim)
#             p_flat = p.view(p.size(0), -1)
#             for j in range(p.size(0)):
#                 category = parent_dirs[j]
#                 if category not in proto_dicts[i]:
#                     proto_dicts[i][category] = torch.zeros_like(p_flat[j]).to(self.device)
#                     count_dicts[i][category] = 0
#                 proto_dicts[i][category] += p_flat[j]
#                 count_dicts[i][category] += 1

#         for i in range(len(proto_dicts)):
#             for key in proto_dicts[i]:
#                 proto_dicts[i][key] /= count_dicts[i][key]

#         return proto_dicts

#     def triplet_loss(self, p_list, paths, proto_dicts):
#         """
#         对每一层计算 Triplet Loss：  
#           - 使用父目录（日期字符串）作为类别标签。  
#           - 在负样本选择时，计算当前样本（anchor）的日期与负类别日期之间的时间差，  
#             并乘以 time_factor 后加入到 margin 中。
#         """
#         parent_dirs = [os.path.basename(os.path.dirname(path)) for path in paths]
#         total_triplet_loss = 0

#         for i, p in enumerate(p_list):
#             p_flat = p.view(p.size(0), -1)
#             triplet_losses = []
#             for j in range(p.size(0)):
#                 anchor = p_flat[j]
#                 category = parent_dirs[j]
#                 pos_proto = proto_dicts[i].get(category, None)
#                 neg_proto = None
#                 neg_category = None
#                 # 选择一个负类别原型（这里简单选择第一个不同类别的原型）
#                 for key in proto_dicts[i]:
#                     if key != category:
#                         neg_proto = proto_dicts[i][key]
#                         neg_category = key
#                         break

#                 if pos_proto is not None and neg_proto is not None:
#                     d_pos = F.pairwise_distance(anchor.unsqueeze(0), pos_proto.unsqueeze(0))
#                     d_neg = F.pairwise_distance(anchor.unsqueeze(0), neg_proto.unsqueeze(0))
#                     # 根据 anchor 与负样本对应的日期计算时间差（单位：天）
#                     time_diff = date_diff_in_days(category, neg_category)
#                     time_margin = self.time_factor * time_diff  # 时间差带来的额外 margin
#                     # 动态 margin：基础 margin 根据 d_pos 的均值计算，再加上时间信息
#                     margin = self.alpha * torch.mean(d_pos).detach() + time_margin
#                     triplet_loss_sample = F.relu(d_pos - d_neg + margin)
#                     triplet_losses.append(triplet_loss_sample)

#             if triplet_losses:
#                 total_triplet_loss += torch.mean(torch.stack(triplet_losses))
#         return total_triplet_loss

#     def contrastive_loss(self, p_list, paths):
#         """
#         对每一层计算 InfoNCE 对比损失：  
#           - 对于每一对样本，计算余弦相似度。  
#           - 根据对应的日期计算时间差，进而设计一个权重 time_weight，  
#             当时间差越大时，相似度约束越强或惩罚越大。
#         """
#         parent_dirs = [os.path.basename(os.path.dirname(path)) for path in paths]
#         total_contrast_loss = 0
#         for i, p in enumerate(p_list):
#             p_flat = p.view(p.size(0), -1)
#             losses = []
#             for j in range(p.size(0)):
#                 for k in range(j + 1, p.size(0)):
#                     cosine_sim = F.cosine_similarity(p_flat[j].unsqueeze(0), p_flat[k].unsqueeze(0), dim=-1)
#                     date1 = parent_dirs[j]
#                     date2 = parent_dirs[k]
#                     time_diff = date_diff_in_days(date1, date2)
#                     # 当时间差越大时，权重越大（这里简单设计为 1 + time_factor * time_diff）
#                     time_weight = 1 + self.time_factor * time_diff
#                     if date1 == date2:
#                         # 同一日期：期望特征更相似，因此目标是 (1 - cosine_sim)
#                         loss_val = F.relu((1 - cosine_sim) / time_weight)
#                     else:
#                         # 不同日期：期望特征更不相似，因此目标是增大 cosine_sim
#                         loss_val = F.relu(cosine_sim * time_weight)
#                     losses.append(loss_val)
#             if losses:
#                 total_contrast_loss += torch.mean(torch.stack(losses))
#         return total_contrast_loss

#     def __call__(self, p_list, paths):
#         """
#         计算最终损失：融合了 Triplet Loss 和 Contrastive Loss，  
#         并分别对各层计算后求和。
#         """
#         proto_dicts = self.compute_prototypes(p_list, paths)
#         t_loss = self.triplet_loss(p_list, paths, proto_dicts)
#         c_loss = self.contrastive_loss(p_list, paths)
#         total_loss = self.proto_weight * t_loss + self.contrast_weight * c_loss
#         return total_loss
