import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import constant_init, xavier_init
from mmdet.core import bbox_overlaps, multi_apply
from mmdet.ops import nms
from ..registry import HEADS
from ..builder import build_loss
from ..utils import ConvModule


# TODO: merge the YOLOAnchorGenerator with the anchor generator
# in other one-stage and two-stage models.
class YOLOAnchorGenerator(object):
    """Anchor Generator for YOLO
    Yolo uses k-means clustering to determine bounding box priors.
    Each level has three base anchors.

    Args:
        base_anchor (Iterable): three kinds of width and height for each level.
    """

    def __init__(self,
                 base_anchors):
        super(YOLOAnchorGenerator, self).__init__()
        self.base_anchors = torch.Tensor(base_anchors)

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        shifts = torch.stack([shift_xx, shift_yy], dim=-1).view(
            1, feat_h, feat_w, 2)
        shifts = shifts.type_as(base_anchors).repeat(3, 1, 1, 1)

        bboxpriors = base_anchors.view(3, 1, 1, 2).repeat(1, feat_h, feat_w, 1)

        all_anchors = torch.cat([shifts, bboxpriors], dim=-1).view(-1, 4)
        return all_anchors

    def anchor_shifts(self, device='cuda'):
        base_anchors = self.base_anchors.to(device)
        anchor_shifts = base_anchors.new_zeros((base_anchors.shape[0], 4))
        anchor_shifts[:, 2:] = base_anchors
        return anchor_shifts


@HEADS.register_module
class YOLOV3Head(nn.Module):
    """YOLOV3 Head

    Args:
        num_classes (int): number of classes for classification.
        in_channels (Iterable): number of channels in the input feature map.
        anchors (Iterable): base anchors.
        anchor_strides (Iterable): anchor strides.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        act_cfg (dict): dictionary to construct
            and config activation layer.
        loss_xy (dict): Config of localization loss.
        loss_wh (dict): Config of localization loss.
        loss_obj (dict): Config of object loss.
        loss_cls (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 scale_x_y=[1, 1, 1],
                 image_size=(512, 512),
                 in_channels=[128, 256, 512],
                 anchors=[([10, 13], [16, 30], [33, 23]),
                          ([30, 61], [62, 45], [59, 119]),
                          ([116, 90], [156, 198], [373, 326])],
                 anchor_strides=[8, 16, 32],
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1,
                                     inplace=True),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(
                     type='MSELoss',
                     reduction='mean',
                     loss_weight=1.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super(YOLOV3Head, self).__init__()
        self.num_classes = num_classes
        self.scale_x_y = scale_x_y
        self.image_size = image_size
        assert len(in_channels) == len(anchors)
        self.num_levels = len(in_channels)
        self.in_channels = in_channels

        self.anchor_generators = []
        for i, anchor in enumerate(anchors):
            self.anchor_generators.append(YOLOAnchorGenerator(anchor))
        self.anchor_strides = anchor_strides
        featmap_sizes = [
            (int(self.image_size[0] / anchor_strides[i]),
             int(self.image_size[1] / anchor_strides[i]))
            for i in range(self.num_levels)
        ]

        anchor_shifts_list, multi_level_anchors = \
            self.get_anchors(featmap_sizes)
        self.anchor_shifts_list = anchor_shifts_list
        self.multi_level_anchors = multi_level_anchors

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        self.loss_obj = build_loss(loss_obj)
        self.loss_cls = build_loss(loss_cls)

        self._init_layers()

    def _init_layers(self):
        self.yolo_heads = []
        for i in range(self.num_levels):
            head_layer = nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i] * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                nn.Conv2d(self.in_channels[i] * 2,
                          (self.num_classes - 1 + 5) * 3,
                          kernel_size=1,
                          stride=1,
                          padding=0)
            )
            name = 'yolo_head{}'.format(i + 1)
            self.add_module(name, head_layer)
            self.yolo_heads.append(name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

    def forward(self, x):
        head_outs = []
        for i in range(self.num_levels):
            head_layer = getattr(self, self.yolo_heads[i])
            head_outs.append(head_layer(x[i]))
        return (head_outs,)

    def loss_single(self,
                    x,
                    targets,
                    targets_weights,
                    targets_scale,
                    num_total_pos):
        """Get bce loss for xy, mse loss for wh, bce loss for obj,
        bce loss for cls.
        The input x, targets, targets_weights have the same shape:
        (b, 3, h, w, c). And the shape of targets_scale is (b, 3, h, w, 2).
        In details, the last dim c is consist of
        bbox (4: x, y, w, h), obj (1), cls (c - 5).
        """

        x[..., 2:4] = x[..., 2:4] * targets_scale
        targets[..., 2:4] = targets[..., 2:4] * targets_scale

        loss_xy = self.loss_xy(
            x[..., :2],
            targets[..., :2],
            weight=targets_scale * targets_scale,
            avg_factor=num_total_pos)
        loss_wh = self.loss_wh(
            x[..., 2:4],
            targets[..., 2:4],
            weight=targets_weights[..., 2:4],
            avg_factor=num_total_pos * 2)
        loss_obj = self.loss_obj(
            x[..., 4],
            targets[..., 4],
            weight=targets_weights[..., 4],
            avg_factor=num_total_pos)
        loss_cls = self.loss_cls(
            x[..., 5:],
            targets[..., 5:],
            weight=targets_weights[..., 5:],
            avg_factor=num_total_pos)

        return loss_xy, loss_wh, loss_obj, loss_cls

    def loss(self, head_outs,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        batch_size = len(gt_bboxes)
        # max_num_gts = max([gt_bboxes[i].shape[0] for i in range(batch_size)])
        max_num_gts = 70
        gt_bboxes_list = gt_bboxes[0].new_zeros((batch_size, max_num_gts, 4))
        gt_labels_list = gt_labels[0].new_zeros((batch_size, max_num_gts))
        for i in range(batch_size):
            num_gts = gt_bboxes[i].shape[0]
            gt_bboxes_list[i, :num_gts] = gt_bboxes[i]
            gt_labels_list[i, :num_gts] = gt_labels[i] \
                if not self.use_sigmoid_cls else gt_labels[i] - 1

        losses_xy, losses_wh, losses_obj, losses_cls = multi_apply(
            self.get_targets,
            head_outs,
            self.multi_level_anchors,
            self.anchor_shifts_list,
            self.anchor_strides,
            list(range(self.num_levels)),
            gt_bboxes=gt_bboxes_list,
            gt_labels=gt_labels_list,
            num_gts_pre_img=max_num_gts,
            cfg=cfg
        )

        return dict(loss_xy=losses_xy,
                    loss_wh=losses_wh,
                    loss_obj=losses_obj,
                    loss_cls=losses_cls)

    def get_anchors(self, featmap_sizes):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.

        Returns:
            list: anchors (wh) of each image, anchors of each level
        """
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        multi_level_anchor_shifts = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i]) / \
                    self.anchor_strides[i]
            anchors = anchors.view(
                1, 3, featmap_sizes[i][0], featmap_sizes[i][1], 4)
            multi_level_anchors.append(anchors)
            anchor_shifts = self.anchor_generators[i].anchor_shifts()
            multi_level_anchor_shifts.append(anchor_shifts)
        multi_level_anchor_shifts = torch.cat(multi_level_anchor_shifts)
        anchor_shifts = []
        for i in range(num_levels):
            anchor_shifts.append(
                multi_level_anchor_shifts / self.anchor_strides[i])
        return anchor_shifts, multi_level_anchors

    def get_targets(self,
                    x,
                    anchors,
                    anchor_shifts,
                    stride,
                    level_index,
                    gt_bboxes,
                    gt_labels,
                    num_gts_pre_img,
                    cfg):
        b, _, h, w = x.size()
        x = x.view(b, 3, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()

        bbox_pred = x[..., :4].data.clone()
        bbox_pred[..., :2] = torch.sigmoid(bbox_pred[..., :2]) + \
            anchors[..., :2]
        bbox_pred[..., 2:4] = torch.exp(bbox_pred[..., 2:4]) * \
            anchors[..., 2:]
        # cx, cy, w, h --> x1, y1, x2, y2
        bbox_pred[..., :2] = bbox_pred[..., :2] - \
            bbox_pred[..., 2:4] * 0.5 + 0.5
        bbox_pred[..., 2:4] = bbox_pred[..., :2] + bbox_pred[..., 2:4] - 1

        gt_bboxes_level = gt_bboxes / stride
        gt_shifts = gt_bboxes_level.clone()
        # x1, y1, x2, y2 --> 0, 0, w, h
        gt_shifts[..., 2:] = gt_shifts[..., 2:] - gt_shifts[..., :2] + 1
        gt_shifts[..., :2] = 0

        overlaps_shifts = bbox_overlaps(
            gt_shifts.reshape(-1, 4), anchor_shifts)
        inds = overlaps_shifts.reshape(b, num_gts_pre_img, 9).argmax(dim=-1)
        valid_flags = (inds >= level_index * 3) & \
            (inds < (level_index + 1) * 3)
        bs_inds, gt_inds = valid_flags.nonzero().t()
        inds = inds[valid_flags]
        anchor_inds = inds % 3

        overlaps_pred = bbox_overlaps(bbox_pred.reshape(-1, 4),
                                      gt_bboxes_level.reshape(-1, 4))
        num_overlaps_pre_img = overlaps_pred.shape[0] // b
        overlaps_pred = [overlaps_pred[
            i * num_overlaps_pre_img: (i + 1) * num_overlaps_pre_img,
            i * num_gts_pre_img: (i + 1) * num_gts_pre_img]
            for i in range(b)
        ]

        overlaps_pred = torch.stack(overlaps_pred)
        best_match_iou, _ = overlaps_pred.max(dim=-1)
        best_match_iou = \
            (best_match_iou.view(b, 3, h, w) > cfg.ignore_thresh)
        obj_mask = gt_bboxes.new_ones((b, 3, h, w))
        obj_mask = ~ best_match_iou

        targets = gt_bboxes.new_zeros(
            (b, 3, h, w, self.cls_out_channels + 5))
        targets_weights = gt_bboxes.new_zeros(
            (b, 3, h, w, self.cls_out_channels + 5))
        targets_weights[..., 4] = obj_mask
        targets_scale = gt_bboxes.new_zeros((b, 3, h, w, 2))

        gt_bboxes_level = gt_bboxes_level[bs_inds, gt_inds]
        cx = (gt_bboxes_level[..., 0] + gt_bboxes_level[..., 2]) * 0.5
        cy = (gt_bboxes_level[..., 1] + gt_bboxes_level[..., 3]) * 0.5
        w_gt = gt_bboxes_level[..., 2] - gt_bboxes_level[..., 0] + 1
        h_gt = gt_bboxes_level[..., 3] - gt_bboxes_level[..., 1] + 1
        x_inds = cx.type_as(gt_inds)
        y_inds = cy.type_as(gt_inds)
        targets[bs_inds, anchor_inds, y_inds, x_inds, 0] = cx - x_inds
        targets[bs_inds, anchor_inds, y_inds, x_inds, 1] = cy - y_inds
        targets[bs_inds, anchor_inds, y_inds, x_inds, 2] = \
            torch.log(w_gt / anchor_shifts[inds, 2] + 1e-16)
        targets[bs_inds, anchor_inds, y_inds, x_inds, 3] = \
            torch.log(h_gt / anchor_shifts[inds, 3] + 1e-16)
        targets[bs_inds, anchor_inds, y_inds, x_inds, 4] = 1
        targets[bs_inds, anchor_inds, y_inds, x_inds,
                5 + gt_labels[bs_inds, gt_inds]] = 1
        targets_weights[bs_inds, anchor_inds, y_inds, x_inds] = 1
        targets_scale[bs_inds, anchor_inds, y_inds, x_inds, 0] = \
            torch.sqrt(2 - w_gt * h_gt / w / h)
        targets_scale[bs_inds, anchor_inds, y_inds, x_inds, 1] = \
            torch.sqrt(2 - w_gt * h_gt / w / h)

        num_total_pos = inds.shape[0]

        return self.loss_single(x,
                                targets,
                                targets_weights,
                                targets_scale,
                                num_total_pos if num_total_pos else 1)

    def get_bboxes(self, head_outs, img_metas, cfg,
                   rescale=False):
        num_levels = len(head_outs)
        mlvl_anchors = self.get_mlvl_anchors(head_outs, num_levels)

        result_list = []
        for img_id in range(len(img_metas)):
            single_output = self.get_image_outs(head_outs,
                                                num_levels,
                                                img_idx=img_id)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            single_result = self.get_bboxes_single(single_output,
                                                   mlvl_anchors,
                                                   img_shape,
                                                   scale_factor,
                                                   cfg,
                                                   rescale)
            result_list.append(single_result)
        return result_list

    def get_mlvl_anchors(self, output, num_levels):
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(output[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        return mlvl_anchors

    def get_image_outs(self, output, num_levels, img_idx=0):
        output_list = [
            output[i][img_idx].detach() for i in range(num_levels)
        ]
        return output_list

    def get_bboxes_single(self,
                          output,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(output) == len(mlvl_anchors)

        predictions = []
        for i, x in enumerate(output):
            _, h, w = x.size()
            x = x.view(3, -1, h, w).permute(0, 2, 3, 1).contiguous()

            anchors = mlvl_anchors[i].view(3, h, w, 4)
            x[..., :2] = torch.sigmoid(x[..., :2])
            x[..., :2] = x[..., :2] * self.scale_x_y[i] - \
                (self.scale_x_y[i] - 1) * 0.5

            x[..., :2] = x[..., :2] * self.anchor_strides[i] + \
                anchors[..., :2]
            x[..., 2:4] = torch.exp(x[..., 2:4]) * anchors[..., 2:]
            torch.sigmoid_(x[..., 4:])
            pred = x.view(-1, x.size()[-1])
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)

        dets = self.yolo_nms(predictions,
                             conf_thres=cfg.score_thr,
                             nms_thres=cfg.nms.iou_thr)
        if dets is not None:
            dets[:, 0].clamp_(min=0, max=img_shape[1] - 1)
            dets[:, 1].clamp_(min=0, max=img_shape[0] - 1)
            dets[:, 2].clamp_(min=0, max=img_shape[1] - 1)
            dets[:, 3].clamp_(min=0, max=img_shape[0] - 1)
            dets[:, :4] /= dets[:, :4].new_tensor(scale_factor)
            return dets[:, :5], dets[:, 5]
        else:
            return torch.Tensor([]), torch.Tensor([])

    def yolo_nms(self, prediction, conf_thres=0.001, nms_thres=0.6):
        min_wh, max_wh = 2, 4096
        inds = (prediction[:, 4] > conf_thres) & \
               ((prediction[:, 2:4] > min_wh) &
                (prediction[:, 2:4] < max_wh)).all(1)
        prediction = prediction[inds]
        bbox = prediction[:, :4]
        # cx, cy, w, h --> x1, y1, x2, y2
        bbox[..., :2] = bbox[..., :2] - bbox[..., 2:4] * 0.5 + 0.5
        bbox[..., 2:4] = bbox[..., :2] + bbox[..., 2:4] - 1

        if not prediction.shape[0]:
            return None

        prediction[..., 5:] *= prediction[..., 4:5]
        valid_inds, cls_ids = (prediction[:, 5:] > conf_thres).nonzero().t()
        det_bboxes = torch.cat(
            (bbox[valid_inds],
             prediction[valid_inds, cls_ids + 5].unsqueeze(1),
             cls_ids.float().unsqueeze(1)), dim=1)

        inds = torch.isfinite(det_bboxes).all(1)
        det_bboxes = det_bboxes[inds]

        if not det_bboxes.shape[0]:
            return None

        boxes = det_bboxes[:, :4].clone() + \
            det_bboxes[:, 5].view(-1, 1) * max_wh
        scores = det_bboxes[:, 4]

        _, inds = nms(torch.cat((boxes, scores.view(-1, 1)), 1), nms_thres)
        weights = (bbox_overlaps(boxes[inds], boxes) > nms_thres) * \
            scores[None]
        det_bboxes[inds, :4] = torch.mm(weights, det_bboxes[:, :4]).float() / \
            weights.sum(1, keepdim=True)

        return det_bboxes[inds]


@HEADS.register_module
class YOLOV4Head(YOLOV3Head):
    """YOLO Head

    Args:
        num_classes (int): number of classes for classification.
        in_channels (Iterable): number of channels in the input feature map.
        anchors (Iterable): base anchors.
        anchor_strides (Iterable): anchor strides.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        act_cfg (dict): dictionary to construct
            and config activation layer.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of object loss.
        loss_cls (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 loss_bbox=dict(
                     type='IoULoss',
                     use_complete=True,
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        super(YOLOV4Head, self).__init__(num_classes, **kwargs)
        self.loss_bbox = build_loss(loss_bbox)

    def init_weights(self):
        super(YOLOV4Head, self).init_weights()

    def loss_single(self,
                    x,
                    targets,
                    targets_weights,
                    num_total_pos):
        """Get iou loss for bbox, bce loss for obj, bce loss for cls.
        The input x, targets, targets_weights have the same shape:
        (b, 3, h, w, c). In details, the last dim c is consist of
        bbox (4: x1, y1, x2, y2), obj (1), cls (c - 5).
        """

        loss_bbox = self.loss_bbox(
            x[..., :4].reshape(-1, 4),
            targets[..., :4].reshape(-1, 4),
            weight=targets_weights[..., 1].reshape(-1),
            avg_factor=num_total_pos)
        loss_obj = self.loss_obj(
            x[..., 4],
            targets[..., 4],
            weight=targets_weights[..., 4],
            avg_factor=num_total_pos)
        loss_cls = self.loss_cls(
            x[..., 5:],
            targets[..., 5:],
            weight=targets_weights[..., 5:],
            avg_factor=num_total_pos)

        return loss_bbox, loss_obj, loss_cls

    def loss(self, head_outs,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        batch_size = len(gt_bboxes)
        # max_num_gts = max([gt_bboxes[i].shape[0] for i in range(batch_size)])
        max_num_gts = 70
        gt_bboxes_list = gt_bboxes[0].new_zeros((batch_size, max_num_gts, 4))
        gt_labels_list = gt_labels[0].new_zeros((batch_size, max_num_gts))
        for i in range(batch_size):
            num_gts = min(gt_bboxes[i].shape[0], max_num_gts)
            gt_bboxes_list[i, :num_gts] = gt_bboxes[i][:num_gts]
            gt_labels_list[i, :num_gts] = gt_labels[i][:num_gts] \
                if not self.use_sigmoid_cls else gt_labels[i][:num_gts] - 1

        losses_bbox, losses_obj, losses_cls = multi_apply(
            self.get_targets,
            head_outs,
            self.scale_x_y,
            self.multi_level_anchors,
            self.anchor_shifts_list,
            self.anchor_strides,
            list(range(self.num_levels)),
            gt_bboxes=gt_bboxes_list,
            gt_labels=gt_labels_list,
            num_gts_pre_img=max_num_gts,
            cfg=cfg
        )

        return dict(
            loss_bbox=losses_bbox,
            loss_obj=losses_obj,
            loss_cls=losses_cls
        )

    def get_targets(self,
                    x,
                    scale_x_y,
                    anchors,
                    anchor_shifts,
                    stride,
                    level_index,
                    gt_bboxes,
                    gt_labels,
                    num_gts_pre_img,
                    cfg):
        b, _, h, w = x.size()
        x = x.view(b, 3, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
        x[..., :2] = torch.sigmoid(x[..., :2])
        x[..., :2] = x[..., :2] * scale_x_y - (scale_x_y - 1) * 0.5

        x[..., :2] = x[..., :2] + anchors[..., :2]
        x[..., 2:4] = torch.exp(x[..., 2:4]) * anchors[..., 2:]
        # cx, cy, w, h --> x1, y1, x2, y2
        x[..., :2] = x[..., :2] - x[..., 2:4] * 0.5 + 0.5
        x[..., 2:4] = x[..., :2] + x[..., 2:4] - 1

        gt_bboxes_level = gt_bboxes / stride
        gt_shifts = gt_bboxes_level.clone()
        # x1, y1, x2, y2 --> 0, 0, w, h
        gt_shifts[..., 2:] = gt_shifts[..., 2:] - gt_shifts[..., :2] + 1
        gt_shifts[..., :2] = 0

        overlaps_shifts = bbox_overlaps(
            gt_shifts.reshape(-1, 4), anchor_shifts)
        inds = (overlaps_shifts.reshape(
            b, num_gts_pre_img, 9) > cfg.iou_thresh).nonzero()
        valid_flags = (inds[..., 2] >= level_index * 3) & \
            (inds[..., 2] < (level_index + 1) * 3)
        inds = inds[valid_flags]
        inds[..., 2] = inds[..., 2] % 3
        del overlaps_shifts

        overlaps_pred = bbox_overlaps(x[..., :4].reshape(-1, 4),
                                      gt_bboxes_level.reshape(-1, 4))
        num_overlaps_pre_img = overlaps_pred.shape[0] // b
        overlaps_pred = [overlaps_pred[
            i * num_overlaps_pre_img: (i + 1) * num_overlaps_pre_img,
            i * num_gts_pre_img: (i + 1) * num_gts_pre_img]
            for i in range(b)
        ]
        overlaps_pred = torch.stack(overlaps_pred)
        best_match_iou, _ = overlaps_pred.max(dim=-1)
        best_match_iou = (best_match_iou.view(b, 3, h, w) > cfg.ignore_thresh)
        obj_mask = gt_bboxes.new_ones((b, 3, h, w))
        obj_mask = ~ best_match_iou
        del overlaps_pred

        targets = gt_bboxes.new_zeros(
            (b, 3, h, w, self.cls_out_channels + 5))
        targets_weights = gt_bboxes.new_zeros(
            (b, 3, h, w, self.cls_out_channels + 5))
        targets_weights[..., 4] = obj_mask

        bs_inds, gt_inds, anchor_inds = inds.t()
        gt_bboxes_level = gt_bboxes_level[bs_inds, gt_inds]
        cx = (gt_bboxes_level[..., 0] + gt_bboxes_level[..., 2]) * 0.5
        cy = (gt_bboxes_level[..., 1] + gt_bboxes_level[..., 3]) * 0.5
        x_inds = cx.type_as(gt_inds)
        y_inds = cy.type_as(gt_inds)
        targets[bs_inds, anchor_inds, y_inds, x_inds, :4] = gt_bboxes_level
        targets[bs_inds, anchor_inds, y_inds, x_inds, 4] = 1
        targets[bs_inds, anchor_inds, y_inds, x_inds,
                5 + gt_labels[bs_inds, gt_inds]] = 1
        targets_weights[bs_inds, anchor_inds, y_inds, x_inds] = 1

        num_total_pos = inds.shape[0]

        return self.loss_single(x, targets, targets_weights,
                                num_total_pos if num_total_pos else 1)
