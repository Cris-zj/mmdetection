import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox_overlaps, multi_apply
from ..anchor_heads import YOLOV4Head
from ..registry import HEADS
from ..builder import build_loss
from ..utils import build_conv_layer


@HEADS.register_module
class YOLOEmbeddingHead(YOLOV4Head):
    """YOLOV3 + embedding Head

    Args:
    num_classes: default is 2.
    dim_embedding: default is 512.
    """

    def __init__(self,
                 num_classes,
                 dim_embedding,
                 num_identities,
                 loss_id=dict(
                     type='CrossEntropyLoss',
                     reduction='mean',
                     loss_weight=1.0),
                 **kwargs):
        super(YOLOEmbeddingHead, self).__init__(num_classes, **kwargs)
        self.loss_id = build_loss(loss_id)
        self.dim_embedding = dim_embedding
        self.num_identities = num_identities

        self.embedding_heads = []
        for i in range(self.num_levels):
            embedding_layer = build_conv_layer(
                self.conv_cfg,
                self.in_channels[i],
                self.dim_embedding,
                kernel_size=3,
                stride=1,
                padding=1
            )
            name = 'embedding_head{}'.format(i + 1)
            self.add_module(name, embedding_layer)
            self.embedding_heads.append(name)

            self.register_parameter(f'bbox_weight{i + 1}',
                                    nn.Parameter(-4.85 * torch.ones(1)))
            self.register_parameter(f'cls_weight{i + 1}',
                                    nn.Parameter(-4.15 * torch.ones(1)))
            self.register_parameter(f'id_weight{i + 1}',
                                    nn.Parameter(-2.3 * torch.ones(1)))

        self.fc = nn.Linear(self.dim_embedding, self.num_identities)

    def init_weights(self):
        super(YOLOEmbeddingHead, self).init_weights()

    def forward(self, x):
        yolo_outs = []
        embedding_outs = []
        for i in range(self.num_levels):
            yolo_layer = getattr(self, self.yolo_heads[i])
            yolo_out = yolo_layer(x[i])
            yolo_outs.append(yolo_out)

            embedding_layer = getattr(self, self.embedding_heads[i])
            embedding_out = embedding_layer(x[i])
            embedding_outs.append(embedding_out)
        return yolo_outs, embedding_outs

    def loss_single(self,
                    x,
                    targets,
                    targets_weights,
                    num_total_pos):
        """
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
            x[..., 5:self.num_classes - 1 + 5],
            targets[..., 5:self.num_classes - 1 + 5],
            weight=targets_weights[..., 5:self.num_classes - 1 + 5],
            avg_factor=num_total_pos)
        loss_id = self.loss_id(
            self.fc(x[..., self.num_classes - 1 + 5:]),
            targets[..., self.num_classes - 1 + 5:],
            weight=targets_weights[..., self.num_classes - 1 + 5:],
            avg_factor=num_total_pos)

        return loss_bbox, loss_obj, loss_cls, loss_id

    def loss(self, head_outs,
             gt_bboxes,
             gt_labels,
             gt_ids,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        batch_size = len(gt_bboxes)
        # max_num_gts = max([gt_bboxes[i].shape[0] for i in range(batch_size)])
        max_num_gts = 70
        gt_bboxes_list = gt_bboxes[0].new_zeros((batch_size, max_num_gts, 4))
        gt_labels_list = gt_labels[0].new_zeros((batch_size, max_num_gts))
        gt_ids_list = gt_ids[0].new_zeros((batch_size, max_num_gts))
        for i in range(batch_size):
            num_gts = min(gt_bboxes[i].shape[0], max_num_gts)
            gt_bboxes_list[i, :num_gts] = gt_bboxes[i][:num_gts]
            gt_labels_list[i, :num_gts] = gt_labels[i][:num_gts] \
                if not self.use_sigmoid_cls else gt_labels[i][:num_gts] - 1
            gt_ids_list[i, :num_gts] = gt_ids[i][:num_gts]

        loss_bbox, loss_obj, loss_cls, loss_id = multi_apply(
            self.get_targets,
            head_outs,
            self.scale_x_y,
            self.multi_level_anchors,
            self.anchor_shifts_list,
            self.anchor_strides,
            list(range(self.num_levels)),
            gt_bboxes=gt_bboxes_list,
            gt_labels=gt_labels_list,
            gt_ids=gt_ids_list,
            num_gts_pre_img=max_num_gts,
            cfg=cfg
        )

        return dict()

    def get_targets(self,
                    x,
                    scale_x_y,
                    anchors,
                    anchor_shifts,
                    stride,
                    level_index,
                    gt_bboxes,
                    gt_labels,
                    gt_ids,
                    num_gts_pre_img,
                    cfg):
        b, _, h, w = x.size()
        x = x.view(b, 3, -1, h, w).permute(0, 1, 3, 4, 2).contiguous()
        x[..., :2] = torch.sigmoid(x[..., :2])
        x[..., :2] = x[..., :2] * scale_x_y - (scale_x_y - 1) * 0.5

        x[..., :2] = x[..., :2] + anchors[..., :2]
        x[..., 2:4] = torch.exp(x[..., 2:4]) * anchors[..., 2:]
        # cx, cy, w, h --> x1, y1, x2, y2
        x[..., :2] -= x[..., 2:4] * 0.5
        x[..., 2:4] += x[..., :2]

        gt_bboxes_level = gt_bboxes / stride
        gt_shifts = gt_bboxes_level.clone()
        # x1, y1, x2, y2 --> 0, 0, w, h
        gt_shifts[..., 2:] -= gt_shifts[..., :2]
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
        targets[bs_inds, anchor_inds, y_inds, x_inds,
                5 + self.cls_out_channels + gt_ids[bs_inds, gt_inds]] = 1
        targets_weights[bs_inds, anchor_inds, y_inds, x_inds] = 1

        num_total_pos = inds.shape[0]

        return self.loss_single(x, targets, targets_weights,
                                num_total_pos if num_total_pos else 1)

    def get_bboxes_and_embeds(self, bbox_outs, embed_outs, img_metas, cfg,
                              rescale=False):
        num_levels = len(bbox_outs)
        mlvl_anchors = self.get_mlvl_anchors(bbox_outs, num_levels)

        result_list = []
        for img_id in range(len(img_metas)):
            single_bbox_output = self.get_image_outs(bbox_outs,
                                                     num_levels,
                                                     img_idx=img_id)
            single_embed_output = self.get_image_outs(embed_outs,
                                                      num_levels,
                                                      img_idx=img_id)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            single_result = self.get_bboxes_and_embeds_single(
                single_bbox_output,
                single_embed_output,
                mlvl_anchors,
                img_shape,
                scale_factor,
                cfg,
                rescale)
            result_list.append(single_result)
        return result_list

    def get_bboxes_and_embeds_single(self,
                                     bbox_output,
                                     embed_output,
                                     mlvl_anchors,
                                     img_shape,
                                     scale_factor,
                                     cfg,
                                     rescale=False):
        assert len(bbox_output) == len(embed_output) == len(mlvl_anchors)

        predictions = []
        for i in range(len(bbox_output)):
            x = bbox_output[i]
            _, h, w = x.size()
            x = x.view(self.num_anchors_per_level, -1, h,
                       w).permute(0, 2, 3, 1).contiguous()

            anchors = mlvl_anchors[i].view(self.num_anchors_per_level, h, w, 4)

            x[..., :2] = anchors[..., 2:] * x[..., :2] + anchors[..., :2]
            x[..., 2:4] = anchors[..., 2:] * torch.exp(x[..., 2:4])

            x[..., :2] -= x[..., 2:4] * 0.5
            x[..., 2:4] += x[..., :2]

            x[..., 4:] = torch.softmax(x[..., 4:], dim=-1)

            e = embed_output[i]
            e = e.unsqueeze(0).repeat(self.num_anchors_per_level, 1, 1, 1)
            e = e.permute(0, 2, 3, 1).contiguous()
            e = F.normalize(e, dim=-1)

            pred = torch.cat([x[..., :4], x[..., 5:6], e], dim=-1)
            pred = pred.view(-1, pred.size()[-1])

            predictions.append(pred)
        predictions = torch.cat(predictions[::-1], dim=0)

        return predictions
