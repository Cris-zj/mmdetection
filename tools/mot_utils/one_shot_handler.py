# from mmdet.ops.nms import nms
from torchvision.ops import nms

from .base import Detection
from .multi_object_tracker import MultiObjectTracker
from .build import build_tracktor_encoder


class OneShotMOT(MultiObjectTracker):

    def __init__(self,
                 cfg,
                 **kwargs):
        super(OneShotMOT, self).__init__(**kwargs)
        self.tracktor = build_tracktor_encoder(**cfg.tracktor)

    def init_tracks(self, **kwargs):
        super(OneShotMOT, self).init_tracks(**kwargs)

    def forward(self, img, cfg):
        prediction = self.tracktor(img)[0]

        valid_inds = (prediction[:, 4] > cfg.min_conf).nonzero().squeeze()
        prediction = prediction[valid_inds]

        det_bboxes = prediction[:, :4]
        det_scores = prediction[:, 4]
        keep_inds = nms(det_bboxes, det_scores, cfg.nms_iou_thr)

        prediction = prediction[keep_inds]

        img_shape = img['img_meta'][0].data[0][0]['img_shape']
        ori_shape = img['img_meta'][0].data[0][0]['ori_shape']
        scale_factor = img['img_meta'][0].data[0][0]['scale_factor']
        pad_left = round(
            (img_shape[1] - ori_shape[1] * scale_factor) * 0.5 - 0.1)
        pad_top = round(
            (img_shape[0] - ori_shape[0] * scale_factor) * 0.5 - 0.1)
        prediction[:, [0, 2]] -= pad_left
        prediction[:, [1, 3]] -= pad_top

        prediction[:, 0].clamp_(min=0, max=img_shape[1] - 1)
        prediction[:, 1].clamp_(min=0, max=img_shape[0] - 1)
        prediction[:, 2].clamp_(min=0, max=img_shape[1] - 1)
        prediction[:, 3].clamp_(min=0, max=img_shape[0] - 1)
        prediction[:, :4] /= prediction[:, :4].new_tensor(scale_factor)

        detections = []
        for det in prediction.cpu().numpy():
            detections.append(Detection(det[:4], det[4], det[5:]))

        return detections
