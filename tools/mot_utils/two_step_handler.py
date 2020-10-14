from mmdet.ops.nms import nms

from .base import Detection
from .multi_object_tracker import MultiObjectTracker
from .build import build_detect_encoder, build_embed_encoder


class TwoStepMOT(MultiObjectTracker):

    def __init__(self,
                 cfg,
                 **kwargs):
        super(TwoStepMOT, self).__init__()
        self.detector = build_detect_encoder(**cfg.detector)
        self.embeddor = build_embed_encoder(**cfg.embeddor)

    def init_tracks(self, **kwargs):
        super(TwoStepMOT, self).init_tracks(**kwargs)

    def forward(self, img, cfg):
        bboxes = self.detector(img)

        valid_labels = cfg.detection.labels
        bboxes = bboxes[valid_labels[0] - 1]

        bboxes = bboxes[bboxes[:, 4] > cfg.min_conf]
        _, keep_inds = nms(bboxes, cfg.nms_iou_thr)
        bboxes = bboxes[keep_inds]

        embeddings = self.embeddor(img, bboxes[:, 0:4]).cpu().numpy()

        detections = []
        for bbox, feat in zip(bboxes, embeddings):
            detections.append(Detection(bbox[0:4], bbox[4], feat))

        return detections
