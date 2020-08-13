import argparse
import os

import colorsys
import cv2
import motmetrics as mm
import numpy as np
import torch

from mmcls.datasets.pipelines import Compose as reid_Compose
from mmcls.models import build_reid
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.ops.nms import nms

from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class VideoStructor(object):

    def __init__(self, detector, extractor, cfg):
        self.detector = detector
        self.extractor = extractor
        self.cfg = cfg
        self.tracker = Tracker(self.cfg.tracktor.metric)
        self.results = {}

    def init_tracker(self):
        self.tracker = Tracker(self.cfg.tracktor.metric)
        self.results = {}

    def __call__(self):
        pass

    def step(self, frame):
        img = cv2.imread(frame.filename)

        det_cfg = self.cfg.tracktor.detection
        if det_cfg.default:
            bbox_result = frame.detections[:, 2:7]
            bbox_result[:, 2:4] += bbox_result[:, :2]
        else:
            bbox_result, other_cls_result = self.get_bboxes(img)

        valid_inds = bbox_result[:, 4] > det_cfg.min_conf
        bbox_result = bbox_result[valid_inds]
        _, keep_inds = nms(bbox_result, det_cfg.nms_iou_thr)
        bbox_result = bbox_result[keep_inds]

        features = self.get_features(img, bbox_result)

        bbox_result[:, 2:4] -= bbox_result[:, :2]
        detections = []
        for bbox, feat in zip(bbox_result, features):
            detections.append(Detection(bbox[0:4], bbox[4], feat))

        self.tracker.predict()
        self.tracker.update(detections)

        frame_results = self.tracker.get_results()
        frame_results = frame_results if frame_results else np.zeros((0, 5))
        self.results[frame.frame_idx] = np.array(frame_results)

    def get_bboxes(self, img):
        bbox_result = self.detector(img)

        needed_labels = self.cfg.detection.labels
        dets_for_reid = bbox_result[needed_labels[0] - 1]
        other_dets = []
        for i in needed_labels[1:]:
            other_dets.append(bbox_result[i - 1])

        return dets_for_reid, tuple(other_dets)

    def get_features(self, img, bbox):
        if bbox.shape[1] <= 5:
            features = self.extractor(img, bbox[:, 0:4]).cpu().numpy()
        else:
            features = bbox[:, 5:]
        return features

    def get_results(self):
        return self.results


def gather_sequence_info(cfg):
    sequence_dir = os.path.join(cfg.data_root)
    sequences = []
    for sequence in sorted(os.listdir(sequence_dir)):
        img_dir = os.path.join(sequence_dir, sequence, cfg.img_prefix)
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(img_dir, f)
            for f in os.listdir(img_dir)}

        gt_file = os.path.join(sequence_dir, sequence, cfg.gt_file)
        groundtruth = None
        if os.path.exists(gt_file):
            groundtruth = np.loadtxt(gt_file, delimiter=',')

        detections = None
        detection_file = os.path.join(sequence_dir, sequence, cfg.det_file)
        if detection_file is not None:
            detections = np.loadtxt(detection_file, delimiter=',')

        if len(image_filenames) > 0:
            image = cv2.imread(next(iter(image_filenames.values())),
                               cv2.IMREAD_GRAYSCALE)
            image_size = image.shape
        else:
            image_size = None

        if len(image_filenames) > 0:
            min_frame_idx = min(image_filenames.keys())
            max_frame_idx = max(image_filenames.keys())
        else:
            raise RuntimeError('Image data is None!')

        if groundtruth is not None:
            groundtruth_dict = {
                frame_idx: groundtruth[groundtruth[:, 0] == frame_idx]
                for frame_idx in range(min_frame_idx, max_frame_idx + 1)
            }
        if detections is not None:
            detections = {
                frame_idx: detections[detections[:, 0] == frame_idx]
                for frame_idx in range(min_frame_idx, max_frame_idx + 1)
            }

        info = {
            frame_idx: {
                'frame_idx': frame_idx,
                'filename': image_filenames[frame_idx],
                'groundtruth': groundtruth_dict[frame_idx],
                'detections': detections[frame_idx]
            }
            for frame_idx in range(min_frame_idx, max_frame_idx + 1)
        }

        info_filename = os.path.join(sequence_dir, sequence, "seqinfo.ini")
        if os.path.exists(info_filename):
            with open(info_filename, "r") as f:
                line_splits = [
                    line.split('=') for line in f.read().splitlines()[1:]
                ]
                info_dict = dict(
                    s for s in line_splits
                    if isinstance(s, list) and len(s) == 2)

            update_ms = 1000 / int(info_dict["frameRate"])
        else:
            update_ms = None

        seq_info = mmcv.Config({
            "sequence_name": sequence,
            "sequence_info": info,
            "image_size": image_size,
            "min_frame_idx": min_frame_idx,
            "max_frame_idx": max_frame_idx,
            "update_ms": update_ms
        })
        sequences.append(seq_info)
    return sequences


def extract_image_patch(image, bbox):
    bbox = bbox.astype(np.int)
    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,)
    print(str_summary)


def create_detect_encoder(cfg, device='cuda:0'):
    # build detector
    print('Building detector ...')
    det_cfg = mmcv.Config.fromfile(cfg.config)
    det_cfg.model.pretrained = None
    detector = build_detector(det_cfg.model, test_cfg=det_cfg.test_cfg)
    _ = load_checkpoint(detector, cfg.checkpoint, map_location='cpu')
    detector.to(device)
    detector.eval()

    img_pipeline = [LoadImage(), cfg.pipeline]
    img_transform = Compose(img_pipeline)

    def detect_encoder(image):
        data = img_transform(dict(img=image))
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        with torch.no_grad():
            return detector(return_loss=False, rescale=not cfg.show, **data)

    return detect_encoder


def create_reid_encoder(cfg, device='cuda:0'):
    # build feature extractor for reid
    print('Building feature extractor ...')
    reid_cfg = mmcv.Config.fromfile(cfg.config)
    reid_cfg.model.pretrained = None
    extractor = build_reid(reid_cfg.model)
    _ = load_checkpoint(extractor, cfg.checkpoint, map_location='cpu')
    extractor.to(device)
    extractor.eval()
    extractor.forward = extractor.extract_feat

    input_size = cfg.input_size
    patch_pipeline = [LoadImage()] + cfg.pipeline
    patch_transform = reid_Compose(patch_pipeline)

    def reid_encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box)
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., input_size).astype(np.uint8)
            data = patch_transform(dict(img=patch))
            image_patches.append(data)
        if len(image_patches) == 0:
            return torch.zeros((0, 512)).to(device)
        data = scatter(collate(
            image_patches, samples_per_gpu=len(image_patches)), [device])[0]
        with torch.no_grad():
            return extractor(**data)

    return reid_encoder


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def parse_args():
    parser = argparse.ArgumentParser(
        description='detection, reid, deepsort.')
    parser.add_argument('--mot_config', help='mot config file path')
    parser.add_argument('--output_dir', help='output result dir')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save_video', action='store_true')

    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = mmcv.Config.fromfile(args.mot_config)
    cfg.output_dir = args.output_dir
    cfg.show = args.show
    if cfg.evaluate:
        mot_accums = []

    print('Parsing sequence info ...')
    sequences = gather_sequence_info(cfg.data)
    num_seqs = len(sequences)

    detector = create_detect_encoder(cfg.detection)
    extractor = create_reid_encoder(cfg.reid)

    videotracker = VideoStructor(detector, extractor, cfg)

    for i, sequence in enumerate(sequences):
        sequence_name = sequence.sequence_name
        sequence_info = sequence.sequence_info
        min_frame_idx = sequence.min_frame_idx
        max_frame_idx = sequence.max_frame_idx
        image_size = sequence.image_size
        print(f'[{i + 1}/{num_seqs}] Processing sequence {sequence_name} ...')
        if args.save_video:
            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            save_video_path = os.path.join(
                cfg.output_dir, f'{sequence_name}.avi')
            writer = cv2.VideoWriter(
                save_video_path, fourcc, 20, (image_size[1], image_size[0]))

        videotracker.init_tracker()

        prog_bar = mmcv.ProgressBar(max_frame_idx)
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            frame_info = sequence_info[frame_idx]
            videotracker.step(frame_info)
            prog_bar.update()

        results = videotracker.get_results()

        print('\n')
        output_file = os.path.join(cfg.output_dir, f'{sequence_name}.txt')
        with open(output_file, 'w') as f:
            for frame_idx, frame_result in results.items():
                if args.save_video:
                    img = cv2.imread(sequence_info[frame_idx].filename)
                for row in frame_result:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                        frame_idx, row[0], row[1], row[2], row[3], row[4]),
                        file=f)
                    if args.save_video:
                        color = create_unique_color_uchar(int(row[0]))
                        pt1 = int(row[1]), int(row[2])
                        pt2 = int(row[1] + row[3]), int(row[2] + row[4])
                        cv2.rectangle(img, pt1, pt2, color, 2)
                        label = str(int(row[0]))
                        text_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_PLAIN, 1, 2)

                        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
                        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                            text_size[0][1]
                        cv2.rectangle(img, pt1, pt2, color, -1)
                        cv2.putText(img, label, center, cv2.FONT_HERSHEY_PLAIN,
                                    1, (255, 255, 255), 2)
                if args.save_video:
                    writer.write(img)

        if cfg.evaluate:
            mot_accum = mm.MOTAccumulator(auto_id=True)

            for frame_idx in range(min_frame_idx, max_frame_idx + 1):
                gt_frame = sequence_info[frame_idx].groundtruth
                res_frame = results[frame_idx]

                gt_ids, gt_bboxes = gt_frame[:, 1], gt_frame[:, 2:6]
                track_ids, track_bboxes = res_frame[:, 0], res_frame[:, 1:5]

                distance = mm.distances.iou_matrix(
                    gt_bboxes, track_bboxes, max_iou=0.5)
                mot_accum.update(gt_ids, track_ids, distance)
            mot_accums.append(mot_accum)

    if cfg.evaluate and mot_accums:
        print('Evaluating results ...')
        evaluate_mot_accums(mot_accums,
                            [sequence.sequence_name for sequence in sequences],
                            generate_overall=True)
