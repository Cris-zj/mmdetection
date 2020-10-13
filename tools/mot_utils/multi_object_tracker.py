from abc import abstractmethod
import numpy as np
import motmetrics as mm

from .base import Track
from .utils.distance import compute_embedding_distance, compute_iou_distance
from .utils.kalman_filter import KalmanFilter
from .utils.matching import gate_cost_matrix, linear_assignment


class MultiObjectTracker(object):

    def __init__(self,
                 similarity_metric='euclidean',
                 similarity_thr=0.7,
                 iou_thr=0.5,
                 budget_size=30):
        self.kalman_filter = KalmanFilter()
        self.matching_similarity_metric = similarity_metric
        self.matching_similarity_thr = similarity_thr
        self.matching_iou_thr = iou_thr
        self.budget_size = budget_size

    def init_tracks(self, frame_rate=30):
        self.tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

        self._track_ids = 1
        self._frame_id = 0

        self.mot_accumulator = mm.MOTAccumulator(auto_id=True)
        self.max_age = int(frame_rate / 30.0 * self.budget_size)

    @property
    def frame_id(self):
        """int: Current frame id."""
        return self._frame_id

    @property
    def track_id(self):
        """int: Next target id."""
        return self._track_ids

    @property
    def mot_accumulation(self):
        """MOTAccumulator: MOT accumulation results."""
        return self.mot_accumulator

    @abstractmethod
    def forward(self, img):
        pass

    def update(self, detections):
        self._frame_id += 1

        activated_tracks = []
        refined_tracks = []
        cur_lost_tracks = []
        cur_removed_tracks = []

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed()]

        candidate_tracks = confirmed_tracks + self.lost_tracks
        for track in candidate_tracks:
            if not track.is_confirmed():
                track.mean[7] = 0
            track.predict(self.kalman_filter)

        # Policy1: matching for targets by feature similarity
        target_features = np.array([t.features for t in candidate_tracks])
        detection_features = np.array([d.feature for d in detections])
        cost_matrix = compute_embedding_distance(
            target_features,
            detection_features,
            metric=self.matching_similarity_metric
        )
        cost_matrix = gate_cost_matrix(self.kalman_filter,
                                       cost_matrix,
                                       candidate_tracks,
                                       detections)
        matches, unmatched_track_indices, unmatched_detection_indices = \
            linear_assignment(cost_matrix, self.matching_similarity_thr)
        for track_idx, detection_idx in matches:
            track = candidate_tracks[track_idx]
            detection = detections[detection_idx]
            if track.is_confirmed():
                track.update(self.kalman_filter, detection, self.frame_id)
            else:
                track.update(self.kalman_filter, detection, self.frame_id)
                refined_tracks.append(track)

        # Policy2: matching for targets by iou distance
        remaining_tracks = [candidate_tracks[i]
                            for i in unmatched_track_indices
                            if candidate_tracks[i].is_confirmed()]
        remaining_tracks.extend(unconfirmed_tracks)
        target_bboxes = np.array([t.tlbr for t in remaining_tracks])
        detections = [detections[i] for i in unmatched_detection_indices]
        detection_bboxes = np.array([d.tlbr for d in detections])
        cost_matrix = compute_iou_distance(target_bboxes, detection_bboxes)
        matches, unmatched_track_indices, unmatched_detection_indices = \
            linear_assignment(cost_matrix, self.matching_iou_thr)
        for track_idx, detection_idx in matches:
            track = remaining_tracks[track_idx]
            detection = detections[detection_idx]
            track.update(self.kalman_filter, detection, self.frame_id)
        for track_idx in unmatched_track_indices:
            track = remaining_tracks[track_idx]
            if track.is_confirmed():
                track.mark_lost()
                cur_lost_tracks.append(track)
            else:
                track.mark_removed()
                cur_removed_tracks.append(track)

        # init new tracks
        for detection_idx in unmatched_detection_indices:
            detection = detections[detection_idx]
            mean, covariance = self.kalman_filter.initiate(detection.xyah)
            track = Track(mean,
                          covariance,
                          detection.feature,
                          self.track_id,
                          self.frame_id,
                          budget_size=self.max_age)
            activated_tracks.append(track)
            self._track_ids += 1

        for track in self.lost_tracks:
            if self.frame_id - track.last_frame_id > self.max_age:
                track.mark_removed()
                cur_removed_tracks.append(track)

        # t is not lost and not removed
        self.tracks = [t for t in self.tracks if t.is_confirmed()]  # confirmed
        self.tracks.extend(refined_tracks)  # not confirmed
        self.tracks.extend(activated_tracks)  # not confirmed

        self.lost_tracks = [t for t in self.lost_tracks if t.is_lost()]
        self.lost_tracks.extend(cur_lost_tracks)

        self.removed_tracks.extend(cur_removed_tracks)

    def output(self):
        track_ids = []
        track_bboxes = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            track_ids.append(track.track_id)
            bbox = track.tlwh
            track_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])

        return dict(
            frame_id=self.frame_id,
            track_ids=track_ids,
            track_bboxes=track_bboxes
        )

    def mot_eval(self,
                 gt_ids,
                 gt_bboxes,
                 gt_labels,
                 pred_ids,
                 pred_bboxes,
                 iou_thr=0.5,
                 valid_labels=None):
        assert valid_labels is None or isinstance(valid_labels, list)
        if valid_labels is not None:
            valid_inds = np.zeros(gt_ids.shape[0], dtype=np.bool)
            for label in valid_labels:
                valid_inds |= (gt_labels == label)
            gt_ids = gt_ids[valid_inds]
            gt_bboxes = gt_bboxes[valid_inds]

        dist = mm.distances.iou_matrix(
            gt_bboxes, pred_bboxes, max_iou=iou_thr)
        self.mot_accumulator.update(gt_ids, pred_ids, dist)

    @ staticmethod
    def save_txt(save_file, results):
        f = open(save_file, 'w')
        for result in results:
            frame_id = result['frame_id']
            track_ids = result['track_ids']
            track_bboxes = result['track_bboxes']
            for i in range(len(track_ids)):
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    frame_id,
                    track_ids[i],
                    track_bboxes[i][0],
                    track_bboxes[i][1],
                    track_bboxes[i][2],
                    track_bboxes[i][3]),
                    file=f)
        f.close()

    @ staticmethod
    def get_summary(results, names, metrics=mm.metrics.motchallenge_metrics):
        mh = mm.metrics.create()
        summary = mh.compute_many(
            results,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        str_summary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        return summary, str_summary

    @ staticmethod
    def save_summary(save_file, summary):
        pass
