import numpy as np
from collections import deque


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `removed` to mark them for removal from the set of active
    tracks.

    """  # noqa: E501

    Tentative = 0
    Confirmed = 1
    Lost = 2
    Removed = 3


class Track(object):

    def __init__(self,
                 mean,
                 covariance,
                 feature,
                 track_id,
                 last_frame_id,
                 budget_size=30):
        self.mean = mean
        self.covariance = covariance
        self.feature_pool = deque([], maxlen=budget_size)
        self.feature_pool.append(feature)
        self.feature = feature

        self.track_id = track_id
        self.last_frame_id = last_frame_id

        self.state = TrackState.Tentative
        self.smooth_factor = 0.9

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(track_id={self.track_id}'
        repr_str += f', last_frame_id={self.last_frame_id}'
        repr_str += f', state={self.state}'
        repr_str += f', mean={self.mean})'
        return repr_str

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def features(self):
        # return np.array(self.feature_pool[-1])
        return self.feature

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance)

    def update(self, kf, detection, last_frame_id):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.xyah)
        self.feature_pool.append(detection.feature)
        self.feature = self.smooth_factor * self.feature + \
            (1. - self.smooth_factor) * detection.feature

        self.state = TrackState.Confirmed
        self.last_frame_id = last_frame_id

    def re_activate(self, kf, detection, last_frame_id):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.xyah)
        self.feature_pool.append(detection.feature)
        self.feature = self.smooth_factor * self.feature + \
            (1. - self.smooth_factor) * detection.feature

        self.state = TrackState.Tentative
        self.last_frame_id = last_frame_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_lost(self):
        """Returns True if this track is lost."""
        return self.state == TrackState.Lost


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlbr : array_like
        Bounding box in format `(x1, y1, x2, y2)`.
    conf : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlbr : ndarray
        Bounding box in format `(top left, top left, top left, bottom right)`.
    conf : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """  # noqa: E501

    def __init__(self, tlbr, conf, feature):
        self.tlbr = np.asarray(tlbr, dtype=np.float)
        self.conf = float(conf)
        self.feature = np.asarray(
            feature / np.linalg.norm(feature), dtype=np.float32)

    @property
    def tlwh(self):
        """Convert bounding box to format `(min x, min y, w, h)`, i.e.,
        `(top left x, top left y, width, height)`.
        """
        ret = self.tlbr.copy()
        ret[2:] -= ret[:2]
        return ret

    @property
    def xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
