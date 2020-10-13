# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import kalman_filter


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[indices[:]]
    matched_mask = (matched_cost <= thresh)

    matches = np.array(list(zip(*indices)))[matched_mask]
    unmatched_a = np.array(
        tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0])))
    unmatched_b = np.array(
        tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1])))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix,
                      thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), \
            np.arange(cost_matrix.shape[0]), \
            np.arange(cost_matrix.shape[1])

    cost_matrix[cost_matrix > thresh] = thresh + 1e-5
    indices = linear_sum_assignment(cost_matrix)
    return _indices_to_matches(cost_matrix, indices, thresh)


def gate_cost_matrix(kf,
                     cost_matrix,
                     tracks,
                     detections,
                     gated_cost=np.inf,
                     only_position=False,
                     fuse_lambda=0.98):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[i]` and `detections[j]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([d.xyah for d in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
