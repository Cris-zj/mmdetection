import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def compute_embedding_distance(input1, input2, metric='euclidean'):
    cost_matrix = np.empty((len(input1), len(input2)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    if metric == 'euclidean':
        _metric = euclidean_distance
    elif metric == 'cosine':
        _metric = cosine_distance
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )
    cost_matrix = _metric(input1, input2)
    # for i in range(len(input1)):
    #     cost_matrix[i, :] = _metric(input1[i], input2).min(axis=0)
    return cost_matrix


def euclidean_distance(input1, input2):
    """Computes euclidean distance.

    Args:
        input1 (numpy.ndarray): 2-D feature matrix.
        input2 (numpy.ndarray): 2-D feature matrix.

    Returns:
        numpy.ndarray: distance matrix.
    """
    m, n = input1.shape[0], input2.shape[0]
    mat1 = np.square(input1).sum(axis=1, keepdims=True).repeat(n, axis=1)
    mat2 = np.square(input2).sum(axis=1, keepdims=True).repeat(m, axis=1)
    distmat = -2. * np.dot(input1, input2.T) + mat1 + mat2.T
    return np.sqrt(distmat).clip(min=0.0)


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (numpy.ndarray): 2-D feature matrix.
        input2 (numpy.ndarray): 2-D feature matrix.

    Returns:
        numpy.ndarray: distance matrix.
    """
    distmat = 1 - np.dot(input1, input2.T)
    return distmat


def compute_iou_distance(input1, input2):
    """Comput iou distance.

    Args:
        input1 (numpy.ndarray): 2-D bbox matrix.
        input2 (numpy.ndarray): 2-D bbox matrix.

    Returns:
        numpy.ndarray: distance matrix.
    """
    iou_mat = bbox_overlaps(input1, input2)
    distmat = 1 - iou_mat
    return distmat
