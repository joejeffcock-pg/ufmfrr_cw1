from typing import Tuple

import torch
import numpy as np
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment


def match_predictions_to_ground_truths(boxes_pred, boxes_gt, threshold=0.0):
    # maximum matching by linear sum assignment
    iou_matrix = box_iou(torch.Tensor(boxes_pred), torch.Tensor(boxes_gt)).numpy()
    matched_indices = linear_sum_assignment(-iou_matrix)

    # keep matches above IOU threshold
    matches = []
    for k in range(matched_indices[0].size):
        i = matched_indices[0][k]
        j = matched_indices[1][k]
        if iou_matrix[i,j] > threshold:
            matches.append((boxes_pred[i], boxes_gt[j]))

    return matches

def test_points_in_boxes(points, boxes):
    boxes = np.expand_dims(boxes, 0)
    points = np.expand_dims(points, 1)

    x_gt_x1 = points[...,0] > boxes[...,0]
    y_gt_y1 = points[...,1] > boxes[...,1]
    x_lt_x2 = points[...,0] < boxes[...,2]
    y_lt_y2 = points[...,1] < boxes[...,3]

    in_matrix = x_gt_x1 * y_gt_y1 * x_lt_x2 * y_lt_y2
    return in_matrix

def dist_points_from_box_centres(points, boxes):
    boxes_np = np.array(boxes)
    centres = np.zeros((len(boxes), 2), dtype=np.float32)
    centres[:,0] = (boxes_np[:,0] + boxes_np[:,2])/2.0
    centres[:,1] = (boxes_np[:,1] + boxes_np[:,3])/2.0

    centres = np.expand_dims(centres, 0)
    points = np.expand_dims(points, 1)

    dist_matrix = np.linalg.norm(points - centres, axis=2)
    return dist_matrix


def match_points_to_box_centres(points, boxes):
    # calculate distance (cost) matrix
    in_matrix = test_points_in_boxes(points, boxes)
    dist_matrix = dist_points_from_box_centres(points, boxes)

    # set high cost for points outside of each box
    float_max = np.finfo(dist_matrix.dtype).max
    dist_matrix[np.invert(in_matrix)] = float_max

    matched_indices = linear_sum_assignment(dist_matrix)

    matches = []
    for k in range(matched_indices[0].size):
        i = matched_indices[0][k]
        j = matched_indices[1][k]
        if(in_matrix[i,j]):
            matches.append((points[i], boxes[j]))

    return matches


def compute_metrics(no_pred: int, no_gt: int, no_matches: int):
    tp = no_matches
    fp = no_pred - no_matches
    fn = no_gt - no_matches
    error = (no_pred - no_gt)/float(no_gt)
    relative_error = abs(error)

    metrics = {}
    metrics["TP"] = tp
    metrics["FP"] = fp
    metrics["FN"] = fn
    if tp + fp == 0:
        metrics["Precision"] = float('nan') # equivalent to accuracy
    else:
        metrics["Precision"] = tp/float(tp + fp) # equivalent to accuracy
    metrics["Recall"] = tp/float(tp + fn)

    # we define counting_accuracy as (1.0 - relative_error)
    metrics["Error"] = error
    metrics["Counting Accuracy"] = 1.0 - relative_error

    return metrics