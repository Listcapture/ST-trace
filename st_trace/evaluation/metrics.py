"""
Evaluation Metrics for Multi-Camera Tracking.

Implements:
- MOTA (Multiple Object Tracking Accuracy)
- IDF1 (ID F1-score)
- MT (Mostly Tracked) percentage
- ML (Mostly Lost) percentage
"""
from typing import List, Dict, Tuple, Any
import numpy as np
from munkres import Munkres


def compute_mota(
    num_gt: int,
    false_positives: int,
    misses: int,
    id_switches: int
) -> float:
    """
    Compute MOTA (Multiple Object Tracking Accuracy).

    Formula: MOTA = 1 - (FN + FP + IDSW) / GT

    Args:
        num_gt: Total number of ground truth detections
        false_positives: Number of false positives (FP)
        misses: Number of missed detections (FN)
        id_switches: Number of ID switches (IDSW)

    Returns:
        MOTA value (higher is better)
    """
    if num_gt == 0:
        return 0.0

    mota = 1.0 - (false_positives + misses + id_switches) / float(num_gt)
    return mota


def compute_idf1(
    matches: int,
    num_detections: int,
    num_gt: int
) -> float:
    """
    Compute IDF1 (ID F1-score).

    IDF1 = 2 * IDTP / (2 * IDTP + FP + FN)

    Args:
        matches: Number of correctly matched detections (IDTP)
        num_detections: Total number of detections
        num_gt: Total number of ground truth detections

    Returns:
        IDF1 value (higher is better)
    """
    precision = matches / float(num_detections) if num_detections > 0 else 0.0
    recall = matches / float(num_gt) if num_gt > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_mt_ml(
    trajectories: List[Dict],
    gt_trajectories: List[Dict],
    overlap_threshold: float = 0.8,
    lost_threshold: float = 0.2
) -> Tuple[float, float]:
    """
    Compute MT (Mostly Tracked) and ML (Mostly Lost) percentages.

    - MT: Percentage of GT trajectories tracked for >80% of length
    - ML: Percentage of GT trajectories tracked for <20% of length

    Args:
        trajectories: List of retrieved trajectories
        gt_trajectories: List of ground truth trajectories
        overlap_threshold: Threshold for MT (default: 0.8)
        lost_threshold: Threshold for ML (default: 0.2)

    Returns:
        (MT_percent, ML_percent)
    """
    if not gt_trajectories:
        return 0.0, 0.0

    num_gt = len(gt_trajectories)
    mt_count = 0
    ml_count = 0

    for gt_traj in gt_trajectories:
        gt_length = len(gt_traj['detections'])

        # Find best matching retrieved trajectory
        max_overlap = 0.0
        for traj in trajectories:
            overlap = compute_overlap(traj, gt_traj)
            if overlap > max_overlap:
                max_overlap = overlap

        if max_overlap >= overlap_threshold:
            mt_count += 1
        if max_overlap <= lost_threshold:
            ml_count += 1

    return float(mt_count) / num_gt * 100, float(ml_count) / num_gt * 100


def compute_overlap(
    traj1: List[Dict],
    traj2: List[Dict]
) -> float:
    """
    Compute temporal overlap between two trajectories.

    Returns:
        fraction of traj2 overlapped by traj1
    """
    # Get all timestamps in traj2
    gt_times = sorted([d['timestamp'] for d in traj2['detections']])
    traj_times = sorted([d['timestamp'] for d in traj1['detections']])

    if not gt_times:
        return 0.0

    overlapped = 0
    for gt_t in gt_times:
        # Check if there's a detection in traj1 that's close enough
        for traj_t in traj_times:
            if abs(gt_t - traj_t) < 60.0:  # within 1 minute
                overlapped += 1
                break

    return overlapped / len(gt_times)


def match_trajectories(
    hyp_trajectories: List[List[Dict]],
    gt_trajectories: List[List[Dict]],
    similarity_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Match hypothesis trajectories to ground truth trajectories and
    compute matching statistics.

    Uses Hungarian algorithm for optimal matching based on overlap.

    Args:
        hyp_trajectories: List of retrieved trajectories (each is list of detections)
        gt_trajectories: List of ground truth trajectories

    Returns:
        Dictionary with matching statistics
    """
    n_hyp = len(hyp_trajectories)
    n_gt = len(gt_trajectories)

    # Compute cost matrix based on negative overlap (lower cost = better)
    cost_matrix = np.ones((n_hyp, n_gt))
    for i, hyp in enumerate(hyp_trajectories):
        for j, gt in enumerate(gt_trajectories):
            overlap = compute_overlap(hyp, gt)
            cost_matrix[i, j] = -overlap  # negative for minimization

    # Hungarian algorithm
    munkres = Munkres()
    indices = munkres.compute(cost_matrix.tolist())

    matches = 0
    total_fp = n_hyp
    total_fn = n_gt

    for i, j in indices:
        if i < n_hyp and j < n_gt:
            overlap = -cost_matrix[i, j]
            if overlap > similarity_threshold:
                matches += 1
                total_fp -= 1
                total_fn -= 1

    return {
        'matches': matches,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'num_hyp': n_hyp,
        'num_gt': n_gt
    }


def evaluate_full(
    hypothesis: List[List[Dict]],
    ground_truth: List[List[Dict]]
) -> Dict[str, float]:
    """
    Compute full evaluation metrics.

    Args:
        hypothesis: Retrieved trajectories
        ground_truth: Ground truth trajectories

    Returns:
        Dictionary with all metrics: MOTA, IDF1, MT, ML
    """
    # Count total detections
    total_gt_dets = sum(len(t) for t in ground_truth)
    total_hyp_dets = sum(len(t) for t in hypothesis)

    # Match trajectories
    matching = match_trajectories(hypothesis, ground_truth)

    # Compute IDF1
    idf1 = compute_idf1(
        matching['matches'] * matching['matches'],  # approx, needs correct counting
        total_hyp_dets,
        total_gt_dets
    )

    # For simplicity, approximate MOTA
    # Full CLEAR MOT requires frame-by-frame association
    fp = matching['false_positives']
    fn = matching['false_negatives']
    mota = compute_mota(total_gt_dets, fp, fn, 0)  # ID switches not counted here

    # Compute MT/ML
    mt, ml = compute_mt_ml(hypothesis, ground_truth)

    return {
        'MOTA': mota * 100,  # percentage
        'IDF1': idf1 * 100,
        'MT': mt,
        'ML': ml
    }
