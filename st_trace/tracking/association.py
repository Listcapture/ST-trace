"""
Trajectory Association for Multi-Camera Tracking.

This module handles association of retrieved candidates into full trajectories
after retrieval.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from munkres import Munkres


def associate_trajectories(
    detections: List[Dict],
    similarity_threshold: float = 0.5,
) -> List[List[Dict]]:
    """
    Associate retrieved detections into full trajectories.

    Uses greedy matching based on ReID similarity.

    Args:
        detections: List of detections across all cameras and timestamps
        similarity_threshold: Minimum similarity for association

    Returns:
        List of full trajectories
    """
    if not detections:
        return []

    # Sort by timestamp
    detections.sort(key=lambda d: d['timestamp'])

    trajectories: List[List[Dict]] = []
    unassigned = detections.copy()

    while unassigned:
        # Start new trajectory with earliest unassigned detection
        current_traj = [unassigned[0]]
        unassigned.pop(0)

        while True:
            last_det = current_traj[-1]
            last_cam = last_det['camera_id']
            last_feat = np.array(last_det['feature'])
            best_sim = 0.0
            best_idx = -1

            for i, cand in enumerate(unassigned):
                if cand['camera_id'] == last_cam:
                continue  # Same camera already handled by single-camera tracking

                sim = float(np.dot(last_feat, np.array(cand['feature'])))
                if sim > best_sim and sim > similarity_threshold:
                    best_sim = sim
                    best_idx = i

            if best_idx >= 0:
                current_traj.append(unassigned[best_idx])
                unassigned.pop(best_idx)
            else:
                break

        trajectories.append(current_traj)

    return trajectories


def hungarian_association(
    cost_matrix: np.ndarray,
    max_cost: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Hungarian algorithm for minimum cost matching.

    Args:
        cost_matrix: Cost matrix shape (n, m)
        max_cost: Maximum cost to allow

    Returns:
        List of (row, col) matches
    """
    n, m = cost_matrix.shape
    # Add dummy rows to make square if needed
    size = max(n, m)
    padded = np.full((size, size), fill_value=max_cost)
    padded[:n, :m] = cost_matrix

    # Convert to list for munkres
    munkres = Munkres()
    cost_list = padded.tolist()
    indices = munkres.compute(cost_list)

    # Filter matches that are within bounds and below max cost
    matches = []
    for i, j in indices:
        if i < n and j < m and padded[i, j] < max_cost:
            matches.append((i, j))

    return matches


def compute_similarity_matrix(
    features1: List[np.ndarray],
    features2: List[np.ndarray]
) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of features.

    Args:
        features1: List of features, shape (d,) each
        features2: List of features, shape (d,) each

    Returns:
        Similarity matrix shape (n1, n2) with values in [0, 1]
    """
    n1 = len(features1)
    n2 = len(features2)
    sim_matrix = np.zeros((n1, n2))

    for i in range(n1):
        f1 = features1[i]
        f1_norm = f1 / np.linalg.norm(f1)
        for j in range(n2):
            f2 = features2[j]
            f2_norm = f2 / np.linalg.norm(f2)
            sim_matrix[i, j] = float(np.dot(f1_norm, f2_norm))

    return sim_matrix
