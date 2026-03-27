"""
Spatio-Temporal Contrastive Learning for ReID.

This module implements the topology-aware contrastive loss where:
- Positive pairs: same identity AND spatio-temporally reachable
- Negative pairs: different identity OR not reachable
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class SpatioTemporalContrastiveLoss(nn.Module):
    """
    Spatio-Temporal Contrastive Loss for ReID.

    A pair (i, j) is considered positive if:
    1. Same identity (y_i = y_j)
    2. Spatio-temporally reachable: exists a path between cameras with |t_i - t_j| in feasible range

    The loss is InfoNCE over positives and negatives.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_st: float = 0.3,
        check_reachability: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_st = lambda_st
        self.check_reachability = check_reachability

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        cameras: torch.Tensor,
        timestamps: torch.Tensor,
        camera_graph,
    ) -> torch.Tensor:
        """
        Compute the spatio-temporal contrastive loss.

        Args:
            features: Feature tensor shape (batch_size, feature_dim)
            labels: Identity labels shape (batch_size)
            cameras: Camera IDs shape (batch_size)
            timestamps: Timestamps in seconds shape (batch_size)
            camera_graph: CameraGraph object for reachability checks

        Returns:
            Contrastive loss value
        """
        batch_size = features.size(0)
        device = features.device

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.t()) / self.temperature

        # Compute mask for positive pairs
        # First condition: same identity
        same_identity = (labels.unsqueeze(0) == labels.unsqueeze(1))

        # Second condition: spatio-temporal reachability
        if self.check_reachability:
            reachable = torch.zeros_like(same_identity, dtype=torch.bool)
            camera_graph_cpu = camera_graph

            for i in range(batch_size):
                cam_i = cameras[i].item()
                t_i = timestamps[i].item()
                for j in range(batch_size):
                    if i == j:
                        reachable[i, j] = False
                        continue
                    cam_j = cameras[j].item()
                    t_j = timestamps[j].item()
                    time_diff = abs(t_j - t_i)
                    reachable[i, j] = camera_graph_cpu.is_reachable(
                        cam_i, cam_j, time_diff
                    )
            # Positive must satisfy both conditions
            positive_mask = same_identity & reachable
        else:
            # Without reachability check: just same identity (standard contrastive)
            positive_mask = same_identity
            # Mask out self
            positive_mask.fill_diagonal_(False)

        # Remove all-diagonal (self comparisons)
        positive_mask.fill_diagonal_(False)

        # Compute loss
        loss = 0.0
        n_valid = 0

        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i])[0]
            if len(pos_indices) == 0:
                continue

            # For each anchor i, compute loss over all positives
            for pos_idx in pos_indices:
                # Log softmax over all other samples
                log_prob = F.log_softmax(sim_matrix[i], dim=0)
                loss += -log_prob[pos_idx]
                n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = loss / n_valid
        return self.lambda_st * loss


def triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Standard triplet loss for ReID.

    Args:
        features: (batch_size, feature_dim)
        labels: (batch_size)
        margin: Triplet margin

    Returns:
        Triplet loss
    """
    batch_size = features.size(0)
    device = features.device

    # Compute pairwise distance
    dist_matrix = torch.cdist(features, features, p=2) ** 2

    # For each anchor, find hardest positive and negative
    loss_total = 0.0
    n_valid = 0

    same_identity = labels.unsqueeze(0) == labels.unsqueeze(1)

    for i in range(batch_size):
        # Get all positives and negatives for anchor i
        pos_mask = same_identity[i].clone()
        pos_mask[i] = False  # exclude self
        neg_mask = ~pos_mask

        if not torch.any(pos_mask) or not torch.any(neg_mask):
            continue

        # Hardest positive: maximum distance among positives
        pos_dist = dist_matrix[i][pos_mask]
        hardest_pos = pos_dist.max()

        # Hardest negative: minimum distance among negatives
        neg_dist = dist_matrix[i][neg_mask]
        hardest_neg = neg_dist.min()

        # Triplet loss: max(0, hardest_pos - hardest_neg + margin)
        loss = F.relu(hardest_pos - hardest_neg + margin)
        loss_total += loss
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_total / n_valid


class STContrastiveReIDLoss(nn.Module):
    """
    Combined loss for ReID training:
    L_total = L_ID + λ_triplet * L_triplet + λ_ST * L_ST
    """
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 2048,
        lambda_triplet: float = 0.5,
        lambda_st: float = 0.3,
        temperature: float = 0.07,
        triplet_margin: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_triplet = lambda_triplet
        self.lambda_st = lambda_st

        # Identity classification loss (cross entropy)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.cross_entropy = nn.CrossEntropyLoss()

        # Triplet loss
        self.triplet_margin = triplet_margin

        # Spatio-temporal contrastive loss
        self.st_contrastive = SpatioTemporalContrastiveLoss(
            temperature=temperature,
            lambda_st=1.0  # lambda_st is already in the total combination
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        cameras: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        camera_graph = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            (total_loss, loss_dict) with individual components for logging
        """
        # Identity classification loss
        logits = self.classifier(features)
        loss_id = self.cross_entropy(logits, labels)

        # Triplet loss
        loss_tri = triplet_loss(features, labels, self.triplet_margin)

        # Spatio-temporal contrastive loss (if data provided)
        if cameras is not None and timestamps is not None and camera_graph is not None:
            loss_st_val = self.st_contrastive(
                features, labels, cameras, timestamps, camera_graph
            )
        else:
            loss_st_val = torch.tensor(0.0, device=features.device)

        # Combined loss
        total_loss = (
            loss_id +
            self.lambda_triplet * loss_tri +
            self.lambda_st * loss_st_val
        )

        loss_dict = {
            'loss_id': loss_id.item(),
            'loss_triplet': loss_tri.item(),
            'loss_st': loss_st_val.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


def temporal_aggregate(
    features: torch.Tensor,
    timestamps: torch.Tensor,
    center_time: float,
    sigma_t: float = 30.0
) -> torch.Tensor:
    """
    Temporally-weighted feature aggregation for video sequences.

    As described in paper: emphasizes frames close to the predicted center time.

    f_video = Σ w_k * f_k / Σ w_k, where w_k = exp(-|t_k - t_center| / σ_t)

    Args:
        features: (K, feature_dim) features from K keyframes
        timestamps: (K,) timestamps of keyframes
        center_time: Predicted center time of the temporal window
        sigma_t: Gaussian bandwidth for temporal weighting

    Returns:
        Aggregated feature tensor (feature_dim,)
    """
    if features.ndim == 1:
        return features

    K = features.size(0)
    if K == 0:
        return None

    timestamps = timestamps.to(features.device)
    weights = torch.exp(-torch.abs(timestamps - center_time) / sigma_t)
    weights = weights / weights.sum()

    # Weighted average
    aggregated = (weights.unsqueeze(1) * features).sum(dim=0)
    return aggregated
