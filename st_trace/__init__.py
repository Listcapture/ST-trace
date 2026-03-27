"""
ST-Trace: Neural Graph Search with Spatio-Temporal Priors for Efficient Multi-Camera Tracking.
"""

__version__ = '0.1.0'

from .data.graph import CameraGraph
from .data.dataset import BaseMCTDataset, Trajectory, Detection
from .data.nlpr_mct import NLPRMCTDataset
from .search.st_anbs import STANBS, ExhaustiveBFS, FixedBeamSearch
from .models.transition_net import TransitionNet
from .models.reid.st_contrastive import SpatioTemporalContrastiveLoss, STContrastiveReIDLoss
from .tracking.pipeline import CoarseToFineRetrieval, RetrievedTrajectory, CandidateDetection

__all__ = [
    'CameraGraph',
    'BaseMCTDataset',
    'Trajectory',
    'Detection',
    'NLPRMCTDataset',
    'STANBS',
    'ExhaustiveBFS',
    'FixedBeamSearch',
    'TransitionNet',
    'SpatioTemporalContrastiveLoss',
    'STContrastiveReIDLoss',
    'CoarseToFineRetrieval',
    'RetrievedTrajectory',
    'CandidateDetection',
]
