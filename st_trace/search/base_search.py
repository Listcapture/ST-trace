"""
Base class for graph search algorithms.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
from ..data.graph import CameraGraph


class SearchCandidate:
    """
    A candidate trajectory in the beam search.

    Attributes:
        current_camera: Current camera ID
        path: List of camera IDs in the path so far
        t_min: Minimum possible arrival time at current camera
        t_max: Maximum possible arrival time at current camera
        score: Cumulative path score
        reward: Cumulative reward
    """
    def __init__(
        self,
        current_camera: int,
        path: List[int],
        t_min: float,
        t_max: float,
        score: float,
        reward: float = 0.0
    ):
        self.current_camera = current_camera
        self.path = path
        self.t_min = t_min
        self.t_max = t_max
        self.score = score
        self.reward = reward

    def __lt__(self, other: 'SearchCandidate') -> bool:
        # For sorting, higher score is better
        return self.score < other.score


class BaseGraphSearch(ABC):
    """
    Abstract base class for graph search algorithms.
    """

    def __init__(
        self,
        camera_graph: CameraGraph,
        max_depth: int = 6,
        max_duration_min: float = 30.0,
        gamma: float = 0.9,
        lambda_length: float = 0.1
    ):
        self.camera_graph = camera_graph
        self.max_depth = max_depth
        self.max_duration = max_duration_min * 60.0  # convert to seconds
        self.gamma = gamma
        self.lambda_length = lambda_length

    @abstractmethod
    def search(
        self,
        start_camera: int,
        start_time: float
    ) -> Tuple[List[SearchCandidate], Dict[int, Tuple[float, float]]]:
        """
        Perform graph search starting from given camera and time.

        Args:
            start_camera: Starting camera ID
            start_time: Starting timestamp in seconds

        Returns:
            (all_candidates, temporal_map):
            - all_candidates: List of all candidate trajectories
            - temporal_map: Dict mapping camera -> (t_min, t_max) reachable interval
        """
        pass

    def compute_path_info(
        self,
        candidate: SearchCandidate,
        next_camera: int
    ) -> Tuple[float, float, float, float]:
        """
        Compute new t_min, t_max, score, and length penalty for extending path.

        Args:
            candidate: Current candidate
            next_camera: Next camera to visit

        Returns:
            (new_t_min, new_t_max, new_score, length_penalty)
        """
        c = candidate.current_camera
        dt_min, dt_max = self.camera_graph.get_travel_time_range(c, next_camera)
        distance = self.camera_graph.get_distance(c, next_camera)

        new_t_min = candidate.t_min + dt_min
        new_t_max = candidate.t_max + dt_max
        length_penalty = self.lambda_length * (distance / 10.0)  # normalized

        return new_t_min, new_t_max, length_penalty
