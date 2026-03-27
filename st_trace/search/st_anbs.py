"""
Adaptive Neural Beam Search (ST-ANBS) Implementation.

This is the core algorithm of ST-Trace:
- Uses learned transition model to predict next camera probabilities
- Keeps top-B candidates at each depth level (beam pruning)
- Maintains temporal reachability intervals
- Has theoretical suboptimality bound
- Complexity: O(B * D_max * d̄) where d̄ is average out-degree
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from ..data.graph import CameraGraph
from .base_search import BaseGraphSearch, SearchCandidate
from ..models.transition_net import TransitionNet


class STANBS(BaseGraphSearch):
    """
    Adaptive Neural Beam Search (ST-ANBS) for multi-camera trajectory search.

    Implements Algorithm 1 from the paper.
    """

    def __init__(
        self,
        camera_graph: CameraGraph,
        transition_model: TransitionNet,
        beam_width: int = 5,
        max_depth: int = 6,
        max_duration_min: float = 30.0,
        gamma: float = 0.9,
        lambda_length: float = 0.1,
        score_threshold: float = -10.0,
        device: torch.device = None
    ):
        """
        Initialize ST-ANBS.

        Args:
            camera_graph: Camera topology graph
            transition_model: Learned transition prediction network
            beam_width: Beam width B for pruning
            max_depth: Maximum search depth D_max
            max_duration_min: Maximum trajectory duration in minutes
            gamma: Discount factor for path scoring
            lambda_length: Length penalty coefficient
            score_threshold: Minimum score threshold to continue search
            device: PyTorch device for model inference
        """
        super().__init__(
            camera_graph=camera_graph,
            max_depth=max_depth,
            max_duration_min=max_duration_min,
            gamma=gamma,
            lambda_length=lambda_length
        )
        self.transition_model = transition_model
        self.beam_width = beam_width
        self.score_threshold = score_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transition_model.eval()
        self.transition_model.to(self.device)

    @torch.no_grad()
    def search(
        self,
        start_camera: int,
        start_time: float
    ) -> Tuple[List[SearchCandidate], Dict[int, Tuple[float, float]]]:
        """
        Perform ST-ANBS search starting from probe.

        Args:
            start_camera: Starting camera ID (probe camera)
            start_time: Starting timestamp (probe time) in seconds

        Returns:
            (all_candidates, temporal_map):
            - all_candidates: All candidate trajectories explored
            - temporal_map: Maps camera to (t_min, t_max) reachable interval
        """
        # Initialize beam with starting point
        # Format: (current_camera, t_min, t_max, score, path)
        initial_candidate = SearchCandidate(
            current_camera=start_camera,
            path=[start_camera],
            t_min=start_time,
            t_max=start_time,
            score=0.0,
            reward=0.0
        )

        current_beam = [initial_candidate]
        all_candidates = [initial_candidate]
        temporal_map: Dict[int, Tuple[float, float]] = {
            start_camera: (start_time, start_time)
        }

        # Depth-first beam search iteration
        for depth in range(1, self.max_depth + 1):
            candidates: List[SearchCandidate] = []

            # Expand each candidate in current beam
            for candidate in current_beam:
                # Check stopping conditions
                current_duration = candidate.t_max - start_time
                if current_duration > self.max_duration:
                    continue
                if candidate.score < self.score_threshold:
                    continue

                # Get all possible neighbor cameras
                neighbors = self.camera_graph.get_neighbors(candidate.current_camera)
                if not neighbors:
                    continue

                # Predict transition probabilities for all neighbors
                probs = self._predict_transition_probabilities(
                    candidate.current_camera,
                    neighbors,
                    candidate.t_max,
                    candidate.path
                )

                # Keep top-B neighbors by probability
                num_keep = min(self.beam_width, len(neighbors))
                top_indices = np.argsort(-probs)[:num_keep]

                # Expand each top neighbor
                for idx in top_indices:
                    next_cam = neighbors[idx]
                    prob = probs[idx]

                    # Skip if probability is extremely low
                    if prob < 1e-6:
                        continue

                    new_t_min, new_t_max, length_penalty = self.compute_path_info(
                        candidate, next_cam
                    )

                    # Compute new score: R = score + gamma^d * log(prob) - lambda * length
                    gamma_d = self.gamma ** (depth - 1)
                    log_prob = np.log(max(prob, 1e-10))
                    new_score = candidate.score + gamma_d * log_prob - length_penalty

                    new_path = candidate.path + [next_cam]

                    new_candidate = SearchCandidate(
                        current_camera=next_cam,
                        path=new_path,
                        t_min=new_t_min,
                        t_max=new_t_max,
                        score=new_score,
                        reward=gamma_d * log_prob
                    )

                    candidates.append(new_candidate)

            if not candidates:
                break

            # Prune to top-B candidates by score for this depth
            candidates.sort(reverse=True)
            num_keep = min(self.beam_width, len(candidates))
            current_beam = candidates[:num_keep]

            # Add to result collection
            all_candidates.extend(current_beam)

            # Update temporal map with reachability intervals
            for candidate in current_beam:
                cam = candidate.current_camera
                if cam not in temporal_map:
                    temporal_map[cam] = (candidate.t_min, candidate.t_max)
                else:
                    # Union the interval
                    curr_min, curr_max = temporal_map[cam]
                    new_min = min(curr_min, candidate.t_min)
                    new_max = max(curr_max, candidate.t_max)
                    temporal_map[cam] = (new_min, new_max)

        return all_candidates, temporal_map

    def _predict_transition_probabilities(
        self,
        from_camera: int,
        to_cameras: List[int],
        current_time: float,
        path: List[int]
    ) -> np.ndarray:
        """
        Predict transition probabilities using the learned model.

        Args:
            from_camera: Current camera ID
            to_cameras: List of candidate next cameras
            current_time: Current timestamp
            path: Current trajectory path

        Returns:
            numpy array of probabilities for each candidate
        """
        # Compute edge features
        edge_features = TransitionNet.compute_edge_features(
            from_camera, to_cameras, self.camera_graph, current_time
        )
        edge_features = edge_features.to(self.device)

        # Compute temporal context: normalized hour and day
        hour = (current_time / 3600.0) % 24 / 24.0
        day = 0.0  # can be set based on actual data
        temporal_context = torch.tensor(
            [[hour, day]], dtype=torch.float32, device=self.device
        )

        # TODO: Encode trajectory history with LSTM
        # For now, simplified: no history encoding for first step
        # This will be extended to use full LSTM encoding
        history = None

        # Forward pass through model
        with torch.no_grad():
            probs = self.transition_model(edge_features, temporal_context, history)

        return probs[0].cpu().numpy()


class ExhaustiveBFS(BaseGraphSearch):
    """
    Exhaustive BFS baseline - no pruning, explores all reachable paths.

    Used as a baseline to compare against ST-ANBS.
    """

    def __init__(
        self,
        camera_graph: CameraGraph,
        max_depth: int = 6,
        max_duration_min: float = 30.0,
        gamma: float = 0.9,
        lambda_length: float = 0.1
    ):
        super().__init__(camera_graph, max_depth, max_duration_min, gamma, lambda_length)

    def search(
        self,
        start_camera: int,
        start_time: float
    ) -> Tuple[List[SearchCandidate], Dict[int, Tuple[float, float]]]:
        """Exhaustive BFS search."""
        from collections import deque

        initial = SearchCandidate(
            current_camera=start_camera,
            path=[start_camera],
            t_min=start_time,
            t_max=start_time,
            score=0.0
        )

        queue = deque([initial])
        all_candidates = [initial]
        temporal_map = {start_camera: (start_time, start_time)}
        visited = set([(start_camera,)])

        while queue:
            candidate = queue.popleft()

            if len(candidate.path) > self.max_depth:
                continue

            current_duration = candidate.t_max - start_time
            if current_duration > self.max_duration:
                continue

            neighbors = self.camera_graph.get_neighbors(candidate.current_camera)
            for next_cam in neighbors:
                # Check for cycles (avoid revisiting same camera)
                if next_cam in candidate.path:
                    continue

                new_t_min, new_t_max, length_penalty = self.compute_path_info(
                    candidate, next_cam
                )

                depth = len(candidate.path)
                gamma_d = self.gamma ** depth
                uniform_prob = 1.0 / len(neighbors)
                new_score = candidate.score + gamma_d * np.log(uniform_prob) - length_penalty

                new_path = candidate.path + [next_cam]

                if tuple(new_path) in visited:
                    continue
                visited.add(tuple(new_path))

                new_candidate = SearchCandidate(
                    current_camera=next_cam,
                    path=new_path,
                    t_min=new_t_min,
                    t_max=new_t_max,
                    score=new_score
                )

                queue.append(new_candidate)
                all_candidates.append(new_candidate)

                if next_cam not in temporal_map:
                    temporal_map[next_cam] = (new_t_min, new_t_max)
                else:
                    curr_min, curr_max = temporal_map[next_cam]
                    temporal_map[next_cam] = (
                        min(curr_min, new_t_min),
                        max(curr_max, new_t_max)
                    )

        return all_candidates, temporal_map


class FixedBeamSearch(BaseGraphSearch):
    """
    Fixed beam search baseline with uniform or handcrafted probabilities.

    Used for ablation study "w/o Neural Scoring".
    """

    def __init__(
        self,
        camera_graph: CameraGraph,
        beam_width: int = 5,
        max_depth: int = 6,
        max_duration_min: float = 30.0,
        gamma: float = 0.9,
        lambda_length: float = 0.1,
        use_distance_based: bool = True
    ):
        super().__init__(camera_graph, max_depth, max_duration_min, gamma, lambda_length)
        self.beam_width = beam_width
        self.use_distance_based = use_distance_based

    def search(
        self,
        start_camera: int,
        start_time: float
    ) -> Tuple[List[SearchCandidate], Dict[int, Tuple[float, float]]]:
        """Fixed beam search with handcrafted scoring."""
        initial_candidate = SearchCandidate(
            current_camera=start_camera,
            path=[start_camera],
            t_min=start_time,
            t_max=start_time,
            score=0.0
        )

        current_beam = [initial_candidate]
        all_candidates = [initial_candidate]
        temporal_map = {start_camera: (start_time, start_time)}

        for depth in range(1, self.max_depth + 1):
            candidates = []

            for candidate in current_beam:
                current_duration = candidate.t_max - start_time
                if current_duration > self.max_duration:
                    continue

                neighbors = self.camera_graph.get_neighbors(candidate.current_camera)
                if not neighbors:
                    continue

                # Score neighbors with handcrafted function
                scores = []
                for n in neighbors:
                    if self.use_distance_based:
                        # Closer distance = higher probability
                        dist = self.camera_graph.get_distance(candidate.current_camera, n)
                        score = -dist  # closer is better
                    else:
                        # Uniform probability
                        score = 0.0
                    scores.append(score)

                # Keep top-B
                num_keep = min(self.beam_width, len(neighbors))
                sorted_indices = np.argsort(-np.array(scores))[:num_keep]

                for idx in sorted_indices:
                    next_cam = neighbors[idx]
                    if next_cam in candidate.path:
                        continue

                    new_t_min, new_t_max, length_penalty = self.compute_path_info(
                        candidate, next_cam
                    )

                    gamma_d = self.gamma ** (depth - 1)
                    if self.use_distance_based:
                        dist = self.camera_graph.get_distance(candidate.current_camera, next_cam)
                        prob = np.exp(-dist / 50.0)
                    else:
                        prob = 1.0 / len(neighbors)

                    new_score = candidate.score + gamma_d * np.log(prob) - length_penalty
                    new_path = candidate.path + [next_cam]

                    new_candidate = SearchCandidate(
                        current_camera=next_cam,
                        path=new_path,
                        t_min=new_t_min,
                        t_max=new_t_max,
                        score=new_score
                    )

                    candidates.append(new_candidate)

            if not candidates:
                break

            candidates.sort(reverse=True)
            current_beam = candidates[:self.beam_width]
            all_candidates.extend(current_beam)

            for candidate in current_beam:
                cam = candidate.current_camera
                if cam not in temporal_map:
                    temporal_map[cam] = (candidate.t_min, candidate.t_max)
                else:
                    curr_min, curr_max = temporal_map[cam]
                    temporal_map[cam] = (
                        min(curr_min, candidate.t_min),
                        max(curr_max, candidate.t_max)
                    )

        return all_candidates, temporal_map
