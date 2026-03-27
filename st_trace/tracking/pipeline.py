"""
Coarse-to-Fine Video Retrieval Pipeline.

This implements the full cascaded pipeline:
Stage 1: Topology-Guided Temporal Filtering
Stage 2: Keyframe Extraction (1fps)
Stage 3: Lightweight Detection (YOLOv8n)
Stage 4: ReID Matching (TransReID) with adaptive threshold
Stage 5: Iterative Refinement
"""
import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

from ..data.graph import CameraGraph
from ..search.st_anbs import STANBS, SearchCandidate
from ..models.detector import YOLOPersonDetector
from ..models.reid.st_contrastive import temporal_aggregate


class CandidateDetection:
    """Detected person candidate with ReID feature."""
    def __init__(
        self,
        camera_id: int,
        frame_id: int,
        timestamp: float,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        feature: Optional[np.ndarray] = None
    ):
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.bbox = bbox
        self.confidence = confidence
        self.feature = feature

    def to_dict(self) -> Dict[str, Any]:
        return {
            'camera_id': self.camera_id,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'bbox': self.bbox,
            'confidence': self.confidence
        }


class RetrievedTrajectory:
    """Retrieved trajectory with matching scores."""
    def __init__(
        self,
        path: List[int],
        candidates: List[CandidateDetection],
        total_score: float,
        max_similarity: float
    ):
        self.path = path
        self.candidates = candidates
        self.total_score = total_score
        self.max_similarity = max_similarity


class CoarseToFineRetrieval:
    """
    Full coarse-to-fine video retrieval pipeline for ST-Trace.

    Implements the 5-stage pipeline as described in the paper.
    """

    def __init__(
        self,
        camera_graph: CameraGraph,
        st_anbs: STANBS,
        detector: YOLOPersonDetector,
        reid_model: torch.nn.Module,
        similarity_threshold: float = 0.75,
        keyframe_fps: float = 1.0,
        alpha_adaptive: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the pipeline.

        Args:
            camera_graph: Camera topology graph
            st_anbs: ST-ANBS search instance
            detector: YOLO person detector
            reid_model: Trained ReID model
            similarity_threshold: Base similarity threshold for matching
            keyframe_fps: Keyframe sampling rate (default 1fps)
            alpha_adaptive: Adaptive threshold parameter: τ = μ + α·σ
                           0 = use fixed threshold, >0 = adaptive
            device: Torch device
        """
        self.camera_graph = camera_graph
        self.st_anbs = st_anbs
        self.detector = detector
        self.reid_model = reid_model
        self.similarity_threshold = similarity_threshold
        self.keyframe_fps = keyframe_fps
        self.alpha_adaptive = alpha_adaptive
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reid_model.to(self.device)
        self.reid_model.eval()

    def retrieve(
        self,
        start_camera: int,
        start_time: float,
        probe_feature: np.ndarray,
        video_root: str,
        fps: int = 30,
        do_iterative_refinement: bool = True
    ) -> Tuple[List[RetrievedTrajectory], Dict[int, Tuple[float, float]]]:
        """
        Perform full coarse-to-fine retrieval starting from a probe.

        Args:
            start_camera: Probe camera ID
            start_time: Probe timestamp in seconds
            probe_feature: Pre-extracted ReID feature of the probe
            video_root: Root directory containing video files
            fps: Original video FPS
            do_iterative_refinement: Whether to do iterative refinement

        Returns:
            (retrieved_trajectories, temporal_map)
        """
        # Stage 1: ST-ANBS search to get candidate trajectories and temporal map
        candidates, temporal_map = self.st_anbs.search(start_camera, start_time)

        retrieved: List[RetrievedTrajectory] = []

        # Keep track of visited cameras for iterative refinement
        visited_cameras = {start_camera}

        current_probe_feature = probe_feature.copy()

        # Process each camera with predicted temporal interval
        for camera, (t_min, t_max) in temporal_map.items():
            if camera == start_camera:
                continue

            # Get video path for this camera
            video_path = os.path.join(video_root, f'cam{camera:02d}.mp4')
            if not os.path.exists(video_path):
                continue

            # Stage 2: Keyframe extraction - sample 1fps in temporal window
            keyframes = self._sample_keyframes(video_path, t_min, t_max, fps)
            if not keyframes:
                continue

            # Stage 3: Lightweight detection
            detections = self._detect_persons(keyframes)
            if not detections:
                continue

            # Stage 4: ReID feature extraction and matching
            matched = self._match_candidates(
                detections, current_probe_feature, video_path
            )

            if matched:
                for det, similarity in matched:
                    # Add to result
                    trajectory_path = [start_camera, camera]
                    retrieved.append(RetrievedTrajectory(
                        path=trajectory_path,
                        candidates=[det],
                        total_score=similarity,
                        max_similarity=similarity
                    ))

                    # Stage 5: Iterative refinement
                    if do_iterative_refinement:
                        # Update probe feature with matched detection
                        if det.feature is not None:
                            current_probe_feature = det.feature
                            visited_cameras.add(camera)

        # Sort by similarity
        retrieved.sort(key=lambda x: x.max_similarity, reverse=True)
        return retrieved, temporal_map

    def _sample_keyframes(
        self,
        video_path: str,
        t_min: float,
        t_max: float,
        original_fps: int
    ) -> List[Tuple[np.ndarray, float, int]]:
        """
        Stage 2: Sample keyframes at 1fps within temporal window.

        Returns:
            List of (frame_image, timestamp, frame_id)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        keyframes = []
        sample_interval = int(original_fps / self.keyframe_fps)
        frame_start = int(t_min * original_fps)
        frame_end = int(t_max * original_fps)

        frame_id = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        while frame_id <= (frame_end - frame_start):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % sample_interval == 0:
                timestamp = (frame_start + frame_id) / original_fps
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.append((frame_rgb, timestamp, frame_start + frame_id))

            frame_id += 1

        cap.release()
        return keyframes

    def _detect_persons(
        self,
        keyframes: List[Tuple[np.ndarray, float, int]]
    ) -> List[Tuple[CandidateDetection, np.ndarray]]:
        """
        Stage 3: Detect persons in keyframes.
        """
        detections = []
        for frame, timestamp, frame_id in keyframes:
            boxes = self.detector.detect(frame)
            for (x1, y1, x2, y2, conf) in boxes:
                det = CandidateDetection(
                    camera_id=0,  # filled later
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )
                # Crop detection patch
                h, w = frame.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                crop = frame[y1:y2, x1:x2]
                detections.append((det, crop))
        return detections

    def _match_candidates(
        self,
        detections: List[Tuple[CandidateDetection, np.ndarray]],
        probe_feature: np.ndarray,
        video_path: str
    ) -> List[Tuple[CandidateDetection, float]]:
        """
        Stage 4: Extract ReID features and match with adaptive threshold.
        """
        if not detections:
            return []

        # Preprocess all crops and extract features
        features = []
        valid_dets = []
        current_camera = int(os.path.basename(video_path).split('.')[0][3:])

        from ..data.transforms import get_val_transform
        transform = get_val_transform()

        with torch.no_grad():
            for det, crop in detections:
                pil_crop = Image.fromarray(crop)
                tensor = transform(pil_crop).unsqueeze(0).to(self.device)

                # Extract feature
                feature = self.reid_model.forward_features(tensor)
                feature = feature.cpu().numpy()
                feature = feature / np.linalg.norm(feature)

                det.camera_id = current_camera
                det.feature = feature
                features.append(feature)
                valid_dets.append(det)

        if not valid_dets:
            return []

        # Compute similarities
        probe_norm = probe_feature / np.linalg.norm(probe_feature)
        similarities = [float(np.dot(probe_norm, f)) for f in features]

        # Adaptive threshold
        if self.alpha_adaptive > 0:
            sim_arr = np.array(similarities)
            mu = sim_arr.mean()
            sigma = sim_arr.std()
            threshold = mu + self.alpha_adaptive * sigma
        else:
            threshold = self.similarity_threshold

        # Keep above threshold
        matched = []
        for det, sim in zip(valid_dets, similarities):
            if sim >= threshold:
                matched.append((det, sim))

        # Sort by similarity
        matched.sort(key=lambda x: x[1], reverse=True)
        return matched

    def compute_vrr(
        self,
        temporal_map: Dict[int, Tuple[float, float]],
        total_duration: float,
        fps: int = 30
    ) -> float:
        """
        Compute Video Retrieval Ratio (VRR = processed_frames / total_frames).

        Lower VRR means more efficient.
        """
        processed_seconds = 0.0
        for t_min, t_max in temporal_map.values():
            processed_seconds += (t_max - t_min)

        total_frames = int(total_duration * fps)
        processed_frames = int(processed_seconds * self.keyframe_fps)

        return 100.0 * processed_frames / total_frames
