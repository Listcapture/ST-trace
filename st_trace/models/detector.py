"""
YOLOv8 Detection Wrapper for Person Detection.

This module wraps YOLOv8 for person detection in the coarse-to-fine pipeline.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


class YOLOPersonDetector:
    """
    Wrapper for YOLOv8 person detection.

    Uses YOLOv8n (nano) for lightweight detection as per Stage 3 of the pipeline.
    """

    # COCO class index for 'person' is 0
    PERSON_CLASS = 0

    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[torch.device] = None
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model_name: YOLOv8 model size ('yolov8n.pt', 'yolov8s.pt', etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: NMS IoU threshold
            device: Torch device
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(
        self,
        image: np.ndarray,
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect persons in an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS],
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        return detections

    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Detect persons and return bounding boxes in numpy array.

        Args:
            frame: Input frame (H, W, 3)

        Returns:
            Numpy array shape (N, 5) where each row is [x1, y1, x2, y2, conf]
        """
        detections = self.detect(frame)
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return np.array(detections, dtype=np.float32)
