"""
Efficiency Metrics for Multi-Camera Tracking.

Implements:
- VRR (Video Retrieval Ratio): lower = better, more pruning
- FPS: Frames per second processing speed
- Runtime breakdown per component
"""
import time
import numpy as np
from typing import Dict, List, Optional, Callable
import torch


class Timer:
    """Simple timer for measuring execution time."""
    def __init__(self):
        self.start_time = None
        self.total_time = 0.0
        self.count = 0

    def start(self):
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        self.total_time += elapsed
        self.count += 1
        self.start_time = None
        return elapsed

    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_time / self.count

    def total(self) -> float:
        return self.total_time

    def reset(self):
        self.total_time = 0.0
        self.count = 0


def compute_vrr(
    processed_frames: int,
    total_frames: int
) -> float:
    """
    Compute Video Retrieval Ratio (VRR).

    VRR = (processed_frames / total_frames) * 100%
    Lower VRR means more effective pruning.

    Args:
        processed_frames: Number of frames actually processed
        total_frames: Total number of frames in dataset

    Returns:
        VRR percentage (0-100, lower better)
    """
    if total_frames == 0:
        return 100.0
    return 100.0 * processed_frames / total_frames


def compute_fps(
    total_frames: int,
    total_time_seconds: float
) -> float:
    """
    Compute FPS (frames per second).

    Higher FPS = faster processing.

    Args:
        total_frames: Number of frames processed
        total_time_seconds: Total wall-clock time

    Returns:
        FPS value (higher better)
    """
    if total_time_seconds == 0:
        return 0.0
    return float(total_frames) / total_time_seconds


class EfficiencyTracker:
    """
    Tracks efficiency metrics during execution:
    - Timing breakdown by component
    - Counts of processed frames/candidates
    - Computes VRR and FPS
    """

    def __init__(self, total_frames_total: int):
        """
        Initialize tracker.

        Args:
            total_frames_total: Total frames in the whole dataset
        """
        self.total_frames_total = total_frames_total
        self.timers: Dict[str, Timer] = {}
        self.processed_frames = 0
        self.processed_candidates = 0
        self.start_time_total = None

    def start_total(self):
        """Start timing total runtime."""
        self.start_time_total = time.time()

    def stop_total(self) -> float:
        """Stop timing total runtime and return total seconds."""
        if self.start_time_total is None:
            return 0.0
        return time.time() - self.start_time_total

    def add_timer(self, name: str) -> Timer:
        """Add a timer for a specific component."""
        timer = Timer()
        self.timers[name] = timer
        return timer

    def get_timer(self, name: str) -> Timer:
        """Get existing timer or create new."""
        if name not in self.timers:
            return self.add_timer(name)
        return self.timers[name]

    def increment_processed(self, frames: int = 1):
        """Increment count of processed frames."""
        self.processed_frames += frames

    def increment_candidates(self, count: int = 1):
        """Increment count of processed candidates."""
        self.processed_candidates += count

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all efficiency metrics.

        Returns:
            Dictionary with:
            - VRR: Video Retrieval Ratio percentage
            - FPS: Overall frames per second
            - total_time_seconds: Total runtime
        """
        vrr = compute_vrr(self.processed_frames, self.total_frames_total)
        total_time = self.stop_total() if self.start_time_total else 0.0
        fps = compute_fps(self.processed_frames, total_time)

        result = {
            'VRR': vrr,
            'FPS': fps,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60.0,
            'processed_frames': self.processed_frames
        }

        # Add component breakdown
        for name, timer in self.timers.items():
            result[f'time_{name}'] = timer.average()
            result[f'total_time_{name}'] = timer.total()

        return result

    def print_breakdown(self):
        """Print timing breakdown."""
        metrics = self.compute_metrics()
        print("\n=== Efficiency Metrics ===")
        print(f"VRR: {metrics['VRR']:.1f}% (lower = better)")
        print(f"FPS: {metrics['FPS']:.1f} (higher = faster)")
        print(f"Total runtime: {metrics['total_time_minutes']:.1f} min")
        print("\nComponent timing (average per query):")
        for name in sorted(self.timers.keys()):
            avg_ms = self.timers[name].average() * 1000
            pct = 100 * self.timers[name].total() / metrics['total_time_seconds']
            print(f"  {name}: {avg_ms:.1f} ms ({pct:.1f}%)")
