"""# 🔗 Multi-Object Tracker (Deep SORT / BoT-SORT)

Assigns and maintains persistent IDs for detected vehicles across frames.
Uses appearance Re-ID embeddings + Kalman filtering for robust tracking.

## TODO
- [ ] Wrap boxmot or custom Deep SORT implementation
- [ ] Implement appearance feature extractor (ResNet-18)
- [ ] Add velocity-aware association cost
- [ ] Handle occlusion recovery
"""

import numpy as np


class Track:
    """A single tracked object."""

    def __init__(self, track_id: int, bbox: tuple, features: np.ndarray):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.features = features
        self.age = 0
        self.is_confirmed = False

    def predict(self):
        """Kalman predict step."""
        raise NotImplementedError("TODO: Kalman filter state prediction")

    def update(self, bbox: tuple, features: np.ndarray):
        """Kalman update step with new detection."""
        raise NotImplementedError("TODO: Kalman filter state update")


class MultiObjectTracker:
    """Orchestrates tracking across frames."""

    def __init__(self, max_cosine_distance: float = 0.2, max_age: int = 30):
        raise NotImplementedError("TODO: Initialize Deep SORT / BoT-SORT tracker")

    def update(self, detections: list, frame: np.ndarray) -> list:
        """Update tracks with new detections from current frame.

        Args:
            detections: List of (x1, y1, x2, y2, conf, class_id)
            frame: Current camera frame (for Re-ID feature extraction)

        Returns:
            List of active Track objects with updated positions.
        """
        raise NotImplementedError("TODO: Implement track association pipeline")
