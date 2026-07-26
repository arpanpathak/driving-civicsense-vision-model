"""# 🎨 Visualization Utilities

Debug overlay rendering for bounding boxes, tracks, speed labels,
and intersection occupancy grid.

## TODO
- [ ] Draw detections with labels and confidence
- [ ] Render BEV occupancy grid overlay
- [ ] Add track ID and speed labels
- [ ] Color-code alerts (green/yellow/red)
"""

import numpy as np


def draw_detections(frame: np.ndarray, detections: list, class_names: dict) -> np.ndarray:
    """Draw bounding boxes and class labels on frame.

    Args:
        frame: BGR image array.
        detections: List of (x1, y1, x2, y2, conf, class_id).
        class_names: Mapping of class_id -> name string.

    Returns:
        Annotated frame.
    """
    raise NotImplementedError("TODO: Implement overlay rendering")


def draw_alerts(frame: np.ndarray, alerts: list) -> np.ndarray:
    """Draw alert text overlays on frame."""
    raise NotImplementedError("TODO: Implement alert overlay")
