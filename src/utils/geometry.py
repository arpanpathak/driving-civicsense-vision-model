"""# 📐 Geometry & Projection Utilities

Inverse Perspective Mapping (IPM), distance estimation, and
bounding box manipulation for the driving vision pipeline.

## TODO
- [ ] Implement pinhole distance estimation: Z = (f * W) / w
- [ ] Add IPM for BEV grid projection
- [ ] Add IoU and bbox utility functions
"""

import numpy as np


def estimate_distance(pixel_width: float, real_width: float, focal_length: float) -> float:
    """Estimate distance using pinhole camera model.

    Args:
        pixel_width: Width of object in pixels (from bounding box).
        real_width: Known real-world width of object class (e.g., car=1.8m).
        focal_length: Camera focal length in pixels.

    Returns:
        Distance Z in meters.
    """
    raise NotImplementedError("TODO: Z = (focal_length * real_width) / pixel_width")


def compute_relative_velocity(
    prev_distance: float, curr_distance: float, dt: float
) -> float:
    """Compute relative velocity from distance change over time.

    Args:
        prev_distance: Distance in previous frame (m).
        curr_distance: Distance in current frame (m).
        dt: Time delta (seconds).

    Returns:
        Relative velocity (m/s). Positive = approaching, negative = receding.
    """
    raise NotImplementedError("TODO: V_rel = dZ / dt")


def low_pass_filter(value: float, prev_value: float, alpha: float = 0.3) -> float:
    """First-order low-pass filter for smoothing noisy measurements.

    Args:
        value: Current measurement.
        prev_value: Previous filtered value.
        alpha: Smoothing factor (0 = max smooth, 1 = no smoothing).

    Returns:
        Filtered value.
    """
    return alpha * value + (1 - alpha) * prev_value
