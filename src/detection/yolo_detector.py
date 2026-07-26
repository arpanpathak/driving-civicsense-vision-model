"""# 🎯 YOLOv8 / YOLOv11 Object Detector

Wraps Ultralytics YOLO for real-time inference on edge hardware.
Supports both PyTorch (training) and ONNX/TensorRT (deployment).

## Target Classes
- stop_sign, traffic_light, crosswalk
- vehicle, truck, bus
- intersection_zone

## TODO
- [ ] Load model from ONNX/TensorRT at init
- [ ] Implement pre-processing (letterbox resize to 640x640)
- [ ] Add NMS with class-specific thresholds
- [ ] Profile FPS on target hardware (Raspberry Pi 5, Coral, Qualcomm AR1)
"""

import numpy as np


class YOLODetector:
    """Real-time object detector wrapping a YOLO model."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Args:
            model_path: Path to .onnx or .pt weights.
            conf_threshold: Minimum confidence for a detection.
            iou_threshold: NMS IoU threshold.
        """
        raise NotImplementedError("TODO: Load YOLO model (ONNX/TensorRT)")

    def detect(self, frame: np.ndarray) -> list:
        """Run inference on a single frame.

        Returns:
            List of detections: [(x1, y1, x2, y2, conf, class_id), ...]
        """
        raise NotImplementedError("TODO: Implement forward pass + NMS")

    def __del__(self):
        """Cleanup model resources."""
        pass
