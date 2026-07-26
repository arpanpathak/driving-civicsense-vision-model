"""🔍 Object Detection Module

Wraps YOLOv8 / YOLOv11 for real-time detection of traffic signs,
vehicles, and intersection zones on edge hardware.

TODO:
    - [ ] Load and cache the INT8 quantized model
    - [ ] Implement async inference pipeline
    - [ ] Add confidence calibration for edge cases (night, rain)
"""
