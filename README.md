<div align="center">

# 🚗 Driving-CivicSense-Vision-Model

> *Turning windshield cameras into proactive traffic rule advisors.*

**AI-driven auxiliary perception for intersection discipline and lane-awareness — built in Rust 🦀**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8/v11-00BBFF)](https://github.com/ultralytics/ultralytics)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Cloud GPU](https://img.shields.io/badge/☁️-Cloud%20GPU%20Guide-8A2BE2)](CLOUD_TRAINING.md)

</div>

---

## 🧠 What Is This?

**Driving-CivicSense-Vision-Model** is an aftermarket Computer Vision system for AR glasses or dashcam accessories. It processes real-time forward-facing video to solve **two critical, under-addressed driving failures**:

| # | Problem | Solution |
|---|---------|----------|
| 🛑 | **Stop & Intersection Violations** (T‑bone collisions) | Real‑time stop sign compliance + intersection occupancy detection |
| 🚗 | **Left‑Lane Camping** (traffic compression) | Relative lane speed estimation + "Merge Right" reminders |

The system runs **entirely on-device** (Edge AI) — privacy, < 50ms latency, no cloud dependency.

---

## 🚨 The Problems We Solve

### 1. The Intersection "Blocking" Crisis
- **~40%** of all crashes occur at intersections (NHTSA)
- "Blocking the box" — entering a congested intersection on green and failing to clear — causes T‑bone impacts
- Current ADAS detects vehicles but **fails** to semantically interpret intersection occupancy

### 2. The "Left-Lane Camping" Epidemic
- Many drivers don't realize the **speed differential** between their lane and the lane to their right
- Camping in the left lane causes traffic compression, road rage, and reduces highway throughput
- GPS doesn't provide dynamic, real-time lane-level speed awareness

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DRIVING-CIVICSENSE PIPELINE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Camera Frame]                                             │
│       ↓                                                     │
│  [YOLOv8/v11 ONNX] → [NMS] → [Detections]                  │
│       ↓                                                     │
│  [Deep SORT Tracker] → [Kalman Filter] → [Tracks]           │
│       ↓                                                     │
│  ┌─────────────────┐  ┌──────────────────────────┐          │
│  │ Intersection     │  │ Lane Speed               │          │
│  │ Module           │  │ Module                   │          │
│  │  • Stop sign     │  │  • Lane assignment       │          │
│  │  • Occupancy     │  │  • Relative velocity     │          │
│  │  • Deceleration  │  │  • Hysteresis timer      │          │
│  └────────┬────────┘  └───────────┬──────────────┘          │
│           ↓                       ↓                         │
│  ┌──────────────────────────────────────────┐               │
│  │ Alert Priority Engine                    │               │
│  │   → LED / Haptic / Audio / Log           │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚧 Project Status

**⚠️ Pre‑alpha / Skeleton.** All modules stubbed with `todo!()`.

- [x] Project structure & module layout
- [x] YOLO detector skeleton (`Detection`, `YoloDetector`)
- [x] Deep SORT tracker skeleton (`Track`, `MultiObjectTracker`)
- [x] Intersection module skeleton (`IntersectionAnalyzer`)
- [x] Lane speed module skeleton (`LaneSpeedAnalyzer`)
- [x] Geometry utilities skeleton (`estimate_distance`, filters)
- [x] Visualization utilities skeleton
- [x] Typed config with defaults
- [ ] **ONNX Runtime integration** ![WIP](https://img.shields.io/badge/-TODO-red)
- [ ] **Deep SORT association logic** ![WIP](https://img.shields.io/badge/-TODO-red)
- [ ] **BEV occupancy grid** ![WIP](https://img.shields.io/badge/-TODO-red)
- [ ] **Relative speed estimation** ![WIP](https://img.shields.io/badge/-TODO-red)
- [ ] **100-mile real-world validation** ![WIP](https://img.shields.io/badge/-TODO-red)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🦀 Language | Rust (edition 2021) |
| 👁️ Detection | YOLOv8n / YOLOv11n via ONNX Runtime |
| 🔗 Tracking | Deep SORT (custom Rust implementation) |
| 📐 Geometry | Pinhole camera model + IPM |
| ⚡ Edge AI | Qualcomm AR1, Coral, RPi5 + Hailo-8L |
| 🚀 Inference | ONNX Runtime / TensorRT |

### Performance Targets

| Device | Inference Time | Power | Use Case |
|--------|---------------|-------|----------|
| Qualcomm Snapdragon AR1 | ~22 ms | < 500 mW | AR Glasses |
| Google Coral Dev Board | ~15 ms | 2 W | Dashcam |
| Raspberry Pi 5 + Hailo-8L | ~18 ms | 8 W | DIY Kit |

---

## 🚀 Quick Start (Development)

```bash
# Prerequisites: Rust 1.85+, ONNX Runtime
git clone https://github.com/arpanpathak/driving-civicsense-vision-model.git
cd driving-civicsense-vision-model

# Build (will fail on todo!() stubs — expected!)
cargo build --bin civicsense
```

```bash
# Run (once modules are implemented)
cargo run --bin civicsense -- --source path/to/video.mp4 --visualize
```

> **Note:** `cargo build` will fail on `todo!()` — this is intentional. Pick a module and start hacking!

---

## ☁️ No GPU? No Problem.

Don't have an NVIDIA GPU for training YOLO models?

**Rent one for $0.19/hr.** Full guide with provider comparison, step-by-step RunPod setup, and cost estimates:

| Dataset | Epochs | GPU | Est. Cost |
|---------|--------|-----|-----------|
| 5,000 images | 100 | RTX 3090 | **~$0.60** |
| 15,000 images | 150 | RTX 4090 | **~$2.04** |
| 50,000 images | 200 | RTX 4090 | **~$6.80** |

👉 **[CLOUD_TRAINING.md](CLOUD_TRAINING.md)**

---

## 📊 Alert Types

| Alert | Trigger | Latency |
|-------|---------|---------|
| 🛑 **Stop Sign Warning** | Stop sign detected, ego > 10 mph, distance < 50 ft, no braking | 40 ms |
| ⛔ **Blocked Intersection** | Occupancy > 70%, ego > 15 mph, distance < 30 ft | 50 ms |
| ➡️ **Lane Courtesy Reminder** | Right lane +5 mph faster for > 3 seconds | 60 ms |

---

## 🔒 Privacy

- **100% on-device** — no video leaves the device
- **No cloud dependency** — inference, tracking, and alerts are all local
- **No recording** — real-time analysis only, no persistent video storage

---

## 📜 License

**GNU AGPL v3** — protecting against proprietary appropriation.

- ✅ **You can** use, fork, modify, and distribute — even commercially
- ✅ **You can** deploy it as a service
- ❌ **You cannot** incorporate it into a closed‑source proprietary product without releasing your source code
- ❌ **Big tech** cannot swallow this into proprietary ADAS without giving back

---

## 🤝 Contributing

We need:
- 🦀 **Rust engineers** — implement the modules behind the `todo!()` stubs
- 📸 **Data labelers** — annotate "blocked intersection" scenarios
- 🚗 **Testers** — run on your dashcam and report performance

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<div align="center">

*Built with ❤️ + 🦀 to make every mile a socially aware mile.*

⭐ **Star this repo if you believe traffic should be cooperative, not competitive.**

</div>
