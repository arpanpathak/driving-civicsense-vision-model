<div align="center">

<img src="assets/logo.svg" alt="CivicSense" width="200"/>

# CivicSense

> *Aftermarket AI vision for your windshield. Voice-guided, edge-native, socially aware.*

**Edge AI perception for intersection discipline, lane courtesy, road hazard alerts, and cooperative safety — running on 3D-printed smart glasses or dashcam hardware.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8/v11-00BBFF)](https://github.com/ultralytics/ultralytics)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Cloud GPU](https://img.shields.io/badge/Cloud%20GPU%20Guide-8A2BE2)](CLOUD_TRAINING.md)
[![KMP Companion](https://img.shields.io/badge/KMP-Companion-purple)](frontend/)

</div>

---

## The Vision

CivicSense is an aftermarket edge-AI accessory that clips onto glasses or mounts on a dashcam. It watches the road, understands traffic behavior, detects hazards, and talks to you — politely but firmly.

It's not a self-driving system. It's a **co-pilot that cares about traffic civility**.

### What it says to you

| Situation | Voice Alert |
|-----------|-------------|
| Someone merging slowly ahead | *"Move to the middle lane, someone slow is merging."* |
| You're crawling in the left lane | *"Speed up — too slow! You're holding up traffic."* |
| Someone passing on the right | *"You're getting passed from the right. Maybe choose the correct lane."* |
| Stop sign ahead, you're not slowing | *"Stop sign in 200 feet. You need to brake."* |
| Bear or deer on the road | *"Large animal ahead. Slow down."* |
| Fallen tree or debris | *"Obstruction in the road ahead. Stop or take evasive action."* |
| Emergency vehicle approaching | *"Emergency vehicle behind you. Pull right."* |
| Lane change with no signal | *"Turn signal? Or do you expect everyone to read your mind?"* |
| Cutting across multiple lanes | *"That's three lanes without a signal. Pick a lane and commit."* |
| Late signal during merge | *"Signal first, then merge. That's the deal."* |

### What it does for everyone else

Beyond just alerting you, CivicSense broadcasts to the mesh:

- **Hazard beacons** — Detects fallen trees, animals, debris, crashes and broadcasts their GPS coordinates to nearby vehicles via short-range radio (or cellular fallback).  
  *"Fallen tree reported 500m ahead on Highway 1. Approach with caution."*

- **Officer notification** — When a serious road hazard, blocked intersection, or erratic driving is detected, CivicSense can relay an anonymous report to the nearest patrol unit. Not a dashcam upload — just a data beacon: *"Intersection blocked at Main & 5th. High likelihood of gridlock."*

- **Cooperative awareness** — If three CivicSense units detect the same hazard independently, the system auto-escalates to a verified road condition alert.

This turns every unit from a personal assistant into a **distributed sensor node** — making roads safer for everyone, not just the person wearing them.

---

## Product Form Factors

| Form | Hardware | Target Price |
|------|----------|-------------|
| **Smart glasses clip** | 3D-printed frame + camera module + Qualcomm AR1 | < $50 BOM |
| **Dashcam puck** | Raspberry Pi Zero + camera + Hailo-8L NPU | < $80 BOM |
| **Phone companion** | Uses existing phone camera (GPS + alerts via Bluetooth) | Free app |

All variants process **100% on-device**. No cloud upload. No subscription.

---

## Architecture

```
=====================================================================
                     DRIVING-CIVICSENSE PIPELINE                     
=====================================================================

  [Camera Frame]
       |
  [YOLOv8/v11 ONNX] ---> [NMS] ---> [Detections]
       |
  [Deep SORT Tracker] ---> [Kalman Filter] ---> [Tracks]
       |
  +--------------------+  +---------------------------+  +----------------------------+
  | Intersection       |  | Lane Speed                |  | Turn Signal / Lane         |
  | Module             |  | Module                    |  | Change Module              |
  |  * Stop sign       |  |  * Lane assignment        |  |  * Amber light detection   |
  |  * Occupancy       |  |  * Relative velocity      |  |  * Lateral motion track    |
  |  * Deceleration    |  |  * Hysteresis timer       |  |  * Multi-lane cut detect   |
  +---------+----------+  +-------------+-------------+  +--------------+-------------+
           |                             |                             |
  +--------------------------------------------------------------------+
  | Alert Priority Engine                                              |
  |   -> Voice / Haptic / LED / Beacon                                |
  +--------------------------------------------------------------------+
=====================================================================
```

---

## Companion App (Kotlin Multiplatform)

Get real-time alerts from the Rust pipeline right on your phone. The companion app connects over gRPC to display intersection violations, lane warnings, hazards, and more.

**Repo:** [github.com/arpanpathak/civicsense-companion](https://github.com/arpanpathak/civicsense-companion)  
**License:** Apache 2.0 (separate from the pipeline's AGPL v3)

| Platform | UI Framework | Transport |
|----------|-------------|-----------|
| Android | Jetpack Compose + Material 3 | gRPC + OkHttp |
| iOS | SwiftUI | Ktor HTTP (gRPC-web) |

**Shared layer:** Domain models, service interface, and ViewModel live in `frontend/shared/` — one Kotlin codebase, two native UIs.

```bash
# Clone everything (pipeline + companion) in one shot
git clone --recurse-submodules https://github.com/arpanpathak/driving-civicsense-vision-model.git

# Or if already cloned
git submodule update --init --recursive
```

> Full build instructions in the [`frontend/` README](frontend/).

---

## The Problems We Solve

### 1. The Intersection Crisis
- **~40%** of all crashes occur at intersections (NHTSA).
- "Blocking the box" causes T-bone impacts.
- Current ADAS detects vehicles but **fails** to semantically interpret intersection occupancy.

### 2. Left-Lane Camping
- Drivers don't realize the **speed differential** between lanes.
- Camping in the left lane causes compression, road rage, reduced throughput.
- GPS doesn't provide real-time lane-level speed awareness.

### 3. Turn Signal? None. I Turn Now.
- The Family Guy maneuver is real: drivers cut across multiple lanes with zero warning.
- **Missing or late turn signals** cause 25% of lane-change crashes (NHTSA).
- CivicSense detects amber turn-signal lights, tracks lateral vehicle motion, and flags unsignaled lane changes before they become collisions.
- Three specific violations:
  - **No signal** — lane change with zero blinker activation.
  - **Late signal** — blinker comes on after the vehicle is already moving laterally.
  - **Multi-lane cut** — vehicle crosses two or more lanes in a single continuous path.

### 4. Road Hazards Go Unreported
- Fallen trees, debris, wildlife, crashes — often no one reports them until it's too late.
- CivicSense turns every unit into a distributed hazard sensor network.

---

## Alert Types

| Alert | Trigger | Output |
|-------|---------|--------|
| **Stop Sign Warning** | Stop sign detected, ego > 10 mph, distance < 50 ft | *"Stop sign ahead. Brake now."* |
| **Blocked Intersection** | Occupancy > 70%, ego > 15 mph, distance < 30 ft | *"Intersection blocked. Don't enter."* |
| **Merge Right Reminder** | Right lane +5 mph faster for > 3 seconds | *"You're being passed on the right. Move over."* |
| **Slow Traffic Ahead** | Lead vehicle speed < threshold for > 5 seconds | *"Someone slow ahead. Prepare to merge."* |
| **Lane Change No Signal** | Vehicle moves laterally, no amber blinker detected | *"Turn signal? Or do you expect everyone to read your mind?"* |
| **Multi-Lane Cut** | Vehicle crosses 2+ lanes in one path without signal | *"That's three lanes without a signal. Pick a lane and commit."* |
| **Late Signal** | Blinker activates after lateral movement begins | *"Signal first, then merge. That's the deal."* |
| **Road Hazard** | Detected debris / animal / obstruction | Voice + broadcast beacon to mesh |
| **Emergency Vehicle** | Flashing lights detected (future) | *"Emergency vehicle behind. Pull right."* |
| **Speed Feedback** | Ego significantly below traffic flow | *"Speed up — you're holding up traffic."* |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Pipeline Language** | Rust (edition 2021) |
| Detection | YOLOv8n / YOLOv11n via ONNX Runtime |
| Tracking | Deep SORT (custom Rust, Kalman filter + IoU matching) |
| Geometry | Pinhole camera model |
| Voice | Edge TTS (espeak / piper) |
| Mesh | LoRa / BLE / cellular fallback |
| Edge AI | Qualcomm AR1, Coral, RPi5 + Hailo-8L |
| Inference | ONNX Runtime |
| **Companion Language** | Kotlin Multiplatform (Kotlin 2.1.20) |
| Shared logic | KMP common module (coroutines, StateFlow) |
| Android UI | Jetpack Compose + Material 3 |
| iOS UI | SwiftUI |
| App Transport | gRPC (Android: OkHttp, iOS: Ktor) |
| Build | Gradle 8.12, Version Catalog |

### Performance Targets

| Device | Inference Time | Power | Use Case |
|--------|---------------|-------|----------|
| Qualcomm Snapdragon AR1 | ~22 ms | < 500 mW | AR Glasses |
| Google Coral Dev Board | ~15 ms | 2 W | Dashcam |
| Raspberry Pi 5 + Hailo-8L | ~18 ms | 8 W | DIY Kit |

---

## Privacy

- **100% on-device** — no video leaves the device.
- **Hazard beacons are anonymous** — they contain only GPS coordinates and hazard type, no video, no identifier.
- **Officer notifications are data-only** — structured reports, not surveillance footage.
- **No cloud dependency** — everything runs locally.

---

## Quick Start

```bash
# Prerequisites: Rust 1.85+, JDK 17 (for companion app)
git clone --recurse-submodules https://github.com/arpanpathak/driving-civicsense-vision-model.git
cd driving-civicsense-vision-model

# Download a test YOLO model
./scripts/download_test_model.sh

# Run the pipeline on a test video
cargo run -- run --source test_video.mp4 --visualize
```

### Companion App (Android)

```bash
cd frontend
./gradlew :androidApp:assembleDebug
# Install the APK on a device connected to the same network as the pipeline
```

---

## Cloud Training

Don't have an NVIDIA GPU for training YOLO models?

**Rent one for $0.19/hr.** Full guide in [`CLOUD_TRAINING.md`](CLOUD_TRAINING.md):

| Dataset | Epochs | GPU | Est. Cost |
|---------|--------|-----|-----------|
| 5,000 images | 100 | RTX 3090 | **~$0.60** |
| 15,000 images | 150 | RTX 4090 | **~$2.04** |
| 50,000 images | 200 | RTX 4090 | **~$6.80** |

---

## License

**GNU AGPL v3** — protecting against proprietary appropriation.

- You can use, fork, modify, and distribute — even commercially.
- You can deploy it as a service.
- You cannot incorporate it into a closed-source proprietary product without releasing your source code.

---

## Contributing

We need:
- **Rust engineers** — implement detection, tracking, voice output, mesh networking
- **Kotlin / Android engineers** — Jetpack Compose UI, gRPC integration
- **iOS engineers** — SwiftUI views, KMP framework integration
- **Hardware designers** — 3D-printable glasses clip, dashcam enclosure
- **Data labelers** — annotate intersection blocking, wildlife, road debris
- **Testers** — run on your commute and report real-world performance

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<div align="center">

*Built to make every mile a socially aware mile.*

**Star this repo if you believe traffic should be cooperative, not competitive.**

</div>
