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
[![KMP Companion](https://img.shields.io/badge/KMP-Companion-purple)](https://github.com/arpanpathak/civicsense-companion)
[![Book: Seeing Machines](https://img.shields.io/badge/Book-Seeing%20Machines-FF6B6B)](https://arpanpathak.github.io/seeing-machines-book/foreword.html)

</div>

---

## The Vision

CivicSense is an aftermarket edge-AI accessory that clips onto glasses or mounts on a dashcam. It watches the road, understands traffic behavior, detects hazards, and talks to you — politely but firmly.

It's not a self-driving system. It's a **co-pilot that cares about traffic civility** — a civic sense teacher on your dashboard.

### The engineering vision

- **Privacy-first** — 100% on-device inference. No video ever leaves the device.
- **Ultra-fast & low latency** — a perception pipeline that must react in real time, from frame to voice in a blink.
- **Low power-hungry** — squeezing serious computer vision onto watts, not kilowatts.
- **Distributed edge pipeline** — don't cram YOLO into 512 MB; a Pico triggers, a Pi Zero streams, and a Pi 5 / desktop GPU runs the heavy inference.
- **3D-printed hardware accessories** — open, printable frames and dashcam pucks, not black-box gadgets.
- **Copilot as civic sense teacher** — alerts that correct, teach, and nudge good road citizenship.

### What it says to you

| Situation | Voice Alert |
|-----------|-------------|
| Someone merging slowly ahead | *"Move to the middle lane, someone slow is merging."* |
| You're crawling in the left lane | *"Speed up — too slow! You're holding up traffic."* |
| Someone passing on the right | *"You're getting passed from the right. Maybe choose the correct lane."* |
| Stop sign ahead, you're not slowing | *"Stop sign in 200 feet. You need to brake."* |
| Green light, but the box is still full | *"Green light — but the intersection's still blocked. Hold back, don't block the box."* |
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

## The Book: *Seeing Machines*

> **["Seeing Machines: Deep Learning & Computer Vision from Python to Bare Metal"](https://arpanpathak.github.io/seeing-machines-book/foreword.html)** — the companion book to this project, written by the same author.

This repo is the capstone project behind the book, and the book is the engineer's diary behind the repo: every line of code written after a mistake, every equation derived after a model failed to converge, a tracker lost its target, or a pipeline got squeezed onto a tiny edge device.

Together they tell one story — **how to take computer vision from cloud to bare metal**: from Python prototypes to an ultra-fast, privacy-first, low-power Rust pipeline, plus a [Kotlin Multiplatform companion app](https://github.com/arpanpathak/civicsense-companion) on Android and iOS.

**Read the foreword → [arpanpathak.github.io/seeing-machines-book/foreword.html](https://arpanpathak.github.io/seeing-machines-book/foreword.html)**

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

## The Full Pipeline: Distributed Edge Vision

Squeezing a YOLO model into 512 MB of RAM on a Pi Zero is a losing game — you trade accuracy for 2 FPS and watch it thermal-throttle. **Don't embed — distribute.** Each node does the job it's best at, and the heaviest brain in the room runs the real inference:

<p align="center">
  <img src="assets/pipeline.svg" alt="CivicSense distributed edge pipeline: Pico triggers, Pi Zero streams, the brain infers, KMP app alerts" width="860"/>
</p>

| Tier | Node | Job | Stack |
|------|------|-----|-------|
| **1 · Trigger** | Raspberry Pi Pico | Physical sensing & control: PIR motion, buttons, ultrasonic, power states, LED/buzzer | RP2040 **PIO** state machines |
| **2 · Capture & Stream** | Raspberry Pi Zero | CSI camera capture, MJPEG/H.264 encode, low-latency streaming | libcamera, Rust/Go stream daemon |
| **3 · Brain** | Pi 5 + Hailo-8L / desktop GPU | Heavy inference: YOLO ONNX (INT8), NMS, Deep SORT + Kalman, alert engine | ONNX Runtime, Rust |
| **4 · Companion** | Phone (KMP) | Live alerts, violations, map view | Kotlin Multiplatform, gRPC |

**Why this wins over one-board-everything:**

- **The Pico's PIO** handles triggers with zero CPU cost — the camera only wakes when there's something to see.
- **The Pi Zero** stays a dumb, low-power camera node: capture, encode, stream. No model to squeeze, no RAM anxiety, no thermal throttling.
- **The brain** (Pi 5 or your desktop GPU) runs the full-fat model — you never trade accuracy to fit in 512 MB.
- **Every hop stays on your network** — frames leave the Pi Zero, but they never leave the car.

```text
[ Pi Pico ] --GPIO trigger--> [ Pi Zero ] --UDP frames--> [ Brain: Pi 5 / GPU ]
     ^                                                         |
     +---------------------- ack / re-trigger -----------------+
                                                               v
                                                   [ KMP Companion App ]
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

> Full build instructions in the [companion repo README](https://github.com/arpanpathak/civicsense-companion).

---

## The Problems We Solve

### 1. The Intersection Crisis
- **~40%** of all crashes occur at intersections (NHTSA).
- "Blocking the box" causes T-bone impacts.
- **Misjudged green lights** — drivers see green, misjudge the gap, roll in, and trap themselves when the light flips. The worst kind of "blocking the box."
- Current ADAS detects vehicles but **fails** to semantically interpret intersection occupancy.

### 2. Left-Lane Camping
- Drivers don't realize the **speed differential** between lanes.
- Camping in the left lane causes compression, road rage, reduced throughput.
- GPS doesn't provide real-time lane-level speed awareness.

### 3. Turn Signal? None. I Turn Now.
- The Family Guy maneuver is real — *"How much turn signal? ... good luck everybody!"* — drivers cut across multiple lanes with zero warning.

<p align="center">
  <a href="https://www.youtube.com/watch?v=yCdGeElhCK4">
    <img src="assets/family-guy-meme.svg" alt="Family Guy driving meme — I turn now. Good luck everybody else!" width="560"/>
  </a>
  <br/>
  <em>The Family Guy maneuver, in the wild — <a href="https://www.youtube.com/watch?v=yCdGeElhCK4">watch the scene</a></em>
</p>
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
| **Blocked Box on Green** | Green light + intersection still occupied, no room to clear (misjudged the gap) | *"Green light — but the box is still full. Hold back, don't block the box."* |
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
| Raspberry Pi 5 + Hailo-8L | ~18 ms | 8 W | **The brain** — DIY Kit |
| Desktop GPU (training) | full-fat YOLO | 200–350 W | Model training & heavy inference |
| **Raspberry Pi Zero 2 W** | capture + stream (MJPEG/H.264) | ~1 W | Camera node — feeds the brain |
| **Raspberry Pi Pico** | trigger plane: PIO sensors, power states | < 0.5 W | Always-on trigger co-processor |

> **Squeeze mission:** Pi Zero + Pi Pico on my desk. The goal is *not* to cram YOLO into 512 MB of RAM — it's to **distribute the pipeline**: the Pico triggers, the Pi Zero streams, and the strongest brain in the room runs inference. Performance-per-watt-per-dollar is not a toy metric.

---

## Privacy

- **100% on-device** — no video leaves the device.
- **Hazard beacons are anonymous** — they contain only GPS coordinates and hazard type, no video, no identifier.
- **Officer notifications are data-only** — structured reports, not surveillance footage.
- **No cloud dependency** — everything runs locally.

---

## Performance per Watt per Dollar — Civic Sense STONKS 📈

Edge AI is a three-way squeeze: **fast enough, cheap enough, low-power enough**. Cloud ADAS vendors monetize you with subscriptions and siphon your video to a server. CivicSense flips the graph:

| Metric | Cloud ADAS | CivicSense (edge) |
|--------|-----------|-------------------|
| **Dollar** | $10–30/mo subscription, forever | ~$0 — one-time hardware, no subscription |
| **Watt** | server rack somewhere + 4G upload | < 8 W on-device, no uplink |
| **Performance** | network RTT + cloud queue | frame-to-voice in real time, on the device |
| **Privacy** | your video leaves the car | 100% on-device, nothing leaves |

<p align="center">
  <img src="assets/stonks.svg" alt="Civic Sense STONKS — performance per watt per dollar goes up" width="640"/>
</p>

The whole point of squeezing onto a Pi Zero / Pi Pico is this: if you can do civic sense on **watts and one-time dollars**, it stops being a luxury feature and becomes a civic right.

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
