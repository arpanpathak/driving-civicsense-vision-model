# ☁️ Cloud GPU Training Guide

> *"I need an NVIDIA GPU to train this, but I don't have one. What do I do?"*

You don't need to drop $2,000+ on a GPU. Rent one by the hour for pocket change. This guide covers the cheapest options and exact setup steps.

---

## 🏆 Cheapest Cloud GPU Providers (2026)

For YOLOv8/v11 training, a **single RTX 3090 (24GB)** is more than enough. You don't need A100s or H100s.

| Provider | GPU | VRAM | Hourly Cost | Best For | Catch |
|----------|-----|------|-------------|----------|-------|
| **Vast.ai** 🥇 | RTX 3090 | 24GB | **~$0.15–0.22/hr** | Absolute cheapest | Peer-to-peer — check reviews before renting |
| **RunPod** 🥇 | RTX 3090 | 24GB | **~$0.19/hr** | Best price + reliability combo | Community Cloud tier (not dedicated) |
| **RunPod** | RTX 4090 | 24GB | ~$0.34/hr | Faster training | Worth it if you train frequently |
| **AutoDL** | RTX 3090 | 24GB | ~¥1.50/hr (~$0.20) | Cheapest in Asia | Chinese UI, Alipay required |
| **Lambda Labs** | RTX 4090 | 24GB | ~$0.35/hr | Reliable, good support | Slightly pricier |
| **Google Colab Pro** | T4 | 16GB | **$10/month** | Prototyping | 8-hour session limit |
| **Google Colab Pro+** | A100 | 40GB | $50/month | Heavy training | Still time-limited |
| **Paperspace** | RTX 4000 Ada | 16GB | ~$0.23/hr | Good UI, easy setup | Less VRAM than 3090 |

### 💰 Cost Estimate for YOLOv8 Fine-Tuning

| Dataset Size | Epochs | GPU | Est. Time | Est. Cost |
|-------------|--------|-----|-----------|-----------|
| 5,000 images | 100 | RTX 3090 | ~3 hours | **~$0.60** |
| 15,000 images | 150 | RTX 4090 | ~6 hours | **~$2.04** |
| 50,000 images | 200 | RTX 4090 | ~20 hours | **~$6.80** |

A full YOLOv8n fine-tuning run costs **less than a coffee**.

---

## 🚀 Quick Start: RunPod (Recommended)

### Step 1: Sign Up & Get Credits

1. Go to [runpod.io](https://runpod.io)
2. Sign up (GitHub account works)
3. Add $10 in credits — this will last you many training runs

### Step 2: Launch a GPU Instance

1. Click **"Pod"** → **"Community Cloud"**
2. Filter: GPU = `RTX 3090`, Disk ≥ `50 GB`
3. Pick an instance with `~$0.19/hr` pricing
4. Template: Select **"RunPod PyTorch 2.x"** (comes with CUDA + torch pre-installed)
5. Click **"Deploy On-Demand"**

### Step 3: Connect & Train

```bash
# SSH into the instance (RunPod gives you the command)
ssh -p <PORT> root@<IP_ADDRESS>

# Clone the project
git clone https://github.com/arpanpathak/driving-civicsense-vision-model.git
cd driving-civicsense-vision-model

# Install dependencies
pip install -r requirements.txt

# Download pretrained YOLOv8n weights
wget -O weights/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Train the model
yolo train \
    model=weights/yolov8n.pt \
    data=configs/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0

# Export to INT8 ONNX for edge deployment
yolo export model=runs/train/exp/weights/best.pt format=onnx int8=true
```

### Step 4: Download Your Weights

```bash
# Compress and download the trained weights
tar -czf trained-model.tar.gz runs/train/exp/weights/

# Download via SCP or use RunPod's web file browser
```

---

## ⚡ Alternative: Vast.ai (Cheapest)

Vast.ai is a peer-to-peer marketplace — cheaper but less consistent.

```bash
# 1. Browse: https://vast.ai/ → search "RTX 3090"
# 2. Filter: rentable = yes, verified = yes
# 3. Pick an instance ~$0.15-0.18/hr
# 4. Launch with PyTorch template
# 5. SSH in and follow same steps as RunPod above
```

### ⚠️ Vast.ai Pro Tips

- **Check "Verified" hosts only** — avoids bad actors
- **Look for "Docker: pytorch/pytorch"** — saves setup time
- **Download speed matters** — pick instances with ≥ 500 Mbps
- **Use `tmux`** — so your training doesn't die if SSH disconnects

```bash
tmux new-session -s training
# ... run training commands ...
# Ctrl+B then D to detach
# tmux attach -t training to reattach
```

---

## 🧪 Option: Google Colab (For Quick Experiments)

Best for prototyping, not serious training runs.

```python
# In a Colab notebook cell:
!git clone https://github.com/arpanpathak/driving-civicsense-vision-model.git
%cd driving-civicsense-vision-model
!pip install -r requirements.txt
!yolo train model=yolov8n.pt data=configs/dataset.yaml epochs=50 imgsz=640
```

| Plan | Price | GPU | Limit |
|------|-------|-----|-------|
| Free | $0 | T4 (16GB) | 2-hour sessions, slow |
| Pro | $10/mo | T4/V100 | 8-hour sessions |
| Pro+ | $50/mo | A100 (40GB) | 24-hour sessions, priority |

---

## 📦 What About the Safeguard Vision Project?

Safeguard Vision's GPU needs are different — it uses **CUDA-Oxide (Rust → PTX)** for GPU kernels, not PyTorch training.

| Need | Cloud Approach |
|------|---------------|
| **CUDA-Oxide kernel development** | Use **RunPod** with a template that has NVIDIA drivers + CUDA toolkit. Install Rust via `rustup` and compile with `cargo oxide build`. |
| **Mistral 7B fine-tuning** | **Vast.ai** RTX 4090 ($0.30/hr) — 24GB VRAM is enough for 7B parameter LoRA fine-tuning. |
| **Whisper model fine-tuning** | **Lambda Labs** RTX 4090 ($0.35/hr) — or Google Colab Pro+ A100. |

For Safeguard Vision kernel work:
```bash
# On a RunPod RTX 3090:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
git clone https://github.com/arpanpathak/safeguard-vision-friendshaped-but-unfriendly.git
cd safeguard-vision-friendshaped-but-unfriendly
cargo build
```

---

## 🧠 Pro Tips

### Save Money

| Tip | Saves |
|-----|-------|
| Use **spot/community** instances, not on-demand | 60–70% |
| **Stop instances when idle** — set auto-stop timers | 100% of idle cost |
| **Mount network storage** (S3, Dropbox) — keep data between sessions | No re-upload costs |
| **Use `tmux`** — detach without killing training | Prevents wasted runs |
| **Monitor with `nvidia-smi`** — if GPU < 80% utilized, increase batch size | Better $/epoch |

### Avoid Pitfalls

- **Don't pick instances with < 50GB disk** — YOLO datasets + weights fill up fast
- **Don't train on spot instances without checkpointing** — they can terminate anytime
- **Don't use A100 for YOLOv8n** — it's overkill. RTX 3090 is the sweet spot
- **Do set `project` and `name` in YOLO config** — so you don't overwrite previous runs

### Session Persistence (RunPod)

RunPod's Community Cloud wipes your disk when the pod stops. For persistent storage:

```bash
# Mount a network volume (RunPod offers this in the pod config)
# Or use S3:
pip install awscli
aws s3 sync runs/train/exp s3://my-bucket/yolo-training/
```

---

## 🔗 Quick Links

| Resource | URL |
|----------|-----|
| RunPod | https://runpod.io |
| Vast.ai | https://vast.ai |
| Lambda Labs | https://lambdalabs.com |
| Google Colab | https://colab.research.google.com |
| GPU Price Comparison | https://gpus.io |

---

<div align="center">

*You don't need a $3,000 GPU. You need $0.19 an hour and this guide.*

⭐ **Star the project if this helped you train your first model.**

</div>
