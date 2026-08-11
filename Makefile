# ── Driving-CivicSense Makefile ──────────────────────────
# Targets for local dev (macOS) and cloud GPU cross-compilation.
#
# Workflow:
#   1. Capture frames:        make collect
#   2. Label images externally (CVAT / labelImg / etc.)
#   3. Prepare dataset:       civicsense train prepare --split data/raw
#   4. Cross-compile:         make build-linux-x86_64
#   5. Upload binary to VM, run: ./civicsense train run
#   6. Validate ONNX:         civicsense train validate

SHELL := /bin/bash
CARGO  := cargo

# Local (macOS) --------------------------------------------

.PHONY: all build test clean lint doc release

all: build

## Build the debug binary (macOS native)
build:
	$(CARGO) build

## Build the release binary (macOS native, LTO)
release:
	$(CARGO) build --release

## Run all Rust tests
test:
	$(CARGO) test

## Run Rust tests with logging output
test-verbose:
	RUST_LOG=debug $(CARGO) test -- --nocapture

## Lint with clippy
lint:
	$(CARGO) clippy -- -D warnings

## Format code
fmt:
	$(CARGO) fmt

## Generate docs
doc:
	$(CARGO) doc --no-deps

## Clean build artifacts
clean:
	$(CARGO) clean
	rm -rf output/

# Cross-compilation (Mac → Linux x86_64 for cloud GPU) ----

## Install cross-compilation toolchain (one-time setup)
cross-setup:
	@echo "=== Installing Linux x86_64 cross-compilation toolchain ==="
	rustup target add x86_64-unknown-linux-gnu
	@echo ""
	@echo "You also need a cross-linker. On macOS:"
	@echo "  brew install SergioBenitez/osxct/x86_64-unknown-linux-gnu"
	@echo ""
	@echo "Then create .cargo/config.toml with:"
	@echo "  [target.x86_64-unknown-linux-gnu]"
	@echo "  linker = \"x86_64-unknown-linux-gnu-gcc\""

## Build release binary for Linux x86_64 (cloud GPU target)
build-linux-x86_64:
	./scripts/cross_compile.sh build

# Training (all Rust CLI, no Python scripts) -------------

## Validate / split a labelled dataset and write YAML config
train-prepare:
	$(CARGO) run --release -- train prepare --dataset data/civicsense

## Train YOLO model on cloud GPU (run this on the VM)
train-run:
	$(CARGO) run --release -- train run --data configs/dataset.yaml --epochs 100

## Validate an exported ONNX model with ort
train-validate:
	$(CARGO) run --release -- train validate

# Data Collection -----------------------------------------

## Capture frames from a camera for training data
collect:
	$(CARGO) run --release -- collect --source 0 --output data/raw --rate 2

# Pipeline -------------------------------------------------

## Run the full perception pipeline on a test video
run:
	$(CARGO) run --release -- run --source test_video.mp4 --visualize

# Help ----------------------------------------------------

help:
	@echo "Targets:"
	@echo "  build              , debug build (macOS)"
	@echo "  release            , release build (macOS)"
	@echo "  test               , run all Rust tests"
	@echo "  lint               , clippy"
	@echo "  fmt                , cargo fmt"
	@echo "  doc                , documentation"
	@echo "  cross-setup        , install Linux x86_64 toolchain"
	@echo "  build-linux-x86_64 , cross-compile for cloud GPU"
	@echo "  train-prepare      , validate/split dataset"
	@echo "  train-run          , train YOLO on GPU"
	@echo "  train-validate     , verify ONNX model"
	@echo "  collect            , capture training frames"
	@echo "  run                , run perception pipeline"
