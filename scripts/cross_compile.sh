#!/usr/bin/env bash
# ── Cross-Compile CivicSense for Linux x86_64 (Cloud GPU) ──
#
# Builds a statically-linked or dynamically-linked binary
# from macOS that runs on Linux x86_64 (e.g. cloud GPU VM).
#
# Prerequisites (one-time):
#   rustup target add x86_64-unknown-linux-gnu
#   brew install SergioBenitez/osxct/x86_64-unknown-linux-gnu
#
# OR for musl (fully static, no linker deps):
#   brew install filosottile/musl-cross/musl-cross
#   rustup target add x86_64-unknown-linux-musl
#
# Usage:
#   ./scripts/cross_compile.sh build          # dynamic link (default)
#   ./scripts/cross_compile.sh build-musl     # fully static
#   ./scripts/cross_compile.sh build-cuda     # with CUDA ONNX Runtime
#
# Output:
#   target/x86_64-unknown-linux-gnu/release/civicsense
#   (or ...-linux-musl/... for static build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TARGET="${1:-build}"

# ── Cargo config for cross-linker ───────────────────────
#
# If you installed x86_64-unknown-linux-gnu via homebrew,
# uncomment or create .cargo/config.toml with:
#
#   [target.x86_64-unknown-linux-gnu]
#   linker = "x86_64-unknown-linux-gnu-gcc"
#
# For musl:
#   [target.x86_64-unknown-linux-musl]
#   linker = "x86_64-linux-musl-gcc"

case "$TARGET" in
    build)
        echo "=== Cross-compiling for x86_64-unknown-linux-gnu ==="
        cargo build --release --target x86_64-unknown-linux-gnu
        echo ""
        echo "✅ Binary at: target/x86_64-unknown-linux-gnu/release/civicsense"
        echo "   Copy it to your cloud VM and run:"
        echo "   scp target/x86_64-unknown-linux-gnu/release/civicsense user@host:~/"
        ;;

    build-musl)
        echo "=== Cross-compiling for x86_64-unknown-linux-musl (static) ==="
        cargo build --release --target x86_64-unknown-linux-musl
        echo ""
        echo "✅ Static binary at: target/x86_64-unknown-linux-musl/release/civicsense"
        echo "   No runtime deps needed on the target."
        ;;

    build-cuda)
        echo "=== Cross-compiling with CUDA ONNX Runtime ==="
        echo ""
        echo "NOTE: ONNX Runtime CUDA EP requires the CUDA toolkit and"
        echo "cuDNN to be present at compile time. This is best done"
        echo "natively on the cloud VM, not cross-compiled."
        echo ""
        echo "For on-device CUDA inference, build natively on the VM:"
        echo "  cargo build --release --features ort/cuda"
        echo ""
        echo "Alternatively, use:"
        echo "  cargo build --release --target x86_64-unknown-linux-gnu \\"
        echo "      --features ort/cuda"
        echo ""
        echo "See CLOUD_TRAINING.md for cloud VM setup instructions."
        ;;

    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 {build|build-musl|build-cuda}"
        exit 1
        ;;
esac
