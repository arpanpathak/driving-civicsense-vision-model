# 📐 Coding Standards & Best Practices

> *"Reducing the distance between userspace and kernel space through great systems programming abstractions that don't leak."*

Every line of code in this project carries a responsibility — to the drivers whose safety depends on real-time inference, to the contributors who build upon it, and to the open-source ethos that protects it from proprietary appropriation.

**PRs that do not adhere to these standards will be rejected.** Full stop.

---

## 🧠 Core Philosophy

### Userspace ↔ Kernel Space Continuum

As systems programmers, our job is to **build abstractions that don't leak**. Every layer — from the YOLO ONNX session to the haptic alert driver — should be:

- **Correct** — formally verifiable where possible
- **Performant** — zero-cost unless explicitly traded off
- **Transparent** — the abstraction hides complexity, not behavior

> *"A great abstraction lets you forget what's underneath. A leaking abstraction forces you to remember everything."*

### Safety First

- **No `unsafe` without `// SAFETY:`** — every unsafe block must have a justification that a reviewer can independently verify
- **No undefined behavior** — run Miri (`cargo miri test`) on any code that touches raw pointers, FFI, or unions
- **No silent panics** — use `Result` for fallible operations; document `unwrap()` calls with a reason

---

## 📏 Rust Idioms

### 1. Types Over Comments

Let the type system express invariants the compiler can enforce:

```rust
// ❌ Bad: magic constants and runtime assertions
fn process(val: f32) {
    assert!(val >= 0.0 && val <= 1.0);
    // ...
}

// ✅ Good: type enforces the invariant at compile time
struct Normalized(f32);

impl Normalized {
    pub fn new(val: f32) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&val) {
            return Err(format!("{val} is not in [0, 1]"));
        }
        Ok(Self(val))
    }
}
```

### 2. Enums Over Booleans

```rust
// ❌ Bad: boolean blindness
fn should_merge(urgent: bool, left_lane: bool) { /* ... */ }

// ✅ Good: readable, exhaustive, compile-time checked
enum LanePosition { Left, Center, Right }
enum Urgency { Normal, Warning, Critical }

fn alert(position: LanePosition, urgency: Urgency) { /* ... */ }
```

### 3. Match Over If-Else Chains

```rust
// ❌ Bad: if-else cascade for lane assignment
fn lane_from_centroid(x: f32) -> &'static str {
    if x < 0.33 { "Left" }
    else if x < 0.66 { "Center" }
    else { "Right" }
}

// ✅ Good: explicit intent, easy to adjust thresholds
fn lane_from_centroid(x: f32) -> LanePosition {
    match x {
        x if x < 0.33 => LanePosition::Left,
        x if x < 0.66 => LanePosition::Center,
        _ => LanePosition::Right,
    }
}
```

### 4. Iterators Over Loops

```rust
// ❌ Bad: imperative loop
let mut mean_speed = 0.0;
for t in &tracks {
    mean_speed += t.speed;
}
mean_speed /= tracks.len() as f32;

// ✅ Good: functional, self-documenting
let mean_speed = tracks.iter()
    .map(|t| t.speed)
    .sum::<f32>() / tracks.len() as f32;
```

### 5. Traits for Dependency Inversion

```rust
// ✅ Good: swap YOLO backends via trait
pub trait ObjectDetector {
    fn detect(&self, frame: &[u8], width: u32, height: u32) -> Result<Vec<Detection>, String>;
}

struct OnnxYolo(YoloConfig);
struct TensorRtYolo(YoloConfig);
impl ObjectDetector for OnnxYolo { /* ... */ }
impl ObjectDetector for TensorRtYolo { /* ... */ }
```

---

## 🧹 Clean Code Principles

### Naming

| Principle | ❌ Bad | ✅ Good |
|-----------|--------|---------|
| Pronounceable | `fn calc_vel(d: &[Trk])` | `fn compute_relative_velocity(tracks: &[Track])` |
| Searchable | `let t = 5.0` | `let speed_diff_threshold_mph = 5.0` |
| No abbreviations | `intersxn_cfg` | `intersection_config` |
| Domain language | `fn chk_blkd()` | `fn detect_blocked_intersection()` |

### Function Size

A function should do **one thing** and fit on one screen (~40 lines). If you need a comment to explain a block, extract that block into a named function.

```rust
// ❌ Bad: 80-line function doing detection + tracking + alert
fn pipeline_step(frame: &[u8]) { /* 80 lines */ }

// ✅ Good: one function per pipeline stage
fn run_detection(frame: &[u8]) -> Result<Vec<Detection>, String> { /* ... */ }
fn associate_tracks(detections: &[Detection]) -> Vec<Track> { /* ... */ }
fn evaluate_alerts(tracks: &[Track], ego_speed: f32) -> Vec<Alert> { /* ... */ }
```

### No Dead Code

- Every `pub` item must have a use
- No commented-out code (that's what git history is for)
- No `#[allow(dead_code)]` without a reason — use it only for stubs during active development, remove before PR

---

## ⚡ Performance

### Zero-Cost Abstractions

Prefer abstractions that compile away to nothing. If a trait or closure adds runtime overhead, document the trade-off.

```rust
// ✅ Good: monomorphized, zero-cost
fn process<T: AsRef<[u8]>>(frame: T) { /* compiled to direct memory access */ }

// ❌ Bad: unnecessary heap allocation on hot path
fn process(frame: &[u8]) {
    let vec = frame.to_vec(); // Every. Single. Frame.
}
```

### Real-Time Discipline

The inference loop runs at 30 fps. Every millisecond counts:

- **No allocations in the hot path** — pre-allocate buffers at startup
- **No I/O in the hot path** — log asynchronously or batch
- **No syscalls in the hot path** — pin threads, lock memory

### Profile Before Optimizing

```rust
// ❌ Bad: guessing about bottlenecks
fn micro_optimization() { /* premature */ }

// ✅ Good: data-driven optimization
// Run: cargo bench or perf record / samply
// Profile with: cargo instruments --open
// Then optimize the hot path
```

---

## 🧪 Formal Verification & Testing

We treat correctness as a **compile-time property** wherever possible.

### Property-Based Tests

Use `proptest` or `quickcheck` to verify invariants:

```rust
/// Invariant: distance must be inversely proportional to pixel width
proptest! {
    #[test]
    fn distance_monotonically_decreases_with_larger_bbox(
        pw1 in 10..1000u32, pw2 in 10..1000u32,
    ) {
        // f and W are constant; larger pw → smaller Z
        let f = 650.0;
        let w = 1.8;
        let z1 = estimate_distance(pw1 as f32, w, f);
        let z2 = estimate_distance(pw2 as f32, w, f);
        if pw1 > pw2 {
            prop_assert!(z1 < z2);
        } else if pw1 < pw2 {
            prop_assert!(z1 > z2);
        }
    }
}
```

### Panic-Free Guarantees

All public API functions must be panic-free unless documented otherwise. Use:

- `.ok()?` over `.unwrap()`
- `.get(index)` over `[index]`
- `checked_add()` / `checked_mul()` over `+` / `*`
- `saturating_sub()` where underflow is semantically valid
- `if let` over `.unwrap()` on Options

### Invariant Documentation

Every `struct` with internal invariants must document them:

```rust
/// Kalman filter state for a tracked vehicle.
///
/// INVARIANT: `covariance` must always be symmetric positive-definite.
/// Violating this produces incorrect (and possibly NaN) state estimates.
/// All update steps must enforce this via `covariance = (covariance + covariance.t()) / 2.0`.
pub struct KalmanState {
    mean: [f32; 8],
    covariance: [[f32; 8]; 8],
}
```

### Latency Budget Tests

```
Inference:   < 25 ms  (yolo model forward pass)
Tracking:    < 5 ms   (association + kalman update)
Intersection:< 3 ms   (grid occupancy + deceleration check)
Lane Speed:  < 3 ms   (velocity estimation + hysteresis)
Overhead:    < 4 ms   (pre/post processing, frame copy)
Total:       < 40 ms  (target: 25 fps minimum)
```

Benchmark each stage with `cargo bench` and reject PRs that exceed the budget without justification.

---

## 🚫 PR Rejection Criteria

Your PR **will be rejected** if it contains any of the following:

| # | Violation | Example |
|---|-----------|---------|
| 1 | **Unsafe without SAFETY comment** | `unsafe { ... }` with no `// SAFETY:` |
| 2 | **Dead code** | Commented-out code, unused imports, `#[allow(dead_code)]` without reason |
| 3 | **Silent unwrap in hot path** | `.unwrap()` or `.expect()` in the inference loop |
| 4 | **Boolean blindness** | `fn alert(urgent: bool, merge_right: bool)` instead of an enum |
| 5 | **Magic numbers** | `if speed > 5.0 { ... }` with no named constant |
| 6 | **Function > 50 lines** | Without justification in a doc comment |
| 7 | **No tests for new logic** | Any non-trivial function without a `#[test]` |
| 8 | **Allocation on hot path** | `to_vec()`, `clone()`, `format!()` in the inference loop |
| 9 | **Exceeds latency budget** | Without profiling data to justify the regression |
| 10 | **Leaking unsafe abstraction** | Wrapping unsafe code in a safe function without upholding safety invariants |

### Pre-Submission Checklist

Before opening a PR, verify:

- [ ] `cargo test` passes (all tests, including doc-tests)
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt` has been run
- [ ] No `todo!()` or `unimplemented!()` remain in production code
- [ ] All new public items have doc comments
- [ ] `// SAFETY:` comments on every `unsafe` block
- [ ] No dead code or commented-out code
- [ ] Property-based tests added for critical math/logic
- [ ] Benchmarks added for hot-path functions

---

## 🔗 References

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Clean Code (Martin)](https://www.oreilly.com/library/view/clean-code/9780136083238/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Rustonomicon — Unsafe Code Guidelines](https://doc.rust-lang.org/nomicon/)

---

<div align="center">

*"A program is never wrong in the way you expect. Formalize your assumptions, verify your invariants, and never trust a runtime assertion you could have made a compile-time guarantee."*

</div>
