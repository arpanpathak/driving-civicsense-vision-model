# 🤝 Contributing to Driving-CivicSense-Vision-Model

Thank you for helping make every mile a socially aware mile! 🚗

## 🧭 Roadmap

- [x] **Phase 1:** YOLOv8 training on Intersection + Stop sign data
- [x] **Phase 2:** Deep SORT integration and Relative Speed proxy
- [ ] **Phase 3:** Real-world validation (100-mile test drive)
- [ ] **Phase 4:** Hardware porting to Qualcomm AR1 Platform
- [ ] **Phase 5:** Beta fleet testing (50 users)

## 🔧 How to Contribute

1. **Pick a module** with `todo!()` stubs
2. **Branch off `main`**: `git checkout -b feat/your-feature`
3. **Replace stubs with real Rust code**
4. **Add tests** (use `#[cfg(test)]` mods or `proptest!`)
5. **Verify**: `cargo test && cargo clippy -- -D warnings && cargo fmt`
6. **Open a PR**

## 📏 Coding Standards

**PRs that do not adhere to our coding standards will be rejected.**

Read the full standards document: [CODING_STANDARDS.md](CODING_STANDARDS.md)

Key rules at a glance:
- **Idiomatic Rust** — enums, match, iterators, traits
- **No unsafe without `// SAFETY:`** — every unsafe block needs justification
- **No dead code** — commented-out code gets rejected
- **Property-based tests** for all critical math/geometry
- **Zero allocations on the hot path** — pre-allocate at startup
- **Latency budget** — stay under 40ms total pipeline latency
- **Panic-free public API** — use `Result`, not `unwrap()`

---

<div align="center">⭐ Even a documentation fix helps someone drive safer.</div>
