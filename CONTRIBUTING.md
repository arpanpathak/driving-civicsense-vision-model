# 🤝 Contributing to Driving-CivicSense-Vision-Model

Thank you for helping make every mile a socially aware mile! 🚗

## 🧭 Roadmap

- [x] **Phase 1:** YOLOv8 training on Intersection + Stop sign data
- [x] **Phase 2:** Deep SORT integration and Relative Speed proxy
- [ ] **Phase 3:** Real-world validation (100-mile test drive)
- [ ] **Phase 4:** Hardware porting to Qualcomm AR1 Platform
- [ ] **Phase 5:** Beta fleet testing (50 users)

## 🔧 How to Contribute

1. **Pick a module** with `raise NotImplementedError` stubs
2. **Branch off `main`**: `git checkout -b feat/your-feature`
3. **Replace stubs with real code**
4. **Add tests** in `tests/`
5. **Run**: `pytest tests/`
6. **Open a PR**

## 📏 Standards

- Type hints on all public functions
- Docstrings (NumPy style)
- Keep it simple — no premature optimization
- Profile before optimizing

---

<div align="center">⭐ Even a documentation fix helps someone drive safer.</div>
