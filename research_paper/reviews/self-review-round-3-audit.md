# Round 3: Pre-Submission Audit

A mechanical, checklist-style audit to run immediately before any submission (arXiv, conference, or journal). It does not find new ideas; it finds holes in execution.

## A. Claims audit

- [ ] Every sentence that asserts a guarantee cites a theorem, an assumption, or an explicit boundary.
- [ ] "Zero external dependencies" is scoped to the core decision logic (true) and not to the whole pipeline (false).
- [ ] "Operates in real time" is supported by a measurement or by an explicit complexity argument; if only the latter, soften the claim.
- [ ] "Recall-first keeps false negatives at zero" is qualified with "within the operational envelope".
- [ ] No claim of "no labelled data" appears without the decision-layer scope.
- [ ] The +-1.5 m bound is described as an engineering bound to be confirmed, not a measured guarantee.

## B. Notation and cross-reference audit

- [ ] Every macro defined in the preamble is used; every symbol used is defined (notation table is complete).
- [ ] All `\ref` targets resolve (checked: 0 unresolved in current build).
- [ ] All `\cite` keys exist and are cited at least once (checked: 16/16 in current build).
- [ ] Figure captions reference the correct sections (verified for Figures 1-4).
- [ ] Table numbering (I-IV) is sequential and referenced from the text.

## C. Structure audit

- [ ] Abstract matches the body (the perception-consumption clause was updated to match Section V).
- [ ] The contributions list matches what the paper actually does (5 theorems, appendix, Rust, verification suite, sensitivity analysis).
- [ ] Related Work covers: dilemma zone (classical + data-driven), RSS, reachability/formal methods, perception stack, simulation.
- [ ] Discussion order is logical: safety engineering, vulnerabilities, scope, design boundaries, baselines, pedagogy, ethics, limitations, future work.
- [ ] Conclusion does not introduce new claims.

## D. Formatting audit

- [ ] IEEEtran conference class, 2-column, 10 pt, no overfull boxes (checked: 0).
- [ ] No content em dashes (checked: 0 in source; the IEEEtran template labels are class-generated).
- [ ] Page count is stable at 13.
- [ ] Bibliography is IEEE style; page ranges use en dashes as is conventional.

## E. Ethics and safety audit

- [ ] Advisory-only framing is explicit (yes: "Ethics and safety" paragraph).
- [ ] False-positive trust hazard is discussed (yes: Table III row 4).
- [ ] Automation-complacency risk is documented (open item R2-9, not yet in the paper).
- [ ] No claim that the system is certified; "suitable for ISO 26262" is phrased as suitability, not compliance.

## F. Open risks register (carry into any submission)

| Risk | Severity | Status |
|---|---|---|
| Simulation-based evaluation absent (only analytical sensitivity) | High | Future Work item (iv) |
| Perception noise not measured on the rig | High | Stated as engineering bound |
| Class-dependent monocular depth bias | Medium | Open (R2-3) |
| Epsilon margin (0.8 s) lacks derivation | Medium | Open (R2-4) |
| Driver reaction treated as a point value | Medium | Open (R2-5) |
| Cut-in detection latency not bounded | Medium | Open (R2-6) |
| End-to-end latency unmeasured | Medium | Open (R2-7) |
| Automation complacency not stated in HMI | Medium | Open (R2-9) |
| Baseline comparison qualitative only | Low | Open (R2-10) |

## G. Sign-off

- [ ] Author has re-read the final PDF cover to cover.
- [ ] Author has re-run the 3-pass compile and confirmed 0 overfull, 0 unresolved.
- [ ] Author has confirmed the GitHub Pages site mirrors the paper (checked after revision `c541e59`).
- [ ] Author has decided the arXiv license option and ORCID.
