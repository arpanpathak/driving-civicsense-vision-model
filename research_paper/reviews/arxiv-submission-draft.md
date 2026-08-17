# arXiv Submission Draft

Ready-to-paste metadata and abstract for an arXiv submission. The companion PDF is `paper.pdf` in the parent directory.

## Metadata

| Field | Value |
|---|---|
| Title | Deterministic Intersection Blockage Prediction: A Kinematic Framework with Mathematical Proofs and a Modular Rust Implementation |
| Author | Arpan Pathak |
| Affiliation | Driving CivicSense Research |
| Categories | `cs.CV` (Computer Vision and Pattern Recognition), `cs.RO` (Robotics), `eess.IV` (Image and Video Processing) |
| Comments | 13 pages, 5 theorems with complete constructive proofs in the appendix, IEEE conference format (IEEEtran, 2-column); companion implementation at https://github.com/arpanpathak/driving-civicsense-vision-model |
| Keywords | intersection blockage, dilemma zone, kinematic safety, mathematical proofs, Rust, ADAS |

## Abstract (as submitted)

> This paper addresses the problem of predicting whether a vehicle will become blocked inside an intersection when approaching a stale green or yellow traffic signal. Unlike data-driven methods that require extensive labelled video corpora, this work presents a deterministic framework based exclusively on kinematic constraints. The system ingests a sliding window of video frames from a forward-facing camera, applies object detection to obtain physical measurements (ego speed, stop-line distance, lead-vehicle state), and evaluates five mathematically derived criteria on those estimates. Each criterion is expressed as a theorem with a constructive proof; the complete mathematical development is given in the appendix. The implementation is written in idiomatic Rust, adhering to SOLID principles, utilising exhaustive pattern matching, and requiring zero external dependencies for its core logic. The decision pipeline is a severity-ordered composition of single-responsibility rules evaluated with $O(n)$ complexity per frame. The system operates in real time, is fully interpretable, and is suitable for deployment in ISO 26262-compliant automotive systems.

## Comments for the submission form

- Source: LaTeX (IEEEtran conference class), compiles cleanly with `pdflatex` (3 passes), 13 pages, 0 overfull boxes.
- The proofs appendix is part of the same document (Appendix A), not a separate file.
- No external datasets are used; the evaluation suite (Table II) is deterministic and every row is an executable test case.
- Companion repositories (MIT license):
  - https://github.com/arpanpathak/civicsense-pi-stream (Pi Zero 2 W MJPEG streaming server)
  - https://github.com/arpanpathak/civicsense-stream-client (Candle YOLOv8n detection client)
- Full HTML rendering of the paper and proofs: https://arpanpathak.github.io/driving-civicsense-vision-model/

## Pre-submission checklist

- [ ] Upload `paper.tex`, all TikZ figures are inline (no external figure files), plus `paper.pdf`.
- [ ] Confirm the license option: arXiv non-exclusive license is the default; choose "arXiv perpetual" if preferred.
- [ ] Verify all 16 references render (0 unresolved citations confirmed in the current build).
- [ ] Confirm author metadata and add ORCID if available.
- [ ] Note that the Abstract and Index Terms heading labels (including the dash after them) are produced by the IEEEtran class itself and are the IEEE convention, not content em dashes.

## Post-submission hooks

- The PDF on the site is served from `research_paper/paper.pdf` via raw GitHub; after arXiv acceptance, swap the site link to the arXiv abstract page.
- Add the arXiv badge to the README badge row once the paper is live.
