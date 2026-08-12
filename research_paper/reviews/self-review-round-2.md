# Round 2: Deeper Technical Audit

A second, harder pass over the revised paper. This round attacks assumptions the first pass took for granted. Findings are graded by severity. Items marked "Addressed" were folded into the paper in this round; items marked "Open" are tracked as honest limitations or future work.

## Findings

### R2-1. Temporal staleness: the theorems assume a synchronized snapshot [High, Addressed]

The kinematic theorems relate quantities (speed, distance, time-to-red) as if they were measured at the same instant. In practice the detector, tracker, and signal phase produce estimates at slightly different times, and inference itself takes time. The formal guarantee silently assumes synchronization.

**Fix (applied):** the paper now states that the engine treats the input vector as a synchronized snapshot, and that a residual staleness of one frame period (about 33 ms at 30 fps) is absorbed by the +-1.5 m operating bound.

**Location:** Section VII, "Scope of the formal results".

### R2-2. "Zero-training" claim is too broad [Medium, Addressed]

The paper's title-adjacent claim of a "zero-training system" is true for the decision layer but not for the pipeline as a whole: the perception layer needs a calibrated camera (focal length, mounting, lane geometry), and YOLOv8n itself is trained (on COCO). An unqualified claim invites a reviewer to find the contradiction.

**Fix (applied):** the introduction now scopes the claim explicitly: "zero training for the decision layer (the perception layer still requires camera calibration)".

**Location:** Section I, introduction.

### R2-3. Class-dependent bias in monocular depth [Medium, Open]

Distance from a monocular camera is typically inferred from bounding-box scale against a width prior. A truck and a car have very different widths, so a class-agnostic width prior introduces systematic bias in lead-vehicle distance, which the +-1.5 m random-jitter bound does not cover.

**Disposition:** Open. Mitigation is class-aware width priors per detected class; noted for future work. The paper's conditional framing ("given the inputs") already contains this: if the distance estimate is biased, the theorem's precondition is not met.

### R2-4. The epsilon margin (0.8 s) is asserted, not derived [Medium, Open]

The clearance time uses t_y - epsilon with epsilon = 0.8 s. The paper never justifies the value. Is it perception-reaction time, actuator latency, or a safety margin against the box geometry?

**Disposition:** Open. A derivation should be added (suggested: epsilon = perception-reaction floor + actuator latency + box-shading margin). Do not hard-code without a source.

### R2-5. Driver reaction is a distribution, not a point [Medium, Open]

The perception-reaction constant t_r = 1.0 s is a point estimate. Human reaction time varies from about 0.5 s (expectant) to 2.5 s (surprised). The fixed value makes the "safe stop" guarantee conditional on a lucky driver.

**Disposition:** Open. The paper's Future Work item (ii) (dynamic thresholds adapted to reaction time) is the right direction; a distributional treatment (e.g., warn earlier for a measured slow reactor) would strengthen it.

### R2-6. Cut-in rule latency condition [Medium, Open]

The cut-in rule uses lateral velocity from the box-centroid trajectory, which needs at least two frames. A fast cut-in could complete a lane change inside the decision window. The theorem bounds the geometry but not the detection latency.

**Disposition:** Open. Add an explicit latency condition to the cut-in theorem's assumptions (max lateral velocity over a minimum observation window).

### R2-7. No measured end-to-end latency [Low-Medium, Open]

The paper claims "operates in real time" but reports no measured detect-to-alert latency on target hardware.

**Disposition:** Open. A short benchmark (Pi Zero 2 W vs desktop) should be added before a journal submission.

### R2-8. Calibration drift and re-mounting [Low, Open]

The pinhole model assumes a fixed camera mounting. Vibration, re-mounting, or dashcam repositioning invalidates the calibration silently.

**Disposition:** Open. Online re-calibration (vanishing-point tracking) is future work; the paper's type-level enforcement item is related.

### R2-9. Automation complacency [Medium, Open]

A driver may trust the "guardian" beyond its operational envelope. The paper's recall-first claim is true only inside the envelope; outside it, silent failure is possible, and the HMI must communicate the envelope, not just the warning.

**Disposition:** Open. The HMI trust item (Table III row 4) covers false-positive erosion; a positive statement about envelope disclosure should be added to the HMI design.

### R2-10. Baseline "dominance" is analytical, not numerical [Low, Open]

The comparison with threshold baselines is qualitative. A one-page table computing false-positive and false-negative counts for a few threshold rules over the 8-scenario suite would make the claim quantitative and cheap to produce.

**Disposition:** Open. Easy win before submission.

## Round 2 summary

- 2 findings addressed in this round (R2-1, R2-2).
- 8 findings tracked as open limitations or future work, all consistent with the paper's conditional-framing philosophy.
- The paper is honest about its boundaries; the open items are the honest price of that boundary.
