# Round 1 Response Letter

The reviewer is the author's own critical thinking, red-teamed with LLM critics (DeepSeek R1). Two review passes were produced in Round 1: a project-level critique ("the elephants in the room") and a 9-point document review. Every comment was either fixed in the paper, or explicitly deferred with a reason. Revision `c541e59` implements the fixes.

## Part A: Project-level critique ("elephants in the room")

| Comment | Response | Where fixed |
|---|---|---|
| "Where is the vision?" The math assumes perfect state estimation; the perception layer is glossed over | Agreed, and now explicit: the decision layer is input-agnostic, and the perception pipeline that produces the inputs is described in its own subsection with a derived noise bound | New "Perception pipeline" subsection (Section V); "Design boundaries" paragraph (Section VII) |
| Sensor fusion (radar/LiDAR) is missing | The decision layer is modality-agnostic: camera, radar, and LiDAR are interchangeable providers of the same physical quantities; fusion improves the state estimate, not the decision's correctness | "Design boundaries" paragraph (Section VII) |
| Friction: wet leaves, black ice, bald tyres break the stopping-distance equation | Addressed twice: the grip value is a configurable parameter, and a new sensitivity table quantifies how the stopping distance and the regime (dilemma vs forced stop) shift with deceleration | Table III row 6; "Sensitivity to braking capability" (Section VI, Table IV) |
| A tailgating follower makes "brake now" unsafe | The brake decision is recall-first for the ego vehicle and its leader; rear-end protection from an aggressive follower is a different control problem, explicitly out of scope | "Design boundaries" paragraph (Section VII); Theorem 4 scope |
| Verdict: this is a Safety Monitor / Guardian Angel, not a Vision Model | Accepted. The paper now states the guardian-angel framing explicitly and names the true open question: proving the vision pipeline that extracts the inputs | "Design boundaries" paragraph; "Future work" paragraph (Section VII) |

## Part B: 9-point document review

| Point | Comment | Response | Where fixed |
|---|---|---|---|
| 1 | Evaluation is only synthetic | Added an analytical sensitivity study (required stopping distance vs speed and deceleration, Table IV) and committed to a simulation campaign (SUMO/CARLA) in Future Work. A full CARLA/SUMO study is deliberately not faked here | Section VI (Table IV); Section VII "Future work", item (iv) |
| 2 | Perception layer is hand-waved | New "Perception pipeline" subsection: YOLOv8n (COCO-pretrained), Deep SORT-style tracking with a Kalman smoother, calibrated pinhole model, and a concrete derivation of the +-1.5 m bound (one-pixel jitter at 30 fps plus calibration tolerance up to 60 m) | Section V, "Perception pipeline" |
| 3 | Signal timing assumption is unrealistic | A vision-based signal-phase classifier can replace V2I, degrading gracefully to the worst-case bound when uncertain | Section VII, "Scope of the formal results" |
| 4 | Friction fixed at 4.0 m/s^2 | The deceleration is now explicitly a configurable parameter driven by external friction cues, with a sensitivity table showing the effect of halving it | Section VI (Table IV); Section VII |
| 5 | No comparison to baselines | New paragraph: any fixed speed/distance threshold either over-warns or under-warns, because the feasibility region is a coupled conjunction; the kinematic rule is parameter-free and reproduces the exact boundary | Section VII, "Comparison with threshold baselines" |
| 6 | Figures and visuals | No changes needed. Figure captions were verified to reference the correct sections (the reviewer misread one) | n/a |
| 7 | References are sparse (6) | Expanded to 16 real citations across dilemma-zone engineering, reachability verification, detection, tracking, depth, traffic-light perception, simulation, conformal prediction, and Rust | Related Work; bibliography |
| 8 | Rebuttal-style list reads like a letter | Converted the list into a flowing prose paragraph ("Design boundaries") and added normal paragraphs for baseline comparison, ethics, and future work | Section VII |
| 9 | Typos and minor issues | Abstract now states that the decision logic consumes physical estimates; other cited items were verified as non-issues | Abstract |

## What was NOT done, and why

| Deferred item | Reason |
|---|---|
| CARLA/SUMO simulation with thousands of scenarios | Requires a real simulation harness and time; documented as Future Work item (iv) rather than fabricated |
| Measured perception noise statistics (mAP, distance std-dev) | Requires the physical rig; the paper states the +-1.5 m bound is an engineering bound to be confirmed by field calibration |
| Measured end-to-end latency | Requires a benchmark on the target hardware; tracked as an open item in Round 2 |
