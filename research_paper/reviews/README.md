# Critical Self-Review Process

The paper is developed through structured adversarial self-review. The reviewer is the author's own critical thinking, red-teamed with the help of LLM critics (DeepSeek R1). Every round produces findings; actionable findings are folded back into `paper.tex` and the GitHub Pages site, and each round is archived here so the record is traceable.

## Rounds

| Round | Focus | Status |
|---|---|---|
| [Round 1](response-letter.md) | Project critique (perception, fusion, friction, tailgating, guardian-angel framing) plus a 9-point paper review | Addressed in revision `c541e59` |
| [Round 2](self-review-round-2.md) | Deeper technical audit: temporal staleness, calibration scope, class-dependent bias, HMI latency, the epsilon margin | Partially addressed; open items tracked with dispositions |
| [Round 3](self-review-round-3-audit.md) | Pre-submission audit: claims, notation, references, ethics, open risks | Checklist to run before any submission |

## How to contribute a review

1. Read `paper.tex` (or the PDF) and the GitHub Pages site.
2. Attack the strongest claim, not the weakest.
3. For each finding, record: severity, location in the paper, and a concrete fix.
4. Add it to a new round file; if it is actionable, fold it into the paper and mark the disposition.
