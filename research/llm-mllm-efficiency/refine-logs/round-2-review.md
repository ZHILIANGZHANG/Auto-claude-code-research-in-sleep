# Round 2 Review

- Reviewer: the round-1 reviewer, resumed with its context intact (same-thread follow-up). Fresh-context Claude subagent; **same family as executor, not an ARIS acceptance receipt.**
- Proposal reviewed: `round-1-refinement.md`. Anchor verified word for word. 2 WebSearch spot-checks.

| Dimension | R1 | R2 |
|---|---|---|
| Problem Fidelity | 8 | 8 |
| Method Specificity | 6 | 7 |
| Contribution Quality | 6 | 7 |
| Frontier Leverage | 6 | 6 |
| Feasibility | 7 | 8 |
| Validation Focus | 7 | 8 |
| Venue Readiness | 5 | 6 |
| **Overall** | **6.4** | **7.1** |

**Verdict: REVISE.** Drift: NONE.

<details><summary>Reviewer response (transcribed, structure preserved)</summary>

## Did the round-1 fixes land?

| Item | Status |
|---|---|
| Renewal-walk τ | Fixed and correct. Assign each step to the bucket where it *starts*; pool as a ratio of sums. |
| EAL off-by-one | Fixed. Knock-on effect on the G0 thresholds (N4). |
| G renamed | Fixed. |
| O_K | Partial; new errors (N1–N3). |
| Currency E | Mostly fixed (N5, N6). |
| Within-model test | Fixed in design; power and covariate issues remain (N7). |
| Loop and coverage controls | Fixed. |
| ρ identity | Correct; also S_c/S_v = ρ_SD/ρ_noSD (N8). |
| Compute | Plausible; eviction slightly low (~105–110 GPU-h vs 96). |
| G0 | Written; needs recalibration (N4). |
| 2×2 demotion | Fixed. |

## New problems

- **N1 (IMPORTANT): the per-head union 1−(1−s)^H is the wrong form.**
  - R1-Distill-Qwen-1.5B has 28 layers × 2 KV heads; 7B has 28 × 4.
  - A literal any-(layer, KV head) union predicts ≈0.9999 recall for RA's 57-round passcode: s = 0.941^57 ≈ 0.031, over 288 draws for Qwen3-4B. RA reports 0.000. A per-layer union over 8 KV heads gives 0.22 for a single token.
  - What fits instead: retrieval is head-local, and facts are multi-token (Qwen splits every digit), giving 0.031^5 ≈ 3e-8.
  - The analytic per-head form is not identifiable from current evidence. Drop it. Use the joint mention survival s^{n_tok} under a single draw, and keep P1 empirical. State whether shared-draw is shared across layers.
- **N2 (IMPORTANT): multi-token facts.** Mention survival should be s^{n_tok}.
- **N3 (MINOR): the exponent.**
  - The actual round count is ⌊u/r⌋−⌊m/r⌋−b, with b=1 only if a protecting buffer exists.
  - At small K, one round shifts s by ~11%.
  - Make the logged empirical curve primary.
  - Confirm whether the prompt counts against K.
  - Say which K is used for each budget point.
- **N4 (IMPORTANT): G0 thresholds.**
  - τ≥1 and n-gram τ is roughly 1.3–1.8, so a 15% drop in τ equals a 35–50% drop in τ−1. State the gate on τ−1.
  - Define the middle zone and restore the "composition-only" branch.
  - Define O_K(c)≈O_K(v) as within ±10% with a CI.
- **N5 (IMPORTANT): E is censored at 8×.** RA's MATH500 numbers suggest the parents keep ≥95% at 8×. Add a 16× point for the parents, or pre-register censored bounds. Bootstrap E over problems; the CI is likely ±10–20%. Take full accuracy from the same harness, or show HF and vLLM agree.
- **N6 (MINOR): memory formula mislabelled.**
  - The total saving vs parent full-KV is (T_v/T_c)·E_c.
  - (T_v/T_c)·(E_c/E_v) is the saving vs parent+evict.
  - Peak KV equals typical length divided by E.
- **N7 (IMPORTANT): power and specification of the within-model test.**
  - About 25 flip events per child and about 0 per parent.
  - Pool budgets within each model with a budget fixed effect.
  - Use the model's own full-KV pass rate as the covariate, minus the resample flip rate.
  - Use the same O_K for both regressions; on the τ side use only the 1.5B drafter and G. Name as a risk that O_K–τ is weaker by construction.
- **N8 (MINOR): the identity is exact only at b=1.**
  - At b>1 it is a model, licensed only by validation. Validate at 128 or drop 128.
  - Restate Claim 2 as ρ_SD < ρ_noSD ⇔ S_c/S_v < 1, and keep ρ_SD < 1 as a separate deployment statement.

## Dimensions < 7
- **Frontier (6)**:
  - No QwQ-32B EAGLE-3 head or ThinkPrune-QwQ found in spot-checks.
  - The only verified public head is R1-Distill-Llama-8B, and it has no compressed child.
  - Run one checkpoint search; otherwise keep it as a named risk and drop the QwQ line.
- **Venue (6)**: inherent to a measurement paper.
  - Promote Laser-D-7B into the main eviction grid (~15 GPU-h) to satisfy "≥3 pairs at 2 scales" strictly.

## Simplification
1. Delete the analytic per-head union; use a single O_K with s^{n_tok}.
2. Use the same O_K for both regressions.

## Modernization
The R1-Distill-Llama-8B EAGLE-3 head, only if a compressed child is found; otherwise NONE.

OVERALL: 7.1 — VERDICT: REVISE
</details>

## Executor's assessment
- Accept N1–N8.
- N1 was verified by recomputation: 0.941^57 ≈ e^{−3.47} ≈ 0.031, and 1−(1−0.031)^288 ≈ 1, against RA's measured 0.000. The formula is falsified.
- N3: RA App. D's own survival formula is ((K−ℓ_p)/(K+r−ℓ_p))^n, so the prompt does count against K. Adopted directly.
- Frontier: executor search (2 queries) found no compressed R1-Distill-Llama-8B child and no Nemotron-Nano-8B EAGLE-3 head ([search 1](https://featherless.ai/models/unsloth/DeepSeek-R1-Distill-Llama-8B), [search 2](https://arxiv.org/pdf/2604.19767)). The QwQ line is dropped.
- New option: train ONE EAGLE-3 head for the 7B parent only (~100 A100-h, SpecForge) to measure the deployment cell parent-head → child.

### Ceiling analysis (why READY ≥ 9 may be unreachable without changing the paper's nature)
Frontier Leverage and Venue Readiness are capped at ~6 by the paper type: no new frontier component, by design per the Non-goals. Even with Specificity 9, Contribution 8, Fidelity 9, Feasibility 8, Validation 9:

0.15·9 + 0.25·9 + 0.25·8 + 0.15·6 + 0.10·8 + 0.05·9 + 0.05·6 = **8.05**

The rubric's READY bar cannot be reached without changing the paper's nature.
