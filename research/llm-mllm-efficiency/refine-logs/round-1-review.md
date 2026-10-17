# Round 1 Review

- Reviewer: fresh-context Claude subagent (general-purpose), 7-dimension research-refine rubric. **Same model family as executor; Codex/GPT reviewer unavailable — not an ARIS acceptance receipt.**
- Bundle: `codex_round_1_review_bundle.md`; proposal: `round-0-initial-proposal.md`
- Reviewer tooling: read proposal + research_contract + IDEA_REPORT (review, full-text sections); 5 WebSearch spot-checks.

| Dimension | Score |
|---|---|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 6 |
| Frontier Leverage | 6 |
| Feasibility | 7 |
| Validation Focus | 7 |
| Venue Readiness | 5 |
| **Overall (weighted)** | **6.4** |

**Verdict: REVISE.** Drift: NONE.

<details><summary>Reviewer response (transcribed in full, structure preserved)</summary>

## Summary judgment
Anchor is well defined and scope is disciplined. The trace-coupled EAL estimator is correct, but as written it averages the wrong quantity, with a bias that favours the verbose model, which could manufacture the effect. Smaller definitional errors: an off-by-one against Leviathan, oracle-argmax mislabelled as an upper bound, and O(K) conflating text distance with cache budget. The mechanism test is confounded (a cross-model Spearman over ~8–10 points with compression ratio as a common cause). All of these are fixable at the definition level. Compute is underestimated for the eviction grid and for batch-1 timing.

## A. Trace-coupled EAL
- Correct: with a deterministic drafter, lossless SD accepts with p(d) and resamples from p restricted to ≠d, which is the same law as "sample y~p, accept iff y=d". So the acceptance length is the prefix-match length, unbiased provided p is the same processed distribution the trace was sampled from (T=0.6, top-p 0.95). One teacher-forced drafter pass suffices, and n-gram needs none. The self-correction of the product formula was right.
- **Error 1 (CRITICAL)**: SD visits only block-boundary positions. Tokens/step = N/#steps is a token-weighted harmonic mean of (L_t+1), which is ≤ the uniform per-position arithmetic mean, and the gap grows with heterogeneity. Example: τ=5 bursts vs τ=1 regions at 50/50 gives a per-position mean of 3.0 but an actual rate of 1.67. Verbose traces are hypothesised to be burstier, so this inflates the verbose side differentially. Fix: **renewal walk**. Start at t=0, compute L by prefix match, advance t←t+L+1 (under the coupling the correction or bonus token equals y), and report tokens/steps per bucket, averaged over the k+1 start offsets. Keep α_t only as an i.i.d. diagnostic.
- **Error 2 (IMPORTANT)**: (1−α^{k+1})/(1−α) is tokens/step including the bonus token; E[L] under i.i.d. is (α−α^{k+1})/(1−α). Define EAL := E[L]+1 everywhere, consistent with vLLM's 1 + accepted/steps.
- Variance (MINOR): a Rao-Blackwellised estimator Σ_j 1[d_{<j}=y]·p(d_j|y_{<t+j−1}) is optional. Bootstrap must resample problems or traces, not positions.
- Variable-length drafts (MINOR): cap L at the draft length; no proposal yields 1 token.

## B. Oracle-argmax is not an upper bound (IMPORTANT)
A stochastic drafter with q=p accepts with probability 1, and greedy is not block-optimal. Rename it "target greedy-predictability"; it is entangled with temperature and RL entropy sharpening, which pushes against the hypothesis and should be stated.

## C. O(K) (IMPORTANT)
(1) Text distance vs cache budget are different K's; random-eviction survival decays smoothly with distance. (2) Absolute K biases against the hypothesis; use each model's own relative budget. (3) O and R overlap. Merge them into O_K = mean over reuse events of Π_{prior mentions}(1 − s_K(u−m)), with s_K the policy's survival curve. (4) Trivial literals collide; restrict them, and report annotation precision on reuse *events*.

## D. Drafter 2×2 (IMPORTANT)
The parents were distilled from the same R1 data, so the parent diagonal is aligned. The children come from independent RL runs at different compression rates (DLER −77/−69, LC-R1 −52/−46, AdaptThink −53/−40), so the child diagonal is not aligned. The 2×2 bounds the mismatch share but does not identify it. Make the target-side measure primary and drafter-free, make d_v→m_c vs d_v→m_v the primary deployment number, and treat the child cells as secondary. Claim "just retrain the drafter" only if the KD drafter is run.

## E. Compute
- Traces (8 h): plausible but optimistic; cap unstated; AIME tails.
- Offline (5 h): plausible; store argmax and p(y) only.
- vLLM timing (25 h): underestimated if full-set batch 1 (7B at batch 1 ≈ 80–100 tok/s → ~25 h per config on the full set).
- Eviction (80 h): 1.5–3× too low (~0.5B tokens main grid plus sweeps; HF R-KV/SnapKV ~1–2K tok/s at 7B → 150–260 h).
- Resample (10 h): plausible.

Fixes: a timing subset, a per-step cost grid c(context, batch, k) composed with walk-τ and validated end-to-end on a subset, and a trimmed eviction grid.

## F. ρ
The direction is correct. Make it an exact identity: wall = Σ_steps c, ρ = (τ_c/τ_v)·(c̄_v/c̄_c), and τ_c/τ_v = (target-side ratio) × (drafter residual), which is log-additive. Use a ratio of sums over a fixed workload; throughput at batch 32/128; and add a per-solved-problem variant.

## Dimensions < 7
- **Specificity (6)**:
  - CRITICAL: renewal walk.
  - IMPORTANT: off-by-one and renaming oracle-argmax; O_K.
  - IMPORTANT: "gap at equal random-baseline accuracy" is undefined. Interpolate K at a target retention from the 2×/4×/8× sweep, which needs the sweep on every model in the gap analysis.
  - IMPORTANT: loops and truncated traces inflate the verbose side. Report with and without cap-hit or repetition-flagged traces, with a pre-fixed threshold.
  - MINOR: late buckets are selection-confounded. Restrict to problems covered by both models and report N per bucket.
  - MINOR: G0 is undefined. Restate the P0/C5 thresholds.
- **Contribution (6)**:
  - The cross-model Spearman over ~8–10 points is confounded by compression ratio. Only 8 points remain with AdaptThink excluded. Make the primary mechanism test within-model (per problem, controlling for length) and treat cross-model as descriptive with a partial correlation.
  - Eviction has no shared currency. Define the eviction gain as the relative compression achievable at ≤X accuracy loss, so composition = token ratio × eviction ratio.
- **Frontier (6)**: production uses EAGLE-3/MTP. The public `yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B` exists. If a compressed Llama-8B child exists, it gives a zero-training EAGLE-3 cell; otherwise record it as a named risk.
- **Venue (5)**: inherent to a measurement paper. Airtight measurement plus one crisp composition number helps more than extra experiments.

## Keep
- The coupling argument and one-pass claim.
- The ρ sign flip.
- n-gram kept out of the O prediction.
- ≥3/4 both-correct plus a resample control.
- AdaptThink stratified.
- Relative budgets.
- Shared-draw vs per-head as the P1 discriminator.
- The matrix meets ≥3 pairs at 2 scales.
- DLER-1.5B (−77%) verified, and vLLM `draft_model` exists.
- Qwen2 vs Qwen3 also differ by QKV bias; the port is still small.

## Named risks
- vLLM draft-model vocab mismatch: R1-Distill-1.5B 151,936 vs 7B 152,064 embedding rows (to verify); offline EAL is unaffected.
- Entropy sharpening.
- vLLM #56101 is unverified by the reviewer, and moot if acceptance is computed from committed tokens.

## Simplification
1. Merge O and R into the survival-kernel O_K.
2. Trim the eviction grid to full / per-head / shared / R-KV. SnapKV and recency get one appendix budget; the 2/4/8× sweep runs on MATH500 only; absolute K runs on 2 pairs.
3. Replace batch-1 full-set timing with a cost grid × walk-τ; demote the child-drafter cells.

## Modernization
The public R1-Distill-Llama-8B EAGLE-3 head, if a compressed Llama-8B child exists; otherwise NONE.

## Drift
NONE.

OVERALL: 6.4 — VERDICT: REVISE
</details>

## Executor's assessment of the review
I accept all CRITICAL and IMPORTANT points after checking each:
- **Renewal walk**: correct. Under the coupling, the rejection or bonus token equals y_{t+L}, so the walk reproduces the SD step count exactly.
- **Off-by-one**: correct.
- **Oracle-argmax is not an upper bound**: correct; renamed.
- **O_K survival kernel**: adopted. Under RA's schedule each candidate survives a round with probability K/(K+r), so s(Δ) ≈ (K/(K+r))^{Δ/r}, i.e. RA's 0.94^n at K=1024, r=64. Per-head union over H KV heads is 1−(1−s)^H, while shared-draw uses s. This makes P1 an analytic prediction.
- **Within-model mechanism test**: adopted.
- **Eviction currency**: adopted.

Pushback: none. EAGLE-3 stays a named risk. Searches found no compressed R1-Distill-Llama-8B child. The only zero-training EAGLE cell candidate is QwQ-32B → `Shiyu-Lab/QwQ-32B-thinkprune-3k` with `reinforce20001/QwQ-32B-Eagle`, but that head's version and quality are unverified, so it goes to nice-to-have.
