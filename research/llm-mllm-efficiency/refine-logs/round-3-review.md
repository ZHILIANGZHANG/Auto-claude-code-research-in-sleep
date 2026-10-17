# Round 3 Review

- Reviewer: same reviewer thread (resumed). Same model family as the executor, so **not an ARIS acceptance receipt**. No new searches this round; all checks are arithmetic on the proposal's own figures and RA's figures.

| Dimension | R1 | R2 | R3 |
|---|---|---|---|
| Problem Fidelity | 8 | 8 | 9 |
| Method Specificity | 6 | 7 | 8 |
| Contribution Quality | 6 | 7 | 8 |
| Frontier Leverage | 6 | 6 | 6 |
| Feasibility | 7 | 8 | 8 |
| Validation Focus | 7 | 8 | 8 |
| Venue Readiness | 5 | 6 | 7 |
| **Overall** | **6.4** | **7.1** | **7.8** |

**Verdict: REVISE (converged).** No drift. The reviewer recommends fixing the three IMPORTANT items and then **stopping refinement and running G0**.

<details><summary>Reviewer response (transcribed, structure preserved)</summary>

## Checks
1. **O_K**
   - The algebra is right. The per-round factor (K−ℓ_p)/(K+r−ℓ_p) is exact. Rounds are independent, so the product form holds. The round count is right.
   - s^{n_tok} holds to within ~n/K.
   - At K=1024 the per-round factor is 0.932–0.941, which matches RA's "0.94^n".
   - It is consistent with Table 3 (s^5 ≈ 3e-8 → 0) and with shared ≈ per-head on verbose MATH500.
   - **IMPORTANT: P1 is no longer implied.** Table 3 is equally consistent with a union model of joint survival (1−(1−3e-8)^288 ≈ 1e-5). The adopted single-draw model predicts shared ≈ per-head on children too.
   - Reframe P1 as a pre-registered discriminator between the two survival models. Remove it from Claim 1's evidence. Run the eviction regression primarily on the shared-draw arm, which matches O_K exactly, and confirm cross-layer sharing.
2. **G0**
   - The unit A is correct, and 25% for Promote is reasonable.
   - **IMPORTANT:** both pairs dropping 10–25% has no branch, and this is the most likely outcome. Add "7B offline re-check, then proceed as a measured moderate per-token tax".
   - MINOR: say which drafter's A the gate uses; define "matched buckets" (pooled vs every bucket).
3. **E**
   - Correct, and there is no left-censoring problem.
   - If E_v ≥ 16 is censored and E_c is finite, E_c/E_v is a one-sided upper bound in the hypothesis direction.
   - MINOR: use a crossing rule for non-monotone retention (isotonic fit, then first crossing).
   - MINOR: the 100-problem full-KV check has a CI of about ±4 points, as large as the 5% margin. Use ≥4 samples, a pre-registered equivalence margin of |Δ| ≤ 1.5, and fall back to HF.
4. **Memory**
   - Correct for per-sequence *peak* KV; label it so.
   - Time-integrated KV scales differently (T_v²/2 vs T_c²/E_c).
   - MINOR: restore the length rebound as a reported quantity.
5. **Regression**
   - Power is now adequate: about 3–4K eviction samples per model.
   - **IMPORTANT:** subtracting a flip rate from a logit outcome is ill-defined, and the selection uses the same samples as the covariate.
   - Fix: fit one per-sample model, correct ~ policy × O_K + log length + budget FE + pass rate. Take the pass rate from independent resample samples, with a problem random effect or clustered SEs. The mechanism is the policy × O_K term.
   - MINOR: make one stratified regression with model FE primary; report the VIF.
6. **Claim 2**
   - S_c/S_v = ρ_SD/ρ_noSD is correct.
   - Lossless SD leaves accuracy unchanged, so the per-solved normalization cancels in S_c/S_v.
7. **Compute**
   - 198 → 257 is consistent; about 33 h wall-clock on 8 GPUs.
   - Promoting EAGLE (~105) pushes it to ~365, over 336. Trim AIME ×8 → ×4 in the eviction grid and the margin to 20%.

## Ceiling analysis
- The conclusion is roughly right, but the reason is wrong.
- **FL is not structurally capped.** Its 6 came from the drafters being non-production. Promoting the single 7B-parent EAGLE-3 head (parent head reused on the children) moves FL to about 8, with no drift.
- **VR is not structurally capped.** It is now 7. Remaining issues: model vintage (R1-Distill era) and unknown effect size.
- **The real cap is pre-results uncertainty.** The best credible pre-results profile is PF9, MS9, CQ8, FL8, F8, VF9, VR7, which gives 8.4. Reaching 9 needs CQ≈9 and FL≈9, which is not credible before data.
- Do not add a method to chase 9; that would be drift.
- Fix the IMPORTANT items, optionally promote EAGLE, **then stop and run G0**.

## Simplification
- Use the shared-draw arm as the primary mechanism arm.
- Use one stratified regression with model FE.

## Modernization
- Promote the 7B-parent EAGLE-3 head.

OVERALL: 7.8 — VERDICT: REVISE
</details>

## Executor decision
- **Adopt all three IMPORTANT items and all MINOR items.**
- **Promote EAGLE-3** as a conditional must-run: train it only if G0 promotes. Trim the budget to stay within 336 GPU-h: AIME ×8 → ×4 in the eviction grid, resample control on MATH500 only, margin 15%.
- **Stop the review loop after round 3** (MAX_ROUNDS=5 not used). Reason: the reviewer judged the proposal converged and recommended running G0. The remaining uncertainty is empirical, and further rounds would add process cost without information.
- **Final score: 7.8 REVISE.** Remaining uncertainty is recorded in FINAL_PROPOSAL's named risks.
