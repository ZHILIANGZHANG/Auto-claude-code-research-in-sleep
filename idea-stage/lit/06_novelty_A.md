# 06 · Phase 3 deep novelty check: Ideas A–D (TTS-1, TTS-2, CL-1, CL-2)

> Date: 2026-10-17. Method: `skills/novelty-check/SKILL.md`, judged under the NOVELTY VERDICT LIMITS in that file.
> **There was no Codex/GPT reviewer. I am the only judge.** Phase C (cross-model check) did not run, so read these verdicts as one model's calibrated call, not as a cross-validated one.
> **Sources searched:**
> - arXiv listing search, about 55 short queries using `all` and `abstract`, sorted newest first.
> - arXiv API helper `tools/arxiv_fetch.py`, 11 queries.
> - WebSearch, 7 queries, 4 of them in extended mode.
> - Semantic Scholar API: every call returned 429, including retries with backoff, so it contributed nothing.
> - ICLR 2027 / OpenReview through WebSearch: no matching submissions are indexed yet.
>
> **Verification:** all 33 arXiv ids cited below pass `tools/verify_papers.py` (`verified`).
>
> **Full texts read** (pdftotext, in `/tmp/claude-0/-home-user-Auto-claude-code-research-in-sleep/e7e19eec-aa61-5636-81ec-3cbec12524d6/scratchpad/novelty_A/pdfs/`): 2609.32035, 2508.03363, 2603.25450, 2609.11799 (grep only), 2610.09835, 2510.18874, 2610.08718, 2502.11019, 2406.12227.

## Verdict summary

| Idea | Verdict | Closest paper (new = not in the survey notes) | One-line delta |
|---|---|---|---|
| A Prior-Discounted Voting | **PROCEED WITH CAUTION** | **2508.03363 JointThinking (new)**, 2609.32035 | No paper treats agreement between think-mode votes and the no-think mode as *negative* evidence. JointThinking treats it as *confirmation*, and its own data argue against the premise on math. |
| B SpecSignal | **PROCEED WITH CAUTION** | **2603.25450 Cross-Model Disagreement / CMP (new)**, 2609.11799 | CMP already uses one teacher-forced pass of a second model as a correctness signal. Still open: a *weaker* same-family draft, *long sampled thinking traces*, *within-question* selection, and an acceptance (1−TV) formulation instead of surprisal. |
| C Zero-mean advantage → absent-token drift | **Original hypothesis refuted at the behavioural level by 2510.18874 (ABANDON-grade for that claim). Reframed version: PROCEED WITH CAUTION** | **2510.18874 Retaining by Doing**, 2610.09835 | 2510.18874 Table 1 already runs the decisive ablation: REINFORCE with {0,1} reward and no baseline (non-negative weights, so nothing cancels) forgets almost as little as GRPO. Still unmeasured: whether the SFT-vs-RL gap lives in absent-token output rows at all. |
| D Steer-Back | **PROCEED WITH CAUTION** | **2502.11019 Function Vectors for CF (ICLR'25, new)**, 2406.12227 (new), 2610.08718 | Inference-time subtraction or addition of a vector that undoes continual-instruction-tuning forgetting already appears as analysis evidence (FV Finding I/II; IV App. F). Still open: a *task-agnostic*, base-vs-FT mean shift computed on generic text, with no old-task data and no head selection, benchmarked as a method against WiSE-FT, DG-Hard and FV subtraction. |

---

## Idea A — Prior-Discounted Voting ("intuition trap")

**Core claims**
1. When think-mode majority voting is wrong, the wrong answer coincides with the same model's no-think modal answer more often than chance.
2. Discounting the no-think modal answer when think votes are split beats maj@N, DeepConf and length baselines.
3. This "anchoring" helps explain 2609.32035's log-prob sign flip and its error concentration.
4. The trigger is conditional: it fires only when think votes are split and the leading answer equals the no-think mode.

**Search:** 14 arXiv queries, among them "no-think majority vote", "nothinking calibration", "hybrid thinking self-consistency", "non-reasoning answer prior", "surprisingly popular" (title and all), "error concentration reasoning distractor", "System 1 answer System 2 voting", and "wrong answer attractor". Also 3 arXiv API queries and 3 WebSearches, one of them on the surprisingly-popular algorithm for LLMs. No LLM application of surprisingly-popular voting turned up. The only matches are classic social-choice papers: 2105.09386 and 2411.08367.

**Closest work, read in full**
- **2508.03363 JointThinking (Thinking with Nothinking Calibration):** samples one think answer and one no-think answer, *accepts* when they agree, and re-thinks when they disagree. It tracks "Both Incorrect & Equiv" (Table 6, Table 8). Key data on R1-7B: think errors coincide with a no-think answer *less* often than with another think sample. Error rates are 4.06% vs 4.90% on GSM8K and 3.68% vs 5.23% on MATH500. On AIME24 and AMC23, equal-wrong pairs are about 0–1 out of 10–18 both-wrong problems. These are single samples, from R1-Distill with a prefilled empty think block rather than a native toggle. **This is the specific thing that makes the delta thin: on open-answer math, a published paper already shows the premise is weak.**
- **2609.32035:** a within-weights toggle on Qwen3-8B/32B, k=8. It measures collision *within* each arm and **never cross-matches think errors with no-think answers** (grep found no cross-arm analysis). It releases per-sample graded answers for all 74,944 generations, which may allow a zero-GPU first diagnostic if the supplementary material is downloadable. I did not check access.
- **2606.23181 DART:** no-think draft agreement is used to *skip* thinking, the opposite direction. **2607.16451:** a single-item pre-commitment probe. **2604.06613:** early-exit work, not related to voting. **2610.10049 (new):** dual-process biases do not shrink with reasoning budget, and anchoring is the one bias in the human direction. It covers cognitive-bias vignettes, not TTS voting, but supports the "intuition persists" story.

**Delta (verifiable):** using the same weights' no-think answer distribution as a *conditional negative prior* inside think-mode voting, rather than as confirmation (JointThinking) or as a routing signal (DART), has not been done.

**What would make it carry**
- Show excess coincidence where JointThinking did not look: bounded answer sets (GPQA-D, MMLU-Pro), at k≥16 per mode, with Qwen3's native toggle. Compare against the right base rate, P(no-think mode = a random think-wrong cluster), matched by the number of options.
- Beat JointThinking and maj@N on held-out sets.
- If the effect appears only on multiple-choice questions, frame the paper as a finding about "reasoning failing to override priors on bounded answer spaces".

**Cheapest pilot**
- Setup: Qwen3-8B, think ×32 and no-think ×32, on GPQA-D (198), MMLU-Pro (500 sampled), AIME24/25 and HMMT25, with vLLM on 1×H100.
- Cost: about 0.4–0.6 GPU-day. The no-think arm costs under 5% of that. The same traces are shared with Idea B.
- Step 0, free: re-read JointThinking Table 6. If possible, recompute the cross-arm coincidence from 2609.32035's released records.
- **Kill criterion:** on problems where think-maj@32 is wrong, P(think-maj answer = no-think mode) does not exceed the option-matched base rate by ≥10 pp (cluster-bootstrap CI excluding 0) on GPQA-D or MMLU-Pro. Also kill if the discount rule's net flips on held-out sets are ≤0 versus maj@32 after Holm correction.

## Idea B — SpecSignal

**Core claims**
1. The expected acceptance rate Σ_x min(p,q) = 1−TV of a small same-family draft over a large model's sampled trace carries information about correctness.
2. It costs nothing: one teacher-forced 0.6B pass, or a byproduct of serving-side speculative decoding.
3. After position-detrending and length residualization it adds to self-certainty, DeepConf and length.
4. It improves trace selection, pruning or early stopping over maj@N.

**Search:** 17 arXiv queries, among them "acceptance rate correctness", "speculative decoding acceptance signal", "draft model acceptance reasoning", "speculative decoding hallucination detection", "cross-model perplexity reasoning", "small model likelihood reasoning trace", "weaker model predictability chain-of-thought", "surprisal reasoning trace correctness", "speculative decoding uncertainty estimation", and "draft model verifier best-of-n". Also 2 arXiv API queries and 2 WebSearches.

**Closest work**
- **2603.25450 Cross-Model Disagreement as a Label-Free Correctness Signal (CMP/CME), read in full (new):** a verifier model's surprisal (CMP) or entropy (CME) over the generator's answer from a single forward pass, with no generation, used as a correctness signal on MMLU, TriviaQA and GSM8K (CMP AUROC 0.75 on MMLU). It includes a full-CoT vs answer-only ablation on GSM8K. Its limitations section says: all pairs have a verifier *at least as strong* as the generator; "the reverse case (weaker verifier) is not directly tested", and the signal degrades as the capability gap shrinks. Decoding is greedy only, and "sampled or long-form chain-of-thought" is listed as future work. This covers the conceptual core of claim 1 for a *stronger* verifier, and it warns that a *weaker* draft may fail.
- **2609.11799 SpecGuard:** per-query acceptance rate as a backdoor detector, not a correctness signal.
- **2512.23765 EASD** and **2602.18447 ConfSpec:** token- and step-level decoding control.
- **2606.01682 CGS (new):** a large-model log-prob minus a small-model log-prob scores chunks from a *small* generator, which is the inverse direction.
- **2604.14682 (new):** acceptance depends on domain and is lexical rather than semantic: chat has the highest acceptance despite the highest entropy. This is a confound warning.
- **2605.09329:** acceptance length decays with position.

**Delta (verifiable):** no paper tests whether a *weaker same-family draft's* acceptance (1−TV) or surprisal over *sampled long thinking traces* discriminates *within-question* correct from incorrect samples and improves selection over maj@N. CMP covers greedy, short outputs, between-question AUROC, and verifiers of equal or greater strength.

**Thin because:** CMP already establishes the mechanism, a second model's single forward pass, and predicts that it degrades for weaker verifiers.

**What would make it carry:**
- Within-question, length-residualized AUROC ≥0.6 with a 0.6B draft.
- Clear added value over CMP computed with the same draft (TV vs surprisal) and over a strong-verifier CMP at matched cost.
- A selection gain over maj@64 that survives Holm correction.
- A clean negative result ("acceptance tracks predictability, not correctness") is publishable only as part of Idea A's shared trace study.

**Cheapest pilot**
- Reuse Idea A's Qwen3-8B think ×64 traces and log top-20 target logprobs during generation.
- Run teacher-forced passes with Qwen3-0.6B and 1.7B, about 0.1–0.2 GPU-day.
- Features: per-position 1−TV computed from the top-k overlap, draft NLL (= weak-CMP), worst window, tail window, all detrended by position.
- **Kill criterion:** pooled within-question AUROC after length residualization <0.55, or ΔAUROC over weak-CMP <0.02, or no filter/weight rule beats maj@64 on held-out sets.

## Idea C — Zero-mean advantages cancel absent-token output-embedding drift

**Core claims**
1. Under SFT, absent-token unembedding rows drift far more than under GRPO.
2. GRPO's zero-mean group advantages cancel the one-sided softmax push, and this explains a large part of the RL-vs-SFT forgetting gap.
3. Raising output-ε or freezing lm_head for SFT closes much of the gap.
4. Positive-only RL (RAFT / REINFORCE with 0/1 reward) should bring back the drift and the forgetting.

**Search:** 14 arXiv queries, among them "RL forgets less", "forgetting SFT GRPO", "SFT RL forgetting mechanism", "output embedding drift", "unembedding drift fine-tuning", "lm_head forgetting", "advantage forgetting GRPO", and "softmax gradient unseen tokens". Also 2 arXiv API queries and 1 extended WebSearch. None of them found a paper measuring absent-token output-row drift under RL.

**Closest work, read in full**
- **2510.18874 Retaining by Doing:**
  - §4.1 and Table 1 compare SFT, **REINFORCE "without an advantage estimator"** and GRPO, with r∈{0,1}. Without a baseline, every gradient weight is ≥0, so the zero-mean cancellation the idea relies on is absent.
  - Result: REINFORCE forgetting stays near GRPO's level. On Llama-3.1-8B, IFEval drop is 7.7 vs 3.4 (SFT: 27.8), MMLU is −0.1 vs −0.2, and Countdown is −0.8 vs −0.5. On Qwen2.5-7B, IFEval is 2.9 vs 0.2.
  - The authors conclude that on-policy data is the cause, "ruling out ... advantage estimation". They also contrast their view with Lai et al. (2507.05386), who had proposed an advantage-based explanation.
  - Self-SFT, trained on the initial policy's own filtered samples, forgets severely. Iterative-SFT, approximately on-policy and positive-only, forgets mildly.
  - **This contains the decisive behavioural test of claims 2 and 4, and the answer is negative.** At most the zero-mean term can explain the REINFORCE-vs-GRPO residual: (7.7−3.4)/(27.8−3.4) ≈ 18% of the gap on Llama-8B IFEval.
- **2610.09835 A Deafening Silence:** covers only continual pre-training, instruction SFT and LoRA. There is no RL or GRPO anywhere in the text (grep). Its limitations restrict the mechanism to Adam-type optimizers. It reports that post-hoc row edits recover under 5% of the forgetting.
- **2509.04259**, **2605.28860**, **2610.02437**, **2607.04364**, **2601.02151 (EAFT, new)** and **2604.25011 (new)** each offer behavioural, circuit-level, feature-level or theoretical explanations. None is at the output-row level.

**Verdict:** the claim "zero-mean advantage cancellation is the mechanism" is already tested, with a negative result, in 2510.18874 Table 1. Under the verdict limits that is ABANDON-grade for *that framing*. The idea is salvageable only as a reframe, which gets **PROCEED WITH CAUTION**: "Does the SFT–RL forgetting gap live in 2610.09835's absent-token output rows, and is it on-policy-ness (Self-SFT vs Iterative-SFT vs REINFORCE vs GRPO), not the sign of the advantage, that removes the drift?"

Retaining-by-Doing's data already make this risky. Self-SFT and RL share roughly the same vocabulary and the same absent-token set, yet forget very differently. That predicts the output-row locus will *not* separate them.

**What would make it carry:** row drift that tracks forgetting across all five regimes (Spearman ≥0.7), *and* output-ε closing ≥30% of the SFT–GRPO gap at matched target gain.

**Cheapest pilot**
- Model: OLMo-2-1B-Instruct, chosen because it has untied embeddings. Llama-3.2-1B and Qwen2.5-1.5B are tied.
- Data: Tulu-3 IF subset or Countdown, following the 2510.18874 recipe.
- Runs: SFT (expert data), Self-SFT, Iterative-SFT, REINFORCE-0/1, GRPO, and SFT+output-ε. About 6 runs × 1–2 h on 1×H100, under 0.5 GPU-day.
- Log per-row Δ‖u_i‖ against token count. Evaluate MMLU, HellaSwag and ARC.
- **Kill criterion:** Self-SFT and GRPO absent-row drift differ by less than 1.5× while their forgetting differs by more than 3×, or output-ε closes less than 20% of the SFT–GRPO gap.

## Idea D — Steer-Back

**Core claims**
1. Much of the forgetting after continual instruction tuning or MCIT is a shared, reversible shift.
2. Subtracting a per-layer mean residual shift μ_ℓ = E[h_ft − h_base], estimated on unlabeled generic text, restores old and general abilities without hurting the new task.
3. It needs no weight changes and no old-task data.
4. It beats or matches WiSE-FT and DG-Hard.

**Search:** 15 arXiv queries, among them "steering forgetting", "activation steering catastrophic forgetting", "spurious forgetting", "steering vector restore capabilities fine-tuning", "mean activation difference fine-tuned base", "training-free forgetting recovery", "representation surgery merging", and "inference-time intervention forgetting". Also 2 arXiv API queries and 2 WebSearches, one of them aimed at ICLR 2027.

**Closest work**
- **2502.11019 (ICLR 2025, read; new to the survey):** continual instruction tuning on SuperNI with Llama2-7B/13B, Llama3-8B and Mistral-7B.
  - Finding I: *adding* the source function vector at one layer during inference restores performance, e.g. +7.3 on HellaSwag.
  - Finding II: *subtracting* the current-task FV (h_l ← h_l − θ) "reduces forgetting", e.g. HellaSwag 39.5→49.1.
  - The authors use these only as analysis evidence. Their method is a training-time FV regularizer.
- **2406.12227 Instruction Vector (read; new):** App. F adds an IV to a TRACE-finetuned model at inference, and CommonsenseQA recovers from 0.03 to 0.47–0.49.
- **2610.08718:** subtracts the common shift in activation/logit space, but **only in a synthetic Transformer**. For pretrained OLMo-2-1B the authors explicitly *declined* activation-space averaging ("there is no single set of states over which to average") and edited weights instead, removing the top singular direction of each update. The setting is facts, a single model.
- **2501.13453:** a training-time freeze. **2605.20296 DG-Hard:** a weight-space SVD threshold. **2608.24988:** the reverse question. **2608.28547 DARTS (new):** a *trained* per-position bias that corrects representation drift after model merging.

**Delta (verifiable):** a *task-agnostic*, base-minus-fine-tuned mean residual shift estimated on generic text, applied to all layers with no old-task prompts, no head or layer search per task, and no training, benchmarked as a forgetting-repair method on TRACE and on general ability against WiSE-FT, DG-Hard and FV subtraction. None of the papers above does this.

**Thin because:** FV and IV already show that inference-time vector add/subtract undoes continual-instruction-tuning forgetting.

**What would make it carry:**
- The generic-text shift recovers at least as much as task-specific FV subtraction, which needs task ICL data.
- It Pareto-dominates WiSE-FT at matched new-task accuracy.
- It works where 2610.08718 says activation averaging should be ill-posed (varied prompts in a pretrained LM), which ties it to the common-shift mechanism.

**Cheapest pilot**
- Setup: Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, sequential LoRA on 3 TRACE tasks (about 3 GPU-hours).
- Estimate μ_ℓ on 1k C4 or WikiText samples (minutes), sweep α∈{0.25, 0.5, 1}, and optionally gate on the first k tokens.
- Evaluate MMLU, GSM8K, IFEval and BBH, plus TRACE OP and BWT.
- Baselines: FV subtraction (2502.11019 recipe), WiSE-FT at matched new-task accuracy, DG-Hard.
- About 0.5 GPU-day in total.
- **Kill criterion:** recovers less than 25% of the general-ability drop at ≤2 pt new-task loss, or is dominated by WiSE-FT or FV subtraction.

---

## What changes the picture versus the survey notes (03, 04)
1. **A** is not NOVEL-LIKELY-clean. JointThinking (2508.03363) uses think/no-think agreement in the *opposite* direction, and its data contest the premise on math. Restrict the claim to bounded answer spaces, or let the pilot decide.
2. **B** has a direct conceptual precursor the survey missed: CMP (2603.25450). Position SpecSignal as a "weak-verifier, long-trace, within-question CMP/acceptance study" and make CMP the baseline to beat.
3. **C**'s headline mechanism is pre-empted by 2510.18874 Table 1 (REINFORCE without a baseline still forgets little). The survey's suggested "first pilot" (SFT vs GRPO row drift) is still worth 4 GPU-hours, but it must include Self-SFT and REINFORCE-0/1 arms. Otherwise a positive result would be confounded with on-policy-ness.
4. **D** has ICLR'25 precursors (2502.11019, 2406.12227) that already intervene at inference to undo continual-instruction-tuning forgetting. The delta rests on task-agnostic estimation from generic text plus a benchmarked comparison.
5. Ranking for execution: **A+B** share one trace set, so run the pilot once and kill fast. **D** next, the cheapest training-free method paper. **C** only in its reframed form, and as a quick falsification.
