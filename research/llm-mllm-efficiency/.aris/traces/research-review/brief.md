# Research review brief — LLM/MLLM efficiency idea discovery (2026-10-17)

You are reviewing three candidate research directions that survived literature survey, lens-based ideation (18 → 13 deduped), devil's-advocate triage and a deep novelty check. No pilot experiments have been run (no GPU in this environment). All literature evidence comes from WebSearch abstracts/snippets — no full texts were read, and 2026 preprints are unverified.

Primary artifacts (read them; executor notes are not evidence beyond what they cite):
- /home/user/Auto-claude-code-research-in-sleep/research/llm-mllm-efficiency/idea-stage/IDEA_REPORT.md — sections "Literature Landscape", "Ranked Ideas", "Novelty Verification"
- /home/user/Auto-claude-code-research-in-sleep/research/llm-mllm-efficiency/idea-stage/candidate_pool.md — full candidate descriptions (C1–C13)

The three directions:
1. **C1+C5 (recommended)** — "Length compression eats the redundancy": does length-penalty RL (ThinkPrune / L1 / AdaptThink / LC-R1) lower speculative-decoding acceptance (n-gram, small drafter, EAGLE-3; decomposed into drafter shift vs intrinsic) and make random decode-KV eviction stop being free (vs R-KV / SnapKV), with a single redundancy statistic (n-gram restatement rate) predicting both? Novelty 7/10 PROCEED. Named risks: position confound in EAGLE τ; "filler is low-entropy" may be false; "just a measurement / obvious".
2. **C4** — KV eviction applied only to the softmax layers of hybrid (linear/SSM + softmax) LLMs vs size-matched/parent dense models, with per-layer recency-mass/entropy profiling. Novelty 6/10 PROCEED; crowded Sep–Oct 2026 (GateSight, HeadWiseKV, DeltaS); prior evidence (2605.18053, HeadWiseKV, Hymba) leans against the "window collapses" hypothesis.
3. **C2** — Thinking MLLMs vs matched Instruct (Qwen3-VL 2/4/8B) under prefill visual-token pruning, late-evidence rate by CoT position with causal ablation, thinking-budget sweep. Novelty 6/10 PROCEED; DSTP may already contain Qwen3-VL-4B Thinking rows; counter-evidence that late look-backs carry little visual information (TVC, VisualSwap, 2606.28864).

Constraints: academic lab, ≤ 8 × A100/4090-class GPUs, main experiments ≤ ~2 GPU-weeks per idea, open models only.

## What we need from you
Act as a senior NeurIPS/ICML reviewer. Start from the assumption each idea is broken somewhere — find where. The core hypotheses are not up for rewriting; your risks choose the next tests, they do not add components.

For EACH of the three directions:
1. Strongest case for it (what would make it the paper people cite).
2. Logical gaps / unjustified claims in the plan; missing controls; confounds.
3. The single cheapest *discriminating* experiment (≤ 2 GPU-hours) and its decision rule (what result kills it, what result promotes it).
4. A results-to-claims matrix: for each plausible outcome of the main experiment, what claim is allowed and is it a paper.
5. Mock review score (1–10) for the paper as planned, assuming the main result comes out as hypothesized, and the venue tier it fits.
6. **Bottom line: PROCEED or ABANDON** (one word plus one sentence).

Then overall:
- Which ONE direction should the user start with, and the order for the others.
- Minimal experiment package for the chosen direction (highest acceptance lift per GPU-week), with rough GPU-hour costs.
- Anything in the literature landscape synthesis you think is wrong or overstated.

Scope limits: do not propose extra modules, gates, or machinery to absorb an objection — record it as a named risk. Say plainly when something holds up. Do not manufacture findings.
