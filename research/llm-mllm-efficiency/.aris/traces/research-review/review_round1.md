# Research review — round 1 (trace)

- Date: 2026-10-17
- Brief: `brief.md` (same directory)
- Reviewer: fresh-context Claude subagent (general-purpose, same model family as executor). **Not a cross-family (Codex/GPT) review** — Codex MCP unavailable in this environment. Not usable as an ARIS acceptance receipt.
- Reviewer tooling: read IDEA_REPORT.md + candidate_pool.md; 11 standard WebSearch queries; no full texts.

## Verified facts reported by reviewer
- DeepScaleR-1.5B-Preview is fine-tuned from R1-Distill-Qwen-1.5B. ThinkPrune has two checkpoints (`DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k`, `DeepScaleR-1.5B-Preview-thinkprune-iter2k`); pair explicitly.
- Random Attention (2609.03430): prompt kept, rest evicted uniformly at random independently per head; Qwen3 + Phi-4-reasoning at ~4×; code public; attributes success partly to "the model restates what it still needs" and per-head redundancy.
- DSTP (2604.12358, ECCV'26) has Qwen3-VL-4B numbers at 33% retention; no Thinking rows found.
- 2605.18053 is a single-author preprint; PAGE (2609.22157) finds NIAH-MK3 still collapses at the protected floor.
- Nemotron-H-8B-Base is an 8K-context model (4 attention layers of 52). Falcon-H1 is a parallel hybrid (attention + Mamba-2 heads in each block).
- No public EAGLE-3 head for R1-Distill-Qwen-1.5B/7B found.
- DREAM-R (2605.28678) reports, in a multimodal ablation, that a stronger length reward lowers verifier acceptance.

## C1+C5 — VERDICT: PROCEED (mock 5/10 as planned; ~6/10 with controls + 7B + measured wall-clock; Findings / main borderline)
Gaps:
1. Primary metric not length-bias-free: prompt-lookup/suffix drafters improve with history length; shorter compressed traces mechanically lower n-gram τ. Kill criterion "relative τ drop <10%, unmatched" is biased toward PROCEED. Report matched absolute-position buckets + fixed lookup window (e.g. last 2K).
2. Position matching may remove the effect (redundancy may live late in verbose traces). Pre-register: (a) lower τ at matched positions = per-token draftability tax; (b) equal matched τ but deleted high-τ segment = composition effect.
3. "One redundancy statistic" partly circular for n-gram τ; restrict predictive claim to EAGLE/model-drafter τ and the eviction gap.
4. Opposing effect ignored: attention/KV-read cost grows with context (Kinetics 2506.05333), so halving trace cuts wall-clock by more than half before SD; only measured wall-clock can support the headline.
5. SD on a 1.5B target barely deployment-relevant; wall-clock & EAGLE claims belong at 7B with R1-Distill-1.5B as natural drafter.
6. Drafter re-distillation must be token-matched; evaluate all cross pairs.
7. AdaptThink is a mode mixture — stratify by think/no-think.
8. Use sampled traces at T≈0.6 (greedy loops inflate n-gram τ); compute expected acceptance from target probabilities (exact for lossless SD, one forward pass).
9. Both-correct conditioning → regression to the mean; define as ≥3/4 correct + unperturbed re-sample control.
10. C5: floor effects (compare gap at matched random-accuracy across budget sweep); R-KV also loses signal on less-redundant traces (gap direction genuinely uncertain); per-head redundancy is untouched by length compression and caps the effect — random baseline must be per-head independent; report relative and absolute budgets; per-problem regression confounded by difficulty, per-model has only ~6–8 points.
11. Novelty slightly overstated: cite DREAM-R as partial precedent.

Cheapest discriminating experiment (~1.5 GPU-h): DeepScaleR-1.5B→L1-1.5B-Max and R1-Distill-1.5B→R1-Distill-Qwen-1.5B-thinkprune-iter2k; 200 MATH-500 × 4 samples, T=0.6, 16K cap, vLLM; one forward pass for target log-probs; prompt-lookup expected accepted length (n=3, k=5) unmatched / bucketed by absolute position (0–1K, 1–2K, 2–4K) / last-2K window; L1 at 3 target lengths (same weights); both-correct ≥3/4.
- Kill SD half: <10% drop in every matched bucket in both pairs AND unmatched drop vanishes after matching AND L1 sweep flat.
- Promote: ≥15% drop at matched position with fixed window in both pairs AND τ decreasing with L1 target length at fixed weights → go to 7B + EAGLE-3.
- In between: continue only as "deleted high-τ segment" claim, pre-registered.

Results-to-claims matrix:
| SD outcome | Eviction outcome | Claim | Paper? |
|---|---|---|---|
| A ≥15% matched drop for n-gram and re-distilled EAGLE-3 at 7B; measured effective compression worse than token ratio | F gap widens at matched random-accuracy, relative budgets, tracks redundancy over ≥5 points | Full thesis | Main borderline (~6) |
| A | G gap flat | SD tax real; restatement isn't what compression removes (per-head redundancy dominates) | Findings / main borderline |
| B drop vanishes after matching | any | Length-artifact methodology note | Workshop; Findings if F |
| C EAGLE-3 drop vanishes after token-matched re-distillation | any | "Retrain your drafter" | Workshop |
| D τ unchanged/higher | G | Multiplicative composition | Short/negative; workshop |
| E τ drops but wall-clock ratio ≤ token ratio (KV-read savings) | F | Tax masked by attention-cost savings; eviction no longer free | Findings |
| any | H gap only at absolute budgets | Budget artifact | — |

Minimal package (~300 GPU-h): P0 discriminating test 1.5 GPU-h (gate) → P1 traces incl. R1-Distill-7B→LC-R1-7B / AdaptThink-7B (mode-stratified) ~25 → P2 offline expected acceptance n-gram (matched + windowed) + R1-Distill-1.5B drafter for 7B ~10 → P3 token-matched EAGLE-3 heads + cross-eval ~150 (only if P0/P2 promote) → P4 measured vLLM wall-clock at batch 1/32/128 ~20 → P5 on-policy eviction (Random Attention code, per-head random) vs R-KV vs SnapKV, relative 10/25/50% + absolute 1K/2K/4K, gap at matched random-accuracy, ≥6 model points ~80 → P6 re-sample controls + bootstrap CIs ~10.

## C4 — VERDICT: ABANDON (mock 4/10; workshop/Findings)
- Headline pair infeasible: Nemotron-H-8B-Base is 8K-context; RULER-32K / 8K–128K would test extrapolation, not eviction. Feasibility gate missed this.
- Falcon-H1 is a parallel hybrid — breaks the "few softmax layers" premise.
- Memory case weak: Nemotron-H-8B softmax KV ≈ 16 KiB/token ≈ 2 GiB at 128K vs ≈16 GiB for Llama-3.1-8B; evicting 90% saves ~1.8 GiB/seq. Triage condition (3) fails.
- Pairs confounded; only parent-derived pairs defensible (HALO/HypeNet Qwen3 conversions, Jet-Nemotron) — context lengths unverified.
- Two-sided hypothesis in a space with 3 concurrent method papers; recency mass not causal.
- Cheapest test (~2 GPU-h): RULER 4K/8K within trained context, sink+window / SnapKV / per-head random at 10%/25% softmax-KV + recency profiling. Kill if hybrid window drop ≤ dense drop; promote if hybrid loses ≥20 pts more under window while SnapKV gap <5 pts.

## C2 — VERDICT: PROCEED, second (mock 5/10; ~6 with causal late-masking at 3 sizes)
- Regression to the mean from both-correct filter is the most likely false positive (long sampled thinking traces less self-consistent); require keep=100% re-sample control and ≥k/n definition.
- MVE lacks the causal mechanism test; counter-evidence (TVC, VisualSwap, 2606.28864) lives exactly there.
- Thinking-budget sweep confounded by base-accuracy level; report absolute loss, matched base accuracy where possible.
- Qwen3-VL DeepStack multi-level injection interacts with in-LLM FastV-style pruning (reviewer from memory, unverified).
- DSTP may already cover claim 1 — check tables first (0 GPU).
- Cheapest test (~2 GPU-h): Qwen3-VL-4B-Thinking, 300 MathVista × 3 samples, 4K cap: (a) mask all visual KV after 50% of own CoT length; (b) random 25% prefill pruning on Thinking and Instruct-4B with keep=100% re-sample control; both-correct ≥2/3. Kill if (a) <3 pts or (b) net ratio <1.3; promote if (a) ≥8 pts and (b) net ratio ≥1.5.

Results-to-claims matrix:
| Outcome | Claim | Paper? |
|---|---|---|
| Ratio ≥1.5 net at 3 sizes, late-masking causal, loss grows with budget | Thinking MLLMs need decode-time visual access; prefill-pruning evaluations invalid for them | Main borderline (~6) |
| Ratio high, late-masking cheap | More sensitive, not via late evidence | Findings |
| Ratio ≈1 | Prefill pruning transfers to thinking models | Workshop |
| Thinking loses less | Test-time compute substitutes for visual tokens | Findings/main if robust |
| Vanishes after re-sample control | Selection artifact | Not a paper |

## Overall
Order: C1+C5 → C2 (DSTP table check, then late-masking test) → drop C4 unless a parent-matched ≥32K hybrid appears and the in-context test shows a cliff. C9 (~1.5 GPU-h) is a better side pilot than C4.

Errata on the literature synthesis (reviewer):
- §1.B gap 1 ("hybrid text eviction unstudied, only DeltaS") is stale vs the novelty section (GateSight, HeadWiseKV, 2605.18053).
- "All 13 passed feasibility" inaccurate for C4.
- C1 "nobody measures SD acceptance after length-penalty RL" needs qualifying (DREAM-R, multimodal).
- n-gram/suffix τ is free of training bias but strongly history-length dependent.
- Synthesis item 1 merges two random-baseline crises with different proposed causes; a shared regime theory is a hypothesis.
- Synthesis item 4 too strong: Kinetics and OckBench report cost-grounded efficiency; Kinetics' finding cuts against C1's headline.
- 2605.18053 over-weighted (single-author; PAGE partly contradicts).
- Holds up: L1/LC-R1/AdaptThink parent pairings (ThinkPrune with matching checkpoint); offline expected-acceptance method; Qwen3-VL Instruct/Thinking pairs at 2/4/8B; Random Attention description; overcrowded-area calls.

VERDICT C1+C5: PROCEED
VERDICT C4: ABANDON
VERDICT C2: PROCEED
