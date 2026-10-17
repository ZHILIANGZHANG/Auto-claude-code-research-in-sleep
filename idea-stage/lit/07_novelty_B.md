# 07 — Phase 3 deep novelty check, batch B (ideas A–E)

> Date: 2026-10-17. Method: `skills/novelty-check/SKILL.md`, judged under its NOVELTY VERDICT LIMITS.
> **Reviewer: none.** No Codex/GPT reviewer was available. Phase C (cross-model check) was skipped, so I am the only judge. Treat every verdict as one model's call.
> **Sources searched:**
> - arXiv HTML search, sorted newest first, with `all` and `abstract` fields: about 60 short queries.
> - WebSearch: 12 queries.
> - arXiv abs pages for about 45 abstracts.
> - GitHub and vLLM-forum pages on how serving engines handle thinking tokens.
> - Semantic Scholar was tried but returned nothing (probably rate-limited).
>
> **Full text read** (pdftotext, PDFs in scratchpad `novelty_B/`): KVEraser 2606.17034, RoPERelay 2610.21126, Models-Take-Notes 2606.17107, RAZOR 2609.30465, Puzzle-MoE 2602.11937, 2609.25809, DeepEyes 2505.14362, Mini-o3 2509.07969, PRCR 2606.26631, Illusion-of-Tool-Use 2608.06270, VisionPulse 2605.31457, ViMoD 2610.12060, DSTP 2604.12358, Dropout Decoding 2412.06474, ParVTS 2511.18875.
>
> **ID check:** all 42 arXiv ids cited here passed `tools/verify_papers.py` (status: verified). Venue status is arXiv unless stated.
>
> **Coverage gaps:** ICLR 2027 OpenReview submissions could not be searched directly. ICLR 2027 and NeurIPS 2026 work that is not on arXiv may be missing.

## Summary

| Idea | Verdict | Closest prior work | Score |
|---|---|---|---|
| A ReThink-Reuse | **PROCEED** | KVEraser 2606.17034 (its delete-and-shift baseline); Models-Take-Notes 2606.17107 | 6/10 |
| B MoE-pruning token inflation + phase-aware k | **PROCEED WITH CAUTION** | RAZOR 2609.30465 (§3.4 / App. E.6 length + termination); Puzzle 2602.11937 | 4.5/10 |
| C GeoDedup | **PROCEED** (gated by a token-budget pilot) | PRCR 2606.26631; Illusion of Visual Tool-Use 2608.06270 | 6.5/10 |
| D Pruning → thinking MLLMs think longer | **ABANDON as standalone** | VisionPulse 2605.31457, Table 1 / 3 (+ ViMoD 2610.12060 Fig. 3, DSTP 2604.12358 Tab. 4) | 2/10 |
| E Complementary-view ensembling | **PROCEED WITH CAUTION** | Dropout Decoding 2412.06474; ParVTS 2511.18875 | 4.5/10 |

---

## A. ReThink-Reuse — PROCEED

**Core claims**
1. Some reasoning-model chat templates drop previous-turn thinking. This breaks prefix caching, so every answer after the first dropped thinking span is recomputed.
2. A training-free fix can work: drop the thinking KV, re-rotate RoPE on the answer-span KV, and optionally recompute the top-r% most-deviating tokens. This should approach exact re-prefill quality at a fraction of the TTFT and FLOPs.
3. Answer KV that "saw" the thinking may help the next turn (a "residue" effect), not hurt it.

**Does the problem still exist? Yes, on default settings.**
- **Qwen3 / Qwen3.5 / Qwen3.6 templates** still drop prior-turn `<think>` blocks by default. Qwen3.6 added an opt-in `preserve_thinking` kwarg, which defaults to false (community template pages and a pi-agent GitHub issue).
- **DeepSeek V4** encoding has a `drop_thinking` path with tool-use exceptions (vLLM forum thread, 2026-09-10).
- **Interleaved-thinking trend:** within a tool-call loop, thinking is kept, so the problem only appears across *user* turns. MiniMax-M2 and Kimi-style templates keep thinking.
- So the pain point is real on default Qwen/DeepSeek deployments. It is also partly optional now: users can keep thinking via `preserve_thinking` and pay in context length.
- **Angle this suggests:** ReThink-Reuse aims for the context size of stripping plus most of the cache hits of preserving.

**Do serving engines already do this? No, not with quality control.**
- **vLLM:** issue #39321 (Apr 2026) states the mechanism outright: "A1 must be recomputed despite its KV being in the cache." The fixes proposed and merged since then only *evict* thinking blocks: SGLang PR #23315, merged 2026-04-21; vLLM PR #39806 and feature request #41938.
- **llama.cpp:** `--cache-reuse N` (off by default) reuses chunks by KV shifting. That is a generic zero-recompute delete-and-shift, which is arm (c) of this idea, but it comes with no quality study for reasoning models.

**Closest papers**
- **KVEraser 2606.17034 (full text read).**
  - Learned (LoRA) span erasure on Qwen3-8B.
  - Its "delete-and-shift" baseline is exactly arm (c), and it fails badly *for erasure*: exact match is near 0 on NIAH. The reason is that the suffix KV stays "contaminated" by the erased span.
  - It never studies thinking-strip. Its tasks are needle/distractor removal and malicious skill files, where contamination is the failure. In thinking-strip, contamination may be the *benefit*.
- **Models Take Notes at Prefill 2606.17107.**
  - Shows downstream KV memoizes the conclusions of earlier spans.
  - Shows RoPE-repositioned spliced KV can be indistinguishable from recompute.
  - This *predicts* the residue effect, so cite it. It does not study multi-turn reasoning or thinking removal.
- **RoPERelay 2610.21126 (full text read):** post-eviction RoPE compaction on LongMemEval-type memory. It has no thinking-strip setting and no recompute.
- **Others:** CacheBlend 2405.16444, SAFE-KV 2610.15186, Request-Order 2610.05833, KV-Repurpose 2610.13060 (non-prefix RAG/agent reuse); CommitKV 2608.07855, IntentKV 2606.09916, AgentKV 2609.14872 (agent KV eviction, not thinking-strip); AbridgeKV 2610.20641 (trained, single-generation).

**Delta (verifiable):** No paper measures quality versus TTFT for reusing answer-span KV after template-mandated thinking removal in multi-turn reasoning models. None compares exact re-prefill, preserve-thinking, delete-and-shift, and selective recompute in that setting. None tests whether "contamination" by the dropped thinking helps the next turn.

**What makes it carry**
1. Show where delete-and-shift is fine and where it breaks (math follow-ups vs tool or multi-turn coding), plus a cheap recompute rule that closes the gap.
2. A clean residue result: answer-KV-that-saw-thinking beats exact re-prefill on next-turn accuracy, while using the same visible context.
3. Report TTFT as a function of the turn count.

**Risks**
- Arm (c) is so good that the method looks like engineering. If so, lead with the residue finding.
- Savings are bounded by how long the history is after the first thinking span.

**Cheapest pilot**
- **Setup:** Qwen3-8B thinking mode in HF transformers with a custom DynamicCache (re-rotate keys). 1×A100/H100, ~0.5 GPU-day.
- **Data:** 200 three-turn chains: AIME/MATH follow-up questions plus 100 BFCL-v4 multi-turn items.
- **Arms:** (a) re-prefill; (b) preserve_thinking; (c) delete+shift; (d) recompute r ∈ {5, 10, 20}% by KV deviation.
- **Metrics:** turn-2/3 accuracy, KL of the first 64 tokens vs (a), TTFT.
- **Kill if** either holds:
  - (d) at r=20% stays more than 2 points below (a) on both task types, *and* (c) shows no residue gain.
  - In realistic histories (WildChat-like lengths), the tokens recomputed after the divergence point are under 2k, so the TTFT saving is under 1.5×.

---

## B. Token inflation from MoE expert pruning + phase-aware expert budgets — PROCEED WITH CAUTION

**Core claims**
1. Dynamic expert pruning (top-k truncation, ACE-style skipping) on *thinking-mode* MoE reasoning models lengthens CoT and raises the length-cap hit rate.
2. As a result, per-token speedups of 1.2–1.7× overstate per-problem speedups.
3. Thinking, answer, and "wait/verify" spans differ in how sensitive they are to pruning.
4. A training-free phase/entropy-aware k schedule reduces inflation at the same average k.

**What I found (this changes the picture)**
- **RAZOR 2609.30465 (Sep 2026, full text §3.4 and App. E.6) already measures length and termination under expert pruning.**
  - Qwen3.6-35B-A3B, LiveCodeBench, *static* expert removal.
  - At 50% removal, REAP raises median length from 2.43k to 3.94k tokens.
  - Length-limited finishes rise from 16.0% (unpruned) to 24–26%.
  - At 75% removal, 42–79% of completions hit the cap.
  - Caveat: thinking was **disabled**. The method is static pool pruning, not per-token dynamic top-k.
- **Puzzle for MoE reasoning 2602.11937 (NVIDIA).**
  - Expert pruning plus NAS on gpt-oss, explicitly framed as "a 2× throughput gain is erased if traces grow 2×".
  - Proposes request-level efficiency metrics and reports 1.09–1.57× more tokens than the parent (Table 3).
  - Requires KD and RL training.
- **2609.25809 (full text):** dynamic top-k truncation on 12 models, but it reports only accuracy and per-token speed (it mentions native mean output length). No length change under pruning. Its finding that thinking models are more robust to pruning is relevant.
- **Others:**
  - ACE 2609.05228 and ZEDA 2605.18643 (trained): no length analysis.
  - LayerTrim 2610.19121: reports generation length, but for dense structured pruning.
  - 2610.20826: per-phase (prompt/reasoning/response) MoE routing statistics are *not* a reliable proxy for quantization sensitivity. This is a warning sign for phase-aware budgets.
  - LiteStage 2510.14211: stage-wise layer budgets on dense small models.

**Why the delta is thin, specifically:** "Expert pruning inflates output length and request-level cost" is no longer a new finding (RAZOR, Puzzle). What is left:
- (i) the *dynamic per-token* top-k regime with thinking *enabled*, where 2609.25809 says thinking models are robust;
- (ii) the phase/entropy-aware k schedule as a method.

**What makes it carry**
- The phase-aware schedule must beat uniform truncation on **request-level** cost: wall-clock per solved problem at matched accuracy, in the aggressive regime (k' ≤ 50% of k).
- It must win by at least 10% cost or 2 points accuracy. 2609.25809 already shows dynamic rules beat uniform by less than 1% at conservative budgets.
- The cleanest framing is the inflation-adjusted speedup curve for *dynamic* pruning in thinking mode, with RAZOR and Puzzle as the static and trained counterparts.

**Cheapest pilot**
- **Setup:** GPT-OSS-20B (MXFP4) and Qwen3-30B-A3B-Thinking-2507 (FP8) in vLLM (top-k is a one-integer change per 2609.25809). ~1 GPU-day.
- **Data:** AIME24/25 + MATH500-200 at k' ∈ {100, 67, 50, 33}% of k.
- **Metrics:** mean and p90 length, cap-hit rate, accuracy, seconds per solved problem.
- **Kill if** either holds:
  - At the largest k' with accuracy retention ≥ 97%, length inflation is ≤ 5%. The diagnosis is then weak and the paper becomes a RAZOR footnote.
  - The phase-aware schedule fails the margin above.

---

## C. GeoDedup — PROCEED (magnitude-gated)

**Core claims**
1. In zoom/crop "thinking-with-images" agents, global-view visual tokens inside a later crop's bbox (and coarser crops nested inside finer ones) are redundant.
2. Because the geometry is known from the bbox, they can be merged or downsampled at zero scoring cost.
3. Trajectory-level cost (tokens, latency, turns) drops while accuracy holds.
4. Under a fixed context window, the freed context buys more turns.

**Search result:** no paper found doing cross-round, bbox-geometry-based dedup of earlier visual context in zoom agents. Nearest neighbours:
- **PRCR 2606.26631 (full text):** reuses or rebinds *prefill* visual KV instead of replaying it. Its stated limitation (§7) is that it *fails* when crops are re-encoded at a new scale, which is exactly the zoom-agent case.
- **AwaRes 2603.16932, AdaptVision 2512.03794:** trained, on-demand crop acquisition.
- **GUIPruner 2602.23235:** temporal decay over GUI screenshots.
- **TTSP 2604.11025:** accuracy-oriented.
- **Thinking-Once 2607.27830:** single-pass alternative to zooming.
- **Others:** BEE 2607.11106 and TiCO 2610.14265 (trained efficient TwI); LOOK-M / KVCapsule (generic multimodal KV).

**New fact that changes the pilot — Illusion of Visual Tool-Use 2608.06270 (full text):**
- **DeepEyes is "structurally inactive."** Replacing *all* returned crops with random crops changes accuracy by ≤ 1.6 points (e.g. V* 83.3 → 83.8), and nearly all DeepEyes trajectories have a single tool call.
- **Mini-o3 and Qwen3-VL-8B depend heavily on their crops:** V* 87.8 → 23.6 and 91.1 → 30.4 under random-crop corruption.
- **Consequence:** do **not** pilot on DeepEyes. Dropping global tokens covered by a crop that the model ignores can only hurt. Use Mini-o3 or Qwen3-VL-8B with a crop tool.

**Typical tokens per trajectory** (from the papers, plus Qwen2.5-VL's ≈ 1 token per 28×28 px):
- **DeepEyes 2505.14362:**
  - About 1 tool call per example; Mini-o3 reports an average of ~1 turn on HR-Bench-4K.
  - About 241 text response tokens (Table 6).
  - The global image dominates: V* (~2246×1582) gives ≈ 4.5k tokens; 4K/8K images hit the ≈ 12.8 MP cap, giving ≈ 10–16k tokens.
  - A crop is typically a small fraction of that, so dedup could save only ~1–10% of the global tokens. Too small, and the crops don't matter anyway.
- **Mini-o3 2509.07969:**
  - Max 2 MP per image, i.e. ≤ ~2.5k tokens per image or observation.
  - 32K context; average 6.3–6.5 turns on VisualProbe (Table 3); trajectories reach tens of turns at test time.
  - Crops can be taken *from earlier observations*, so nested-crop dedup is natural.
  - Estimated ≲ 10–19k visual tokens per trajectory, depending on how big the crops are once re-encoded (not reported; must be measured).
  - Table 3 shows the 32K context binds: lowering max pixels to allow more turns raised accuracy on VisualProbe-Hard (12 MP: 36.1 vs 2 MP: 48.0).

**Delta (verifiable):** A training-free, geometry-driven (bbox-overlap) compression of earlier visual context *across* zoom rounds. PRCR cannot handle re-encoded crops, and the other methods either train the acquisition policy or score tokens without geometry.

**What makes it carry**
1. At least 20–30% of trajectory visual tokens are eligible for dedup in Mini-o3-class agents.
2. Accuracy holds at ≤ 1 point loss.
3. Ideally a turn-scaling win: at the 32K cap, dedup leads to more turns and higher VisualProbe-Hard accuracy.
4. Report trajectory-level total cost, including any change in turn count.

**Cheapest pilot**
- **Step 0 (CPU-cheap, ~0.2 GPU-day):** run Mini-o3-7B on V* (191) and VisualProbe-hard. Log per-trajectory visual tokens split into global vs observations, and the bbox overlap area.
- **Step 1:** pool overlapped regions 2×2/4×4 with an anchor token, keeping the original position ids per PRCR. Compare against no compression and against FastV/DivPrune applied to the full context.
- **Kill if** either holds:
  - Overlap-eligible tokens are under 20% of the trajectory's visual tokens.
  - GeoDedup costs more than 1.5 points accuracy at the same turn cap.

---

## D. "Does visual token pruning make thinking MLLMs think longer?" — ABANDON as a standalone paper

**Core claims (as proposed)**
1. ViMoD and VisionPulse contradict each other.
2. Attention selectors remove evidence and lengthen CoT; diversity selectors remove distractors and shorten it.
3. Response-level total cost is the right metric.
4. A length or hesitation trigger should restore pruned tokens.

**Finding (VisionPulse full text):** there is no contradiction, and VisionPulse already contains the headline result.
- **VisionPulse 2605.31457, Table 1** (Qwen3-VL-Thinking-4B, 7 benchmarks):
  - Static pruning lengthens generation. At ≤ 10% retention: FastV +63%, VisionZip +54%, LOOK-M +59%. At ≤ 5%: FastV +30%, VisionZip +18%, LOOK-M +108%.
  - Accuracy falls 24–51% in the same settings.
  - Its text: static methods "remove critical tokens prematurely, forcing the model to compensate via longer reasoning".
- **VisionPulse, Table 3:** the same holds on Qwen3-VL-Thinking-8B (+19.3%) and InternVL3.5-Thinking-4B (+11.0%).
- **Where the "contradiction" came from:** VisionPulse's claim that "redundant context lengthens reasoning" compares *dense* context against *its own step-wise dynamic* pruning (−12% length). It is not about static pruning.
- **ViMoD 2610.12060, Fig. 3:** FastV on MathVerse; longer responses make total decoder FLOPs exceed dense at several retention ratios. This is the same direction and already a response-level FLOPs analysis.
- **DSTP 2604.12358, Table 4:** FastV produces more tokens than DSTP, and DSTP wins latency per example.
- **Hypothesis 2 is already contradicted:** VisionZip, a diversity/context-retaining selector, *also* inflates length in VisionPulse's data.
- **Hypothesis 4** (decode-time restore) is what DSTP, VisionPulse and ViMoD already do.

**Verdict:** ABANDON standalone. The specific prior paper is **VisionPulse 2605.31457** (Tables 1 and 3), with ViMoD 2610.12060 Fig. 3 for response-level FLOPs.

**Salvage**
- Use response-level accounting (prefill + decode FLOPs, cap-hit rate) as the *evaluation protocol* inside C or E.
- The only untested residue is moderate retention (25–50%) across many selectors (DivPrune, CDPruner, random) on thinking MLLMs. That is an appendix, not a paper.

---

## E. Complementary-view ensembling — PROCEED WITH CAUTION

**Core claims**
1. Split the visual tokens into K complementary, individually diverse subsets.
2. Run each subset as its own branch and average the logits.
3. At matched compute (K×r total retention), this beats a single pruned pass at the same total retention on accuracy.
4. It also has fewer hallucinations than full tokens.

**Closest papers**
- **Dropout Decoding 2412.06474 (full text).**
  - Ensembles K = 3 decoding contexts, each with an *independently sampled*, uncertainty-guided mask over visual tokens, and combines them by token-level majority vote.
  - Targets hallucination (CHAIR, THRONE, MMBench) on LLaVA-1.5 and InstructBLIP.
  - Adds overhead (one extra preliminary pass, then K parallel passes). It is not compute-matched and the masks are not complementary.
  - Random masks performed badly (repetition).
- **ParVTS 2511.18875 (full text).**
  - Splits visual tokens into subject and non-subject groups and runs them as parallel batched branches.
  - Fuses the question-token states mid-network, then *discards* one branch. Hidden-state fusion aimed at speed, not an output-level ensemble.
- **Others (all single-pass):** PruneHal 2510.19183, OOD-VTP 2607.22716, 2603.10360 (pruned tokens as negatives), TOPS 2606.27161.

**Why the delta is thin, specifically**
1. "Ensemble over visual-token-masked contexts" already exists (Dropout Decoding), and so do "parallel visual-token-subset branches" (ParVTS).
2. Per-step logit averaging needs K parallel *decodes*, so compute is matched only for prefill-dominated, short-answer tasks (POPE, MCQ). For long generations, decode cost is K×.

**Delta (verifiable):** A coverage-balanced *disjoint* partition with output-level aggregation, evaluated at iso-total-FLOPs (prefill + decode) against the best single-pass selector at the same total retention. Prior work is either not compute-matched with independent masks (Dropout Decoding) or uses mid-layer fusion that discards a branch (ParVTS).

**What makes it carry**
1. A win at iso-total-FLOPs on *general* benchmarks (MMBench, GQA, TextVQA), not only POPE/CHAIR.
2. Complementary partitions beat K independent masks (the Dropout-Decoding-style control), backed by an error-decorrelation analysis between branches.
3. Results on modern models (Qwen2.5/3-VL).

**Cheapest pilot**
- **Setup:** Qwen2.5-VL-7B, ~0.5–1 GPU-day.
- **Data:** 1k items each from POPE, MMBench, TextVQA, ChartQA.
- **Arms:** K ∈ {2, 3}; per-branch retention ∈ {11, 22}%; compared against single-pass DivPrune and VisionZip at K·r, full tokens, and Dropout Decoding.
- **Kill if** either holds:
  - K = 2×25% fails to beat single-pass 50% by ≥ 1 point averaged over the non-hallucination benchmarks.
  - It fails to beat Dropout Decoding on POPE at lower cost.

---

## Prior-work table (all ids verified)

| Paper | arXiv | Relevant to | Overlap | Key difference |
|---|---|---|---|---|
| KVEraser | 2606.17034 | A | span erase + delete-and-shift baseline | learned; erasure goal (contamination = failure) |
| Models Take Notes at Prefill | 2606.17107 | A | downstream KV memoizes; RoPE splice | no multi-turn thinking strip |
| RoPERelay | 2610.21126 | A | RoPE re-rotation after eviction | memory QA, no recompute |
| AgentKV / CommitKV / IntentKV | 2609.14872 / 2608.07855 / 2606.09916 | A | multi-turn agent KV | eviction, not thinking-strip reuse |
| SAFE-KV / Request Order Matters | 2610.15186 / 2610.05833 | A | non-prefix reuse + repair | RAG / rolling docs |
| RAZOR | 2609.30465 | B | length / termination under expert pruning | static pruning, thinking off |
| Puzzle for MoE reasoning | 2602.11937 | B | request-level efficiency, token growth | NAS + KD + RL |
| Dynamic expert pruning study | 2609.25809 | B | uniform top-k truncation | no length-change analysis |
| ACE / ZEDA | 2609.05228 / 2605.18643 | B | expert skipping | no length analysis |
| Phase-routing ≠ quant proxy | 2610.20826 | B | per-phase routing statistics | negative result for quantization |
| PRCR | 2606.26631 | C | visual KV reuse in interleaved reasoning | fails on re-encoded crops |
| Illusion of Visual Tool-Use | 2608.06270 | C | causal audit of crops | DeepEyes crops inert; Mini-o3 crops vital |
| AwaRes / AdaptVision / GUIPruner | 2603.16932 / 2512.03794 / 2602.23235 | C | crop acquisition / GUI history | trained / temporal decay |
| VisionPulse | 2605.31457 | D | static pruning lengthens thinking CoT | **contains D's headline** |
| ViMoD | 2610.12060 | D | response-level FLOPs > dense | trained aggregator |
| DSTP | 2604.12358 | D | fewer tokens than FastV, latency per example | — |
| Dropout Decoding | 2412.06474 | E | ensemble of visual-masked contexts | independent masks, not compute-matched |
| ParVTS | 2511.18875 | E | parallel visual-subset branches | mid-layer fusion, branch discarded |

## Recommended order
1. **C** (on Mini-o3, not DeepEyes) and **A**: both clear the check, and both pilots fit in under 1 GPU-day.
2. **E**: quick, unambiguous pilot; abandon on the kill criterion.
3. **B**: run only the dynamic-k length pilot; keep it only if inflation exceeds 5% where accuracy retention is ≥ 97%.
4. **D**: drop. Reuse its cost-accounting protocol in C and E.
