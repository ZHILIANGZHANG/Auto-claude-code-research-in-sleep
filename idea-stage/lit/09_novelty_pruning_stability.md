# 09 — Novelty Check D: stability audit of query-guided visual token pruning (MLLM Idea 4)

> Date: 2026-10-18. Method: `skills/novelty-check/SKILL.md`, with its NOVELTY VERDICT LIMITS applied verbatim. Phase C, the cross-model check by a Codex/GPT reviewer, was **not run**: no Codex MCP exists in this session, so **Claude is the only judge**. ABANDON requires a named published paper that already contains the result. Crowded ground with a delta is PROCEED. A thin delta is PROCEED WITH CAUTION, stated together with what would make it carry.
> Search: arXiv HTML search (all-fields and abstract-only, newest first; about 45 queries), the arXiv API helper (4 queries), WebSearch (5 queries, extended mode), arXiv abs pages, and Hugging Face dataset API checks. The Semantic Scholar API returned 429 on every call. The GitHub API is not enabled for this session, so READMEs were read through raw.githubusercontent.com.
> Full texts were read with pdftotext for **2511.02650** (UniPruneBench), **2608.00077** (Spatial Provenance Audit), 2502.11501, 2607.02484 (EADP), 2609.35002 (CIRA), 2607.27700 (CaRe) and 2412.13180 (FEATHER). Targeted sections were read for 2609.34861 and 2506.11073. PDFs are in `/tmp/claude-0/-home-user-Auto-claude-code-research-in-sleep/e7e19eec-aa61-5636-81ec-3cbec12524d6/scratchpad/novelty_D/`.
> All 49 arXiv ids below were checked with `tools/verify_papers.py` (verdict PASS, hallucination rate 0.0). Titles for the ids cited from memory (benchmarks, selectors) were also resolved through the arXiv API.

## Summary

| Claim | Verdict | Closest paper(s) | One-line delta |
|---|---|---|---|
| C1. Paraphrase instability of kept sets → answer flips caused by pruning | **PROCEED** | 2610.00204, 2609.20139, 2607.02484, 2609.35002 | No paper measures kept-set agreement or pruning-induced flips under meaning-preserving rewrites of the query. |
| C2. Multilingual pruning tax and English-pivot selection | **PROCEED** (pilot-gated) | 2608.26175 (text-prompt compressors), 2506.11073 (CLAIM), 2502.11501 / 2511.02650 (tables only) | No paper isolates query language as a factor in *visual* token pruning. Published tables already hint at the effect but nobody has analysed it. |
| C3. Blind-floor-normalized retention | **PROCEED WITH CAUTION** | 2403.20330 (MMStar multi-modal gain), 2412.13180 (FEATHER text-only), 2601.02316 (DatBench), 2606.30220 | Blind floors are known. Nobody has normalized pruning retention by them or shown that method rankings change. |
| C4. Kept-set disagreement as a label-free fallback trigger | **PROCEED WITH CAUTION** | 2604.12035 (confidence cascade + CRC), 2609.10346 (VIP-Router), 2608.01979 (ET-Prune) | A pre-answer, selection-side signal. It must beat or complement confidence at matched compute. |

**Overall: PROCEED, 6/10.** No paper contains the core result, and none meets the ABANDON bar. The territory is crowded: pruning-evaluation-pitfall papers are appearing at about 4–5 per month in 2026H2. Even so, the query-surface axis (wording and language) is open. Treat this as an early gate; the pilot below decides the headline.

---

## C1. Paraphrase instability of query-guided kept sets

**Closest work**
- **2610.00204** (Query Independent Variable Rate Visual Token Coding, Sep 2026). States that attention-ranked selection "makes the ranking a function of the question being asked" and shows attention-ranked pruning loses once the compressed image must answer *a different question*. This covers different questions, not paraphrases, and it measures neither kept-set stability nor flips. Its fix is query-independent rate allocation, not selection.
- **2609.20139** (Cross-Modal Attention Acts as a Frequency Filter, Sep 2026). Question wording changes question-conditioned cross-modal attention: verbose paraphrases broaden its frequency support and reduce drift under image corruption (Qwen3-VL, LLaVA-OV; GQA, CLEVR). This is mechanism evidence that paraphrases move text→visual attention. It does not cover pruning.
- **2607.02484** (EADP, ECCV 2026). "Textual noise" from function words and punctuation disperses dense text→visual relevance, and the paper filters it with entropy. That is a plausible mechanism for paraphrase sensitivity, since paraphrases mostly change function words. EADP does not test paraphrases.
- **2609.35002** (CIRA, Sep 2026). Retained-set allocation causally changes compressed correctness. Its defense, Translation-Consensus Selection (TCS), aggregates priority scores across *image* views to stabilise selection. That is the image-side analogue of our paraphrase-ensembled relevance fix. It perturbs images adversarially, not queries benignly.
- **2502.11501** (ACL Findings 2025): language guidance helps only on language-correlated tasks. **2511.02650** (UniPruneBench): random pruning is strong and OCR is most fragile; no query perturbation. **2608.00077**: no query perturbation, although its "Focus" query variant (content words only) shows that the query representation is a design knob.
- General MLLM paraphrase fragility is documented in 2602.21428 (medical VLMs) and 2603.28301 (VLA), without pruning.

**Verdict: PROCEED.** **Delta (one sentence):** for a fixed image, budget and model, we measure how much the kept set of query-guided selectors (FastV, SparseVLM, CDPruner, a QCPruner/text-similarity selector) changes across meaning-preserving rewrites of the question (Jaccard), and how many answer flips this adds beyond those the unpruned model already shows. Query-agnostic selectors serve as the zero-instability control.
**Design musts:** (i) Report *pruning-induced* flips, meaning flips under pruning minus flips under full tokens. Qwen2.5-VL already flips under some paraphrases. (ii) Validate the paraphrases: generate with an LLM, filter by back-translation or NLI, and have a human check 100. (iii) Use an IoU-tolerant Jaccard (dilate by one patch) so that neighbouring-patch swaps are not counted as instability.

## C2. Multilingual pruning tax

**Closest work**
- **2608.26175** (Lost in Compression, Jul 2026). This is the exact template, but for **extractive text-prompt compressors** (LLMLingua-2 and others) on text LLMs. With parallel data in 10 languages, English-supervised learned compressors lose far more outside English: Chinese retains essentially nothing at a 0.33 keep-rate, deterministic methods show no gap, and translate-then-compress matches or beats native compression in 3 of 5 languages. It covers no visual tokens and no MLLMs. → Our C2 is the visual-token, query-conditioned analogue, and the translate-pivot fix has precedent. **Cite it prominently.** Reviewers will see the paper as "2608.26175 for VLMs" unless the mechanism (text→visual attention) and the selection-only pivot are clearly distinct.
- **2506.11073** (CLAIM, ACL 2025). In LLaVA-1.5, attention to query-relevant image regions is lower for non-English queries in middle layers (about 10–25), with almost no difference before layer 10. It fixes this with attention intervention for multilingual hallucination, not pruning. **Design implication:** FastV-style selection at layer 2 sits where CLAIM sees little language difference. Mid-layer selectors (DeFT 2609.34861, PDrop) may show a larger tax, so the selection layer is a variable to sweep.
- 2610.21383 (Oct 2026): English calibration skews **block** (weight) pruning for Indian languages. This is the analogous phenomenon for depth pruning, not tokens. 2608.06506 measures cross-lingual comprehension gaps in text LLMs.
- **Evidence already in published tables, which nobody has analysed (computed here from their numbers; retention = pruned/full):**

| Source, model, budget | Selector | MMB-EN ret. | MMB-CN ret. | EN−CN gap |
|---|---|---|---|---|
| 2502.11501 T1, LLaVA-1.5-7B, 144 tok | FastV (query-guided) | 91.7% | 72.5% | **19.2** |
| same | SparseVLM (query-guided) | 92.0% | 87.8% | 4.2 |
| same | Random | 96.1% | 93.1% | 3.0 |
| same | Pooling | 96.6% | 95.0% | 1.6 |
| 2502.11501 T8, Qwen2-VL-7B, −88.9% | FastV | 87.1% | 80.3% | **6.8** |
| 2607.02484 T3, Qwen2.5-VL-7B, 128 tok | CDPruner (query-guided) | 87.4% | 86.6% | 0.8 |
| same | DivPrune (agnostic) | 92.3% | 93.4% | −1.1 |
| 2607.02484 T4, Qwen3-VL-8B, 128 tok | CDPruner | 90.3% | 91.6% | −1.3 |
| 2511.02650 T1, LLaVA-1.5-7B, 192 tok | FastV / DivPrune (agnostic) | 95.2% / 95.0% | 53.1% / 28.2% | 42 / 67 (anomalous) |

  Reading: on LLaVA-1.5 and Qwen2-VL, the attention-based selector pays a Chinese tax 3–6× larger than random or pooling, which supports C2. On bilingual Qwen2.5-VL and Qwen3-VL with CDPruner, the **Chinese tax is ~0**. UniPruneBench's LLaVA MMB-CN collapses even for query-agnostic DivPrune, which almost certainly comes from answer extraction or formatting, not selection. **Consequences:** (a) Chinese on Qwen is the *wrong* headline test, because Qwen is natively bilingual. The tax should be sought in languages where the full model is weaker (xGQA bn/id/ko; CVQA low-resource subsets; MMMB ar/tr/ru). (b) Option-likelihood scoring is mandatory to remove format confounds. (c) The tax must be defined *differentially*, as query-guided minus query-agnostic at matched budget, so that "pruning hurts weaker languages more for any selector" is not mistaken for a selector effect.

**Verdict: PROCEED** (pilot-gated). **Delta:** using paired same-image questions across languages (xGQA, MMBench-EN/CN, CVQA), we show whether text→visual selection imposes a language-dependent retention loss beyond query-agnostic selection, explain it by selection-layer cross-lingual attention divergence, and test a selection-only English pivot (oracle and MT) and a query-agnostic floor quota as training-free fixes. **Risk:** the effect may be null for strong multilingual 2025–26 models. The pilot measures this directly.

## C3. Blind-floor normalization

**Closest work**
- **2403.20330** (MMStar). Defines **multi-modal gain** (score with image minus score without image) and multi-modal leakage, and shows that many benchmarks can be solved without images. Our normalization (pruned − blind)/(full − blind) is exactly "fraction of multi-modal gain retained", so the *formula* is MMStar's MG applied as a denominator.
- **2412.13180** (FEATHER, ICCV 2025). Already plots a **text-only (0 visual token)** model next to pruned models (Fig. 4). It argues that benchmarks do not test fine-grained vision, and that the vision-enabled versus disabled gap is "insufficient to determine a benchmark's ability". It does not normalize retention or re-rank methods.
- **2601.02316** (DatBench): blind-solvable items make up as much as 70% of some evaluations, and DatBench filters them. **2606.30220**: "Blind Gap" and "Visual Gain" for traffic VideoQA. **2608.00077**: answers survive from language priors after evidence removal, at the item level for OCR. **2610.17530** (TwinPrune): a separate evaluation pitfall, where prefill-only probes cannot validate decode-phase pruning.
- No pruning paper found reports blind-normalized retention or a change in method ranking under it. UniPruneBench, 2502.11501, EADP, QCPruner, DeFT and VPRune all report pruned/full.

**Verdict: PROCEED WITH CAUTION.** The delta is thin because the blind floor and the multi-modal-gain formula are both published, so the remaining novelty is applying them to pruning retention. **What would make it carry:** (i) normalized retention reorders the 8 selectors (Kendall τ < 0.8 between raw and normalized rankings on at least two benchmarks), or turns a headline claim such as "95% retained" into something materially lower, for example below 80% on POPE, MME, SQA-IMG or MMBench. (ii) Item-level decomposition, separating the drop on blind-solvable items from the drop on vision-necessary items, to show where pruning loss actually lives. (iii) It is needed to make C1 and C2 visible, because high blind floors dilute paraphrase and language effects in raw accuracy. Use C3 as **evaluation infrastructure, not the headline.**

## C4. Kept-set disagreement as a risk signal

**Closest work**
- **2604.12035** (Confidence under Visual Token Pruning). Pruning errors concentrate where the selector removed evidence, and confidence does not register the removal. Its confidence-triggered cascade with conformal risk control falls back to full tokens, needs about a third of the prefill tokens with coverage-based selection, and *more than full* with FastV. → The cascade framework and CRC are taken. Our trigger signal is different. Notably, 2604.12035 reports that confidence is *weakest* for exactly the attention-ranked selectors where paraphrase disagreement should be largest.
- **2609.10346** (VIP-Router): a *trained* router (0.017% params) that picks a pruning strategy per sample, with a full-token option. **2608.01979** (ET-Prune): evidence uncertainty and density set a sample-specific token floor (training-free). **2609.35002**: cross-view support separates stable from view-specific tokens, but for adversarial defense.

**Verdict: PROCEED WITH CAUTION.** The delta is thin because the fallback cascade and per-sample budgeting both exist, so only the trigger signal is new. **What would make it carry:** at matched *total* compute (counting the K−1 extra selection passes, which only need prefill up to the selection layer), paraphrase-disagreement triggering beats confidence triggering on risk–coverage (AUROC for "pruned wrong, full right" ≥ 0.05 higher), or a combination of the two adds ≥ 0.03 AUROC. Report under the 2604.12035 CRC protocol for a direct comparison. The signal is label-free and available before decoding.

---

## Recommended framing

**Headline:** "Query-conditioned visual token pruning makes *what the model sees* depend on *how, and in which language, you ask*." C1 and C2 are one phenomenon, query-surface sensitivity of the selector, with the multilingual tax as the flagship, fairness-relevant instance (the visual analogue of 2608.26175). C3 is the evaluation protocol that makes the effect measurable. C4 plus the paraphrase-ensemble or English-pivot selection are the practical payoff.
**Pilot-gated fallback:** if the differential language tax is null on Qwen2.5-VL and InternVL3 for weaker languages, lead with C1 and C4 (stability and a free risk signal), and keep C2 as a negative or bounded result covering LLaVA-era models only.

## Pilot (phase 0 ≈ 4 GPU-h; full ≈ 1.5 A100-days per model)

- **Models:** Qwen2.5-VL-7B-Instruct (main), InternVL3-8B (second), LLaVA-1.5-7B (positive control, which reproduces the published FastV EN/CN gap).
- **Selectors:** 25% and 11% budgets; keep original M-RoPE ids on Qwen.
  - *Query-guided:* FastV-style last-text-token attention at layer 2, plus a mid-layer variant at the decoder midpoint (DeFT-like); SparseVLM v1.5; CDPruner-style conditional DPP; a text-embedding-similarity top-k selector (QCPruner/2608.00077 "Target"-like).
  - *Query-agnostic:* Random, pooling, DivPrune, VisionZip.
  - Add **0-token blind** and **full** runs.
- **Data:**
  - C1: GQA testdev-balanced 1,000 items, TextVQA-val 500 and MMBench-dev-EN 500, each with 4 validated paraphrases.
  - C2: xGQA (same GQA items; en + de/pt/ru/id/bn/ko/zh), MMBench-dev EN/CN (paired 4,329; use 1,000), and CVQA local/English pairs (about 2k, low-resource subsets).
  - C3: blind runs on POPE, MME, SQA-IMG, MMBench, GQA, TextVQA and MMStar.
- **Metrics:**
  - Kept-set Jaccard and dilated Jaccard.
  - Pruning-induced flip rate.
  - Differential language tax, Δ_L = [ret_EN − ret_L]_query-guided − [ret_EN − ret_L]_agnostic.
  - Blind-normalized retention and Kendall τ between raw and normalized rankings.
  - Disagreement-AUROC versus confidence-AUROC, and risk–coverage at matched compute.
  - Paired image-cluster bootstrap 95% CIs throughout. Multiple-choice items are scored by option likelihood.
- **Fixes:** paraphrase-ensembled relevance (mean of K=4 relevance maps); English-pivot selection using the oracle English from xGQA/CVQA and machine translation from the MLLM itself, used for selection only while the answer is still produced from the native-language query; and a 30% query-agnostic floor quota.
- **Phase 0:**
  - Qwen2.5-VL-7B only, 300 GQA items × 4 paraphrases plus xGQA zh/bn/id.
  - Four selectors (FastV, text-sim, Random, DivPrune) at 11%.
  - Blind floor on GQA, MMBench and POPE.
  - About 300×(5+4)×(4+2) ≈ 16k forwards ≈ 2–4 A100-hours.

**Kill criteria**
- C1 dies if query-guided mean paraphrase Jaccard is ≥ 0.80 at 11%, **or** the pruning-induced flip rate is ≤ 1 pt with a CI including 0.
- C2 dies if Δ_L < 2 pts with a CI overlapping 0 for every non-English xGQA/CVQA language on Qwen2.5-VL and InternVL3. The pivot fix is weak if it recovers < 25% of Δ_L.
- C3 is reduced to a footnote if Kendall τ is ≥ 0.9 everywhere and headline retention shrinks by < 5 pts.
- C4 dies if disagreement-AUROC is ≤ confidence-AUROC + 0.03 and the combination adds < 0.02.
- **Whole idea:** kill if C1 and C2 both die. Two or more surviving claims justify the full run.

## Feasibility notes

- **Code:**
  - VisionZip has an official Qwen2.5-VL branch (README, 2025-05-26).
  - DART is officially integrated into Qwen2.5-VL (2025-08-30).
  - DivPrune points to a Qwen2.5-VL port in the MMTok repository.
  - SparseVLM v1.5 claims FlashAttention-2 compatibility, but its README covers only LLaVA and VideoLLaVA, so a Qwen port is needed.
  - CDPruner's official repo covers LLaVA-1.5 and LLaVA-NeXT only. EADP (2607.02484) evaluated CDPruner on Qwen2.5-VL and Qwen3-VL, so ports exist, but their availability is unverified.
  - Official FastV code supports HF LLaVA only. UniPruneBench (2511.02650) runs FastV, DART, DivPrune and GPrune on Qwen2.5-VL and InternVL3 under VLMEvalKit; its code was not inspected.
  - QCPruner (2609.19990) reports Qwen2.5-VL results but no code link was found.
  - Simplest route: one HF-transformers harness implementing all eight selectors (each is under 100 lines).
- **Eager attention:** FastV, SparseVLM v1 and PDrop need attention maps. 2502.11501 notes that FastV, SparseVLM and MustDrop break FlashAttention. Avoid global eager attention by computing q·k only at the selection layer (Qwen2.5-VL uses GQA, so expand the KV heads). DivPrune, VisionZip, DART, CDPruner and the similarity selectors are FlashAttention-compatible. For InternVL3, selection runs over concatenated dynamic tiles plus the thumbnail; fix `max_num` tiles for parity.
- **Benchmarks (Hugging Face API checked):**
  - `lmms-lab/MMBench`: en/cn dev 4,329 each, test 6,666 each, plus cc (CCBench) test 2,040. CCBench has no English pair, so it cannot be used for a paired tax.
  - `floschne/xgqa`: en, de, pt, ru, id, bn, ko, zh. This is the best paired set; GQA images must be fetched separately.
  - `afaji/cvqa`: fields include `Question` and `Translated Question` plus translated options, so it is paired for 31 languages.
  - `floschne/maxm`: 7 languages, not parallel, so absolute numbers only.
  - `ByteDance/MTVQA`: question language follows the scene-text language and changes the image, so it is a poor fit for isolating query language.
  - M3Exam (2306.05179): few image items and not parallel, so skip it.
  - MMMB (Parrot 2406.02539; 6 parallel languages, 12k questions) was not found on the Hugging Face hub. It is presumably available via the Parrot GitHub repository [hosting UNVERIFIED].
- **GPU:** 7–8B models in bf16 on one A100-80GB, about 0.2–0.3 s per multiple-choice or short-answer item at batch 1. The full pilot is about 1.5 A100-days per model; phase 0 is about 4 GPU-hours.

## Closest prior work (all verified)

| Paper | Year | Venue | Overlap | Key difference |
|---|---|---|---|---|
| 2511.02650 UniPruneBench | 2025 | preprint | Unified pruning eval; Qwen2.5-VL/InternVL3; MMB-cn on LLaVA | No paraphrase, language or blind analysis; CN anomaly not discussed |
| 2608.00077 Spatial Provenance Audit | 2026 | preprint (code public) | Audit beyond accuracy; answers survive via priors | OCR provenance only; fixed prompt template; English |
| 2604.12035 Confidence under pruning | 2026 | preprint v4 | Fallback cascade + CRC | Confidence trigger, not selection disagreement |
| 2610.00204 Query-independent token coding | 2026 | preprint | Selection depends on the question | Different questions, not paraphrases; no stability metric |
| 2609.20139 Cross-modal attention as frequency filter | 2026 | preprint | Wording changes cross-modal attention | No pruning |
| 2607.02484 EADP | 2026 | ECCV'26 | Textual noise corrupts relevance; MMB-EN/CN tables | No paraphrase or language analysis |
| 2609.35002 CIRA / TCS | 2026 | preprint | Selection stabilization across views | Image-side adversarial perturbations |
| 2608.26175 Lost in Compression | 2026 | preprint | Cross-lingual audit of compressors; translate-then-compress | Text-prompt compressors, not visual tokens |
| 2506.11073 CLAIM | 2025 | ACL'25 | Cross-lingual divergence in cross-modal attention | Hallucination, not pruning |
| 2502.11501 Are we solving the right problem? | 2025 | ACL Findings | Language guidance is task-dependent; MMB/MMB-CN tables | No language-specific analysis |
| 2403.20330 MMStar | 2024 | NeurIPS'24 | Multi-modal gain (blind floor) | Not applied to pruning |
| 2412.13180 FEATHER | 2025 | ICCV'25 | Text-only baseline next to pruning | No normalization or re-ranking |
| 2601.02316 DatBench | 2026 | preprint | Blind-solvable items up to 70% | Benchmark curation, not pruning |
| 2609.10346 VIP-Router | 2026 | preprint | Per-sample strategy choice with full-token fallback | Trained router |
| 2608.01979 ET-Prune | 2026 | preprint | Uncertainty sets sample token floor | Not query-surface stability |
| 2610.21383 Calibration language & block pruning | 2026 | preprint | Language-dependent pruning decisions | Weight/depth pruning, text LLMs |

Other verified ids used above: 2608.03649, 2609.19990, 2607.27700, 2609.37581, 2609.34861, 2610.16615, 2512.07580, 2606.30220, 2607.22716, 2610.09703, 2606.12412, 2609.26484, 2608.06506, 2607.27952, 2608.10489. Benchmarks and selectors: 2109.06082 xGQA, 2209.05401 MaXM, 2405.11985 MTVQA, 2306.05179 M3Exam, 2406.05967 CVQA, 2307.06281 MMBench, 2406.02539 Parrot/MMMB, 2403.06764 FastV, 2410.04417 SparseVLM, 2506.10967 CDPruner, 2412.04467 VisionZip, 2503.02175 DivPrune, 2502.11494 DART, 2410.17247 PDrop, 2406.16860 Cambrian-1, 2405.05803 VTW, 2502.13923 Qwen2.5-VL, 2504.10479 InternVL3. Also verified, but known only from their listing titles: 2602.21428, 2603.28301 (51 ids in total).

## Limitations of this check
- Semantic Scholar was unavailable (429), so conference-only papers without arXiv versions may be missed.
- Whether the CDPruner, SparseVLM and QCPruner Qwen ports exist was not verified, because GitHub API access is disabled.
- The table under C2 is computed from other papers' published numbers, with different eval stacks. It shows direction only.
