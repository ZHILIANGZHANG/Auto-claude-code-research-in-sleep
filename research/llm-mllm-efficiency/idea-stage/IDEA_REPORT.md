# Idea Discovery Report — LLM / MLLM Efficiency

**Direction**: LLM / MLLM 效率（KV cache 压缩、MLLM 视觉 token 剪枝/合并、量化、稀疏/剪枝、投机解码、高效长上下文、高效推理 / CoT 压缩、MoE 效率）— 文献调研 + 选题
**Date**: 2026-10-17
**Pipeline**: research-lit → idea-creator → novelty-check → research-review（research-refine-pipeline 留作下一步）
**Run ID**: `llm-mllm-efficiency-20261017`（`.aris/runs/`）

> ⚠️ **本次运行的环境限制（如实记录）**
> - 学术 API（arXiv / Semantic Scholar / OpenAlex）被环境网络策略拦截（CONNECT 403）。文献检索全部通过 WebSearch 完成（4 个分片，约 160 次查询）。`verify_papers.py` 的三层校验无法运行：表中论文都在检索结果里出现过并附了链接，但**状态为 ⚠️ UNVERIFIED（verification unavailable）**，数字均为摘要/摘要片段中的声称值。2026 年的高编号预印本（2606–2610）尤其需要在引用前到 arXiv 上复核。
> - 没有 Codex MCP（`codex` CLI 未安装、未登录）、没有 Gemini CLI。ARIS 要求的**跨模型家族评审无法执行**。triage、新颖性复核、外部评审改由**上下文独立的 Claude 子代理**完成，属于同家族评审，不能作为 ARIS 的 acceptance receipt。
> - 没有 GPU，Phase 5 pilot 全部 **SKIPPED: needs GPU**。

## Executive Summary
<!--EXEC-->

<a id="literature-landscape"></a>
## Literature Landscape

### 0. 跨方向综合（四个分片共同指向的结构性规律）

1. **"随机基线"危机遍布各子方向。** 视觉 token 剪枝里，random/pooling/下采样常常打平或胜过精心设计的打分（2502.11501, VTC-Bench 2510.07143, UniPruneBench 2511.02650, Worse-than-Random 2512.07580）。KV 驱逐里，对推理轨迹做随机驱逐约等于最强的 evictor（Random Attention 2609.03430）。共同原因是：主流 benchmark 分不开方法，而且冗余度高的场景里，"选哪些"远不如"留多少"重要。**方法论缺口**：一个能预测"打分何时比随机有用"的 regime 理论或诊断。
2. **decode-heavy 的长推理 regime 正在打破 prefill 时代的效率假设。** 量化后的推理模型想得更久（2606.00206），剪枝会让 TTS 崩掉（2510.22228），剪枝后的模型输出更长（2509.12464），视觉相关性在解码中途会漂移（DSTP 2604.12358），CoT SFT 会抹掉混合架构的长程召回（Attention Amnesia 2606.11052）。大多数方法是在输入重建 / prefill 上校准的。
3. **组合效应几乎没人测。** CoT 压缩 × 投机解码、量化 × token 剪枝（QAPruner 2604.02816 是唯一的警告）、MLA 转换 × 量化 × 驱逐、量化 × 剪枝 × 激活稀疏：各自的论文都声称"正交"，没人检验误差是加性、超加性还是相互抵消。
4. **代价核算单位不对。** FLOPs、token 数、压缩率被当作代理指标；但真实部署中，验证开销、batching、prefix caching、FlashAttention 兼容性、显存访问才决定 wall-clock（Kinetics 2506.05333, SD Performance-or-Illusion 2601.11580, FlashVLM 2512.20561, 2607.05399）。"cost per solved problem"（bits × tokens × latency）没人报。
5. **后训练会改变效率相关的性质。** RL / CoT SFT 会改变量化敏感度、置信度校准（early exit 依赖它）、长程注意力（稀疏/混合架构依赖它），以及可投机性（drafter 依赖它）。这些影响几乎都没被测量过。

**过度拥挤（避免）**：又一个 LLaVA-1.5 上的无训练视觉 token 打分准则；离线视频 token 合并；W4A4 rotation 在 LLaMA PPL 上抠小数点；长度惩罚 RL 用在 R1-Distill-1.5B 数学上；attention-score KV 驱逐在 LongBench 上刷分；dLLM 缓存小技巧；EAGLE 式 drafter 微调。
**欠探索（机会）**：解码期 / 推理期动态；组合效应；wall-clock 与服务栈层面的评测；后训练对效率性质的侵蚀；混合架构里少量 softmax 层的 KV 压缩；压缩 × TTS 曲线（pass@k）；"何时打分优于随机"的 regime 诊断。

### 1.A MLLM / Video-LLM / VLA token efficiency

| Paper (arXiv) | Year/Venue | Method | Key claimed result | URL |
|---|---|---|---|---|
| FastV (2403.06764) | ECCV'24 | Prune visual tokens after layer K by early-layer attention | ~45% FLOPs cut on LLaVA-1.5-13B | https://arxiv.org/abs/2403.06764 |
| LLaVA-PruMerge (2403.15388) | 2024 | [CLS]-attention keep + merge dropped tokens | Claim changed across versions (14.4x → 4x) | https://arxiv.org/abs/2403.15388 |
| Matryoshka MM (M3) (2405.17430) | ICLR'25 | Nested coarse-to-fine visual tokens | ~9 tokens ≈ 576 on COCO-style; large oracle gap | https://arxiv.org/abs/2405.17430 |
| SparseVLM (2410.04417) | 2024 | Text-guided layer-adaptive pruning + recycling | ~4.5x compression at ~97% | https://arxiv.org/abs/2410.04417 |
| LongVU (2410.17434) | ICML'25 | DINOv2 dedup + query frame selection | Hour-long video gains | https://arxiv.org/abs/2410.17434 |
| VisionZip (2412.04467) | CVPR'25 | Dominant + contextual tokens at encoder output | 8x faster prefill | https://arxiv.org/abs/2412.04467 |
| Feather the Throttle (2412.13180) | ICCV'25 | Critique: FastV drops top-of-image tokens due to RoPE | >5x better localization than FastV | https://arxiv.org/abs/2412.13180 |
| FastVLM (2412.13303) | CVPR'25 | Hybrid encoder emits fewer high-res tokens | 85x faster TTFT vs LLaVA-OV at 1152px | https://arxiv.org/abs/2412.13303 |
| LLaVA-Mini (2501.03895) | ICLR'25 | Modality pre-fusion → 1 vision token | Beats LLaVA-1.5 with 1 vs 576 tokens | https://arxiv.org/abs/2501.03895 |
| DART (2502.11494) | EMNLP'25 | Keep tokens least similar to pivots (duplication) | 88.9% pruned, 1.99x speedup | https://arxiv.org/abs/2502.11494 |
| Are We Solving the Right Problem? (2502.11501) | ACL-F'25 | Analysis: random/pooling vs importance pruning | Many methods fail to beat random | https://arxiv.org/abs/2502.11501 |
| AirCache (2503.23956) | 2025 | Text-to-vision relevance KV compression | Full-cache quality at 10% visual KV | https://arxiv.org/abs/2503.23956 |
| HoliTom (2505.21334) | 2025 | Global temporal segmentation + merging | 99.1% perf at 6.9% FLOPs | https://arxiv.org/abs/2505.21334 |
| EfficientVLA (2506.10100) | NeurIPS'25 | Layer skip + diverse tokens + action-head cache | 1.93x on CogACT | https://arxiv.org/abs/2506.10100 |
| InfiniPot-V (2506.15745) | NeurIPS'25 | Fixed-budget KV for streaming video | Flat memory, 3.6x at 500 s | https://arxiv.org/abs/2506.15745 |
| GAP (2506.21873) | 2025 | Grounding collapse = misaligned position IDs | RefCOCO 56.1 → 15.3 under pruning; GAP recovers | https://arxiv.org/abs/2506.21873 |
| VisionThink (2507.13348) | NeurIPS'25 | RL: start at 1/4 res, request full res | OCR needs full res; VQA does not | https://arxiv.org/abs/2507.13348 |
| VTC-Bench (2510.07143) | ACL'26 | Split benchmarks into simple/hard via downsampling | Plain downsampling beats many methods | https://arxiv.org/abs/2510.07143 |
| UniPruneBench (2511.02650) | 2025 | 10 methods × 6 abilities × 3 backbones + latency | Random strong; OCR most sensitive | https://arxiv.org/abs/2511.02650 |
| VLA-Pruner (2511.16449) | 2025 | Semantic + temporally smoothed action relevance | Up to 1.99x | https://arxiv.org/abs/2511.16449 |
| When Token Pruning is Worse than Random (2512.07580) | CVPR'26 | "Information horizon" layer | Past ~layer 20 methods ≈ random; horizon deeper for stronger models | https://arxiv.org/abs/2512.07580 |
| D2Pruner (2512.19443) | 2025 | Debiased importance + diversity | Bias direction depends on PE type | https://arxiv.org/abs/2512.19443 |
| FlashVLM (2512.20561) | 2025 | One-shot text-guided selection at encoder boundary, FA-compatible | 313 → 74 ms vs FastV 112 ms | https://arxiv.org/abs/2512.20561 |
| IVC-Prune (2602.03060) | ICLR'26 | Keep implicit spatial-coordinate tokens | ≥99% at 50%; much better grounding | https://arxiv.org/abs/2602.03060 |
| FlashVID (2602.08024) | 2026 | Tree-based spatiotemporal merging | 99.1% at 10% tokens | https://arxiv.org/abs/2602.08024 |
| ReDiPrune (2603.24680) | 2026 | Pre-projector relevance + diversity | +2.0 EgoSchema at 15% tokens | https://arxiv.org/abs/2603.24680 |
| DSTP (2604.12358) | ECCV'26 | Relevant visual tokens shift during decoding | Static pruning fails on complex reasoning | https://arxiv.org/abs/2604.12358 |
| LLaVA-UHD v4 (2605.08985) | 2026 | Early in-ViT compression for high-res | −55.8% encoding FLOPs | https://arxiv.org/abs/2605.08985 |
| KVCapsule (2605.16439) | 2026 | Compress visual KV, rebuild before attention | Tracks shifting importance | https://arxiv.org/abs/2605.16439 |
| TOPS (2606.27161) | 2026 | Info-theoretic preservation set | ~100% at −77.8% tokens (LLaVA-NeXT); ~92% at 90% on Qwen2.5-VL | https://arxiv.org/abs/2606.27161 |
| Bifocal (2610.18399) | Oct 2026 | Split budget image-driven vs question-driven | At 1/3 tokens general OK, grounding lost | https://arxiv.org/abs/2610.18399 |
| VLA-ACL (2610.08133) | Oct 2026 | Action-supervised pruning policy | 87.5% pruned, 1.5x | https://arxiv.org/abs/2610.08133 |
| CrisPrune | ACL-F'26 | Contextual relevance + intrinsic saliency | 13x FLOPs, 97% kept | https://aclanthology.org/2026.findings-acl.663/ |

Also seen (2026 "*Prune" wave): SinkPruner 2609.01004, EntropyPrune 2602.17196, QCPruner 2609.19990, ET-Prune 2608.01979, IDPruner 2602.13315, RoRA 2608.07088, Layer-Aware PE 2609.23715; video: OTT-Vid 2605.11803, EarlyTom 2605.30010, Visual Token Coding 2608.28008; streaming: StreamMem 2508.15717, LiveVLM 2505.15269, StreamKV 2511.07278; unified models: FlashU 2603.15271, CE-Router 2608.29291; video CoT × compression 2512.09616.

**Landscape.** Training-free image token pruning dominates; the criterion moved from attention-importance (FastV, SparseVLM) to duplication/coverage (DART, DivPrune, CDPruner, TOPS) after early-2025 analyses showed importance pruning often loses to random. Attention scores carry position bias and are FlashAttention-incompatible, so selection is moving to the encoder boundary. A strong critique literature (VTC-Bench, UniPruneBench, Worse-than-Random, Bifocal, GAP, DSTP) indicates that the "~90% pruned at ~97–100% kept" headline numbers mostly live on benchmarks that cannot separate methods; OCR, grounding and multi-step reasoning break first. Video splits into offline spatiotemporal merging (crowded, all "99% at 7–10%") and streaming fixed-size KV memory. High-res efficiency shifts to cheaper encoders / in-ViT compression; VLA and unified models are new, less mature settings (1.5–2x speedups).

**Gaps.**
1. [contradiction] Direction/cause of position bias disagrees across papers; no controlled PE study (absolute vs 1D-RoPE vs M-RoPE) at a fixed backbone.
2. [untested-assumption] Pruning once at prefill assumes fixed relevance; DSTP shows it shifts. Untested on long-CoT "thinking" MLLMs (Qwen3-VL-Thinking etc.) where evidence may be needed late.
3. [diagnostic] Benchmark validity: no stress suite where full-token ≫ low-res/random.
4. [scaling-regime] Prunability may shrink with stronger / native-resolution backbones; no retention-vs-backbone-strength curve.
5. [diagnostic] FLOPs vs real latency under continuous batching / prefix caching unmeasured.
6. [method-transfer] Position-ID bookkeeping (GAP, IVC-Prune) may explain many grounding failures; untested across earlier methods.
7. [untested-assumption] Post-encoder pruning ignores encoder cost dominating TTFT at high res/video; no joint budget across ViT/projector/LLM.
8. [contradiction] Trained extreme compressors (LLaVA-Mini 1 token, M3) vs fine-grained needs; never evaluated on hard subsets.
9. [method-transfer] Streaming bounded KV memory not brought to long-horizon closed-loop VLA.
10. [diagnostic] Side effects (hallucination, safety, calibration) of pruning barely measured.
11. [scaling-regime] Unified und+gen and audio-visual models barely studied.
12. [contradiction] When does query-agnostic coverage beat query relevance?

**Saturation.** Overcrowded: another scoring criterion for training-free image pruning on LLaVA-1.5 (20+ such 2026 papers); offline video merging. Under-explored: decode-time / reasoning-time visual-token dynamics; serving-level evaluation; PE/position-ID diagnostics; hard-subset benchmarks; VLA memory; prunability scaling.

### 1.B KV cache & long-context efficiency

| Paper (arXiv) | Year/Venue | Method | Key claimed result | URL |
|---|---|---|---|---|
| H2O (2306.14048) | NeurIPS'23 | Recent + heavy-hitter tokens | Up to 29x throughput | https://arxiv.org/abs/2306.14048 |
| StreamingLLM (2309.17453) | ICLR'24 | Attention sinks + window | Stable to 4M tokens; no long-range recall | https://arxiv.org/abs/2309.17453 |
| SnapKV (2404.14469) | NeurIPS'24 | Observation-window voting per head | 3.6x speed, 8.2x memory at 16K | https://arxiv.org/abs/2404.14469 |
| PyramidKV (2406.02069) | COLM'25 | Depth-decreasing layer budget | Full-KV LongBench at 12% cache | https://arxiv.org/abs/2406.02069 |
| Ada-KV (2407.11550) | NeurIPS'25 | Adaptive per-head budget | Improves SnapKV/PyramidKV | https://arxiv.org/abs/2407.11550 |
| DefensiveKV (2510.13334) | ICLR'26 | Worst-case score aggregation | 2.3–4.3x lower loss at 20% cache | https://arxiv.org/abs/2510.13334 |
| Quest (2406.10774) | ICML'24 | Query-aware page selection via min/max key bounds | 7.03x attention speedup | https://arxiv.org/abs/2406.10774 |
| DuoAttention (2410.10819) | ICLR'25 | Full KV only on retrieval heads | 2.55x (MHA) / 1.67x (GQA) memory | https://arxiv.org/abs/2410.10819 |
| KVzip (2505.23416) | NeurIPS'25 | Query-agnostic eviction by context reconstruction | Query-aware methods degrade under reuse | https://arxiv.org/abs/2505.23416 |
| RestoreKV (2608.01247) | 2026 | Restore cache on top of query-agnostic eviction | KVzip 38.2 → 73.2 on RULER-4K at 5% | https://arxiv.org/abs/2608.01247 |
| KIVI (2402.02750) | ICML'24 | 2-bit asymmetric KV quant | 2.6x less peak memory | https://arxiv.org/abs/2402.02750 |
| TurboQuant (2504.19874) | ICLR'26 | Random rotation + optimal scalar quant + 1-bit QJL residual | Quality-neutral at 3.5 bits/channel | https://arxiv.org/abs/2504.19874 |
| Palu (2407.21118) | ICLR'25 | Low-rank K/V projection latents | 1.89x at 50% compression | https://arxiv.org/abs/2407.21118 |
| MHA2MLA (2502.14837) | ACL'25 | MHA→MLA via partial-RoPE + joint SVD | −92% KV, ~0.5–1% LongBench drop; VLM variant 2601.11464 | https://arxiv.org/abs/2502.14837 |
| TransMLA (2502.07864) | 2025 | GQA exactly representable as MLA | ~93% KV compression | https://arxiv.org/abs/2502.07864 |
| MInference (2407.02490) | NeurIPS'24 | Per-head offline sparse patterns for prefill | Successors FlexPrefill, XAttention (13.5x at 256K) | https://arxiv.org/abs/2407.02490 |
| NSA (2502.11089) | ACL'25 | Natively trainable hardware-aligned sparse attention | ≥ full attention; 64K speedups | https://arxiv.org/abs/2502.11089 |
| MoBA (2502.13189) | NeurIPS'25 | MoE-style block routing | Deployed at Kimi | https://arxiv.org/abs/2502.13189 |
| DeepSeek-V3.2 DSA (2512.02556) | 2025 | Lightning indexer + top-k token selection | ~parity with dense V3.1 | https://arxiv.org/abs/2512.02556 |
| SeerAttention-R (2506.08889) | ICLR'26 | Learned gate block-sparse decode for reasoning | Near-full AIME at 4K budget | https://arxiv.org/abs/2506.08889 |
| The Sparse Frontier (2504.17768) | ACL-F'26 | isoFLOPS study of training-free sparse attention | Decode tolerates more sparsity; optimal sparsity grows with length | https://arxiv.org/abs/2504.17768 |
| MiniMax-01 (2501.08313) | 2025 | 7:1 lightning/softmax hybrid MoE | 1M training context | https://arxiv.org/abs/2501.08313 |
| Kimi Linear (2510.26692) | 2025 | KDA (gated DeltaNet) 3:1 with MLA | −75% KV, 6x decode at 1M | https://arxiv.org/abs/2510.26692 |
| Mamba in the Llama (2408.15237) | NeurIPS'24 | Distill Transformer → Mamba hybrid | ≈ teacher with 1/4 attention | https://arxiv.org/abs/2408.15237 |
| HALO / HypeNet (2601.22156) | ICML'26 | Transformer → hybrid distillation with layer selection | Qwen3 conversion with 2.3B tokens | https://arxiv.org/abs/2601.22156 |
| Attention Amnesia (2606.11052) | 2026 | CoT SFT breaks hybrid long-range recall | NIAH 67.2 → 9.4 after CoT SFT | https://arxiv.org/abs/2606.11052 |
| ShadowKV (2410.21465) | 2024 | Low-rank K on GPU, V offloaded | 3.04x throughput | https://arxiv.org/abs/2410.21465 |
| SCBench (2412.10319) | ICLR'25 | KV-lifecycle multi-turn benchmark | Sub-O(n) methods fail multi-turn | https://arxiv.org/abs/2412.10319 |
| Query-visibility audit (2607.11942) | 2026 | Matched-budget audit of query visibility | Query-aware advantage mostly from seeing the question | https://arxiv.org/abs/2607.11942 |
| KVDiagnosis (2608.09412) | 2026 | Per-failure diagnostic benchmark | 10 diagnostics, AUROC 0.68–0.87 | https://arxiv.org/abs/2608.09412 |
| KV benchmark quality+systems (2607.05399) | 2026 | Quality + throughput + TTFT | Compression ratio poor predictor of e2e | https://arxiv.org/abs/2607.05399 |
| R-KV (2505.24133) | NeurIPS'25 | Redundancy-aware eviction for long CoT | Near-full at 10% KV | https://arxiv.org/abs/2505.24133 |
| DMS (2506.05345) | NeurIPS'25 | Delayed eviction to buy more tokens per memory | +9.1 AIME24 at matched memory | https://arxiv.org/abs/2506.05345 |
| Random Attention (2609.03430) | 2026 | Protect prompt, random-evict decode KV | ≈ strongest evictor; +32–43% vLLM throughput | https://arxiv.org/abs/2609.03430 |
| Sparse-RL (2601.10079) | 2026 | KV-compressed RL rollouts with mismatch correction | −35–53% stored KV | https://arxiv.org/abs/2601.10079 |

Also seen: reasoning eviction ForesightKV 2602.03203, KV-Rescue 2608.15797, AvoKV-E 2610.03007, RLKV 2510.08525, LongFlow 2603.11504, CapKV 2604.25975; RL rollouts Shadow Mask Distillation 2605.06850, S3R 2610.13202; hybrids "When Perplexity Lies" 2603.26556, Opal 2609.31947, DeltaS 2609.27470 (hybrid video eviction via linear state); multimodal KV RetentiveKV 2605.04075; serving Mooncake 2407.00079, CacheBlend 2405.16444.

**Landscape.** Eviction grew from H2O/StreamingLLM → SnapKV → layer/head budgets → refined scoring rules (CriticalKV, DefensiveKV). The query-aware vs query-agnostic split matters: KVzip, RestoreKV and the query-visibility audit show query-aware results are inflated whenever caches are reused. The decode-heavy reasoning regime is the 2025–26 hot spot (R-KV, DMS, SeerAttention-R, 10+ preprints), but Random Attention claims random eviction of trace KV matches the best scorers. Representation compression is converging on learned/architectural solutions (TurboQuant; MHA/GQA→MLA conversion; NSA/MoBA/DSA in production). Hybrid linear/softmax architectures are now mainstream (MiniMax-01, Qwen3-Next, Kimi Linear) with a growing Transformer→hybrid distillation literature, and Attention Amnesia warns CoT SFT can wipe hybrid recall. Evaluation critique (SCBench, KVDiagnosis, 2607.05399) agrees aggregate scores and compression ratios mislead.

**Gaps.**
1. [untested-assumption] KV eviction on the few softmax layers of hybrid text models is unstudied (only DeltaS for video); linear-attention state as free eviction signal untested.
2. [contradiction] Random Attention vs scoring refinement — likely a prompt-heavy vs decode-heavy regime boundary.
3. [diagnostic] Reasoning-eviction methods never re-ranked under multi-turn agentic reuse with query hidden.
4. [method-transfer] DSA-style distilled lightning indexer as retrieval scorer for offloading in existing GQA models.
5. [untested-assumption] Does reasoning post-training erode long-range attention in NSA/DSA indexers, MLA-converted models, Sparse-RL policies?
6. [scaling-regime] Length-adaptive budget schedules for 100K+ token decode.
7. [contradiction] Do latent conversion + eviction + quantization stack?
8. [diagnostic] Do NIAH/RULER deltas predict real-task deltas under compression?
9. [method-transfer] R-KV / DMS reasoning-trace methods to long-video KV.
10. [scaling-regime] Eviction vs prefix sharing under continuous batching.
11. [untested-assumption] Stability of hybrid layer selection across lengths and post-training.
12. [diagnostic] Do sparse RL rollouts shift learned behavior (length, retrieval)?

**Saturation.** Overcrowded: attention-score eviction on LongBench/RULER; training-free prefill sparse patterns; 2026 reasoning-model eviction on AIME/MATH. Under-explored: hybrid-model KV compression; composition; agentic reuse evaluation; post-training effects on efficient architectures; length-adaptive budgets; learned indexers for offloading.

### 1.C Model compression: quantization, pruning/sparsity, MoE

| Paper (arXiv) | Year/Venue | Method | Key claimed result | URL |
|---|---|---|---|---|
| AWQ (2306.00978) | MLSys'24 | Activation-aware per-channel scaling protects ~1% salient weights | Weight-only 4-bit, no backprop | https://arxiv.org/abs/2306.00978 |
| SparseGPT (2301.00774) | ICML'23 | One-shot OBS-style pruning | 50–60% sparsity on 175B, 2:4 support | https://arxiv.org/abs/2301.00774 |
| QuaRot (2404.00456) | NeurIPS'24 | Hadamard rotations remove outliers → W4A4KV4 | LLaMA2-70B ≤0.47 PPL loss | https://arxiv.org/abs/2404.00456 |
| SpinQuant (2405.16406) | ICLR'25 | Learned (Cayley) rotations | W4A4KV4 within 2.9 pts of FP | https://arxiv.org/abs/2405.16406 |
| FlatQuant (2410.09426) | ICML'25 | Learned per-layer Kronecker affine transforms | W4A4 LLaMA-3-70B <1% drop; 2.3x prefill | https://arxiv.org/abs/2410.09426 |
| ReSpinQuant (2604.11080) | 2026 | Subspace residual rotations | L3-8B W4A4 PPL 7.24 vs 7.50 SpinQuant | https://arxiv.org/abs/2604.11080 |
| QTIP (2406.11235) | NeurIPS'24 | Trellis-coded VQ at 2–4 bits | Beats QuIP#/AQLM | https://arxiv.org/abs/2406.11235 |
| Scaling Laws for Precision (2411.04330) | ICLR'25 | Loss vs train/inference precision | PTQ degradation grows with data; over-training hurts PTQ | https://arxiv.org/abs/2411.04330 |
| Low-Bit Favors Undertrained LLMs (2411.17691) | ACL'25 | QiD over 1500+ checkpoints | 100T-token models projected to quantize poorly | https://arxiv.org/abs/2411.17691 |
| Scaling FP8 Training (2409.12517) | ICLR'25 | FP8 to 2T tokens, Smooth-SwiGLU | Matches BF16, ~34% faster | https://arxiv.org/abs/2409.12517 |
| NVFP4 Pretraining (2509.25149) | 2025 | NVFP4 (block 16) + RHT pretraining | MXFP4 needs +36% tokens vs NVFP4 | https://arxiv.org/abs/2509.25149 |
| Quartet (2505.14669) | 2025 | Native MXFP4 training of all matmuls | ~2x linear throughput vs FP8 | https://arxiv.org/abs/2505.14669 |
| BitNet b1.58 2B4T (2504.12285) | 2025 | Native ternary 2B trained on 4T tokens | ≈ FP peers at same size | https://arxiv.org/abs/2504.12285 |
| ParetoQ (2502.02631) | 2025 | Unified QAT across 1–4 bits | Learning transition between 2 and 3 bits | https://arxiv.org/abs/2502.02631 |
| Quantization Hurts Reasoning? (2504.04823) | COLM'25 | PTQ study on R1-distilled / QwQ | W8A8, W4A16 ≈ lossless; lower bits risky | https://arxiv.org/abs/2504.04823 |
| Quantized Reasoning Models Think Longer (2606.00206) | 2026 | Diagnoses overthinking under quantization; logit penalty | 3-bit AWQ overthinking errors ×7.3 | https://arxiv.org/abs/2606.00206 |
| QaRL (2604.07853) | ACL-F'26 | Low-bit rollouts + aligned learner | +5.5 over quantized-rollout RL | https://arxiv.org/abs/2604.07853 |
| TRACE (2610.07767) | Oct 2026 | FP4 QAT aligned to FP4 rollouts for MoE RL | ~BF16 RL, 5.4x rollout speedup | https://arxiv.org/abs/2610.07767 |
| MQuant (2502.00425) | ACM MM'25 | Modality-specific static scales for MLLM PTQ | W4A8 <1% drop | https://arxiv.org/abs/2502.00425 |
| LUQ (2509.23729) | 2025 | Layer-wise sub-4-bit MLLM quantization | Visual tokens have higher activation entropy | https://arxiv.org/abs/2509.23729 |
| RAC (2509.12464) | ICLR'26 | Calibrate SparseGPT on self-generated CoT traces | 95% acc at 50% sparsity on reasoning | https://arxiv.org/abs/2509.12464 |
| When Fewer Layers Break More Chains (2510.22228) | COLM'26 | Layer pruning vs long reasoning / TTS | Removing 1–2 layers collapses TTS | https://arxiv.org/abs/2510.22228 |
| Doing More With Less (2604.25098) | 2026 | Pruning under test-time scaling | Unstructured beats structured, sometimes beats dense | https://arxiv.org/abs/2604.25098 |
| ShortGPT (2403.03853) | ACL-F'25 | Block-Influence layer removal | — | https://arxiv.org/abs/2403.03853 |
| Unreasonable Ineffectiveness of Deeper Layers (2403.17887) | ICLR'25 | Prune layer block + QLoRA heal | ~half layers removable on QA | https://arxiv.org/abs/2403.17887 |
| Minitron (2407.14679) | NeurIPS'24 | Prune + KD retrain | 40x fewer tokens than scratch | https://arxiv.org/abs/2407.14679 |
| TEAL (2408.14690) | ICLR'25 | Training-free magnitude activation sparsity | 1.53–1.8x decode at 40–50% | https://arxiv.org/abs/2408.14690 |
| Polar Sparsity (2505.14884) | 2025 | Batched contextual sparsity on heads | MLP sparsity vanishes under batching | https://arxiv.org/abs/2505.14884 |
| Universal Props of Activation Sparsity (2509.00454) | 2025 | Cross-model activation-sparsity study | Realistic speedups ~1.3–1.5x | https://arxiv.org/abs/2509.00454 |
| To 2:4 Sparsity and Beyond (2602.06183) | 2026 | 2:4 W + v:n:m A sparsity in pretraining | 1.4–1.7x training speedup | https://arxiv.org/abs/2602.06183 |
| REAP (2510.13999) | ICLR'26 | Expert pruning by gate×activation norm | ~lossless 50% expert pruning on 480B | https://arxiv.org/abs/2510.13999 |
| REAM (2604.04356) | 2026 | Group-and-merge experts | Merging often beats REAP; calibration-dependent | https://arxiv.org/abs/2604.04356 |
| MoBiLE (2510.12357) | 2025 | Offloaded MoE, halve experts for unimportant tokens | 1.6–1.7x on consumer GPU | https://arxiv.org/abs/2510.12357 |
| Expert-Sample (2602.02443) | 2026 | Dynamic top-k for fine-grained MoE | Halving experts keeps greedy acc but hurts pass@n | https://www.alphaxiv.org/abs/2602.02443 |
| Expert-wise MoE Quant (2604.06515) | 2026 | Per-expert bit allocation | Sub-3-bit uniform hurts; per-expert barely explored | https://arxiv.org/abs/2604.06515 |
| Upcycling scaling laws (2502.03009) | ICML'25 | When dense→MoE upcycling pays | Budget conditions | https://arxiv.org/abs/2502.03009 |
| ACBench (2505.19433) | ICML'25 | Agentic benchmark for compressed LLMs | Real-world app accuracy −10–15% at 4-bit | https://arxiv.org/abs/2505.19433 |
| Accuracy is Not All You Need (2407.09141) | NeurIPS'24 | Flip-rate / KL metrics | ≤2% acc change but up to 13.6% flips | https://arxiv.org/abs/2407.09141 |
| Quantization on Long-Context (2505.20276) | EMNLP'25 | Quant eval at >64K | 4-bit drops up to 59% | https://arxiv.org/abs/2505.20276 |
| Fairness & Safety in Quantized LLMs (2601.12033) | ACL-F'26 | Safety under quantization | Safety degrades, worst in KO/AR | https://arxiv.org/abs/2601.12033 |

**Landscape.** Weight-only 4-bit PTQ is solved for standard benchmarks; W4A4 is an incremental transform race (QuaRot → SpinQuant → FlatQuant → ReSpinQuant) judged mainly on WikiText PPL. The live frontier is *reasoning under compression*: 8-bit/W4A16 are safe, W4A4/3-bit are not (2504.04823); quantized and pruned reasoning models emit *more* tokens (2606.00206, 2509.12464); layer pruning that is "free" on QA collapses test-time scaling (2510.22228), while unstructured pruning may preserve it (2604.25098). FP8 training is mature; FP4 pretraining has two competing recipes (NVFP4 vs MXFP4/Quartet). Precision scaling laws say heavily over-trained models quantize worse post hoc. A 2025–26 offshoot quantizes the RL rollout loop (QaRL, TRACE). MoE work is mostly offloading systems plus an unresolved prune-vs-merge dispute (REAP vs REAM); dynamic top-k for fine-grained MoE is the 2026 growth area. Evaluation critique (flips, agentic, long-context, multilingual safety) is the most consistent theme, yet new methods still headline PPL.

**Gaps.**
1. [contradiction] Layer pruning near-free on QA but catastrophic for long-CoT TTS; no diagnostic predicts which layers carry multi-step function.
2. [method-transfer] RAC-style self-generated-CoT calibration tested only for SparseGPT; untested for W4A4 rotations, GPTQ, MoE expert-pruning criteria.
3. [untested-assumption] Compression is evaluated with greedy single samples; effect on pass@k / self-consistency / best-of-N scaling curves is unmeasured (Expert-Sample is the only warning).
4. [diagnostic] No end-to-end cost-per-solved-problem accounting (bits × tokens × latency) although compressed reasoners emit more tokens.
5. [scaling-regime] Precision scaling laws fit on pretraining loss ≤1.7B; whether RL / long-CoT post-training changes quantization sensitivity is unmeasured.
6. [method-transfer] Rotation transforms designed on text-only dense LLMs; modality- or expert-specific rotations open; token pruning amplifies W4A4 error (QAPruner 2604.02816).
7. [contradiction] Prune vs merge for MoE experts (REAP vs REAM) not settled by a controlled study.
8. [untested-assumption] Compression methods assumed to compose orthogonally; super-/sub-additivity on reasoning untested.
9. [scaling-regime] Activation-sparsity speedups measured at batch 1, short outputs.
10. [untested-assumption] FP4-pretrained models' robustness on long-context / reasoning / RL stability unknown.
11. [diagnostic] No validated cheap proxy predicting long-context / multilingual / safety degradation.
12. [scaling-regime] Native ternary/2-bit models untested on long CoT, long context, RL.

**Saturation.** Overcrowded: W4A16 PTQ; W4A4 rotation papers on LLaMA PPL; MoE offloading systems; depth pruning on MC benchmarks; on-policy-distillation variants. Under-explored: compression × TTS curves; cost-per-solved-problem; reasoning-trace calibration for quantization/MoE; per-expert mixed precision at DeepSeek/Qwen3 scale; composition effects; native low-bit on reasoning.

### 1.D Decoding acceleration & efficient reasoning

| Paper (arXiv) | Year/Venue | Method | Key claimed result | URL |
|---|---|---|---|---|
| Medusa (2401.10774) | 2024 | Extra decoding heads + tree attention | 2.2–3.6x | https://arxiv.org/abs/2401.10774 |
| Lookahead Decoding (2402.02057) | ICML'24 | Jacobi iteration + n-gram cache | Up to 1.8x | https://arxiv.org/abs/2402.02057 |
| LayerSkip (2404.16710) | ACL'24 | Early-exit self-speculation | 1.34–2.16x | https://arxiv.org/abs/2404.16710 |
| EAGLE-2 (2406.16858) | EMNLP'24 | Dynamic draft tree | 3.05–4.26x | https://arxiv.org/abs/2406.16858 |
| EAGLE-3 (2503.01840) | NeurIPS'25 | Direct token prediction, multi-layer fusion, training-time test | 3.0–6.5x | https://arxiv.org/abs/2503.01840 |
| MagicDec (2408.11049) | ICLR'25 | Sparse-KV draft at large batch × long context | 1.84–2.51x at batch 32–256 | https://arxiv.org/abs/2408.11049 |
| TriForce (2404.11912) | 2024 | Hierarchical speculation with retrieval KV | 2.31x at 128K | https://arxiv.org/abs/2404.11912 |
| DeepSeek-V3 MTP (2412.19437) | 2024 | MTP modules as self-drafter | 85–90% 2nd-token acceptance, 1.8x TPS | https://arxiv.org/abs/2412.19437 |
| Lookahead Reasoning (2506.19830) | 2025 | Step-level speculation for reasoning | Token-level SD capped by acceptance decay | https://arxiv.org/abs/2506.19830 |
| SD: Performance or Illusion? (2601.11580) | MLSys'26 | Systematic vLLM study of SD variants | Verification dominates; real ≪ theoretical | https://arxiv.org/abs/2601.11580 |
| SD for RL rollouts (2604.26779) | 2026 | MTP / EAGLE-3 for RL rollouts | 1.8x rollout throughput | https://arxiv.org/abs/2604.26779 |
| DFlash (2602.06036) | ICML'26 | Block-diffusion drafter + AR verifier | >6x lossless; 2.5x over EAGLE-3 | https://arxiv.org/abs/2602.06036 |
| DREAM VLM-SD (2505.19201) | NeurIPS'25 | Vision-aware drafter | Up to 3.6x | https://arxiv.org/abs/2505.19201 |
| LLaDA (2502.09992) | 2025 | 8B masked diffusion LM | ≈ LLaMA3-8B | https://arxiv.org/abs/2502.09992 |
| Block Diffusion (2503.09573) | ICLR'25 | AR over blocks, diffusion within | KV cache + flexible length | https://arxiv.org/abs/2503.09573 |
| Fast-dLLM (2505.22618) | ICLR'26 | Approx. KV cache + confidence parallel decode | Up to 27.6x over dLLM baseline | https://arxiv.org/abs/2505.22618 |
| dLLM vs AR characterization (2510.04146) | 2025 | Systems profiling | Open dLLMs often slower than AR | https://arxiv.org/abs/2510.04146 |
| JustGRPO (2601.15165) | 2026 | Left-to-right GRPO for dLLMs | Arbitrary order narrows reasoning | https://arxiv.org/abs/2601.15165 |
| Overthinking o1-like (2412.21187) | ICML'25 | Efficiency metrics + self-training | ~44.8% fewer tokens | https://arxiv.org/abs/2412.21187 |
| Underthinking (2501.18585) | NeurIPS'25 | Penalize premature thought switching | +10.7% Pass@1 | https://arxiv.org/abs/2501.18585 |
| L1 / LCPO (2503.04697) | COLM'25 | RL with length target in prompt | Precise length control | https://arxiv.org/abs/2503.04697 |
| ThinkPrune (2504.01296) | TMLR'26 | RL with tightening token cap | Half length, ~2% AIME drop | https://arxiv.org/abs/2504.01296 |
| AdaptThink (2505.13417) | 2025 | RL think / no-think by difficulty | −53% length, +2.4% acc | https://arxiv.org/abs/2505.13417 |
| NoThinking (2504.09858) | 2025 | Skip thinking + parallel best-of-N | Beats Thinking at low budget, 9x lower latency | https://arxiv.org/abs/2504.09858 |
| Coconut (2412.06769) | COLM'25 | Continuous latent thought | Better on search-heavy logic | https://arxiv.org/abs/2412.06769 |
| Limits of latent CoT (2602.01148) | 2026 | Theory + experiments | ProsQA 97.0 vs GSM8K 34.1 | https://arxiv.org/abs/2602.01148 |
| Looped-transformer latent reasoning (2606.31779) | 2026 | Recurrent depth | Latent CoT trails explicit above ~1B | https://arxiv.org/abs/2606.31779 |
| DeepConf | ICLR'26 | Confidence filtering + early stop of parallel traces | −84.7% tokens on AIME25 | https://mlanthology.org/iclr/2026/fu2026iclr-deep/ |
| ST-BoN (2503.01422) | NeurIPS'25 | Early-truncated best-of-N | −70–80% compute | https://arxiv.org/abs/2503.01422 |
| Kinetics (2506.05333) | NeurIPS'25 | TTS scaling law counting memory access | Attention dominates cost | https://arxiv.org/abs/2506.05333 |
| Don't Overthink it (2505.17813) | 2025 | Prefer shorter sampled chains | Shortest chain up to +34.5% acc | https://arxiv.org/abs/2505.17813 |
| Thought-free inference (2511.03408) | 2025 | Train with thought, infer without | More no-think data: AIME 76.6 → 43.3 | https://arxiv.org/abs/2511.03408 |
| OckBench (2511.05722) | 2025 | Accuracy + token efficiency benchmark | 26x length variance at similar acc | https://arxiv.org/abs/2511.05722 |
| VIG (2608.21883) | 2026 | Visual-information-gain reward for MM CoT compression | Many CoT tokens never consult the image | https://arxiv.org/abs/2608.21883 |
| ChainV (2511.17106) | 2025 | Visual hints cut reflection loops | −34.1% latency, +3.4% acc | https://arxiv.org/abs/2511.17106 |

Also seen: s1 2501.19393, TALE 2412.18547, Chain of Draft 2502.18600, LC-R1 2506.14755, NEAT 2602.02010, SAIL-RL 2511.02280, V-Skip 2601.13879, MASSV 2505.10526, agentic token-vs-latency 2511.08042, multi-sample SD temperature 2503.05330, dLLM inference survey 2607.12829.

**Landscape.** Speculative decoding is mature for small-batch chat (Medusa → EAGLE-3, LayerSkip, MTP); the open questions are systems behaviour (verification dominates on vLLM; real ≪ theoretical) and reasoning workloads, where token-level acceptance decay pushes toward step-level speculation. Diffusion LLMs moved from "can they model language" to "can they be faster": method papers claim 10–27x over their own baselines while systems profiling finds them slower than optimized AR; diffusion-as-drafter (DFlash) may be the synthesis. Efficient reasoning is the most crowded area: length-penalty RL on R1-Distill-1.5B/7B math (L1, ThinkPrune, AdaptThink, Thinkless…), training-free early exit, confidence-based parallel-sample pruning, and latent reasoning (which trails explicit CoT above ~1B). The central disagreement: within a question, shorter chains are more often correct; across questions, forced brevity collapses on hard problems. Tokens are an imperfect proxy for wall-clock (Kinetics; agentic 10–14x tokens vs 4–6x latency). MLLM efficient reasoning trails text by 6–12 months.

**Gaps.**
1. [contradiction] Within-question "shorter is better" vs across-question "forced brevity collapses"; per-instance oracle-length regret decomposition missing.
2. [untested-assumption] CoT compression assumed to save wall-clock proportionally; compressed CoT may be less draftable (lower SD acceptance). Nobody measures EAGLE-3/MTP acceptance before vs after length-penalty RL.
3. [diagnostic] No efficient-reasoning benchmark reports accuracy + tokens + serving wall-clock + memory together.
4. [scaling-regime] SD at T≈1, large batch, >10K outputs, under policy drift during RL — poorly characterized.
5. [method-transfer] Step-level speculation and confidence early-exit use similar signals but were never unified.
6. [contradiction] dLLM speed claims conflict; matched-quality comparison vs vLLM AR + EAGLE-3 missing.
7. [untested-assumption] dLLM parallel-unmasking heuristics validated only on short outputs; effect on long CoT unknown.
8. [scaling-regime] Latent reasoning at 7B+ with RL and wall-clock gains not shown.
9. [method-transfer] MLLM SD evaluated on short captions, not long visual CoT.
10. [untested-assumption] Early-exit/parallel pruning assume calibrated confidence; length-penalty RL likely changes calibration.
11. [contradiction] Parallel-short (NoThinking) vs sequential-long; no size/hardware/budget rule.
12. [diagnostic] Results concentrated on R1-Distill math; frontier scale / code / agents / MLLMs unreported.

**Saturation.** Overcrowded: length-penalty RL on small math models; think-or-not routing; training-free early exit; dLLM caching tricks; EAGLE-style drafter tweaks. Under-explored: CoT compression × SD × batching; wall-clock-grounded benchmarks; SD for RL rollouts under drift; length control for dLLMs; calibration after RL compression.


---

<a id="ranked-ideas"></a>
## Ranked Ideas

**Ideas evaluated**: 18 个原始候选（3 个 lens 分片 × 6）→ 机械去重得 13 个 → 13 个全部通过客观可行性筛选 → 独立 triage 排序 → 前 3 名做深度新颖性检查 → 0 个跑了 pilot（无 GPU）。
完整候选及标注见 [`candidate_pool.md`](candidate_pool.md)。

> **Triage 的评审身份**：triage 由一个上下文全新的 Claude 子代理完成（魔鬼代言人，看不到生成过程，另做了 12 次检索）。它**不是** ARIS 要求的跨模型家族评审（Codex 不可用）。下面的排序应视为 provisional，需要等 GPT 类评审确认。

### Triage 排序（N = 新颖性，I = 信息价值，F = 可行性，各 1–10）

| # | ID | 题目 | N | I | F | 一句话理由 |
|---|---|---|---|---|---|---|
| 1 | C1(+C5) | 压缩后的 CoT 的"可投机性税"，以及随机驱逐免费午餐的消失 | 8 | 9 | 9 | 直接挑战领域的头条指标（token 数）；正反结果都有用；几天的推理就能出结果 |
| 2 | C4 | 混合架构（linear + softmax）中少量 softmax 层的 KV 驱逐 | 8 | 8 | 6 | 真正部署的架构上的空白 regime；出现断崖就能给出部署规则 |
| 3 | C2 | Thinking MLLM 对视觉 token 剪枝的惩罚更大 | 7 | 8 | 7 | 少见的匹配对照（Instruct/Thinking）；会重置推理 MLLM 剪枝的评测方式 |
| 4 | C3 | MoE 专家剪枝 vs 合并 = pass@1 vs pass@k 之争 | 7 | 8 | 6 | 能平息 REAP vs REAM 的争论（前提是校准数据不是真正原因） |
| 5 | C9 | PTQ 会把 RL 学到的增量量化掉 | 5 | 8 | 9 | 框架部分被抢先，但 pilot 几乎零成本，部署答案很明确 |
| 6 | C5 | （并入 C1）压缩让随机驱逐不再免费 | 6 | 6 | 8 | 方向可以预判；作为 C1 的第二个维度更强 |
| 7 | C12 | 自我转写：多轮对话下与查询无关的视觉 KV 压缩 | 6 | 7 | 7 | 多轮诊断很有价值；方法本身是 KVzip / StreamMem 的迁移 |
| 8 | C8 | 重要性平稳性能预测"打分何时胜过随机" | 6 | 7 | 7 | 野心大的统一解释；与层深 / information horizon 混杂 |
| 9 | C7 | 随机抽样 headroom：任何选择器最多能赚多少 | 6 | 7 | 9 | 便宜；但 best-of-R 测的是方差，不是上限 |
| 10 | C10 | 用 drafter 的 surprisal 当 KV 驱逐信号 | 7 | 5 | 7 | 最可能的结果是"打平随机"，而这已被别人声称过 |
| 11 | C11 | 视觉 grounded token 一致率作为剪枝评测指标 | 6 | 6 | 8 | 接近 VIG 的打分；可能只是复现 hard subset 的结论 |
| 12 | C6 | 分叉 token 的 KL 能预测 TTS 损伤 | 5 | 6 | 8 | 在 2606.00206 之后剩下的空间很薄；约 10 个点上算 ρ |
| 13 | C13 | 量化 × KV 驱逐的长度中介叠加效应 | 4 | 5 | 8 | 大部分已被 2510.10964 和 ThinKV 做过 |

**合并建议**（triage 提出，已采纳）：C1 + C5 合为一篇（同一批模型对、同一批 trace、同一个冗余度统计量）；C7 + C8 可合为一篇关于随机基线问题的诊断论文（备选）。

### 🥇 Idea 1: C1+C5 — "压缩吃掉了冗余"：CoT 长度压缩对投机解码与 KV 驱逐的隐性代价
- **Method（我们实际做什么）**:
  1. 选公开的"冗长 / 压缩"推理模型对，必须是真正的父子关系：R1-Distill-Qwen-1.5B → ThinkPrune-1.5B；DeepScaleR-1.5B → L1-1.5B；R1-Distill-7B → AdaptThink-7B / LC-R1-7B。
  2. 在同一批 MATH-500 / AIME 题目上生成 trace，用 teacher forcing 离线测投机解码的接受长度 τ。drafter 用三种：无模型的 n-gram / 后缀查找、固定的小 drafter、EAGLE-3。再用压缩后的 trace 重新蒸馏 drafter，把"drafter 分布偏移"和"可预测 token 本身被删掉"这两个原因分开。
  3. 在 vLLM 里测 batch=1 和真实 serving batch 下每道题的 wall-clock，算"有效压缩比" = wall-clock 比 / token 比。
  4. 在同一批 trace 上比较随机驱逐（保护 prompt）、R-KV 和 SnapKV，把"随机 vs 打分"的差距对 trace 冗余度（n-gram 复述率）做回归。
- **Hypothesis**: 长度惩罚 RL 删掉的主要是低熵、高可投机、高冗余的填充内容，所以开 SD 时 token 节省会高估 wall-clock 节省，而且随机驱逐在压缩后的 trace 上不再"免费"。
- **Minimum experiment**: R1-Distill-1.5B vs ThinkPrune-1.5B，200 道 MATH-500，离线 n-gram τ + vLLM SD，约 1 GPU-h。
- **Kill criterion（triage 给出）**: 在两边都答对的题上，teacher-forced τ 的相对下降 < 10%。
- **Risk**: MEDIUM。具名风险：(a) drafter 偏移解释了全部下降，结论退化成"重训 drafter 就好"；(b) serving batch 下 SD 本来就收益有限；(c) 父模型配对必须正确。
- **Contribution type**: empirical + diagnostic
- **Pilot**: SKIPPED（无 GPU）

### 🥈 Idea 2: C4 — 混合架构的 softmax 层需要不同的 KV 驱逐策略
- **Method**:
  1. 选尺寸匹配的混合 / 稠密模型对：Nemotron-H-8B vs Llama-3.1-8B / Qwen3-8B、Falcon-H1-7B、Mamba-in-the-Llama 蒸馏模型 vs 其父模型。
  2. 只对 softmax 层的 KV 施加相同的驱逐策略（sink+window、SnapKV、随机、Ada-KV / PyramidKV 预算），预算 5–50%。
  3. 在 RULER（NIAH、multi-key、variable tracking）和 LongBench 上测 8K–128K。
  4. 逐层统计 recency mass（最近 512 token 上的注意力占比）和注意力熵。
- **Hypothesis**: 混合架构里的 softmax 层专职检索，所以窗口类驱逐会断崖式失效，打分类驱逐每字节保留的质量比稠密模型更高，稠密模型的层预算形状是错的。另一种可能是 SSM 已经承担了局部上下文，驱逐几乎免费。
- **Pilot 说服条件（triage 给出）**: (1) 相同相对预算下，混合模型上窗口驱逐比稠密模型多掉 ≥ 20 分，而 SnapKV 的差距小得多；(2) 混合模型 softmax 层的 recency mass 不到稠密模型中位数的一半；(3) 128K 时 softmax 层的 KV 每条序列仍有数 GB。
- **Risk**: MEDIUM–HIGH。具名风险："KV 本来就小，何必驱逐"；混合与稠密模型的训练数据不同（混杂因素）；cache hook 的工程量。
- **Contribution type**: empirical + diagnostic
- **Pilot**: SKIPPED（无 GPU）

### 🥉 Idea 3: C2 — Thinking MLLM 对视觉 token 剪枝更脆弱，且脆弱程度随思考长度增长
- **Method**:
  1. 用 Qwen3-VL-{2,4,8}B 的 Instruct / Thinking 匹配对。
  2. 在 prefill 时一次性剪枝（随机、FastV、DART、下采样），保留 50/25/11%。
  3. 在 MathVista、MathVerse-vision-only、ChartQA、MMMU 和一个简单 VQA 对照集上测，只统计全 token 时两个模型都答对的题。
  4. 记录每个解码步对视觉 token 的注意力，算"晚期证据率"；在 Thinking 模型内部扫思考预算 1K / 4K / 16K。
- **Hypothesis**: 相同保留率下，Thinking 模型的相对精度损失是 Instruct 的 2–3 倍，损失集中在长 trace 的后段回看。另一种可能是更长的思考补偿了缺失的视觉 token（test-time compute 替代视觉 token），这同样值得发表。
- **Kill criterion**: 25% 保留时 Thinking / Instruct 的相对损失比 < 1.3 且置信区间重叠，同时思考预算曲线是平的。
- **Risk**: MEDIUM。具名风险：两者的后训练差异不止 trace 长度。用 Thinking 模型内部的预算扫描来回应这个风险，不加模块。
- **Contribution type**: empirical + diagnostic
- **Pilot**: SKIPPED（无 GPU）

### 备选：C3（MoE 剪枝 vs 合并 pass@k）、C9（PTQ 是否抹掉 RL 增量，约 1.5 GPU-h，可当 side pilot）

<a id="eliminated-ideas"></a>
## Eliminated / Deprioritized Ideas

| Idea | 原因 |
|---|---|
| C13 量化 × KV 驱逐叠加 | 大部分已被 "Not All Bits Are Equal" 2510.10964（ICLR'26，1700+ 组 GPTQ × R-KV 配置）和 ThinKV 2510.01290 覆盖；只剩形式化的中介分析 |
| C6 Fork-token KL 预测器 | 2606.00206 已报告量化 KL 集中在高熵 token；跨家族预测只在约 10 个点上，统计上脆弱 |
| C11 grounded-token 一致率 | grounded 判定与 VIG 2608.21883 的打分几乎相同；可能只复现 hard subset 排名 |
| C10 drafter surprisal 驱逐 | 最可能的结果是"打平随机"，而 Random Attention 2609.03430 已有此结论 |
| C5 | 并入 C1，不再单独成文 |
| C7、C8、C12 | 没有淘汰，保留为候选（C7 + C8 可合并成诊断论文） |
