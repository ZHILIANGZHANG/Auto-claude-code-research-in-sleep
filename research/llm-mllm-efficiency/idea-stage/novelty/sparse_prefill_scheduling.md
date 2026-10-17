# Novelty Check — 稀疏 prefill 调度（Sparse-prefill-aware scheduling for LLM serving）

- 日期：2026-10-17；用户提问："稀疏 prefill 调度能做吗"
- 方法：ARIS `/novelty-check`。Phase B 由检索子代理完成（44 次 WebSearch，以 extended 为主）。Phase C 由同一个子代理在 NOVELTY VERDICT LIMITS 下判定。**它与执行者同属 Claude 家族，不是跨模型评审。** 判断只基于摘要和检索摘录，没有读全文。
- 4 个角度，判定如下：
  - **(a) 稀疏度感知的代价模型 + SLO 调度：PROCEED 7/10**
  - (b) chunked prefill × 稀疏模式估计：作为独立贡献 **ABANDON**，具名论文 CompactAttention 2605.16839、QuoKA 2602.08722（ICLR'26）。其中"chunk 大小与稀疏度联合优化"仍然开放，并入 (a)
  - (c) 近似 KV 的 prefix caching：PROCEED WITH CAUTION 5.5/10（SCBench 已部分回答）
  - (d) 稀疏 prefill 下的 P/D 配比：PROCEED WITH CAUTION 5/10（DistServe 的放置搜索可以直接套用）

## (a) 的关键证据
- 所有找到的 SLO 调度器都**只按长度**估计 prefill 代价：Kairos 2605.02329 原话"prefill time is highly predictable from input length"；SLOWeave 2609.07883；P-PAS 2608.15171；LAPS / PLA-Serve 2601.11589。
- 稀疏 prefill 已经进入服务引擎，但调度器本身没有改：
  - FlashPrefill V2 2608.19758 是 SGLang 后端，密度在 4K 时约 70%、128K 时约 5%；
  - UniPrefill 2605.06221（NeurIPS'26）修改了 vLLM 调度器，但目的是让稀疏 prefill 适配 continuous batching，看不到逐请求的代价预测；
  - SGLang 的 NSA 后端只用一个固定的 2048 token 阈值切换稠密 / 稀疏。
- 相邻工作：
  - RouteSparse 2608.29058：只在单个请求内部做延迟预算，不跨请求调度；
  - LServe、SPIN、SparseServe、Vortex：稀疏注意力服务系统，关注点主要是 decode 侧和 KV 内存；
  - SPEAR 2606.11244：结构上类似（输入相关的计算量 + SLO 调度器），说明这种模式能发表。

## 定位（一句话）
现有 SLO 调度器（Kairos、SLOWeave、P-PAS）按 prompt 长度估计 prefill 代价。在动态稀疏 prefill（FlashPrefill V2、MInference）下，相同长度的代价随内容变化，因此用稀疏度感知的代价预测器来做准入、排序和 chunk 大小选择，可以在同一个稀疏 kernel、相同精度下提高 TTFT-SLO goodput。

## 风险
1. **"只是把 Kairos / SLOWeave 的代价模型换掉"**。需要证明：相同长度下密度的异质性会导致按长度的预测器排错请求顺序、错过 SLO，并且探针本身的开销可以摊销或被 kernel 复用。
2. **收益只在长 prompt 上出现**。2–4K 以下稀疏 prefill 退化为接近稠密，所以必须用长上下文 trace。
3. **如果把稀疏预算当作可调旋钮**，就必须同时报告质量。
4. **抢先风险**：FlashPrefill V2 和 UniPrefill 的作者组。

## 可行性（≤8 × A100 / 4090）
- **引擎**：SGLang 路线最短（FlashPrefill V2、Vortex 都基于 SGLang）。vLLM 也可以（UniPrefill、P-PAS 的 vLLM 分支）。
- **硬件注意**：FlashPrefill V2 的 kernel 与 FA3/4 对齐，可能需要 Hopper 或 Blackwell。在 A100 / 4090 上应改用 MInference、XAttention、FlexPrefill 或 MIT 的 Block-Sparse-Attention（LServe 代码）。
- **模型**：Llama-3.1-8B 或 Qwen2.5/3-7B，在 A100-80GB 上跑 128K。4090 只适合 32–64K 或 3–4B 模型。
- **基准与 trace**：精度用 RULER、LongBench v2、InfiniteBench（SCBench 用于 (c)）。负载用 Mooncake trace（长上下文到达模式，内容用 LongBench / RULER 映射填充）、Azure LLM trace、BurstGPT（突发）。ShareGPT 只作为短请求混合。
- **工程量**：约 2–3 人月：
  - 稀疏度和延迟的 profiling：约 2 周；
  - 预测器 + 调度器钩子：约 3–4 周；
  - 端到端评测：约 3–4 周；
  - 如果做 (c)，再加约 2 周。

## 近 6 个月的同期工作
UniPrefill（5 月）、CompactAttention（5 月）、Vortex（6 月）、SPEAR（6 月）、FlashPrefill V2（8 月 20 日）、RouteSparse（8 月 29 日）、P-PAS（8–10 月，附 vLLM RFC #52906）、SLOWeave（9 月）、CRISP 2609.01925（9 月）、HySparse2 2609.26368（9 月）。

## 开工前需要人工复核
- UniPrefill 和 FlashPrefill V2 的调度器实现（代码树），确认它们确实没有做逐请求的代价预测；
- SCBench 中 shared-context 的结果（全文）。
