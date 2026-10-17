# Pipeline Summary

**Problem**: 长度压缩、投机解码、KV 驱逐三者的收益能否叠加
**Final Method Thesis**: 长度惩罚 RL 删掉的是复述，而复述是 SD 和无信号驱逐共同依赖的资源，所以交互因子 S_c/S_v < 1、E_c/E_v < 1；复述统计量 O_K 在模型内同时预测这两类损失。
**Final Verdict**: REVISE（7.8/10，已收敛。评审建议停止打磨，直接跑 G0。评审为同家族 Claude 子代理，不是跨模型评审）
**Date**: 2026-10-17

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`（另有 `REFINEMENT_REPORT.md`、`score-history.md`、`round-*.md`）
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant contribution**：受控测量加机制解释。仪器为 τ / A（更新步进的位置匹配 tokens/step）、G（target 贪心可预测性）、O_K（提及的联合存活）。
- **Optional supporting contribution**：组合报告规范，即 S_c/S_v、E_c/E_v，以及 ρ 的三项分解。
- **Explicitly rejected complexity**：新 drafter、新驱逐策略、压缩感知 RL、MLLM、长输入、Qwen3.8-27B 对、QwQ-32B。

## Must-Prove Claims
- **C1a**：压缩降低剩余 trace 的位置匹配可投机性，即 A 和 G 下降。
- **C1b**：压缩让无信号驱逐不再免费（E_c/E_v < 1），且 O_K 能预测失败（policy × O_K 交互项为负）。
- **C2**：S_c/S_v = ρ_SD / ρ_noSD < 1。

## First Runs to Launch
1. **R003**：RA harness 移植后复现 Qwen3-4B MATH500 的结果（逐头 0.874 / shared 0.871），约 3 GPU-h。
2. **R004**：walk-τ 与 vLLM 实测接受率在 50 题上对照，约 1 GPU-h。
3. **R010 + R011**：G0-SD 和 G0-Evict，约 6 GPU-h。按预注册的 Kill / Promote / 中间分支执行。

## Main Risks
- **效应量落在 10–25%** → 按预注册的中间分支，叙事改为"实测的中等每 token 税"。
- **熵锐化导致反向偏差** → 报告 G 和熵，作为部分抵消如实写。
- **EAGLE 只有父模型 head，或训练失败** → 部署结论限定为 n-gram / draft-model 类 SD。
- **模型年代较早** → 如实说明 Qwen3 系列没有公开的压缩子模型。
- **抢先风险** → 第 1 周内完成 M0 和 G0。

## Budget
- **合计**：约 289 GPU-h（含 EAGLE），加余量约 332（上限 336）；不含 EAGLE 约 240。
- **G0**：约 6 GPU-h。
- **工程量**：约 2–3 人周。
- **日历时间**：8 卡下，第 1 周出 G0，第 4–5 周出主结果。

## Next Action
- 在有 GPU 的环境中运行 `/run-experiment`，从 R001–R005（M0）开始，然后跑 R010–R011（G0）。
- 同时配置 Codex，对 `FINAL_PROPOSAL.md` 补一轮跨模型评审（`/research-review`），拿到 ARIS 的 acceptance receipt。
