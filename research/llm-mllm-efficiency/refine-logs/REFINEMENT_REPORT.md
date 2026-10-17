# Refinement Report

## Problem Anchor
见 `PROBLEM_ANCHOR.md`，自 2026-10-17 冻结后未改动。

## Output Files
- `round-0-initial-proposal.md`、`round-{1,2}-refinement.md`：每一轮的完整方案
- `round-{1,2,3}-review.md`：评审记录与执行者的判断
- `codex_round_1_review_bundle.md`：评审说明。第 2–3 轮由同一评审续审，请求通过 SendMessage 发送
- `score-history.md`、`REFINE_STATE.json`
- `FINAL_PROPOSAL.md`、`REVIEW_SUMMARY.md`

## Score Evolution
6.4（R1）→ 7.1（R2）→ 7.8（R3）。评审估计修完第 3 轮问题后约 8.4。

## Round-by-Round Review Record
见 `REVIEW_SUMMARY.md` 中的表格；评审原文见各 `round-N-review.md` 的 details 区块。

## Final Proposal Snapshot
- **论点**：长度压缩删掉了复述，而复述是 SD 和无信号驱逐共同依赖的资源，所以交互因子 S_c/S_v < 1、E_c/E_v < 1；复述统计量 O_K 在模型内同时预测两者。
- **仪器**：τ / A（更新步进的位置匹配 tokens/step）、G、O_K（提及的联合存活）。
- **模型**：两个尺度各 3 个父子对，外加 L1 同权重剂量扫描。
- **闸门**：G0 约 6 GPU-h，第 1 周出结果。
- **预算**：合计约 289 GPU-h，含余量约 332，在 336 上限以内。

## Method Evolution Highlights
1. **执行者自查**：连乘形式的 EAL 公式需要草稿分支上的 target 条件概率，teacher forcing 给不出来，改为 trace-coupling 下的前缀匹配估计（发生在第 0 轮送审之前）。
2. **第 1 轮**：改用更新步进，修正逐位置平均的偏差。这是最关键的一处修正。
3. **第 2 轮**：逐头并集公式被 RA 自己的实测数据证伪，改为单次抽样下多 token 提及的联合存活。
4. **第 3 轮**：把"能被统计量推出的结论"和"判别不同模型的实验"分开（P1）；回归写成一个合法的统计模型。

## Pushback / Drift Log
- 评审建议的修改全部采纳，没有出现需要反驳的 drift。
- 两处主动选择：
  1. **EAGLE-3 改为条件性 must-run，而不是无条件 must-run。** 只在 G0 promote 后才训练，以保证总预算不超过 336 GPU-h。
  2. **第 3 轮后停止评审循环**，没有用满 MAX_ROUNDS=5。依据是评审对上限的分析：继续打磨无法消除"还没有数据"带来的不确定性。

## Remaining Weaknesses
效应量未知；熵锐化可能造成反向偏差；模型年代较早；EAGLE 只有父模型 head；E 的置信区间宽；O_K 与 τ 的联系较弱；抢先风险。

## Raw Reviewer Responses
见 `round-{1,2,3}-review.md` 的 details 区块（转写时保留了原有结构）。

## Next Steps
`EXPERIMENT_PLAN.md` → M0 / G0 → `/run-experiment`
