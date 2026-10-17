# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | sanity | 用 `huggingface-cli` 复核 11 个 checkpoint ID 和许可 | — | 可下载 | MUST | TODO | DLER 采用 NSCLv1 |
| R002 | M0 | sanity | RA harness 移植 Qwen2；shared-draw 改为跨层共享 | — | 单测 | MUST | TODO | 注意 QKV bias |
| R003 | M0 | 复现 baseline | RA 逐头 / shared，Qwen3-4B，K=1024 | MATH500 × 2 | acc（目标 0.874 / 0.871 ± 1.5） | MUST | TODO | ~3 GPU-h |
| R004 | M0 | sanity | walk-τ vs vLLM n-gram 实测接受率 | 50 道 MATH500 | 相对误差 ≤ 3% | MUST | TODO | 自己从提交的 token 统计 |
| R005 | M0 | sanity | 事实抽取器的人工标注 | 50 条 trace | 复用事件的 P / R | MUST | TODO | 约 1 人日 |
| R010 | M1 | G0-SD | R1-1.5B → ThinkPrune-iter2k；DeepScaleR → L1-Max（3 个预算） | MATH500 200 题 × 4 | A（n-gram，窗口 2048，匹配桶）、G | MUST | TODO | Kill / Promote / 中间分支按预注册 |
| R011 | M1 | G0-Evict | R1-1.5B vs DLER-1.5B × {逐头, shared, R-KV}，4× | MATH500 × 2 | acc 差、O_K 比 | MUST | TODO | ~4 GPU-h |
| R020 | M2 | B1 | 11 个模型的 trace 生成（vLLM） | MATH500 × 4、AIME × 8 | 长度、acc、循环标记 | MUST | TODO | ~10 GPU-h |
| R021 | M2 | B4 | 重采样对照 | MATH500 × 4 | 翻转率 | MUST | TODO | ~5 GPU-h |
| R022 | M2 | B1 / B4 | 离线 A / G / 父 drafter / 子 drafter | 全部 trace | A、G、熵 | MUST | TODO | ~5 GPU-h |
| R023 | M2 | B1 | L1-Max 剂量扫描 {512, 1K, 2K, 4K} | MATH500 × 4 | A vs 预算 | MUST | TODO | ~10 GPU-h |
| R030 | M3 | B2 | 1.5B：父 + 3 个子 × {shared, 逐头, R-KV} × 2/4/8(/16)× | MATH500 × 2 | E、acc、长度反弹 | MUST | TODO | |
| R031 | M3 | B2 | 7B：父 + 3 个子 × {shared, 逐头, R-KV} × 2/4/8(/16)× | MATH500 × 2 | E、acc、长度反弹 | MUST | TODO | 最大开销项 |
| R032 | M3 | B2 | 8 个模型 × 3 臂，4× | AIME × 4 | acc | MUST | TODO | |
| R033 | M3 | B2 | full-KV 等价性检验（HF vs vLLM） | 100 题 × 4 | TOST，|Δ| ≤ 1.5 | MUST | TODO | 不通过则用 HF full |
| R034 | M3 | B2 / B5 | 分层混合回归 + 预测量比较 | R030–R032 的数据 | 交互项、AUC、VIF | MUST | TODO | 零额外算力 |
| R040 | M4 | B3 | 代价网格 c(ctx, b, k) | 2 个尺寸 × 5 种 ctx × 3 种 b × 3 种配置 | ms / step | MUST | TODO | |
| R041 | M4 | B3 | 端到端验证 | 100 道 MATH500 + 30 道 AIME，DLER 两个对 | 预测误差 | MUST | TODO | b = 1 / 32 / 128 |
| R042 | M4 | B3 / B4 | EAGLE-3 head 训练（7B 父模型，SpecForge） | 父模型自己的 trace | 接受率 | MUST（G0 promote 后） | TODO | ~105 GPU-h |
| R043 | M4 | B3 / B4 | EAGLE 父 head → 父 / 3 个 7B 子 | 离线 + 计时子集 | A、S | MUST（G0 promote 后） | TODO | |
| R050 | M5 | 附录 | AdaptThink 1.5B / 7B 分层 | MATH500 | A、E | NICE | TODO | |
| R051 | M5 | 附录 | SnapKV + prompt / recency + prompt，1.5B 对，4× | MATH500 × 2 | acc | NICE | TODO | ~13 GPU-h |
| R052 | M5 | 附录 | 绝对 K ∈ {1024, 2048}，2 个对 | MATH500 × 2 | acc | NICE | TODO | ~20 GPU-h |
| R053 | M5 | 附录 | GPQA-D 泛化 | GPQA-D × 4 | A、E | NICE | TODO | ~30 GPU-h |
| R054 | M5 | 附录 | KD drafter（token 匹配） | 7B trace | A | NICE | TODO | ~20 GPU-h |
