# 阅读清单：“越推理越脱离证据”

只收录顶会论文或大组的工作，会议均已核实（未核实的已标注）。建议按 A → B → C 的顺序读。

## A. 多模态：推理越长，越不看图
1. **Mitigating Visual Forgetting via Take-along Visual Conditioning (TVC)**（ACL'25）https://arxiv.org/abs/2503.13360
   - 最关键的证据：推理到一半把图片拿掉，MathVista-hard 只掉约 2%，说明后半段推理基本不用图像。
   - 提出的修补方法是周期性重新注入图像。
2. **More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models**（NeurIPS'25；UCSC / Stanford James Zou）https://arxiv.org/abs/2505.21523
   - 提出 RH-AUC 和 RH-Bench，并控制了推理长度：推理越长，对视觉 token 的 attention 越低，对指令 token 的越高。
   - 长度和性能的关系不是单调的。
3. **More Thought, Less Accuracy? On the Dual Nature of Reasoning in VLMs**（ICLR'26；ANU / Melbourne）https://arxiv.org/abs/2509.25848
   - 称这个现象为 visual forgetting，用 VAPO（训练方法）把推理锚定在视觉上。
4. **Are VLMs Seeing or Just Saying?**（ICML'26；即 NOTES #43）
   - 在“再看一眼图”之后偷偷换图，模型察觉不到。插入一个 user turn 能把 attention 拉回图像。
   - thinking 模型比非 thinking 模型约差 3 倍。
   - 和“漂移还是开关”这个问题最相关。
5. **Deeper Thought, Weaker Aim**（CVPR'26，依据是 CVPR 的 poster 页面）https://arxiv.org/abs/2603.14184
   - 机制解释为“注意力发散”。VRGA 是 training-free 方法：选出视觉 head，再重加权它们对相关区域的 attention。
6. **Look Light, Think Heavy**（ACL'26；即 #94）
   - 视觉反思单调衰减，言语反思先升后降。图像被遮挡时模型仍然不弃权。

## B. 文本 / agent：推理越长，越不用外部信息
7. **Inverse Scaling in Test-Time Compute**（TMLR'25；Anthropic Fellows，Ethan Perez 等）https://arxiv.org/abs/2507.14417
   - 推理越长越差，常见原因是被干扰项带偏、抓住虚假特征、丢掉约束。
8. **AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions**（NeurIPS'25 D&B；Meta FAIR）https://arxiv.org/abs/2506.09038
   - 推理微调让弃权平均下降 24%。推理预算越大，准确率越高，但弃权越差。
   - 问题出在 RLVR 阶段；SFT 和 DPO 阶段反而会改善弃权。
9. **The Reasoning Trap**（ACL'26；即 #76）
   - 只在数学上做 RL，工具幻觉也会上升。对照实验显示原因是 think 这一步，而不是 RL 本身。
10. **The Danger of Overthinking: Reasoning-Action Dilemma in Agentic Tasks**（Berkeley，arXiv 2502.08235，会议未核实）https://arxiv.org/abs/2502.08235
    - 推理模型偏好内部模拟，而不是环境反馈（分析瘫痪、擅自行动、提前放弃）。分析了 4018 条 SWE-bench 轨迹。
11. **When Thinking Fails: Pitfalls of Reasoning for Instruction-Following**（NeurIPS'25 Spotlight；即 #2）
    - CoT 稀释了对约束 token 的 attention，导致指令遵循下降。
12. （可选）**The Hallucination Tax of Reinforcement Finetuning**（Findings EMNLP'25；USC）https://arxiv.org/abs/2505.13988
    - RFT 让不可解题的拒答率下降超过 80%。混入约 10% 的不可解题数据就能恢复。

## C. 机制和反面视角
13. **Characterizing and Mitigating Reasoning Drift in LLMs**（ICLR'26；中科院自动化所）https://openreview.net/pdf/1558de6be99c18e95e2391bdbb50f9b23bf439ab.pdf
    - 这里的 drift 指“锁死在错误轨迹上”：早期步骤可塑，后期锁定。用激活 steering（RAAS）修正。
    - 和“脱离证据”不是一回事，但方法上可以借鉴。
14. **Position: Reasoning After Perception Means Reasoning Without Vision**（ICML'26 position）https://arxiv.org/abs/2507.16863
    - 观点是：把视觉推理转写成文本，本身就丢掉了视觉信息。
15. （外围）**Mind Your Step (by Step)**（ICML'25；Griffiths 组）https://arxiv.org/abs/2410.21333
    - 在“人类越想越差”的那类任务上，CoT 同样会伤害模型。

## 未收录（质量或会议不确定）
- Mechanistic Evidence for Faithfulness Decay（Algoverse，会议信息互相矛盾）
- Reflection Anchors（2605.09614）、Cognitive Pivot Points（2604.10219）：只有 preprint
