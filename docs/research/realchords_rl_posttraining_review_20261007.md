# ReaLchords / ReaLJam RL 后训练调研

日期：2026-10-07

## 结论先行

ReaLchords 的核心不是在线时修补历史，而是让 causal online policy 在训练时反复经历自己的输出历史，学习从错误、冷启动和分布外 Melody 中恢复。训练由三部分组成：on-policy REINFORCE、自监督音乐一致性奖励、来自 non-causal offline teacher 的 KL 蒸馏。ReaLJam 没有提出新的 RL 算法，而是把 ReaLchords 的 Online、RL-S、RL-M 三种模型放进真人实时系统做用户实验。

对 StreamMUSE 最重要的判断：当前 Continuation checkpoint 可以作为 policy 初始化，但仓库里没有 reward model、offline teacher、value model 或 on-policy trainer。最稳妥的研究顺序是先建立 rollout/reward 离线审计，再做 KD-only，最后才做完整 RL。不能把现有 Rule-S 分数、PPL 或单个乐理指标直接当作 RL reward。

## 论文与来源

1. Yusong Wu et al., **Adaptive Accompaniment with ReaLchords**, ICML 2024, PMLR 235:53328--53345.
   - 论文：https://proceedings.mlr.press/v235/wu24c.html
   - arXiv HTML：https://arxiv.org/html/2506.14723v1
   - 角色：主要的模型、reward、distillation 和 RL 算法来源。
2. Alexander Scarlatos et al., **ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers**, CHI EA 2025.
   - 论文：https://arxiv.org/html/2502.21267v1
   - 角色：真人交互验证和 RL 模型行为证据，不是独立的 RL 方法。

目前未从论文主页找到原始 ReaLchords/ReaLJam 官方训练代码。公开的 `realchords-pytorch` 来自后续工作，不能自动视为 ICML 2024 原实现。

## 1. MDP 与在线因果约束

论文把 Melody 记为 `x`，伴奏和弦记为 `y`。在线 policy 为：

```
pi_theta(y_t | x_<t, y_<t)
```

在时刻 `t` 生成伴奏时，不能看当前或未来 Melody `x_t:`。状态是过去 Melody 与过去由 policy 自己生成的伴奏；动作是下一时刻的 chord token。训练时从数据集抽固定 Melody 作为模拟用户，但伴奏历史来自当前 policy rollout，而不是 GT teacher forcing。

这与 StreamMUSE Continuation 的目标高度相关：真正要减少的是“训练只见 GT accompaniment、推理却见自身 accompaniment”的 exposure bias。

## 2. 总训练目标

ReaLchords 从 Online MLE checkpoint 初始化 policy，并最大化 KL-regularized RL objective：

```
E_{x~data, y~pi_theta(.|x)} [ R(x,y) ]
  - beta * KL(pi_theta(.|x) || phi_omega(.|x))
```

- `pi_theta`：只能看过去的 online policy。
- `phi_omega`：能看完整 Melody 的 offline teacher。
- rollout 由 student policy 生成；teacher 只在 student trajectory 上计算概率。
- 优化算法是 REINFORCE，不是 PPO/GRPO。
- 使用独立 value model 作为 baseline 降低方差；value model 也从 Online MLE 初始化，用 MSE 拟合 total return。
- reward、penalties 和 KL 只对 policy 生成的输出 token 反向传播，不对输入 Melody token 反向传播。
- 论文把 reward 汇总后施加到最后一个 policy token，而不是报告逐 token reward shaping。

## 3. Reward models

### 3.1 Contrastive reward

分别用 Melody encoder 和 chord encoder 得到 `E_x`、`E_y`。训练使用 InfoNCE：

- 正样本：同一首曲子的 Melody--chord pair。
- 负样本：batch 内不同曲子的 Melody 和 chord 随机重配。
- 推理 reward：`cos(E_x, E_y)`。

论文观察：这种 embedding bottleneck 更偏向全局和声一致性。

### 3.2 Discriminative reward

单个 encoder 输入完整 `(x,y)`，训练区分真实配对与随机错配。输出“真实配对”的概率作为 reward。论文观察：它更偏向局部时间同步。

### 3.3 Multi-scale reward

最大长度为 256 frames，另训练长度为 `1/2, 1/4, 1/8, 1/16` 的 reward models，即 128、64、32、16 frames。RL 时在 50% overlap 的滑窗上计算。

动机不是单纯增加模型数量，而是避免一次短暂错误污染整段 global reward，并改善 temporal credit assignment。

论文的 reward model 规模：

- Contrastive：两个 6-layer、6-head、hidden 512 Transformer encoders；Adam `1e-4`，batch 128，35 epochs，dropout 0.1。
- Discriminative：一个 6-layer、6-head、hidden 512 Transformer encoder；Adam `1e-4`，batch 64，5 epochs，dropout 0.1。

短窗口 contrastive retrieval 明显变弱；16-frame R@1 只有约 0.01，而 full-context 为 0.17。相反 discriminative 的 F1 从 full-context 0.79 到 16-frame 仍约 0.73。因此，对 StreamMUSE 不应假定所有尺度都同样适合 harmonic reward。

## 4. Offline teacher 与蒸馏

Offline teacher 为 encoder-decoder Transformer，完整编码 Melody，再自回归生成 chords：

```
phi_omega(y_t | x_full, y_<t)
```

关键是 **student-driven on-policy distillation**：

1. Online student 在 causal 条件下生成完整 accompaniment trajectory。
2. Offline teacher 对同一条 student trajectory 计算 token probability。
3. 以 KL 约束 student 接近能看未来 Melody 的 teacher。

因此 teacher 不只是提供一个离线伴奏样本，而是在 policy 实际访问到的状态上提供 dense probability guidance。论文报告 KD-only 已显著提升感知质量和 harmony，完整 RL 再进一步提升。

StreamMUSE 当前没有一个已确认满足“完整 Melody 条件 + 与 Continuation action vocabulary 对齐”的 offline teacher。Prompt Model 只生成开头 8 beats，不能直接充当该 teacher；现有 causal Continuation 即使离线运行，也不等于论文的 non-causal teacher。

## 5. 防止 reward hacking 的 penalties

论文额外使用三类明确惩罚：

- Repetition：和弦持续过久。
- Silence：乐句开头之后的长静默。
- Early ending：过早生成 EOS。

这说明只优化“和声匹配”会产生退化捷径。论文还报告：把 offline/online MLE likelihood 直接作为 reward 的尝试，会塌缩为重复同一和弦。

StreamMUSE 对应的最低限度 guards 应覆盖：

- `empty/rest` 连续过长；
- 提前停止；
- 长时间重复相同 accompaniment pattern；
- invalid token grammar；
- 输出密度极端值。

这些应该是训练 penalty 或 hard validity mask，不应混进 harmonic reward 后失去可解释性。

## 6. 已公开的 RL 超参数

| 项目 | 论文设置 |
|---|---:|
| Policy/value learning rate | `1e-4` |
| KL coefficient beta | `0.5` |
| Reward-model reward coefficient | `50` |
| Early-ending penalty coefficient | `20` |
| Repetition/silence：ReaLchords | `2 / 2` |
| Repetition/silence：ReaLchords-M | `10 / 10` |
| Algorithm | REINFORCE + separate value model |
| Reward assignment | summed reward applied to final generated policy token |

论文没有充分公开 RL batch size、rollout 数、总训练 steps、gradient accumulation、完整 normalization/clipping 细节和随机种子配置。因此目前不能声称能逐项精确复现。

## 7. 论文证据应如何解读

### ReaLchords 离线评估

- Online MLE 的 Note-in-Chord 为 36.99%。
- KD-only 为 46.38%。
- ReaLchords 为 48.17%。
- Multi-scale ReaLchords-M 为 54.29%。
- Offline MLE 为 63.73%。

但 harmony 提升不代表所有维度都同步提升。完整 ReaLchords 的 chord-length entropy 从 Online MLE 的 2.21 降至 1.35，论文解释为更偏向“安全”的 2/4-beat chords。也就是说 reward 和 KD 可能以牺牲节奏多样性换取更稳定的和声。

10 位音乐家对 8 秒片段进行了 192 次 pairwise rating。KD、ReaLchords、ReaLchords-M 均显著优于 Online MLE；完整 RL 优于 KD-only，ReaLchords-M 对 ReaLchords 的优势未达到显著。

### ReaLJam 真人实验

6 位有经验的音乐家完成实时 jamming。真人对 Online MLE 明显不偏好：该模型容易出现不可预测和弦及长静默；RL 模型更稳定。用户仍指出 RL 模型缺少 4-bar/8-bar 结构感和终止解决倾向。

因此 ReaLJam 支持“RL 后训练改善真人交互稳定性”，但样本只有 6 人，也不能证明它解决了长程结构。

## 8. 与 StreamMUSE 的结构差异

| ReaLchords | StreamMUSE Continuation |
|---|---|
| 单声部 Melody + chord token | Melody + polyphonic piano accompaniment token |
| 每个 frame 一个 chord-level action | 每 beat 内有可变长度 token sequence |
| 最大 256 frames | 当前训练 cutoff 2048 tokens、32-beat inference context |
| Online policy 8-layer/512 | Continuation 18-layer/768，约 170M |
| Offline T5-style teacher | 当前没有等价 teacher |
| reward 直接评估 Melody--chord | reward 需评估 Melody--polyphonic accompaniment |

最重要的工程差异是 action boundary。StreamMUSE 应把“一拍 accompaniment 的完整 token span”当作环境 action，再由环境注入下一拍 Melody；不能把所有词表 token 当作语义等价的 RL action。loss mask 只覆盖真实采样的 accompaniment output tokens，结构 token/非法状态要单独定义。

## 9. 建议的实施顺序

### Phase A：冻结模型，先验证 reward

1. 用当前 Continuation 对训练/held-out Melody 做 causal on-policy rollout。
2. 保存每拍：可见 Melody、生成 accompaniment tokens、log-prob、结束原因和 MIDI/pianoroll。
3. 训练一个 global contrastive reward 和一个 local discriminative reward。
4. 用现有人工 fail labels 检查 reward 是否能区分 `insufficient_output`、`repetition`、`melody_mismatch`。
5. 做 hard-negative：同调/同风格但不匹配的 accompaniment、局部错位、局部转调、错误 duration。仅随机跨曲配对可能过于容易。

Go/no-go：reward 必须在 held-out piece 上与人工偏好有关，并且不能仅靠密度、静默或调性捷径分类。

### Phase B：KD-only

1. 训练一个与 Continuation 输出空间完全一致、可看完整 Melody 的 offline teacher。
2. Student causal rollout；teacher 在 student token prefix 上评分。
3. 仅优化 on-policy KL，加 validity/repetition/silence/EOS penalties，不先加入 learned reward。
4. 比较原 checkpoint、现有 `initdrop/accdrop` SFT 和 KD-only。

KD-only 是最关键消融，也是比完整 RL 更稳定的第一步。论文中它已经产生最大的单步提升之一。

### Phase C：完整 RL

1. Policy 从当前 Continuation checkpoint 初始化。
2. Value model 从同 checkpoint 初始化，或先做明确标注的 value-head 工程近似。
3. Reward models 和 teacher 全冻结。
4. 使用 REINFORCE + value baseline + offline-teacher KL。
5. 先 global rewards，再增加 multi-scale；每一步都保留 repetition/silence/EOS/grammar 监控。

必须记录：reward 分量、KL、entropy、value error、有效输出率、平均 token/beat、静默长度、重复率，以及固定 Melody 上的周期性试听样本。

## 10. 最小实验矩阵

| 条件 | 目的 |
|---|---|
| Current MLE checkpoint | 基线 |
| Existing dropout/initdrop SFT | 判断数据增强能否部分解决 exposure bias |
| On-policy KD only | 隔离未来可见 teacher 的贡献 |
| Reward RL + online-MLE KL | 隔离 learned reward 的贡献 |
| Reward RL + offline-teacher KL | 完整主要方案 |
| Full + no repetition/silence/EOS penalties | 检测 reward hacking |
| Global reward vs multi-scale reward | 局部 credit assignment 与多样性权衡 |

主评估不能只看 CR/UR。至少同时保留：非空率、early-stop、重复率/Unique Beat Ratio、FMD/JSD、CR/UR、人工 pairwise preference，以及冷启动/中途 Melody 扰动后的逐 beat 恢复曲线。

## 11. 主要风险

1. **Reward hacking**：模型通过少弹、重复安全和弦或复制 Melody 提高分数。
2. **Teacher 不匹配**：teacher 与 student token grammar 不一致时，KL 没有正确语义。
3. **训练成本**：170M policy + value + offline teacher + reward ensemble 同时驻留和 rollout，计算量显著高于 SFT。
4. **高方差**：sequence-level REINFORCE 对长可变 token trajectory 方差很高。
5. **结构缺失**：ReaLJam 已显示 RL 模型仍不理解 4/8-bar 结构，reward 需要显式覆盖。
6. **指标捷径**：随机跨曲负样本过易，reward 可能只学风格/调性/密度，而非真正伴奏匹配。

## 12. 当前代码差距审计

当前仓库已有：

- `PianoLLaMA.forward(..., labels=...)`，可返回标准 causal LM loss/logits；
- 训练 dropout 0.1；
- `acc_drop_prob`、`drop_initial_beats` 和 Melody-loss 选项；
- causal beat-wise generation 与采样 generator。

当前仓库未见：

- on-policy rollout dataset/buffer；
- sampled-token log-prob 与 action mask 持久化；
- reward model 训练代码；
- non-causal offline teacher；
- value model/value head；
- REINFORCE/PPO/GRPO trainer；
- KL-to-teacher 训练实现；
- reward component 与 reward-hacking audit。

因此下一步最适合先写 `RL_EXPERIMENT_SPEC.md` 与 rollout/reward 数据契约，而不是直接启动训练。

## 引用

```bibtex
@inproceedings{wu2024realchords,
  title={Adaptive Accompaniment with ReaLchords},
  author={Wu, Yusong and Cooijmans, Tim and Kastner, Kyle and others},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  series={PMLR}, volume={235}, pages={53328--53345}, year={2024}
}

@inproceedings{scarlatos2025realjam,
  title={ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers},
  author={Scarlatos, Alexander and Wu, Yusong and Simon, Ian and others},
  booktitle={Extended Abstracts of the CHI Conference on Human Factors in Computing Systems},
  year={2025}, doi={10.1145/3706599.3720227}
}
```
