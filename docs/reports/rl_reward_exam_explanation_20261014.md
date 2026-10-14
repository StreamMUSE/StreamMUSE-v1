# ReaLchords 式自监督匹配奖励的已知答案考卷报告（2026-10-14）

## 任务与考卷

老师布置的第③项=基于 ReaLchords 做 RL 后训练，第一块砖=自监督匹配奖励模型
（GT 旋律-伴奏配对训练，InfoNCE+判别器）。本报告回答：**这个 reward 能否胜任
我们项目的两个用途——bar 修复验收官 / best-of-N 候选选择器？**

考卷（known-answer exam，BEAT-40 held-out cohort，240 对=v1/v2 两批 ckpt 的
baseline/oracle 臂，同旋律同 seed 只差 history）：oracle 臂的 history 被替换为
GT 伴奏后由同一模型续写，**已知答案=reward(oracle)>reward(baseline)**，目标排序
≥80%；附 GT 锚点测试（裁判给 GT 真配对的分数应高于模型自生成伴奏）。
oracle 的真值性已由同 harness 的 CR 标尺背书（dCR +13.5 点、40/40 首全正）。

## 四轮受控实验（同一考卷、同一评分协议）

| 版本 | 配方 | 训练侧 | 考卷排序 o0/o8 | GT 胜率@offset8 |
|---|---|---|---|---|
| v0 | 单尺度双塔 InfoNCE+判别（论文核心，跨曲批内负例） | corpus R@1 0.94（曲级身份） | 40.8% / — | — |
| v1 | +同曲难负例（另窗/移调/乱序）+按歌切分 | val hard 0.725 | 46.3% / 39.2% | 37.5% |
| v2 | 单塔联合 2 通道+8 拍局部窗+空伴奏负例 | val 峰在 epoch 0 | 34.2% / 34.2% | 30.8% |
| v3 | **ReaLchords 完整配方**：32/16/8/4 拍逐尺度训练+50% 重叠滑窗求和+silence/early-end/repetition 惩罚 | 各尺度 val_r1 0.06-0.09 | **43.8% / 43.8%** | **17.5%** |

四轮全部低于随机线（50%），且方向一致系统性偏负（Δreward~ΔCR Spearman
−0.1~−0.2）：**裁判一致认为 baseline（模型自伴生历史）优于 oracle（GT 历史）**。

## 机制诊断（三个独立证据互相印证）

1. **GT 锚点**：在修复决策发生的窗口（beat 8-40），四版裁判给"数据集真配对"
   的分数全部低于模型自生成伴奏（GT 胜率 17.5%-37.5%）——裁判学会的偏好是
   "像模型的典型输出"，而模型典型输出恰是 baseline 臂，故考卷方向性偏负。
2. **空伴奏病理**：v1 给空伴奏 roll 打 0.87，高于一切真实伴奏（0.35-0.41）。
   ReaLchords 的 silence/重复惩罚项正是针对这类洞——v3 补上惩罚后空伴奏
   问题缓解，但方向性错误依旧，说明惩罚项治标不治本。
3. **捷径学习**：移调/乱序/另窗/跨曲负例全部可被全局纹理统计（音区、密度、
   节奏轮廓）识别，训练任务（判别真伪配对）从不强制模型学旋律-和声对齐。
   v2 的联合输入+局部窗、v3 的多尺度求和都未能移除这条捷径。ReaLchords
   论文自己观察到 contrastive 短窗退化（R@1≈0.01），与我们互证。

## 诚实边界（实现与论文的差异）

- 惩罚系数 0.25×3 为近似默认（论文未完整公开）；4 尺度非论文 5 尺度。
- 我们只验证了 reward model 本体，未跑完整 RL 循环（policy gradient+KL）。
- **考卷测的是"裁判能否识别 GT-history 条件下的更优续写"，这是比 ReaLchords
  自己的验证方式（训后 policy+真人听感）更严格、且不同的命题**——本报告不
  推翻 ReaLchords 的 RL 结论，只否定"其自监督奖励可直接胜任修复验收/候选
  选择"这一用途。

## 结论与建议

**ReaLchords 式自监督匹配奖励（含完整多尺度+惩罚配方）在我们场景四连败，
机制=损坏式判别的纹理捷径导致偏好与 GT 对齐反相关。** 对项目含义：
1. 不能直接用它当 bar 修复验收官或 best-of-N 选择器；
2. 若仍要走 ReaLchords RL 路线，reward 信号源需换：偏好对监督（GT 正例+
   lekai failcase 负例的 KTO 式训练、oracle/baseline 240 对现成）或人工标签，
   这仍兼容"基于 ReaLchords 做后训练"的框架（换的是 reward 来源，不是 RL 算法）；
3. 备选：直接部署测试（best-of-N 用现裁判排序量 CR）——但现裁判反相关，
   预期选到更差候选，须先换 reward 源。

## 产物

- 代码：`experiments/rl_reward/{reward_data,reward_model,train_reward,eval_exam,joint,multiscale}.py`
- checkpoint/考卷：masdar `~/experiments/{rl_reward_20261013,rl_reward_hardneg_20261014,rl_joint_v2_20261014,rl_ms_20261014}/`
- git：bar-quality 链 b3bcc2c5..c38e659f+fix（已推 origin）
