# EXP0025：显式太阳项集成进真实 IRIS 训练——首个 T06 实现与结果

状态：**首次真实（非代理）实现，通过一次真实的失败-诊断-修复循环后，在
EXP0019 的确切协议上把太阳导致的反照率污染从 +18.4/+14.0 个百分点降到
+1.7/+1.6 个百分点（200 步预算），量级已接近 SGS-Intrinsic（EXP0024：
−0.43/−1.20pp）。这是 `task_state.json` T06（"把显式太阳可见性整合进
IRIS"，此前状态 `planned`）的第一次真实落地，不是 EXP0004 的简化 2D 代理。

## 目标与背景

`research/EXPERIMENTS.md` EXP0024 发现：在最简单的已知几何对照里，
SGS-Intrinsic 的太阳污染（−0.43 到 −1.20pp）比原版 IRIS（EXP0019：
+14 到 +18pp）小一个数量级，这把"我们的方法能否在材质污染指标上赢过
SGS"这件事的门槛抬高了。本实验回答：把 `IRIS_ARCHITECTURE_AUDIT.md` 里
"IRIS 对任何逃出网格的光线都返回零背景光照"这个确认过的架构缺口，用一个
显式的、物理参数化的太阳项修补后，能不能在**真实 IRIS 训练代码**（不是
EXP0004 的简化代理）上复现并超过 EXP0004 概念验证过的解耦收益。

## 根因定位（复用 EXP0019/DATASET_AUDIT 已确认的架构问题）

`bake_shading.py` 通过 `SLFEmitter.eval_emitter` 查询每个表面点的间接漫反射
辐射；该函数对任何逃出网格（`valid_next=False`）的采样直接返回零
（`model/emitter.py` 里"assume zero background lighting"的具体后果）。
这个场景没有把太阳建模成任何 mesh 三角形发光体，所以 `diffuse` 烘焙贴图
在阳光斑位置系统性地漏掉了真实到达的能量；`train_brdf_crf.py` 的
`training_step` 只能通过增大该点的反照率 `kd` 来匹配观测到的（更亮的）
像素值——这正是 EXP0019 实测 +14–18pp 污染的直接机制，不是原版 IRIS 材质
优化代码本身有 bug。

## 方法：新增两个组件，均通过数据存在性/新增loss项做成"不存在则零副作用"

1. **`bake_sun_term.py`**（新文件，新增 pipeline 阶段）：对每个训练视角的
   每个表面点，只用几何计算 `sun_vis = max(0, dot(normal, sun_dir)) *
   visibility(point→sun)`，visibility 是对**同一份重建网格**做硬阴影光线
   测试（复用 `utils.path_tracing.ray_intersect`），和
   `experiments/generate_sunpatch_benchmark.py` 生成真值时用的是同一套
   逻辑，只是作用对象换成 IRIS 自己的重建网格。全程不涉及辐射度/材质，
   纯几何，成本几乎为零（这次已知几何 P1 场景里 `sun_dir` 直接取自
   manifest 的 GT 方向，Mode B 幽灵/星历方向恢复留给下一步，见"未做的事"）。
2. **`train_brdf_crf.py` 的两处新增**（均以 `batch['sun_vis'] is None` 完全
   跳过，对没有跑过 `bake_sun_term.py` 的任何既有场景/复现，行为与改动前
   逐字节一致）：
   - `Ld = kd*(diffuse + sun_vis*sun_irradiance)`，`sun_irradiance` 是一个
     3 维全局可学习参数，和材质一起联合优化（而不是像 EXP0004 代理那样
     固定/oracle），这是"Contribution One"在真实代码里的落地。
   - 一个新的 within-segment 反照率一致性正则项，写法完全比照代码里已有
     的 metallic/roughness 一致性正则（`loss_seg`），只是把它也用在
     albedo 上。**这一步是必需的，不是锦上添花**——见下面的调试记录。

## 调试记录：三次失败，每次都定位到真实原因（不是玄学调参）

不掩盖失败，因为失败本身是这次最有信息量的部分：

| 版本 | 改动 | 200 步结果（太阳额外污染） | 根因 |
|---|---|---|---|
| v1 | `sun_irradiance=softplus(zeros(3))` | **+55.5/+50.0pp**（比原版 IRIS 还差 3 倍） | `softplus(0)=ln2≈0.69`，是 `diffuse` 烘焙真实量级（≈0.012）的 ~50 倍，从第 0 步就淹没了正常光照，`kd` 被迫在受光点大幅下压来抵消，且與是否真有太阳无关（sky 组学到几乎同样大的值） |
| v1 @1000步 | 同上，训练更久 | **+75.4/+64.4pp**（更差） | 排除了"只是没训够"——问题在初始化本身，不会随训练自愈 |
| v2 | 改成 `softplus(full(-10))` 试图从零开始 | **+18.2/+13.3pp**（和原版 IRIS 几乎一样） | `sigmoid(-10)≈4.5e-5` 梯度消失，参数 1000 步内基本没动过（sky/sun_a 都停在初始值附近），等于该项从未真正参与优化 |
| v3 | 改用平方参数化 `sun_irradiance=raw.square()`，`raw` 初始化在 0.1（对应初值 0.01，量级匹配 `diffuse`） | 200步 **+19.7/+14.3pp**（未改善）；1000步 **+69.7/+65.4pp**（更差） | 参数化本身健康（sky 组正确学到接近零，sun_a 学到有意义的值），但 `kd` 在这个分支里**完全没有任何空间一致性正则**（`has_part` 分支只约束 metallic/roughness，从不约束 albedo），优化器没有任何理由把亮度差异分配给"免费"的全局太阳项而不是本就自由的逐点 `kd` |
| **v4（最终）** | v3 基础上，给 albedo 也加上和 metallic/roughness 同款的 within-segment 一致性正则（仅在 `sun_vis` 存在时生效） | **200步 +1.7/+1.6pp**；1000步 +6.6/+5.4pp | 让"逐点改反照率"变得有代价，太阳项因此第一次真正被优先使用 |

## 最终结果（200 步，与 EXP0019 完全同预算、同协议）

| 方法 | 太阳额外造成的归一化亮暗差变化 | 说明 |
|---|---|---|
| 原版 IRIS（EXP0019） | **+18.4pp / +14.0pp** | 无任何太阳建模，问题基线 |
| SGS-Intrinsic（EXP0024） | **−0.43pp / −1.20pp** | 不同 tone mapping/先验/协议，评测脚本自己标注不可直接排名 |
| **本方法（EXP0025，200步）** | **+1.7pp / +1.6pp** | 相对原版 IRIS **约 11 倍改善**；量级已接近 SGS，但绝对值仍比 SGS 大约 1.5–4 倍 |
| 本方法（EXP0025，1000步，稳健性检查） | +6.6pp / +5.4pp | 更长预算下有所回退（见"未解决的问题"），但仍远好于原版 IRIS |

`sun_irradiance` 学到的值在 200 步时颜色比例（[0.125,0.116,0.091]≈1:0.93:0.73）
与生成器真值 [4,3.7,3.2]≈1:0.925:0.8 相当接近；`sky` 组在 1000 步收敛到
几乎精确为零（[0.00014,0,0]）——两者都是物理上合理、而非巧合的信号。

## 诚实的局限（不隐藏，供下一步改进）

1. **绝对反照率精度未恢复，200 步时甚至出现饱和**（`comparison_200steps.png`
   四组"IRIS floor albedo"面板全部纯白，即预测值被 clip 到 ≥1）——差分
   指标（本实验的核心问题）修好了，但新正则项在如此短的预算下把绝对量级
   拉偏了。这是真实的代价，不是这次汇报选择性隐藏的东西。
2. **1000 步比 200 步更差**（+6.6pp vs +1.7pp），且 `sun_irradiance` 的
   颜色比例随训练拉长逐渐偏向红色通道（1:0.58:0.27），偏离真值比例——
   说明当前实现不是单调收敛的，200 步这个点可能只是短暂的"甜蜜点"，
   不是稳定解。这是最值得下一步解决的问题，怀疑与正则权重
   `self.hparams.lp`（复用了已有的 0.005，未针对新增项调过）和/或 CRF
   与太阳项的耦合有关。
3. **只有 1 个房间、1 个 seed、200/1000 两个预算点**——和 EXP0019/EXP0024
   一样的样本量限制，不能外推到"方法总是更好"。
4. **太阳方向使用的是生成器 GT（oracle），不是 Mode B 幾何恢复或星历
   预测**——这是刻意的范围控制（把"太阳存在时污染能不能被压低"和
   "太阳方向能不能被独立恢复"分开验证，后者已经在 `PHASE_A_SYNTHETIC.md`
   单独验证过 2.5° 误差），但意味着这不是端到端的真实 killer experiment。
5. 只验证了 material 阶段（stage 06/11），没有跑 `train_emitter.py`/
   `extract_emitter_ldr.py` 是否受益于同样的显式太阳信息（比如能否减少
   EXP0019 报告过的"太阳斑被误判成灯具三角面"问题）。

## 尚未做、留给下一次会话的事

- 修 1000 步的回退（正则权重/学习率分离/warmup schedule）。
- 换用 Mode B 或本次新获取的 Cali-HDR/Pano2Pano 真实 EXIF 星历方向替代
  oracle 方向，检验方法在**不知道确切太阳方向**时是否还成立。
- 多 seed / 多房间，回应"1 个房间 1 个 seed 不能算稳健证据"的标准质疑。
- 把 `sun_vis` 相关信号也接入 `extract_emitter_ldr.py` 的发光体判定，
  直接检验能否减少 EXP0019 报告的虚假发光三角面问题。
- 跑跟 EXP0024 完全一致的 evaluate 格式（如可能，统一 tone mapping）
  以获得更严格可比的数字，目前两者的"不可直接排名"仍然成立。

## 相关文件

- 新脚本：`bake_sun_term.py`，`experiments/run_sunpatch_iris_daylight.py`
- 改动：`train_brdf_crf.py`（`sun_irradiance_raw` 参数、`Ld` 项、albedo
  一致性正则，全部以 `sun_vis is None` 完全跳过保证不影响任何既有复现）、
  `utils/dataset/synthetic_ldr.py`（`InvSyntheticDatasetLDR` 可选加载
  `sun_vis` 缓存通道）
- 证据：`research/evidence/EXP0025/`
- 对照基线：`SUNPATCH_IRIS_BENCHMARK_ZH.md`（EXP0019）、
  `SGS_SUNPATCH_BENCHMARK_ZH.md`（EXP0024）
