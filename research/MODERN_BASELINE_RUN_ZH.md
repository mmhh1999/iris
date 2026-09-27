# EXP0020：SGS-Intrinsic 实际启动与复现检查

日期：2026-09-20。结论：**启动未通过，尚未进入训练，不能判断其太阳光斑材质恢复表现。**

## 已执行

- 官方仓库固定到 `282f76608dc24ea217ae9eebb05d2141ceae00c2`，工作树未修改。
- 在 `.baseline_deps/sgs` 隔离安装 Kornia 0.8.1、kornia_rs 0.1.9、plyfile 1.0.3；没有覆盖 IRIS 环境或降级 Torch。
- 多次运行入口排障，从缺 Kornia、plyfile 推进到缺 `simple_knn`。
- 最终 `attempt_02` 实际调用 `train.py`，指定 sun_a、12 视角、1 次迭代上限和独立输出目录；2.64 秒后导入失败，返回码 1。外层检查器返回 2 表示未通过验收。
- 这只是正式训练前的启动门槛，1 次迭代本来就不能产生科学结论；此次连该步也未执行。

## 不能靠简单安装解决的缺失

1. `rgb2x` 是指向作者仓库外 `../code_reference_firstwork/ipsm_relighting/rgb2x` 的失效符号链接；自定义 `pipeline_rgb2x_myversion.py` 未提供。
2. `r3dg-rasterization` 的 JIT 源目录缺失；另外引用 `diff_gaussian_rasterization_feature`、`simple_knn_r3dg` 等定制模块，其准确上游版本未锁定。
3. SAM2 权重使用作者机器绝对路径；还需要语义、几何等预处理。现有数据是 IRIS 格式，尚未转换为 SGS 输入。
4. 默认 `readColmapSceneInfo_vggt` 自选 5 个测试视角，与 EXP0019 的固定 12/2 划分不一致。接入时必须显式固定相同 split，不能直接运行默认脚本后横比。

检查了 main、master 及当前唯一公开 fork 的目录树：没有找到所缺 RGB-X 管线；fork 当前与 main 同一提交。
上游 issue [#2](https://github.com/GrumpySloths/SGS_Intrinsic.github.io/issues/2) 和 [#3](https://github.com/GrumpySloths/SGS_Intrinsic.github.io/issues/3) 也提出这些缺失，查询时评论为空。
issue #3 的用户报告自行替换组件后结果欠佳，但那是第三方非受控复现，**不是我们的实验结果，更不是 SGS 存在太阳光斑缺陷的证据**。

## 后续执行条件

普通公开依赖可继续补齐，但忠实 SGS 复现需要补全自定义先验管线和渲染接口。不得用零语义、常量材质、GT 材质或任意通用先验代替后仍标为官方 SGS。
若依据论文重建缺失部分，必须另命名 adaptation，并独立验证，不能与原版结果混为一谈。
AEGIR 仍为待发布代码候选。IRGS 的公开 CVPR 2025 实现不能直接标为 2026 IRGS++；本次未运行这两者。

## 可复查产物

- `experiments/check_sgs_baseline.py`：有超时、输出目录防覆盖、提交与数据 manifest 哈希、日志及退出状态。
- `research/evidence/EXP0020/attempt_02/`：最终有效启动检查。
- `research/evidence/EXP0020/attempt_01/`：保留失败的检查器尝试，曾错误解析 venv Python 符号链接导致丢失虚拟环境；该错误已修复，不能用于判断上游依赖。
- 上游 issue 元数据、空评论及分支/fork 树保存在同一 evidence 目录。

太阳光斑 benchmark 完成状态：false。albedo / roughness / heldout-light 指标均未产生。只记录工程复现阻塞，不作性能结论。
