# EXP0021：SGS 公共上游兼容修复

2026-09-20。**修复版训练入口已通过；完整训练和太阳光斑评测尚未完成。**

## 本次实际解决的问题

- 从公开 R3DG 获取 `simple_knn` 与 `r3dg_rasterization`。原先只读 Python 包装层会误以为缺 `weights`；核查 CUDA 源码后确认它确实输出每高斯累积像素权重，与 SGS 包装层的返回值数量一致。
- 从 Feature-3DGS 获取语义渲染器：四项返回值与 SGS 一致；按 SGS 实际存储维数，将编译通道设为 32，模块名设为 `diff_gaussian_rasterization_feature`。
- 为旧 CUDA/C++ 源补 `<cfloat>` / `<cstdint>`，配置本机 CUDA 与 NVIDIA 头文件路径，成功编译 KNN、两种高斯渲染器及 nvdiffrast。没有降低现有 Torch 版本。
- 从公开 RGB-X 提供返回原始解码 AOV 张量的适配器。SGS 调用方自行归一化和 gamma 转换，因此不能直接接默认 PIL/已转换结果。格式契约测试通过；**尚未运行权重推理，无法证明与作者未公开的 myversion 完全相同。**
- 对 RGB-X、SAM2、StableNormal 实施按需加载；SAM2 路径改由 `SGS_SAM2_CHECKPOINT` 指定。仅查看帮助不再下载或分配这些模型。
- 修复官方 `utils/pose_utils.py:41` 的意外缩进。整个修复副本通过 Python 编译检查，相机变换往返验证通过。

原始 `third_party/sgs_intrinsic` 未修改。修复副本为 `third_party/sgs_adapted`，明确标记 **SGS-public-upstream-adaptation**，不冒充未经修改的官方完整复现。

## 实测验收

| 项目 | 结果 |
|---|---|
| KNN GPU 对暴力三近邻距离 | 最大绝对误差 9.54e-7，通过 |
| nvdiffrast 三角形渲染与梯度 | 242 个前景像素，颜色梯度有限且非零，通过 |
| SGS 原始 R3DG 包装层 | 10 项输出完整，通过 |
| 高斯权重总和对像素 opacity 总和 | 绝对误差 3.81e-6，通过 |
| 颜色梯度对累计贡献权重 | 最大误差 7.63e-6，通过 |
| 32 通道特征前向/反向 | 输出形状正确，梯度有限且非零，通过 |
| RGB-X 后处理契约 | 解码值在 [-1,1] 内两条处理路径误差 0；不是预测精度测试 |
| 相机变换往返 | 误差 0 |
| 修复版 `train.py --help` | 最终退出码 0，3.93 秒，无模型权重加载 |

GPU：RTX 5070 Ti；Torch 2.11.0+cu128；CUDA 编译目标 12.0。
依赖安装到 `.baseline_deps/sgs`；不覆盖 IRIS 的原有 Python 包。

## 可复现入口

公共源码 URL、提交及子模块固定在 `experiments/baselines/sgs_sources.lock.json`。

1. 按锁文件克隆公共源码到指定 `third_party/` 路径。
2. `python3 experiments/baselines/prepare_sgs_extensions.py` 生成可移植 CUDA 源。
3. `build_sgs_extensions.py` 编译指定源码目录到隔离包路径；Python 包版本见证据目录的 `python_packages.json`。
4. `prepare_sgs_compat.py --source third_party/sgs_intrinsic --rgbx third_party/rgbx_upstream --output third_party/sgs_adapted` 创建修复副本；拒绝覆盖已有目录。
5. `python3 experiments/baselines/run_sgs_compat.py --output experiments/out/sgs_help_new -- --help` 运行带超时、日志与退出码的检查。输出目录须为新目录。

GPU 检查脚本：`test_sgs_kernels.py`；RGB-X 格式契约：`test_sgs_rgbx_contract.py`。运行时需将 `.baseline_deps/sgs` 加入 PYTHONPATH，及使用本机 `.toolchain/compiler/lib` 动态库路径，参照启动器。

## 还没有解决/完成的部分

- 太阳光斑数据尚未转换为完整 SGS 输入；需固定原来的 12/2 划分，不能用其默认 5 测试视角选择。
- 需从允许输入照片生成语义特征、类别特征、法线等先验，并准备 SAM2、RGB-X 等权重；不能把材料 GT 填进训练先验目录。
- RGB-X 公共适配器尚未做真实图片/批量推理验收；所有先验同时加载在 16GB 显存下的峰值尚未验证。
- 需完成 Stage I/II、小样本收敛检查和独立换光渲染，才能产生 BRDF/光斑污染指标。
- 原版与公共适配版的科学等价性未建立。后续结果必须注明代码版本、先验和适配差异。

EXP0020 的“缺少依赖，入口不能运行”已由本次进展更新；它仍是真实历史记录。当前不能再将公共 CUDA 内核视为完全不可获得，也不能因入口通过就声称 benchmark 已完成。
