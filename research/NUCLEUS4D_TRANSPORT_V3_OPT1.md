# v3_opt1：保留太阳功能的增量优化

状态：已发布并完成本机 GPU 浏览器验证。当前入口仍为
<http://localhost:8769/nucleus4d_transport_v3/>。

用户要求以贴实现方案之前的全屋光照 v3 为基础优化，并保留太阳功能。
此次继续使用该渲染器、完整 6,100,978 个高斯和原有 float32 光场，未启动后续
stages/world 框架，也未重新训练或重算光场。

## 已实现

- 保留连续时间、太阳方向变化、强度、天气、软阴影、播放及直射/黄昏预设。
- 新增明确的太阳开启/关闭按钮；开启时恢复上次非零强度，保留时间。
  “开启太阳（晴空）”会在云量为 100% 时切到晴空，界面说明直射是否生效。
- 修复太阳强度非零但被 100% 云量屏蔽时，启动播放仍没有直射的问题。
- 仅在 float32 光源权重改变时重混光场；曝光调整不再重混不变光场。
  无有效直射时不生成太阳阴影，恢复太阳时按最新时间重新生成。
- 复用光场混合的 uniform 数组，保持原来的累加顺序、精度和着色程序。
- 新增可选 8192² 太阳深度图。默认仍为原版 4096²；新纹理分配及帧缓冲检查
  成功后才替换旧纹理，失败时保留可用的旧图。8K 深度纹理约 256 MiB，
  相比 4K 的 64 MiB 增加 192 MiB，不含切换时暂存的旧图。
- 保留 25 个独立灯光、原拍摄精确对照及自由漫游。

## 变更文件

主要实现：`experiments/nucleus4d_transport_runtime.js` 和
`experiments/nucleus4d_transport_viewer.py`。浏览器检查：
`experiments/nucleus4d_transport_opt_check.mjs`。

基线要求写入 `experiments/gs_relight_baseline.json`，说明更新至
`experiments/RELIGHTING_CURRENT_BASELINE.txt` 和
`experiments/nucleus4d_transport_README.txt`。页面、脚本及 manifest 已重新发布到
`experiments/out/nucleus4d_transport_v3/`。

原始模型、光场、灯光定义、太阳关键帧均未修改。
`nucleus4d_transport_programs.js` 和 `nucleus4d_transport_shading.glsl`
与优化前备份的 SHA256 一致。

## 执行与验证

构建：`.venv/bin/python experiments/nucleus4d_transport_viewer.py`。
Python 编译检查通过。使用 Windows Node 和现有 Chrome CDP 9367，在实际 GPU 上
运行 `nucleus4d_transport_opt_check.mjs <review目录> before|after`。
首次 after 检查读到旧缓存；检查程序改为先禁用缓存再导航，随后完整重测通过。

证据目录：`experiments/out/nucleus4d_transport_v3/optimization_review/`。

- `before/` 保存优化前实现和已发布小文件；大型模型/光场没有复制或覆盖。
- `before.json`、`after.json` 保存浏览器状态及 GPU 计算计数。
- `pixel_comparison.json`：1920×1080、同一原始相机，拍摄原貌、12:30 直射、
  16:00 直射、18:15 直射、40% 云量、仅人工灯共六组图像，RGB 最大误差均为 0，
  所有像素完全一致。
- 每组 12 次曝光调整：光场混合 draw 次数由 12 降到 0。
  每组 12 次无太阳时间调整：光场混合和阴影 draw 次数均由 12 降到 0。
  这是避免重复计算的计数证据，不代表整帧帧率提升了相同比例。
- 4K 的三个时间以及 8K 的两个时间，太阳方向、阴影矩阵和画面均实际变化。
  8K 切回 4K 后，与原版同状态画面再次逐像素一致。
- 太阳关闭期间改变时间，再开启后，按新时间重新生成阴影。
- 从被全云量遮蔽的非零太阳启动播放后，时间推进且直射有效。
- 连续触发 60 次时间输入，未改变相机、未上传纹理、未发起网络请求。
- 单灯调到 40% 时其他 24 灯保持 100%，光场确实重算；方向键漫游正常。
- 全程为 6,100,978 个高斯；无 WebGL 错误或浏览器运行异常。

汇总及当前发布文件 SHA256：`optimization_review/result.json`，`pass: true`。
历史回退记录继续留在 `out/gs_relight_stages/rollback/`；其中旧 hash 记录的是
回退当时状态，本次授权的增量修改另行记录，未覆盖历史证明。

## 适用范围与剩余问题

本轮验收的是功能保留、默认画面一致性及减少重复工作；可选 8K 只提高阴影纹理
分辨率，不证明日照的物理准确性。太阳仍采用原版 12:30–18:15 示意路径，
没有恢复真实建筑朝向、地点和日期。扫描几何、合成表面位置、灯具对齐及
关灯后墙面斑块问题保留原来的实验状态；没有将本轮结果标成物理真实性验收。

后续应在保留太阳控制和上述画面对照的基础上，分别检查遮挡几何与接收面的
误差，再评估光斑破碎问题。不要仅提高分辨率就宣称已修复几何或材质。

## Git 版本快照

`experiments/releases/nucleus4d_transport_v3_opt1/` 保存实际发布的页面、脚本、
元数据、构建模板、验证 JSON 和界面截图。`manifest.json` 记录其 SHA256，
并列出留在本地的模型、光场及几何资产的路径、大小和 SHA256。大文件继续遵守
现有 `.gitignore`，不将代码同步描述为模型/光场的云端备份。

整理提交时补齐了 v3 源码依赖，并修复启动器对未启用 staged app 的
`scene.json` 的依赖：选中 v3 时改查 v3 的 manifest，并创建缺失的日志目录。
渲染代码与已经逐像素验证的发布脚本保持一致；该启动修复不改变渲染结果。
