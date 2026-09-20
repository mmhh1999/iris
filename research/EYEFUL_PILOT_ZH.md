# Eyeful Tower 真实 HDR 首批接入（EXP0014）

2026-09-20。用户授权自行下载并推进，已完成两套真实空间数据的首批下载、
图像预处理、网格投影和可视化；不是完整数据集下载或材质恢复结果。

## 实际获得

- riverview、apartment 各 32 张 1K 线性 HDR EXR，以及对应官方 JPEG。
- 两个房间的完整 OBJ 网格、纹理、MTL、全部相机标定和官方划分。
- 每个房间 24 张训练、8 张测试，测试保留官方 camera 17。
  训练相机 19/20/21，帧均匀抽样；未根据测试图片表现调整训练划分。
- 官方文档、许可、来源 URL、下载长度和 SHA-256 已保存，最终重新校验所有文件。

## 检查与选择

64 张 HDR 均可解码且数值有限。按官方列优先矩阵读取 K 和世界到相机 T；
进行针孔径向/切向去畸变、保存有效像素掩码，并检查投影往返误差。
原始 HDR 不施加白平衡、不裁剪、不逐图归一化，保留官方线性 DCI-P3 数值。
显示预览采用官方白平衡与显示曲线。IRIS 的 RGB/CRF 约定仍需在接入训练时明确转换，
不能把这些 DCI-P3 数值直接当作线性 sRGB。

在 Mitsuba 上实际加载原网格并对全部 64 个视角发射相机射线，
重投影扫描纹理及法线。首轮因 NumPy scalar / Mitsuba 类型不兼容失败，
显式转换后完整重跑成功。纹理投影不是新光照渲染。

**选择 riverview 为太阳实验主场景**：真实照片有清楚的窗框阴影、地面日照斑，
几何投影中地面和家具基本对应；窗外及部分远处区域缺少几何。
网格约 193 万三角形，32 个视角平均射线命中率约 74.6%，该数值包含窗外区域，
不是几何准确率。太阳入射涉及玻璃和外部环境，后续需要检查与建模。

apartment 网格约 176 万三角形，平均命中率约 99.5%；图片主要呈混合照明，
首批视角没有 riverview 那样清晰的窗框直射斑，保留为第二种条件，不能凭命中率判定质量更好。

## 仍需完成

1. 区分窗框、玻璃、外部缺失区域，明确透射与环境照明表示。
2. 接入 IRIS 并进行共同输入的材质/光照分解，正确处理色彩空间、曝光与掩码。
3. 建立受控太阳重渲染条件，生成端材质仅称为赋予/估计的材质，不称实测真值。
4. 照片输入恢复太阳与材质，对比材料误差和未见光照预测。

HDR 不自动等于绝对光度标定或未饱和的太阳圆盘。原网格纹理带有阴影，
不能直接作为纯反照率。使用官方网格/相机的阶段必须标注为已知重建几何条件，
不能称为严格仅照片输入。

## 其他两套数据

已保存 TexIR 官方说明/配置和 Cali-HDR 官方说明/数据摘要。
TexIR 下载需向 wanglingli008@realsee.com 或 pancihui001@realsee.com 申请，
需要姓名、机构和接收链接邮箱；Cali-HDR 需联系 g.ji@northeastern.edu。
当前没有这些数据的下载链接，未发送任何邮件，也未将它们标为已下载。

## 复现

```bash
.venv/bin/python experiments/fetch_eyeful.py
.venv/bin/python experiments/prepare_eyeful.py
LD_LIBRARY_PATH=$PWD/.toolchain/compiler/lib .venv/bin/python experiments/check_eyeful_geometry.py
.venv/bin/python experiments/report_eyeful.py
```

预览：`experiments/out/EXP0014_eyeful/index.html`。
证据：`research/evidence/EXP0014/summary.json`、逐文件下载清单与脚本快照。
官方来源：https://github.com/facebookresearch/EyefulTower 。
