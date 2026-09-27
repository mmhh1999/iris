# Cali-HDR / Pano2Pano（Guanzhou Ji et al.）数据集获取与 EXIF 审计

状态：EXP0022（清单+EXIF 审计）、EXP0023（真实 EXIF 星历交叉检验）均已完成。
数据搬运（169GB 原始 RAW/JPG 拷贝+解压）在后台进行，见文末"数据落地状态"。
本文档只记录**数据工程与元数据事实**，不包含任何 BRDF/太阳先验方法优劣的结论。

## 背景

`research/DATASET_AUDIT.md` 此前已将 **Cali-HDR dataset**（Ji, Sawyer,
Narasimhan, ISVC 2023, "Virtual Home Staging"）列为候选数据集，评分 3/10，
主要卡点是"only available upon request"以及"单视点全景图，无多视角几何"。
`PROJECT_STATUS.md` 记录了"已保存官方说明，未发邮件"的状态。

用户本次获得了两个数据集的完整下载（作者 Guanzhou Ji，CMU/Narasimhan 组，
现执教 Northeastern）：

- **Cali-HDR Dataset.zip**（88.97 GB）：论文原始数据集，137 张室内全景 +
  配对室外鱼眼，14 个场景（按日期命名的文件夹，2022-10 至 2023-07）。
- **Pano2Pano_Release.zip**（80.21 GB）：同一作者更新/扩展的后续数据集
  （网络检索确认为同一课题组 2024 年数据，141 对室内外全景的量级），
  9 个场景（2024-02 至 2024-05）。

两者均来自 `/mnt/c/Users/XMH/Downloads/`（Windows 侧），已用 EXP0022 的脚本
以只读方式（zip 中心目录 + 单文件流式读取）审计，**未修改或删除原始文件**。

## EXP0022 — 归档清单 + EXIF 审计（不解压全部数据）

脚本：`experiments/audit_ji_hdr_datasets.py`。
方法：只用 `unzip -l`/`zipfile` 读取中心目录得到 4882+3402 个条目；对每个
`.jpg` 条目仅流式读取前 256KiB（覆盖 RICOH THETA Z1 / Ricoh GR 相机的
EXIF+缩略图段），用 PIL 解析 EXIF，不下载/解压 DNG/CR2 大文件。
证据：`research/evidence/EXP0022/exif_inventory.csv`（逐图）、
`exif_inventory_summary.json`（逐场景聚合）。

**关键发现（数据事实，非方法结论）：**

1. 文件类型：Cali-HDR = 2286 JPG + 1269 DNG + 1017 CR2；Pano2Pano = 1593 JPG
   + 1593 DNG（室内外均为 JPG+DNG 配对，比 Cali-HDR 的室外鱼眼仅 CR2 更完整）。
   体积主要来自未合并的多曝光 RAW 包围曝光序列（HDR 合成前的原始输入）。
2. **EXIF 中直接带有 GPS + 绝对时间戳 + 罗盘朝向**（`GPSImgDirection`），
   无需任何街道地址地理编码：3879 张 JPG 中 2853 张（73.6%）含 GPS，
   2169 张（55.9%）含相机罗盘朝向。坐标全部落在匹兹堡地区
   （约 40.36–40.46°N, -79.90–-80.00°W），与 CMU/Narasimhan 组的实地采集
   一致（数据显示如此，而非事先假设）。
3. Cali-HDR 的室外"fisheye"文件夹（Canon 相机，仅 CR2）**没有** GPS/朝向
   EXIF；Pano2Pano 的室外文件夹（改用同款 Ricoh 相机）则**有**完整 GPS+朝向。
   Pano2Pano 在这一点上是比 Cali-HDR 更强的验证数据源。
4. 意外发现：Cali-HDR 压缩包内混入了一个开发目录残留
   （`20221006/theta/.idea/...`、一个内嵌 `.zip`、一个 `.gitignore`），
   应是作者打包时的疏漏，非数据本身，解压后需忽略，不影响审计结论。

## EXP0023 — 真实 EXIF 星历交叉检验（纯计算，不接触像素）

脚本：`experiments/report_ji_hdr_datasets.py`，复用已在合成数据上验证过的
`utils/solar_geometry.py::solar_position`（pvlib 后端，`PHASE_A_SYNTHETIC.md`
中 2.5° 角误差通过的同一实现）。对 2853 张有 GPS+UTC 时间戳的图像逐一计算
真实太阳方位角/高度角，并与相机罗盘朝向做夹角比较。
证据：`research/evidence/EXP0023/ephemeris_crosscheck_summary.json`、
`ephemeris_crosscheck_per_image.csv`。

**结果：**

- 35 个场景、2853 张图像全部完成星历计算。
- **0 个**标记为室外的场景出现"白天拍摄但星历判定太阳在地平线以下"的
  不一致（即无时区换算错误或夜拍异常的迹象）——这是一个真实数据上的
  自洽性检验通过，而非方法优势声明。
- 高度角范围（如 20230528 一天内 67.0–70.8°、20221006 一天内 18.9–44.3°）
  与匹兹堡纬度、对应月份/时刻在物理上吻合。
- 相机朝向与太阳方位角夹角（`camera_to_sun_angle_deg`）在场景间有明显差异
  （如 20230528 平均 10.3° "对着太阳"，20230519 平均 81.2° "侧对太阳"），
  为后续挑选"逆光/顺光"对比场景提供了现成的筛选依据。

**这解锁了什么，以及仍然没有解锁什么：**

- 解锁：`DATASET_AUDIT.md`"Open scientific questions"第一条（星历预测是否
  匹配真实观测）第一次有了**真实、非合成、非用户自采**的 GPS+时间戳数据源
  可用于 Mode A 校验，且精度（GPS 到角秒、时间到秒）优于人工采集协议原计划
  的手机 EXIF/手动记录。这填补的是"照度/朝向的物理自洽性"验证缺口。
- 未解锁：`DATASET_AUDIT.md` 的核心结论——"没有数据集同时具备多视角几何+
  真实窗口+真实太阳斑"——依然成立。Cali-HDR/Pano2Pano 是单视点全景，没有
  重建网格、没有多视角位姿，**不能替代 Phase 0 实地采集**去做完整的
  material-illumination 解耦评测（killer experiment）。它们能做的是更早、
  更便宜地验证"星历预测的太阳位置/朝向关系是否物理合理"这一个组件，以及
  （下一步)"预测的太阳方向是否与照片里实际的太阳斑/顺逆光外观一致"。

## 尚未做、留给下一次会话决策的部分

刻意没有在本次自动化中做的事，及原因：

1. **像素级太阳斑视觉核验**（预测太阳方位 vs. 照片里实际的高光/阴影位置）
   需要解码全景图并做太阳斑检测，属于新的方法性判断（如何在全景投影下
   定义"预测位置"），而不是纯数据搬运，留给用户醒来后决定优先级和框架，
   而非在无监督时段仓促写入 evidence。已提取 2 张代表性预览图
   （`research/evidence/EXP0023/preview_images/`，Pano2Pano
   `20240331/6236 5th ave` 室内+室外各一张，方位角差 137°/145° 的顺光对比
   样本）供下次会话直接使用，不需要重新解压。
2. 未对数据集中出现的真实门牌地址（如"6236 5th ave"）做任何反向地理编码
   或身份关联——EXIF 自带的 GPS 已经足够精确，不需要、也不应该额外把地址
   字符串和坐标做二次核对/扩展查询。
3. 未删除或改动 `/mnt/c/Users/XMH/Downloads/` 下的原始 zip
   （用户原话"移动过来"，但两文件共 169GB 且传输耗时数十分钟，为避免任何
   传输中断导致数据丢失，选择**复制**而非**移动**；确认解压校验通过后本
   会话只删除了本地的 zip 拷贝，Windows 原始文件未动）。

## 数据落地状态 — 已完成（2026-09-23 04:48 UTC）

后台任务 `data_download/import_ji_datasets.sh`（以 `nohup ... & disown`
方式启动为独立进程，不依赖对话存活）已完整跑完并逐字节校验通过：

| 数据集 | 拷贝校验 | 解压后文件数 | 解压后总字节数 | 与 zip 元数据比对 |
|---|---|---|---|---|
| `data_download/cali_hdr/extracted/` | 88,968,381,364 字节，与源 zip 一致 | 4580 | 88,967,371,386 | 与 zip 内 4580 个真实文件条目的 `file_size` 之和逐字节相等 |
| `data_download/pano2pano/extracted/` | 80,211,412,643 字节，与源 zip 一致 | 3187 | 80,210,671,305 | 与 zip 内 3187 个真实文件条目的 `file_size` 之和逐字节相等 |

**过程中的一个插曲**：两个压缩包解压时 `unzip` 命令行工具都报错退出
（`mapname: conversion of  failed`）——原因是两个 zip 内部都有一条字面量为
`/` 的根目录条目，`unzip` 的路径映射逻辑处理不了它，且不是"警告后继续"而是
直接中止解压（cali_hdr 因此只解压出 4580/4882 个条目就停了，pano2pano
同理）。用 Python `zipfile` 模块重新解压（跳过这一条空名条目，其余逐一正常
写入）后两个数据集都做到了字节数与 zip 自身元数据完全一致。已经把
`import_ji_datasets.sh` 里的解压步骤换成同样的 Python 方式，供以后重跑。

本地 zip 拷贝已在校验通过后删除；`/mnt/c/Users/XMH/Downloads/` 下的两个
原始文件全程未被读写以外的方式触碰。磁盘峰值占用约 553GB（共 1007GB），
完成后回落到 478GB 已用 / 478GB 可用。

## 相关文件

- 清单脚本：`experiments/audit_ji_hdr_datasets.py`（EXP0022）
- 星历交叉检验脚本：`experiments/report_ji_hdr_datasets.py`（EXP0023）
- 搬运脚本：`data_download/import_ji_datasets.sh`
- 证据：`research/evidence/EXP0022/`、`research/evidence/EXP0023/`
- 更新：`research/DATASET_AUDIT.md`（Cali-HDR 行状态、新增 Pano2Pano 行）、
  `research/EXPERIMENTS.md`（EXP0022/EXP0023）、`research/DECISIONS.md`（D0012）

## 2026-09-24 更新：WSL 内已删除 RAW

为节省 WSL 磁盘，已删除 `data_download/cali_hdr/` 与 `data_download/pano2pano/`
中全部 DNG/CR2（约 151GB），仅保留 JPG（约 17GB）。原始 zip 仍在
`/mnt/c/Users/XMH/Downloads/`，需要 RAW（如重新合成 HDR）时从 zip 中按需解压。
