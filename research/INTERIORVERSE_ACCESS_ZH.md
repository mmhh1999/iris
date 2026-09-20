# InteriorVerse 下载与用途核验（EXP0015）

2026-09-20。用户提供的旧下载清单 URL 实际返回 HTTP 404。
已检查作者最新官方仓库：2026/08 更新指向 Lez/InteriorVerse 完整备份；
2026/04 的另一个恢复版本只有部分数据。两者不能混为一谈。

来源：
- https://github.com/jingsenzhu/IndoorInverseRendering
- https://huggingface.co/datasets/Lez/InteriorVerse
- https://github.com/jingsenzhu/IndoorInverseRendering/tree/main/interiorverse

## 实际完成

从作者指向的备份下载 `dataset_85/part_0.zip`，2,112,042,290 字节。
SHA-256 与 Hugging Face LFS 公布的对象校验值一致，全部 ZIP 条目通过 CRC。
共 100 个场景、5,988 个 EXR 文件；没有非 EXR 文件。
原始压缩包及下载元数据保存在 `data_download/interiorverse/`，不提交 Git。

抽取按场景名排序的前三个场景，共 26 个视角（21 train、5 val），
六种标注全部可解码，有效像素内数值有限；粗糙度和金属度在 [0,1]。
此次仅做格式/质量检查，没有训练模型或计算方法优劣。
原始 split 清单已取得且三者不交叉，但没有验证整个 222 GB 备份的场景覆盖，
因此不能宣称完整官方评测集已准备好。

## 字段含义

- `im.exr`：HDR 渲染图；预览按官方 clip + gamma，仅用于显示。
- `albedo.exr`：反照率真值。
- `material.exr`：RGB 的 R=roughness、G=metallic、B 不使用。
  OpenCV 读入顺序是 BGR，已转换。GGX 参数与 IRIS 的具体映射仍需核验。
- `depth.exr`：毫米单位，无效位置允许 inf，必须结合 mask。
- `normal.exr`：OpenGL 相机坐标，x 右、y 上、z 朝向观察者。
- `mask.exr`：有效像素。

## 对项目的结论

这是人工设计的合成场景材质/几何标注库，适合作为材质估计预训练或评估候选，
不替代 Eyeful Tower 的真实空间主线。

检查的分片没有 mesh、相机外参、场景文件、环境贴图或显式太阳参数。
不能从“有深度/法线图”推断为“有完整可重渲染房间”。
官方 README 目前仍将 spatially-varying lighting 部分列为未发布。
不能声称已拿到完整太阳/光照真值数据集，也不能把原彩图直接用作反照率。

旧 DATASET_AUDIT 中关于“仅有非官方备份、不可可靠获取”的判断已被作者
2026/08 的备份指向更新；原云链接失效仍属实。网页/邮件的旧说明落后于官方仓库。

## 复现首批检查

1. 从上述官方仓库保存最新 README 与 `interiorverse/README.md`。
2. 使用 Hugging Face API 获取 `datasets/Lez/InteriorVerse/tree/main/dataset_85`，
   保存为 `data_download/interiorverse/dataset_85_tree.json`。
3. 从同一仓库下载 `dataset_85/{train,val,test}.txt`，保存于上述本地目录。
4. 下载 `dataset_85/part_0.zip` 至该目录，运行：

```bash
python experiments/audit_interiorverse.py
```

检查器会核验整包 SHA-256 与 CRC 后只解压指定样本，输出逐视角数据审计和材质对照图。
详细记录：`research/evidence/EXP0015/`。
