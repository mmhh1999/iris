# 太阳光斑 × IRIS 实验入口

> **2026-09-27:本页只记录 EXP0019 这一个实验。项目现状请看 [research/PROJECT_STATUS.md](research/PROJECT_STATUS.md)。**

文件已经位于 `/home/minghao/projects/iris`，对应 Windows：
`\\wsl.localhost\Ubuntu\home\minghao\projects\iris`。

- [打开交互结果](experiments/out/EXP0019_sunpatch/index.html)
- [完整实验报告](research/SUNPATCH_IRIS_BENCHMARK_ZH.md)
- [任务状态](research/task_state.json)

已完成 56 张受控 HDR 数据、三组 IRIS 训练与换光重渲染。
太阳 A 出现光斑污染反照率，太阳 B 未复现同样局部结果。
范围：单房间、已知几何、中性先验、每训练阶段 200 步；不等于 IRIS 普遍失败。
