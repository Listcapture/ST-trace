# ST-Trace 完整研究框架已经搭建完成

我已经按照 ST-trace 方案，完成了从论文idea到代码框架的完整实现，建立了**论文 → 实现 → 实验 → 分析 → 改进 → 论文**的闭环研究框架。

## 当前完成的内容

### 项目结构

```
ST-Trace/
├── st_trace/
│   ├── data/            # 数据加载和相机图
│   │   ├── graph.py          ✅ CameraGraph 相机拓扑图实现
│   │   ├── dataset.py        ✅ 基类 Detection, Trajectory, BaseMCTDataset
│   │   ├── nlpr_mct.py       ✅ NLPR_MCT 数据集加载器
│   │   ├── dukemtmc.py       ✅ DukeMTMC 数据集框架
│   │   ├── cityflow.py       ✅ CityFlow 数据集框架
│   │   └── transforms.py     ✅ 数据增强 (RandomErasing, 训练/验证变换)
│   ├── models/
│   │   ├── transition_net.py ✅ ST-ANBS 转移网络 (LSTM + MLP)
│   │   ├── reid/
│   │   │   ├── transreid.py  ✅ TransReID backbone 实现
│   │   │   └── st_contrastive.py ✅ 时空对比损失实现
│   │   └── detector.py      ✅ YOLOv8 检测器封装
│   ├── search/
│   │   ├── base_search.py    ✅ 搜索基类 SearchCandidate
│   │   ├── st_anbs.py        ✅ 核心 ST-ANBS 算法 + ExhaustiveBFS + FixedBeamSearch
│   │   ├── exhaustive.py     ✅ 穷搜基线导出
│   │   └── beam_search.py    ✅ 固定beam基线导出
│   ├── tracking/
│   │   ├── pipeline.py       ✅ 五阶段 coarse-to-fine 检索 pipeline
│   │   └── association.py    ✅ 轨迹关联 (Hungarian + greedy)
│   ├── evaluation/
│   │   ├── metrics.py        ✅ MOTA, IDF1, MT, ML 评估指标
│   │   └── efficiency.py     ✅ VRR, FPS, 计时分解效率指标
│   └── utils/ visualization/ # 占位，可后续填充
├── configs/                  ✅ YAML配置文件 (default, nlpr_mct, dukemtmc, cityflow)
├── scripts/                  ✅ 训练评估脚本
│   ├── download_data.sh
│   ├── preprocess.py
│   ├── train_transition.py
│   ├── train_reid.py
│   └── evaluate.py
└── 基础配置文件: requirements.txt, setup.py, .gitignore, README.md ✅
```

## 下一步执行流程 (按照您的论文->实现->实验->分析->改进->论文周期)

### Phase 1: 数据准备 (预计 1-2 周)

1. 运行 `./scripts/download_data.sh` 下载三个数据集
2. 运行 `python scripts/preprocess.py --config configs/nlpr_mct.yaml` 预处理标注

### Phase 2: 训练模型 (预计 2-3 周)

1. 训练 ST-ANBS 转移模型: `python scripts/train_transition.py --config configs/nlpr_mct.yaml`
2. 使用时空对比学习训练 ReID: `python scripts/train_reid.py --config configs/nlpr_mct.yaml`
3. 在 DukeMTMC 和 CityFlow 重复

### Phase 3: 完整评估 (预计 1-2 周)

1. 运行 `python scripts/evaluate.py` 得到完整结果
2. 运行所有 ablation studies (不同beam宽度, λ_ST, 组件去除)
3. 收集所有表格数据

### Phase 4: 结果分析与改进 (预计 1-2 周)

1. 对照论文中的 ablation table 分析每个组件贡献
2. 根据结果识别弱点，针对性改进超参数/模型结构
   * 如果准确率低 → 增加模型容量，调整正则化
   * 如果 VRR 太高 → 增加剪枝强度，减小beam宽度
   * 如果 IDF1 低 → 调整 λ_ST，改进负采样
3. 重新实验验证改进

### Phase 5: 论文完善

1. 将所有实验结果填入论文表格
2. 根据实验发现更新讨论/结论
3. 生成图表 (轨迹可视化，Pareto曲线，灵敏度分析)
4. 按照会议格式排版投稿

## 总预计时间:  **~3.5个月** ，符合CCF-B会议投稿时间线

## GitHub关联

你的GitHub仓库 [https://github.com/Listcapture/ST-trace](https://github.com/Listcapture/ST-trace) 可以直接push当前代码：

```bash
git remote add origin https://github.com/Listcapture/ST-trace.git
git add .
git commit -m "Initial commit: complete ST-Trace framework implementation"
git push -u origin main
```

框架已经完全按照论文中的算法设计实现，接下来可以直接开始训练实验阶段了。
