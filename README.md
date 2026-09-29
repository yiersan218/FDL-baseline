# Federated Multi-View Clustering

本项目已由监督式个性化联邦分类重构为横向联邦多视图聚类。每个客户端持有互不重叠的样本子集，每个样本保留全部视图；真实标签只用于 ACC、NMI 和 ARI 评估，不进入客户端损失、样本划分或服务端聚合。

## 方法

- 每个视图使用独立 MLP 自编码器。
- 归一化后的视图表示通过可学习视图权重融合。
- 聚类头维护可训练聚类中心，使用 Student-t 分布产生软分配。
- 阶段一先执行 `representation_warmup_rounds` 轮表示热身，只优化重构和跨视图一致性。
- 热身后客户端上传本地 KMeans 中心与计数摘要，服务端初始化全局中心，不上传原始样本或单样本嵌入。
- 阶段二在 `pretrain_rounds` 边界前同时训练编码器和本地聚类头，DEC 与均衡损失权重从小到大线性增加。
- 每个客户端在每个通信轮开始时，用完整本地数据生成一次固定 DEC 目标分布；不再按 mini-batch 即时重算伪标签目标。
- 阶段三从 `pretrain_rounds + 1` 轮开始，使用完整权重联合优化重构、一致性、DEC 自训练 KL 和簇均衡损失。
- 服务端对普通模型参数执行样本数加权 FedAvg；聚类中心先经 Hungarian 对齐，再按各簇软计数融合。

## 数据集

项目当前直接读取 `dataset/` 下的六个主 MAT 文件；`HW.mat` 已移除，不再参与批量运行：

| 配置名 | 样本数 | 视图维度 | 聚类数 |
|---|---:|---|---:|
| ALOI_100 | 10800 | 77, 13, 64, 125 | 100 |
| flower17 | 1360 | 1360 × 7 | 17 |
| LandUse_21 | 2100 | 20, 59, 40 | 21 |
| Mfeat | 2000 | 216, 76, 64, 6, 240, 47 | 10 |
| NUSWIDE | 5000 | 65, 226, 145, 74, 129 | 5 |
| Scene-15 | 4485 | 20, 59, 40 | 15 |

`dataset/animal.mat` 是另一个已完成三阶段调参的扩展数据集：

| 配置路径 | 样本数 | 视图维度 | 聚类数 |
|---|---:|---|---:|
| `config/backup/animal.json` | 10158 | 4096, 4096 | 50 |

MAT 文件通常包含 `X/Y`；`NUSWIDE.mat` 使用同义字段 `data/labels`，`animal.mat` 使用 `X/gt`。视图字段为多视图密集矩阵集合，标签仅用于评估。项目不再包含在线下载或数据生成脚本。

## 环境

验证环境：

```text
C:\Users\29101\.conda\envs\torch_251_118_39
Python 3.9.23
PyTorch 2.5.1 + CUDA
SciPy 1.13.1
scikit-learn 1.6.1
Matplotlib
```

## 运行

从项目根目录执行单个数据集：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config ALOI_100
```

运行全部可用主数据集（缺少 MAT 文件的遗留配置会被跳过）：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all.py
```

使用统一初始参数运行单个或全部可用主数据集：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main_init.py --config ALOI_100
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all_init.py
```

统一初始参数位于 `config/init/`。除数据集名称、MAT 路径和聚类数外，各配置使用相同参数：5 个客户端、standard 归一化、`[256, 128]` 隐藏层、64 维嵌入、20 轮训练（前 8 轮为预训练区间，其中默认第 1 轮表示热身）、2 个本地 epoch、batch size 128、`1e-3` 表示学习率和默认 `1e-4` 联合学习率。

三阶段训练参数由 `system/config.py` 提供默认值，也可以临时覆盖：

- `training.representation_warmup_rounds=1`：第一阶段开始时仅训练表示的轮数；之后初始化中心并联合训练编码器和聚类头。
- `training.warmup_local_epochs`、`training.joint_local_epochs`、`training.clustering_local_epochs`：分别控制表示热身、联合预训练和正式聚类阶段每个客户端的本地 epoch；未设置时均沿用 `training.local_epochs`。
- `training.joint_learning_rate=1e-4`：联合预训练阶段的基础学习率；默认是表示学习率的 0.1 倍。
- `training.cluster_head_learning_rate_multiplier=1.0`：聚类头学习率相对当前阶段基础学习率的倍数。
- `training.center_momentum=0.0`：服务端中心融合动量，取值范围为 `[0, 1)`。

命令行覆盖会保留上述继承关系：覆盖 `training.local_epochs` 时，配置文件中未显式设置的三个阶段 epoch 会同步更新；覆盖 `training.learning_rate` 时，未显式设置的 `joint_learning_rate` 会重新计算为新值的 0.1 倍，未显式设置的 `clustering_learning_rate` 会继承新的基础学习率。配置文件或同一命令行中显式指定的阶段参数保持独立，并优先于基础参数。

联合预训练期间，聚类损失与均衡损失的有效权重会从第一个联合轮的
`1 / (pretrain_rounds - representation_warmup_rounds)` 线性增长到 1。

临时覆盖配置：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py `
  --config Scene-15 `
  --override training.rounds=5 `
  --override training.pretrain_rounds=3
```

训练并绘制逐轮 ACC、NMI、ARI 及总损失、重构损失、一致性损失、聚类损失和均衡损失：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' visualization/main.py --config Scene-15
```

可视化已有结果而不重复训练：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' visualization/main.py `
  --config Scene-15 `
  --skip-training
```

图片默认保存到 `visualization/output/<dataset>/clustering_metrics.png` 和
`visualization/output/<dataset>/training_losses.png`。可通过 `--plot-directory`、
`--format png|pdf|svg` 和 `--dpi` 调整输出。
曲线会分别标出 `Joint pretraining starts` 与 `Formal clustering starts`，
用于区分表示热身、联合预训练和正式聚类三个阶段。

从 `visualization/main.py` 启动的实验，其 JSON 和模型默认保存在：

```text
visualization/result/<dataset>/summary.json
visualization/result/<dataset>/history.json
visualization/result/<dataset>/model.pt
```

可通过 `--results-directory` 指定其他结果根目录。

结果保存到 `results/<dataset>/`：

- `summary.json`：最终指标、最佳轮次和完整配置。
- `history.json`：逐轮训练损失及聚类指标。
- `model.pt`：最佳 NMI 轮次的模型参数。

训练终端按总通信轮数约 10% 的间隔显示一次进度，并始终显示最后一轮；
逐轮训练、评估以及 `history.json` 记录不受打印间隔影响。

统一初始参数的结果保存到 `results/init_result/<dataset>/`。两个批量入口训练结束后分别生成 `results/summary.md` 和 `results/init_result/summary.md`，其中变化值定义为“最后一轮指标减最佳轮指标”，负数表示后期下降。

本轮稳定性调参结果保存在 `results-3阶段/<dataset>/`，覆盖六个主数据集以及 animal。`results-3阶段/summary.md` 汇总最佳轮与末轮差异，`results-3阶段/tuning_summary.md` 记录初始值、选定参数与提升幅度，未选中的候选保留在 `results-3阶段/tuning/`。稳定标准为最佳轮到末轮的 `|ΔACC|`、`|ΔNMI|`、`|ΔARI|` 均不超过 0.01。

Scene-15 当前正式配置为 30 轮：前 14 轮表示热身，第 15～22 轮联合训练编码器和聚类头，第 23～30 轮正式训练。热身阶段每客户端 2 个本地 epoch，后两个阶段各 3 个；后两个阶段均使用 `5e-8`，中心融合动量为 0.99875，聚类/均衡损失权重为 0.05/0。固定 `seed=42` 的正式复验在第 23 轮达到 ACC/NMI/ARI `0.449052/0.437947/0.282406`，到第 30 轮三项保持不变。

## 测试

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' -m unittest discover -s tests -v
```

六个主数据集的独立参数位于 `config/`，备用大数据集 animal 的参数位于 `config/backup/`。
