# 项目上下文：联邦多视图聚类

每次开始任务先阅读本文档。本项目已由 FedAS 监督分类代码重构为配置驱动的横向联邦多视图聚类；旧分类 Client/Server、分类模型和数据生成代码已经移除。

## 当前目标与边界

- 客户端持有互不重叠且数量尽量相等的样本子集；每个样本保留全部视图。
- 样本通过随机索引划分，不使用标签分层。
- MAT 中的标签字段（`Y`、`labels` 或 `gt`）只允许用于 ACC/NMI/ARI 评估和调优结果比较，不得进入训练损失、客户端划分、中心初始化或聚合。
- 服务端只聚合模型参数与客户端聚类中心摘要，不接收原始样本。
- 当前六个主数据集为 `ALOI_100、flower17、LandUse_21、Mfeat、NUSWIDE、Scene-15`；`HW.mat` 已移除且批量入口会跳过缺失数据。扩展数据集 `animal` 位于 `dataset/animal.mat`。不要恢复在线数据下载脚本或 HW 数据。

## 核心文件

```text
config/*.json                              主数据集独立调优配置（遗留 HW 配置无数据时跳过）
config/init/*.json                         主数据集统一初始参数配置
config/backup/*.json                       animal 调优配置
system/main.py                             单数据集入口
system/main_init.py                        统一初始参数单数据集入口
system/run_all.py                          全数据集批量入口
system/run_all_init.py                     统一初始参数全数据集批量入口
system/summarize_results.py                最佳轮、末轮变化与 Markdown 汇总
visualization/main.py                      单数据集训练及运行过程可视化入口
visualization/plot_history.py              ACC/NMI/ARI 与各项损失曲线绘制
system/config.py                           配置加载、验证与覆盖
system/utils/mat_data.py                   MAT 加载、归一化、无标签客户端划分
system/utils/clustering_metrics.py         Hungarian ACC、NMI、ARI
system/flcore/trainmodel/multiview.py       多视图自编码器、融合、聚类头和损失
system/flcore/clients/clientcluster.py      无监督本地训练与中心摘要
system/flcore/servers/servercluster.py      FedAvg、中心对齐/初始化、评估与保存
tests/test_clustering.py                    核心回归测试
```

## 模型与训练流程

1. 每个视图经独立 MLP 编码器得到嵌入并由解码器重构。
2. 视图嵌入先 L2 归一化，再用可学习 softmax 权重融合。
3. 聚类头以 Student-t 分布计算样本到可训练中心的软分配。
4. 第一阶段先执行 `representation_warmup_rounds` 轮表示热身，仅训练重构和一致性；默认值为 1。
5. 热身后每个客户端对融合嵌入做 KMeans，只上传中心和计数；服务端据此初始化全局中心。
6. 第一阶段余下轮次联合训练编码器与本地聚类头，DEC KL 和均衡损失权重线性增加；`pretrain_rounds` 表示进入正式聚类阶段的边界，`joint_learning_rate` 控制该阶段学习率。
7. 每个客户端每个通信轮开始时，用完整本地数据计算一次固定 DEC 目标 \(P\)，本轮所有本地 batch/epoch 共用，避免按 batch 即时目标造成伪标签抖动。
8. 正式阶段以完整损失权重继续联合优化；聚类头可通过 `cluster_head_learning_rate_multiplier` 使用独立学习率。
9. `warmup_local_epochs`、`joint_local_epochs` 和 `clustering_local_epochs` 可分别控制三个阶段的客户端本地 epoch；缺省时均继承 `local_epochs`。命令行覆盖基础 `local_epochs` 或 `learning_rate` 时，未显式配置的阶段参数必须同步继承新值，其中联合学习率默认为新基础学习率的 0.1 倍、正式聚类学习率默认等于新基础学习率；显式阶段配置或阶段覆盖优先。
10. 服务端对普通参数按样本数 FedAvg；本地中心和软簇计数先按 Hungarian 同步排列，再按每个簇的软计数融合，可选 `center_momentum`。
11. 按 NMI 只保留正式聚类阶段的最佳检查点，同时报告 ACC/NMI/ARI。

## 数据格式

MAT 文件通常包含：

- `X`：形状通常为 MATLAB cell，每个元素为 `N × D_v`；加载器也会识别并转置 `D_v × N`。
- `Y`：长度为 `N`，加载时重映射为从 0 开始的连续整数；加载器也接受同义字段 `labels` 和 `gt`。

`NUSWIDE.mat` 使用同义字段 `data/labels`，且各视图以 `D_v × N` 保存；`animal.mat` 使用 `X/gt`。加载器会统一转为 `N × D_v`。

数据由 `system/utils/mat_data.py` 按配置分别做 `standard`、`minmax`、`l2` 或不归一化。不得修改原始 MAT 文件。

## 环境与命令

默认 Python 环境：`C:\Users\29101\.conda\envs\torch_251_118_39`。如果用户没有另行指定 Python 环境，所有运行、测试和依赖安装都默认使用该环境。已验证 Python 3.9.23、PyTorch 2.5.1、CUDA、SciPy 1.13.1、scikit-learn 1.6.1 可用。缺包时优先使用 Conda 安装到此环境，不要安装到系统 Python。

```powershell
# 单数据集
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config ALOI_100

# 全部数据集
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all.py

# 临时覆盖参数
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config Scene-15 --override training.rounds=5 --override training.pretrain_rounds=3

# 测试
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' -m unittest discover -s tests -v

# 训练并可视化
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' visualization/main.py --config Scene-15

# 仅可视化已有结果
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' visualization/main.py --config Scene-15 --skip-training
```

## 输出与验证

- 正式输出：`results/<dataset>/{summary.json,history.json,model.pt}`。
- 可视化入口实验输出：`visualization/result/<dataset>/{summary.json,history.json,model.pt}`。
- 可视化输出：`visualization/output/<dataset>/{clustering_metrics.png,training_losses.png}`。
- 指标图和损失图分别标出联合预训练与正式聚类的开始边界，完整呈现表示热身、联合预训练、正式聚类三个阶段。
- 统一初始参数输出：`results/init_result/<dataset>/{summary.json,history.json,model.pt}`。
- 两套批量训练分别生成 `results/summary.md` 与 `results/init_result/summary.md`；指标变化值为“末轮减最佳轮”。
- 稳定性调参正式输出：`results-3阶段/<dataset>/{summary.json,history.json}`；覆盖六个主数据集和 animal，汇总见 `results-3阶段/summary.md`，调参对比见 `results-3阶段/tuning_summary.md`。
- 稳定方案要求最佳轮到末轮的 `|ΔACC|`、`|ΔNMI|`、`|ΔARI|` 均不超过 0.01；未选中的候选保存在 `results-3阶段/tuning/`。
- 训练终端按总通信轮数约 10% 的间隔打印进度，并始终打印最后一轮；历史仍逐轮记录。
- Scene-15 当前配置为 30 轮、14 轮表示热身、第一阶段边界 22 轮；热身阶段每客户端 2 个本地 epoch，联合和正式阶段各 3 个。联合与正式阶段学习率均为 `5e-8`、`center_momentum=0.99875`、聚类/均衡权重 0.05/0。固定 `seed=42` 的正式阶段最佳轮为第 23 轮，ACC/NMI/ARI 为 `0.449052/0.437947/0.282406`，到第 30 轮三项保持不变。
- 改动至少运行单元测试和一个两轮冒烟测试；涉及共享训练逻辑时应复验六份主配置及 animal 配置，HW 不在复验范围内。
- 聚类结果受随机种子、客户端划分和 GPU 数值差异影响；报告指标时同时注明配置、seed 和最佳轮次，不能声称理论或全局最优。
- 配置调优采用相同 seed/客户端数比较，选择指标为 NMI。

## 维护要求

- 算法、配置、数据格式、运行命令、依赖或结果语义变化时同步更新 README 和 AGENTS.md。
- 不要将 MAT 数据、模型检查点、临时调优目录或 `__pycache__` 当作源码提交。
