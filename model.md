# 两阶段联邦多视图聚类模型

## 1. 问题定义

数据包含 \(V\) 个视图和 \(N\) 个样本：

\[
\mathcal{X}=\{X^{(1)},X^{(2)},\ldots,X^{(V)}\},
\quad X^{(v)}\in\mathbb{R}^{N\times D_v}.
\]

样本通过随机索引划分到 \(M\) 个客户端，各客户端样本集合互不重叠，每个样本保留全部视图。标签只用于最终评估，不进入任何训练步骤。

## 2. 多视图表示学习

每个视图配置独立的 MLP 编码器和解码器：

\[
z_i^{(v)}=f_v(x_i^{(v)}),
\qquad
\hat{x}_i^{(v)}=g_v(z_i^{(v)}).
\]

重构损失为：

\[
\mathcal{L}_{rec}
=\frac{1}{V}\sum_{v=1}^{V}
\operatorname{MSE}(\hat{X}^{(v)},X^{(v)}).
\]

编码结果先进行 L2 归一化，再使用可学习视图权重融合：

\[
a_v=\frac{\exp(w_v)}{\sum_{j=1}^{V}\exp(w_j)},
\qquad
h_i=\operatorname{norm}\left(
\sum_{v=1}^{V}a_v\operatorname{norm}(z_i^{(v)})
\right).
\]

跨视图一致性损失约束各视图表示接近融合表示：

\[
\mathcal{L}_{con}
=\frac{1}{V}\sum_{v=1}^{V}
\operatorname{MSE}(\operatorname{norm}(Z^{(v)}),H).
\]

## 3. 聚类头

聚类头维护 \(K\) 个可训练中心 \(\mu_j\)。样本到中心的 Student-t 软分配为：

\[
q_{ij}=
\frac{(1+\lVert h_i-\mu_j\rVert^2/\alpha)^{-(\alpha+1)/2}}
{\sum_{k}(1+\lVert h_i-\mu_k\rVert^2/\alpha)^{-(\alpha+1)/2}}.
\]

DEC 目标分布为：

\[
p_{ij}=
\frac{q_{ij}^2/\sum_i q_{ij}}
{\sum_k(q_{ik}^2/\sum_i q_{ik})}.
\]

聚类损失使用 KL 散度：

\[
\mathcal{L}_{KL}=KL(P\Vert Q).
\]

簇均衡损失为：

\[
\mathcal{L}_{bal}
=\sum_{j=1}^{K}\bar q_j\log(K\bar q_j),
\qquad
\bar q_j=\frac{1}{N}\sum_i q_{ij}.
\]

## 4. 总损失

第 \(t\) 轮的总损失为：

\[
\mathcal{L}^{(t)}
=\lambda_{rec}\mathcal{L}_{rec}
+\lambda_{con}\mathcal{L}_{con}
+s_t\lambda_{clu}\mathcal{L}_{KL}
+s_t\lambda_{bal}\mathcal{L}_{bal}.
\]

其中 \(s_t\) 是聚类损失调度系数。中心初始化前 \(s_t=0\)；初始化后在预训练阶段线性增加；正式聚类阶段固定为 1。

## 5. 阶段一：聚类感知预训练

阶段范围为：

\[
1\le t\le T_{pre},
\]

其中 \(T_{pre}=\texttt{pretrain\_rounds}\)。整个区间在历史中统一记录为 `pretraining`。

### 5.1 中心初始化前

前 \(T_c=\texttt{center\_init\_round}\) 轮使用：

\[
\mathcal{L}=\lambda_{rec}\mathcal{L}_{rec}
+\lambda_{con}\mathcal{L}_{con}.
\]

该过程属于阶段一内部的表示学习部分，不是独立训练阶段。

### 5.2 联邦中心初始化

完成第 \(T_c\) 轮后，每个客户端对完整本地融合表示执行 KMeans，上传本地中心和簇计数。服务端对全部摘要执行带权 KMeans，初始化全局聚类中心。原始样本和单样本表示不会离开客户端。

### 5.3 中心初始化后

从第 \(T_c+1\) 轮至第 \(T_{pre}\) 轮启用聚类目标，其权重为：

\[
s_t=\frac{t-T_c}{T_{pre}-T_c}.
\]

若分母为 0，则正式聚类阶段直接使用 \(s_t=1\)。中心初始化后使用 `pretraining_end_learning_rate`，但 phase 仍为 `pretraining`。

每个客户端在通信轮开始时，用完整本地数据计算一次目标分布 \(P\)，并在本轮所有本地 epoch 和 batch 中固定使用，减少伪标签抖动。

## 6. 阶段二：正式聚类

阶段范围为：

\[
T_{pre}<t\le T.
\]

历史记录为 `clustering`，聚类权重固定为 \(s_t=1\)，学习率切换为 `clustering_learning_rate`。编码器、解码器、视图权重和聚类中心继续联合优化。

模型只在该阶段按配置的选择指标保存最佳状态，当前选择指标为 NMI。

## 7. 联邦聚合

### 7.1 普通参数

客户端 \(m\) 拥有 \(n_m\) 个样本，普通参数采用 FedAvg：

\[
\theta=\sum_m\frac{n_m}{\sum_j n_j}\theta_m.
\]

### 7.2 聚类中心

不同客户端的簇编号没有固定语义。服务端先根据中心距离使用 Hungarian 算法将本地中心排列到全局中心顺序，然后按每个簇的软计数聚合：

\[
\mu_k^{agg}=
\frac{\sum_m c_{mk}\mu_{mk}}
{\sum_m c_{mk}}.
\]

可选中心动量为：

\[
\mu_k\leftarrow
\rho\mu_k^{old}+(1-\rho)\mu_k^{agg}.
\]

## 8. 两阶段训练算法

```text
初始化全局模型
for round = 1 ... rounds:
    phase = pretraining if round <= pretrain_rounds else clustering

    if round == center_init_round + 1:
        客户端计算本地 KMeans 中心与计数
        服务端初始化全局中心

    if 中心尚未初始化:
        clustering_scale = 0
    elif phase == pretraining:
        clustering_scale 线性增加到 1
    else:
        clustering_scale = 1

    客户端基于完整本地数据固定本轮 DEC 目标
    客户端执行对应阶段的本地训练
    服务端聚合普通参数和对齐后的聚类中心
    评估 ACC、NMI、ARI

    if phase == clustering:
        按 NMI 更新最佳检查点
```

## 9. 参数语义

| 参数 | 含义 |
|---|---|
| `rounds` | 总通信轮数 |
| `pretrain_rounds` | 阶段一结束轮次 |
| `center_init_round` | 阶段一内部中心初始化时点 |
| `learning_rate` | 中心初始化前预训练学习率 |
| `pretraining_end_learning_rate` | 中心初始化后预训练学习率 |
| `clustering_learning_rate` | 阶段二学习率 |
| `pretraining_local_epochs` | 阶段一本地 epoch |
| `clustering_local_epochs` | 阶段二本地 epoch |
| `cluster_head_learning_rate_multiplier` | 聚类头学习率倍数 |
| `center_momentum` | 聚类中心融合动量 |

## 10. 评估与稳定性

- ACC 使用 Hungarian 匹配后的聚类准确率。
- NMI 为模型选择指标。
- ARI 衡量成对聚类一致性。
- 固定 `seed=42` 和客户端数量进行参数比较。
- 稳定方案要求最佳轮到末轮的三项指标绝对差值均不超过 0.01。
- 七个正式数据集的四项损失权重均保持为正数，不关闭任何损失项。

当前正式结果及逐轮历史位于 `results-2阶段/`。
