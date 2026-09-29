# 服务端、客户端与训练损失完整说明

下面按程序真正的执行顺序，从启动、服务端初始化、表示热身、本地编码器与聚类头联合预训练、正式聚类训练，一直到最佳模型保存，完整说明一次训练是怎样进行的。

核心代码：

- 入口：`system/main.py`
- 服务端：`system/flcore/servers/servercluster.py`
- 客户端：`system/flcore/clients/clientcluster.py`
- 模型与损失：`system/flcore/trainmodel/multiview.py`

## 1. 启动阶段

执行：

```powershell
python system/main.py --config Scene-15
```

程序首先加载对应 JSON 配置，包括：

- 数据集路径和聚类数
- 客户端数量
- 输入归一化方法
- 编码器隐藏层
- 嵌入维度
- 总联邦轮数
- 预训练轮数
- 本地 epoch 数
- batch size
- 学习率
- 四项损失权重
- 随机种子
- 运行设备

随后固定随机种子：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

这会尽可能保证客户端划分、模型初始化、KMeans 和训练顺序可复现。

## 2. 数据加载与客户端划分

MAT 文件中的每个视图被转换为统一形式：

\[
X^{(v)}\in\mathbb{R}^{N\times D_v}
\]

当前正式配置均使用 standard 标准化：

\[
\widetilde{x}_{i,r}^{(v)}
=
\frac{x_{i,r}^{(v)}-\mu_r^{(v)}}{\sigma_r^{(v)}}
\]

然后只使用样本索引随机打乱：

\[
\{1,\ldots,N\}\rightarrow
\mathcal I_1,\mathcal I_2,\ldots,\mathcal I_C
\]

满足：

\[
\mathcal I_a\cap\mathcal I_b=\varnothing
\]

并且客户端之间的样本数量尽量相等。

这里不会根据标签进行分层。每个客户端得到一部分样本，但这些样本的全部视图都会保留。

## 3. 服务端初始化

服务端创建 `FederatedMultiViewClusteringServer`，主要完成三件事。

### 3.1 创建客户端对象

根据划分结果，为每个客户端建立：

```python
FederatedClusteringClient(
    client_id,
    data,
    indices,
    config,
    device,
)
```

客户端持有：

- 本地样本索引
- 本地全部视图
- 配置
- 设备
- 客户端编号

服务端还检查：

\[
N_m\ge K
\]

即每个客户端的样本数必须不少于聚类数，否则客户端无法运行 \(K\) 类 KMeans。

### 3.2 创建全局模型

服务端创建一个全局 `MultiViewClusteringModel`。

模型包含：

- 每个视图一个独立编码器
- 每个视图一个独立解码器
- 可学习的视图融合权重
- 一个 Student-t 聚类头
- \(K\) 个可训练聚类中心

此时聚类中心使用 Xavier uniform 随机初始化：

\[
\boldsymbol\mu_j\sim\operatorname{XavierUniform}
\]

但这个随机中心不会作为正式聚类初始化结果。完成配置指定的表示热身轮后，它会被客户端中心摘要得到的全局 KMeans 中心覆盖。

### 3.3 初始化视图权重

模型保存 \(V\) 个可学习 logits：

\[
a_1,a_2,\ldots,a_V
\]

初始值全部为 0，因此初始视图权重为：

\[
\pi_v
=
\frac{\exp(a_v)}
{\sum_{u=1}^{V}\exp(a_u)}
=
\frac{1}{V}
\]

即训练开始时各视图等权。

## 4. 服务端是否直接训练模型

服务端不执行梯度下降，也没有自己的 AdamW 优化器。

服务端的作用是：

1. 下发全局模型；
2. 选择参与客户端；
3. 接收本地模型参数；
4. 对齐客户端聚类中心；
5. 执行 FedAvg；
6. 初始化全局聚类中心；
7. 评估全局模型；
8. 保存最佳检查点。

真正的反向传播和参数优化全部发生在客户端。

## 5. 第一阶段：渐进式联合预训练

假设：

```json
"rounds": 16,
"pretrain_rounds": 12
```

若默认 `representation_warmup_rounds=1`，那么：

- 第 1 轮只优化表示；
- 第 2～12 轮联合训练编码器与本地聚类头，聚类相关损失逐轮增强；
- 第 13～16 轮使用完整损失权重进入正式聚类训练。

第一阶段每轮执行以下流程。

### 5.1 服务端选择客户端

客户端参与数为：

\[
M
=
\max
\left(
1,
\left\lceil C\cdot\text{join\_ratio}\right\rceil
\right)
\]

当前正式配置的 `join_ratio=1.0`，因此每轮所有客户端都参与。

### 5.2 服务端下发全局模型

服务端把当前全局模型传给客户端。

客户端不会直接修改该对象，而是深拷贝：

```python
model = copy.deepcopy(global_model)
```

因此每个客户端拥有独立的本地模型副本。

### 5.3 客户端创建本地优化器

每个联邦轮都会重新创建 AdamW：

```python
optimizer = torch.optim.AdamW([
    {"params": representation_parameters, "lr": learning_rate},
    {"params": head_parameters, "lr": learning_rate * head_lr_multiplier},
], weight_decay=weight_decay)
```

第一阶段使用：

```text
training.learning_rate
```

通常为：

\[
\eta_{\mathrm{pre}}=10^{-3}
\]

聚类头学习率还可以通过 `cluster_head_learning_rate_multiplier` 单独缩放，默认值为 1.0。AdamW 的动量状态不会跨联邦轮保留，因为每轮都会重新创建优化器。

## 6. 客户端模型前向传播

对一个 batch 中的多视图输入：

\[
\left(
X^{(1)},X^{(2)},\ldots,X^{(V)}
\right)
\]

### 6.1 独立视图编码

第 \(v\) 个视图经过自己的编码器：

\[
\mathbf z_i^{(v)}
=
E_v\left(\mathbf x_i^{(v)}\right)
\]

编码器结构为：

```text
输入
 ↓
Linear
 ↓
LayerNorm
 ↓
ReLU
 ↓
可选 Dropout
 ↓
Linear
 ↓
LayerNorm
 ↓
ReLU
 ↓
最终 Linear
 ↓
视图嵌入
```

最后一个线性层没有激活函数。不同视图拥有独立参数，不共享编码器。

### 6.2 视图重构

原始视图嵌入送入对应解码器：

\[
\widehat{\mathbf x}_i^{(v)}
=
D_v\left(\mathbf z_i^{(v)}\right)
\]

解码器维度与编码器相反。

例如编码器为：

\[
D_v\rightarrow256\rightarrow128\rightarrow64
\]

解码器就是：

\[
64\rightarrow128\rightarrow256\rightarrow D_v
\]

### 6.3 嵌入归一化

每个视图嵌入按样本进行 L2 归一化：

\[
\overline{\mathbf z}_i^{(v)}
=
\frac{\mathbf z_i^{(v)}}
{\|\mathbf z_i^{(v)}\|_2}
\]

### 6.4 可学习视图融合

视图 logits 经过 softmax：

\[
\pi_v
=
\frac{\exp(a_v)}
{\sum_{u=1}^{V}\exp(a_u)}
\]

然后对归一化嵌入加权：

\[
\widetilde{\mathbf z}_i
=
\sum_{v=1}^{V}
\pi_v\overline{\mathbf z}_i^{(v)}
\]

融合结果再进行一次 L2 归一化：

\[
\mathbf z_i
=
\frac{\widetilde{\mathbf z}_i}
{\|\widetilde{\mathbf z}_i\|_2}
\]

这个 \(\mathbf z_i\) 就是最终融合表示。

## 7. Student-t 聚类头

设全局有 \(K\) 个聚类中心：

\[
\boldsymbol\mu_1,\ldots,\boldsymbol\mu_K
\]

计算样本与中心的平方欧氏距离：

\[
d_{ij}
=
\|\mathbf z_i-\boldsymbol\mu_j\|_2^2
\]

然后使用 Student-t 分布计算未归一化相似度：

\[
s_{ij}
=
\left(
1+\frac{d_{ij}}{\alpha}
\right)^{-\frac{\alpha+1}{2}}
\]

当前配置使用：

\[
\alpha=1
\]

归一化后得到软聚类分配：

\[
q_{ij}
=
\frac{s_{ij}}
{\sum_{\ell=1}^{K}s_{i\ell}}
\]

最终预测簇为：

\[
\widehat y_i=\arg\max_jq_{ij}
\]

## 8. 第一项损失：重构损失

每个视图分别计算输入与重构结果之间的均方误差：

\[
\mathcal L_{\mathrm{rec}}^{(v)}
=
\frac{1}{BD_v}
\sum_{i=1}^{B}
\left\|
\widehat{\mathbf x}_i^{(v)}
-
\mathbf x_i^{(v)}
\right\|_2^2
\]

然后对所有视图等权平均：

\[
\mathcal L_{\mathrm{rec}}
=
\frac{1}{V}
\sum_{v=1}^{V}
\mathcal L_{\mathrm{rec}}^{(v)}
\]

作用是让编码器保留原始视图信息，避免表示只追求聚类而丢失数据结构。

虽然不同视图维度可能差异很大，但代码先在每个视图内部计算平均 MSE，再对视图平均，因此每个视图在重构损失中权重相同。

## 9. 第二项损失：跨视图一致性损失

每个归一化视图嵌入都与融合表示对齐：

\[
\mathcal L_{\mathrm{con}}
=
\frac{1}{V}
\sum_{v=1}^{V}
\frac{1}{Bd}
\sum_{i=1}^{B}
\left\|
\overline{\mathbf z}_i^{(v)}
-
\mathbf z_i
\right\|_2^2
\]

它推动不同视图对同一样本产生方向一致的表示。

该损失会同时影响：

- 各视图编码器
- 可学习视图权重
- 最终融合表示

## 10. 第一阶段总损失

前 `representation_warmup_rounds` 轮是表示热身，此时：

\[
\mathcal L_{\mathrm{warm}}
=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}
\]

聚类中心尚未初始化，DEC 和均衡损失不参与反向传播。

热身结束后，服务端先初始化全局中心。第一阶段余下轮次同时训练编码器和聚类头：

\[
\boxed{
\mathcal L_{\mathrm{joint}}^{(t)}
=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}
+
s_t\lambda_{\mathrm{clu}}\mathcal L_{\mathrm{clu}}
+
s_t\lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}}
}
\]

其中 \(s_t\) 从第一个联合预训练轮的
\(1/(\text{pretrain\_rounds}-\text{representation\_warmup\_rounds})\)
线性增加到 1。这样不会让刚初始化的伪标签立即以完整强度推动编码器，同时又能在第一阶段内联合训练本地表示与聚类头。

## 11. 客户端反向传播

每个 batch 执行：

```python
optimizer.zero_grad()
outputs = model(views)
loss, metrics = clustering_objective(...)
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

具体包括：

1. 清空梯度；
2. 前向传播；
3. 计算损失；
4. 检查损失是否为有限数；
5. 反向传播；
6. 梯度裁剪；
7. AdamW 更新参数。

梯度裁剪为：

\[
\|\nabla_\theta\mathcal L\|_2
\leq
\text{gradient\_clip}
\]

当前通常设置为 5.0。

## 12. 客户端完成本地训练后上传什么

客户端完成当前阶段设定的本地 epoch 后上传。三个阶段可分别由
`warmup_local_epochs`、`joint_local_epochs` 和
`clustering_local_epochs` 控制；未单独设置时沿用 `local_epochs`：

```python
{
    "client_id": ...,
    "num_samples": ...,
    "state_dict": ...,
    "metrics": ...,
    "cluster_counts": ...,
}
```

即：

- 客户端编号
- 本地样本数
- 本地模型全部参数
- 本地平均训练指标
- 中心初始化后每个簇的软计数

不上传：

- 原始样本
- 标签
- 单样本嵌入
- 单样本预测

## 13. 服务端第一阶段聚合

服务端按照客户端样本数量做 FedAvg：

\[
\theta^{(t+1)}
=
\sum_{m\in\mathcal S_t}
\frac{N_m}
{\sum_{\ell\in\mathcal S_t}N_\ell}
\theta_m^{(t+1)}
\]

普通模型参数按上述样本数权重聚合，包括：

- 所有编码器参数
- 所有解码器参数
- LayerNorm 参数
- 视图融合 logits
- 除聚类中心外的模型参数

表示热身轮中聚类中心尚未启用。联合预训练开始后，服务端先用 Hungarian 匹配同步本地中心和软簇计数的排列，再对第 \(k\) 个中心按该簇软计数聚合：

\[
\boldsymbol\mu_k^{(t+1)}
=
\frac{
\sum_m n_{m,k}\boldsymbol\mu_{m,k}^{(t+1)}
}{
\sum_m n_{m,k}
}
\]

因此某客户端整体样本很多、但第 \(k\) 个簇样本很少时，不会对该中心产生不合理的大权重。`center_momentum` 可进一步对新中心与旧全局中心做动量平滑，默认 0 表示不平滑。

服务端完成聚合后，用新参数更新全局模型。

## 14. 第一阶段评估

服务端会在完整数据集上计算软分配并得到：

- ACC
- NMI
- ARI
- 平均置信度

但第一阶段的指标只写入历史记录，不参与最佳模型选择。只有 `round_index >= pretrain_rounds` 的正式聚类轮能够更新最佳检查点。

也就是说，即使某个联合预训练轮的 NMI 很高，也不会保存为最终最佳检查点。

## 15. 表示热身后的全局中心初始化

假设 `representation_warmup_rounds=1`，那么服务端完成第 1 轮表示 FedAvg 后，在第 2 轮客户端联合训练开始前初始化全局聚类中心。

### 15.1 客户端生成本地中心摘要

每个客户端复制表示热身后的全局模型，并对本地样本计算融合嵌入：

\[
\mathbf z_i,\quad i\in\mathcal I_m
\]

然后运行 \(K\) 类 KMeans：

\[
\{\mathbf z_i:i\in\mathcal I_m\}
\rightarrow
\{
\mathbf c_{m,1},
\ldots,
\mathbf c_{m,K}
\}
\]

同时统计每个本地簇的样本数：

\[
n_{m,j}
=
\left|
\left\{
i:\widehat y_i^{(m)}=j
\right\}
\right|
\]

客户端上传：

\[
\left\{
(\mathbf c_{m,j},n_{m,j})
\right\}_{j=1}^{K}
\]

不会上传本地嵌入和原始样本。

### 15.2 服务端合并本地中心

服务端收集所有客户端中心：

\[
\{
\mathbf c_{m,j}
\}_{m=1,j=1}^{C,K}
\]

然后以对应本地簇计数作为权重，再执行一次 KMeans：

\[
\{
(\mathbf c_{m,j},n_{m,j})
\}
\rightarrow
\{
\boldsymbol\mu_1,\ldots,\boldsymbol\mu_K
\}
\]

服务端使用：

```python
KMeans.fit(
    centers,
    sample_weight=counts,
)
```

由此得到第一个数据驱动的全局聚类中心，并覆盖随机初始化中心。接下来的每个联合预训练轮，各客户端都从该全局中心出发，在更新编码器的同时更新本地聚类头；服务端再对齐并按软簇计数融合这些本地中心。

## 16. 第二阶段：正式联合聚类训练

到达 `pretrain_rounds` 边界后：

```python
clustering_enabled = True
```

客户端仍然执行：

- 复制全局模型
- 创建 AdamW
- 多个本地 epoch
- batch 训练
- 上传参数

与联合预训练阶段相比，此时 DEC 和均衡损失的缩放系数固定为 1，并切换到 `clustering_learning_rate`。中心不会再次运行 KMeans 初始化，而是继续使用第一阶段已经联合训练和聚合的结果。

## 17. DEC 目标分布

每个客户端在每个通信轮开始时，先用收到的全局模型对完整本地数据前向推理。设客户端 \(m\) 有 \(N_m\) 个样本，计算本地完整软分配：

\[
Q_m\in\mathbb R^{N_m\times K}
\]

再计算每个簇在完整本地数据中的软频率：

\[
f_j
=
\sum_{i=1}^{N_m}q_{ij}
\]

对软分配平方，并除以簇频率：

\[
w_{ij}
=
\frac{q_{ij}^2}{f_j}
\]

最后按样本归一化：

\[
p_{ij}
=
\frac{w_{ij}}
{\sum_{\ell=1}^{K}w_{i\ell}}
\]

即：

\[
p_{ij}
=
\frac{q_{ij}^2/f_j}
{\sum_{\ell=1}^{K}q_{i\ell}^2/f_\ell}
\]

作用包括：

- 对高置信度分配进行平方强化；
- 减少大簇频率带来的偏置；
- 根据当前模型构造伪标签目标。

客户端由 `local_target_cache()` 生成一次完整目标：

```python
local_targets = target_distribution(local_assignments).detach()
```

目标按样本索引缓存。本通信轮中的所有本地 batch 和当前阶段的所有本地 epoch 都使用对应缓存行，不会在每个 mini-batch 中即时重算 \(P\)。因此：

- 簇频率来自完整本地数据，不受单个 batch 构成影响；
- 同一通信轮内伪标签目标固定；
- 梯度不会通过 \(P\) 返回；
- 下一通信轮收到新全局模型后才重新生成目标。

旧实现按当前 mini-batch 即时生成目标，Scene-15 中出现了 DEC KL 持续增大和 NMI 持续下降；该实现已经替换。

## 18. 第三项损失：DEC KL 自训练损失

聚类损失为：

\[
\mathcal L_{\mathrm{clu}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\sum_{j=1}^{K}
p_{ij}
\log
\frac{p_{ij}}{q_{ij}}
\]

即：

\[
\boxed{
\mathcal L_{\mathrm{clu}}
=
\frac{1}{B}
D_{\mathrm{KL}}(P\|Q)
}
\]

代码对应：

```python
F.kl_div(
    assignments.log(),
    target,
    reduction="batchmean",
)
```

它推动当前模型分配 \(Q\) 接近强化后的目标分布 \(P\)。

这是一种自训练过程。因为目标来自模型自身，如果学习率过大，错误预测也可能被持续强化，因此可能出现后期性能下降。

## 19. 第四项损失：簇均衡损失

先计算当前 batch 的平均软分配：

\[
\overline q_j
=
\frac{1}{B}
\sum_{i=1}^{B}q_{ij}
\]

均衡损失为：

\[
\mathcal L_{\mathrm{bal}}
=
\sum_{j=1}^{K}
\overline q_j
\log(K\overline q_j)
\]

它等价于：

\[
\boxed{
\mathcal L_{\mathrm{bal}}
=
D_{\mathrm{KL}}
\left(
\overline{\mathbf q}
\|
\mathbf u
\right)
}
\]

其中均匀分布为：

\[
u_j=\frac{1}{K}
\]

当所有簇平均使用程度相同时：

\[
\overline q_j=\frac{1}{K}
\]

均衡损失达到理论最小值 0。

该损失用于避免：

```text
大量样本 → 少数几个簇
```

但它是在每个客户端的每个 batch 内计算，并不是直接约束整个数据集的全局簇比例。

## 20. 联合聚类阶段总损失

聚类阶段完整损失为：

\[
\boxed{
\mathcal L_{\mathrm{joint}}
=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}
+
\lambda_{\mathrm{clu}}\mathcal L_{\mathrm{clu}}
+
\lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}}
}
\]

四项功能分别是：

| 损失 | 作用 |
|---|---|
| \(\mathcal L_{\mathrm{rec}}\) | 保留各视图输入信息 |
| \(\mathcal L_{\mathrm{con}}\) | 对齐不同视图表示 |
| \(\mathcal L_{\mathrm{clu}}\) | 强化高置信度伪标签 |
| \(\mathcal L_{\mathrm{bal}}\) | 防止聚类塌缩到少数簇 |

例如大部分正式配置使用：

\[
\lambda_{\mathrm{rec}}=1.0
\]

\[
\lambda_{\mathrm{con}}=0.2
\]

\[
\lambda_{\mathrm{clu}}=0.5
\]

\[
\lambda_{\mathrm{bal}}=0.05
\]

但 HW 的一致性权重是 1.0，NUSWIDE 的一致性权重是 0。Scene-15 为抑制伪标签过度强化，当前使用：

\[
\lambda_{\mathrm{clu}}=0.05,\qquad
\lambda_{\mathrm{bal}}=0
\]

Scene-15 关闭 batch 级均衡约束，是因为该约束会要求每个小批次近似均匀使用 15 个簇，与实际局部数据分布不一定一致。

## 21. 两阶段学习率

表示热身使用：

\[
\eta_{\mathrm{pre}}
=
\text{learning\_rate}
\]

联合预训练使用：

\[
\eta_{\mathrm{joint}}
=
\text{joint\_learning\_rate}
\]

若配置未显式提供，加载器默认设为 `0.1 * learning_rate`。

正式聚类阶段优先使用：

\[
\eta_{\mathrm{cluster}}
=
\text{clustering\_learning\_rate}
\]

如果配置中没有 `clustering_learning_rate`，就继续使用第一阶段学习率。聚类头在两个阶段都可以使用：

\[
\eta_{\mathrm{head}}
=
\eta_{\mathrm{stage}}
\times
\text{cluster\_head\_learning\_rate\_multiplier}
\]

默认倍数为 1.0，即暂不放大；调参时可以让聚类头比表示参数更新更快。

单独设置聚类学习率的原因是：

- 预训练需要较快学习重构表示；
- 聚类阶段的目标来自伪标签；
- 过大的更新会强化错误伪标签；
- 较小学习率可以减少后期中心漂移。

例如 Scene-15 当前为：

\[
\eta_{\mathrm{pre}}=10^{-3}
\]

\[
\eta_{\mathrm{joint}}=5\times10^{-8}
\]

\[
\eta_{\mathrm{cluster}}=5\times10^{-8}
\]

Scene-15 的表示热身使用 2 个本地 epoch，联合预训练和正式聚类阶段各使用 3 个本地 epoch。由于后两个阶段的本地更新次数和通信轮数都增加，学习率与中心更新幅度相应减小，以避免伪标签误差被反复放大。

## 22. 聚类阶段的中心排列问题

客户端完成本地聚类训练后，聚类中心的编号可能不同。

例如：

```text
全局中心：    [簇 A, 簇 B, 簇 C]
客户端中心：  [簇 C, 簇 A, 簇 B]
```

虽然几何结构相同，但如果直接平均，会错误混合中心。

因此服务端在 FedAvg 前计算客户端中心与全局中心之间的距离：

\[
D_{ab}
=
\left\|
\boldsymbol\mu_a^{(m)}
-
\boldsymbol\mu_b^{(g)}
\right\|_2^2
\]

然后使用 Hungarian 算法求最小代价匹配：

\[
\sigma_m
=
\arg\min_\sigma
\sum_{j=1}^{K}
D_{\sigma(j),j}
\]

客户端中心按照匹配关系重新排列，再参与 FedAvg。

## 23. 联合训练阶段的参数聚合

中心对齐后，服务端对编码器、解码器、LayerNorm 和视图融合权重执行：

\[
\theta^{(t+1)}
=
\sum_{m\in\mathcal S_t}
\frac{N_m}
{\sum_{\ell\in\mathcal S_t}N_\ell}
\theta_m^{(t+1)}
\]

聚合内容包括：

- 编码器
- 解码器
- LayerNorm
- 视图融合权重
- 聚类中心之外的其他浮点状态

聚类中心不再使用客户端总样本数 FedAvg，而是将同步排列后的软簇计数 \(n_{m,k}\) 作为逐簇权重：

\[
\boldsymbol\mu_k^{(t+1)}
=
\frac{\sum_m n_{m,k}\boldsymbol\mu_{m,k}^{(t+1)}}
{\sum_m n_{m,k}}
\]

客户端只上传每个簇的汇总计数，不上传单样本分配。

## 24. 服务端评估

每轮 FedAvg 后，服务端对完整数据集前向推理：

\[
q_{ij}
=
P(\text{cluster}=j\mid\mathbf x_i)
\]

预测为：

\[
\widehat y_i=\arg\max_jq_{ij}
\]

然后读取真实标签计算：

- ACC
- NMI
- ARI
- 平均置信度

真实标签只在这里使用。

### ACC

聚类编号和真实标签编号不一定相同，所以先用 Hungarian 算法寻找最佳编号映射：

\[
\mathrm{ACC}
=
\frac{1}{N}
\max_\pi
\sum_{i=1}^{N}
\mathbf 1[y_i=\pi(\widehat y_i)]
\]

### NMI

衡量预测簇与真实类别的归一化互信息。

### ARI

衡量样本对聚类一致性，并对随机一致情况进行校正。

## 25. 最佳模型选择

当前配置使用 NMI 选模。

如果当前处于聚类阶段，并且：

\[
\mathrm{NMI}_t
>
\mathrm{NMI}_{\mathrm{best}}
\]

服务端就保存：

- 当前全局模型参数
- 当前轮次
- 当前 ACC/NMI/ARI

表示热身和联合预训练阶段的指标不会参与最佳模型选择。

## 26. 训练结束

完成全部联邦轮次后，服务端不会直接保存最后一轮模型，而是先恢复最佳 NMI 检查点：

```python
global_model.load_state_dict(best_state)
```

然后重新评估，并写出：

```text
results/<dataset>/summary.json
results/<dataset>/history.json
results/<dataset>/model.pt
```

其中：

- `model.pt`：最佳 NMI 轮的模型
- `summary.json`：最佳模型的指标和配置
- `history.json`：每一轮的训练损失和评估指标

因此最终模型不一定来自训练末轮。

## 27. 完整流程总结

```text
读取配置和 MAT 数据
        ↓
逐视图归一化
        ↓
无标签随机均衡划分客户端
        ↓
服务端创建全局多视图模型
        ↓
────────────────────────────────
阶段一：渐进式联合预训练
第一个或指定数量的轮次：
重构损失 + 一致性损失
        ↓
上传本地参数和样本数
        ↓
服务端 FedAvg
        ↓
客户端计算融合嵌入
        ↓
客户端本地 KMeans
        ↓
上传本地中心和计数
        ↓
服务端加权 KMeans
        ↓
初始化全局聚类中心
        ↓
客户端联合训练编码器 + 本地聚类头
聚类损失与均衡损失线性增强
        ↓
上传本地模型参数 + 软簇计数
        ↓
服务端 Hungarian 对齐中心与计数
        ↓
普通参数 FedAvg + 中心按簇计数融合
────────────────────────────────
        ↓
阶段二：正式联邦聚类训练
客户端优化：
重构 + 一致性 + DEC KL + 簇均衡
        ↓
上传本地模型参数 + 软簇计数
        ↓
服务端 Hungarian 对齐中心与计数
        ↓
普通参数 FedAvg + 中心按簇计数融合
────────────────────────────────
        ↓
服务端全局评估 ACC/NMI/ARI
        ↓
按 NMI 保存最佳检查点
        ↓
训练结束后恢复最佳模型
        ↓
保存模型、历史和汇总
```

最关键的一点是：客户端负责真正的神经网络优化，服务端不进行梯度训练，而是负责模型调度、中心初始化、中心对齐、参数聚合、全局评估和最佳模型保存。
