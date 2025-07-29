# TimeXer
## 实现方法
**综合内部和外部的变量预测时间序列**
1. 将时间序列划分成多个patch后，对每个patch进行patch embedding，对每个变量进行variable embedding，形成N个时间token（表示局部时间片段）和1个变量token（表示变量的全局信息）
2. 其次将内部变量的embedding同时作为self-attention的Query和Key，形成$(N + 1) \times (N + 1)$的注意力网格，表示内部变量两两之间的互相影响关系和与全局的关系
3. 接着外部嵌入，即将外部变量变成C个变量token；并通过**内外交叉注意力机制**形成一个$1 \times C$的网格表示每一个外部变量对内部变量的影响
4. 最后将内部和外部注意力的输出特征融合，解码成对未来时间点的预测值
## novelty
现有的模型要么平等对待所有的变量要么忽略外部的变量，TimeXer让外部信息辅助内部变量的预测。
## Loss 计算
$$
L_{TimeXer} = \underbrace{MSE(Y, \hat{Y})}_{主损失} + \lambda \underbrace{\sum_{l=1}^L \|W_l^{\top}W_l - I\|_F^2}_{正交约束}
$$
> $W_l$是第$l$层时间交叉注意力矩阵

# TimesNet
## 实现方法
**通过多个周期预测时间序列**
1. 首先通过嵌入层得到深度特征$X_{1D}^0 \in \mathbb{R}^{T\times d_{model}}$
2. 下面执行n次TimesBlock，第l层输入深度特征为$X_{1D}^{l - 1}$，输出为$X_{1D}^l$
3. $X_{1D}^n \in \mathbb{R}^{T\times d_{model}}$ 将作为输入，通过全连接层映射到预测值空间，得到最终输出 $Y \in \mathbb{R}^{T\times d_{out}}$，$d_{out}$是预测值的维度

> TimesBlock结构
> 1. 对输入 $X_{1D}^{l-1}$ 做快速傅里叶变换，挑选若干 dominant frequencies 作为每个周期折叠的依据，然后将一维时间序列按这些周期 reshape 成二维张量 $X_{2D}^{l-1,i}$
> 2. 二维张量有二维局部性，对每个 $X_{2D}^{l-1,i}$ 使用不同 kernel size 的 1D 卷积进行特征提取，形成多尺度表示 $\hat{X}_{2D}^{l-1,i}$（即 Inception-like 模块）
> 3. 把二维转化成一维
> 4. 将得到的一维表征${\hat{X}^{l, 1}, \cdots ,\hat{X}^{l, k} }$和对应频率的强度加权求和
$\hat{A}_{f_1}^{l-1}, \cdots,\hat{A}_{f_k}^{l-1} = Softmax(A_{f_1}^{l-1}, \cdots, A_{f_k}^{l-1}) \\ X_{1D}^l = \Sigma_{i = 1}^k \hat{A}_{f_i}^{l-1} \times \hat{X}_{1D}^{l, i}$
## novelty
传统的Transformer仅仅通过离散的时间点的attention机制难以找到与时间周期性的关系。TimesNet通过把时间按照多种周期划分为二维张量，张量的列表示周期内时序变化，行表示周期间时序变化。可以多周期分析时间变化。 
## Loss 计算
$$
L_{TimesNet} = \frac{1}{L}\sum_{t=1}^L \left( \underbrace{\alpha \|y_t - \hat{y}_t\|_2^2}_{\text{预测损失}} + \underbrace{\beta \|F_t - F_{t-1}\|_F^2}_{\text{频率一致性损失}} \right)
$$
>$y_t$和$\hat{y}_t$分别表示真实值和预测值

> $\mathbf{F}_t$是时频转换模块输出的频率特征

> $\alpha,\beta$是超参数（默认$\alpha=1.0, \beta=0.5$）

# PatchTST
## 实现方法
**将时间序列分割成多个patches执行transformer**
1. 对每个输入的单变量时间序列分割成多个patches（可重叠可不重叠）
2. 使用transformer编码器将信号由潜空间表示
3. 对数据进行多头注意力计算，建立不同patch之间的关系
4. 根据注意力，通过$Softmax$计算权重，进行后面时间点的预测

> with self-supervised
> PatchTST可以分为有自监督和无自监督两种，上面的实现方法是没有自监督的情况，下面是有自监督的情况
> 有自监督的情况在变成embedding之前通过随机赋值0的方法遮盖了一部分patch，通过self-supervised重建被遮盖的数据
> 这样模型可以更好地捕捉时间序列中的模式和特征
## novelty
无self-supervised版本可以理解成把时序数据切块进行预测，但是确实显著提升了预测结果的准确性
1. 将图片/时序分割成小块可以减少需要处理的维度，让模型在每个块上处理而不是逐个时间点处理，降低了计算复杂度
2. 每个时序/图像块捕捉了局部区域的信息，使模型能够关注不同区域的特征。这有助于模型更好地理解时序/图像的局部信息
## Loss 计算
$$
L_{PatchTST} = \underbrace{\frac{1}{T}\sum_{t=1}^T MAE(y_t, \hat{y}_t)}_{预测损失} + \gamma \underbrace{KL(p_{patch} \| p_{uniform})}_{Patch分布正则}
$$
> $p_{\text{patch}}$是patch embedding的注意力分布

# iTransformer
## 实现方法
**保持不同的变量的独立性利用注意力找到不同变量的相关性**
1. 获取单变量的序列编码：通过MLP层将$X_{:n} \in mathbb{R}^{N \times D}$映射成$H_{:n} \in mathbb{R}^{N \times T}$
2. 通过Multivariate Attention机制，将不同的变量的token分别作为attention中的Query, Key, Value分析不同变量token之间的相关性。生成了Multivariate-correlation map, 是attention系数，用来指示不同变量间相关性的大小。
3. 为了保证变量的独立性，避免采样不对齐或时滞性影响建模变量互相关，对每一个变量单独进行标准化（Layer Normalization）
4. 使用前馈神经网络（Feed Forward）建立每一个变量内部的时序特点
5. 最后映射，实现未来时间点的时序预测
## novelty
1. 不同的变量具有不同的物理意义等，传统的Transformer方法将他们编码为统一的token而不区分不同的通道，这样会导致不同变量之间的关系被忽略并且不适合多变量的预测。iTransformer通过将不同变量独立作为token，可以考虑到不同变量之间的关系。
## Loss 计算
$$
L_{iTransformer} = \underbrace{SmoothL1Loss(Y, \hat{Y})}_{\text{主损失}} + \mu \underbrace{\|E^{\top}E - I\|_2^2}_{\text{嵌入正交损失}}
$$
> $E$是变量嵌入矩阵

# Dlinear 
## 实现方法
1. 对每个变量维度 $x^{(d)} \in \mathbb{R}^T$，进行分解 $x^{(d)} = s^{(d)} + t^{(d)}$
> 其中 $t^{(d)}$ 是趋势序列，通过滑动平均或低通滤波提取；Dlinear使用滑动平均滤波器提取趋势成分：
> 对于输入序列 $x = [x_1, x_2, \dots, x_T]，t_t = \frac{1}{k} \sum_{i = t - \lfloor k/2 \rfloor}^{t + \lfloor k/2 \rfloor} x_i$

> $s^{(d)}$ 是残差序列，表示去除趋势后剩余的短期波动部分,$s^{(d)} = x^{(d)} - t^{(d)}$
2. 两个部分分别输入独立的线性层，拟合趋势和残差序列
3. 最后的预测结果是两个部分预测之和 $\hat{y} = \text{Linear}_T(t) + \text{Linear}_S(s)$
## novelty
1. 相较于基于Transformer的模型，Dlinear模型更加简单
2. 支持多步预测（一次性输出整个预测窗口）， 相较于Transformer-based方法不需要递归预测，不会造成误差累积
3. Dlinear可以避免过度拟合
## Loss 计算
$$
L_{DLinear} = \underbrace{\frac{1}{2N}\sum_{i=1}^N \|y_i - \hat{y}_i\|_2^2}_{\text{趋势项}} + \underbrace{\frac{\lambda}{2N}\sum_{i=1}^N \|y_i - T_i\|_2^2}_{\text{残差项}}
$$

# SFT（Supervised Fine-Tuning）
在训练大模型上，大模型会在一些bad cases上表现很差，显著降低了回答的准确性。SFT从概括上讲通过给一些bad cases提供了回答的标准答案，让模型在面对一些prompt的时候更倾向于生成某些回答，从而提高回答的效果。
为了高效和不需要为大模型储存梯度，下面以基于LoRA（Low-Rank Adaptation）的SFT为例
## 输入输出
输入：bad cases的prompt和标准答案

输出：优化后的LoRA参数，用merge_and_unload()函数融入愿来的模型，获得新模型
## 原理
训练模型本质上是训练$y = Wx$中的系数矩阵W，在LoRA中，我们不改变中间过程的W, 而是用W_{\text{LoRA}} = W + \Delta W = W + BA$,其中$A \in \mathbb{R}^{r \times k},B \in \mathbb{R}^{d \times r}$，通过优化AB矩阵来进行调优

通过在Attention中的Q，V层插入LoRA模块，用$W+BA$前向传播计算得到output和标准答案中的交叉熵损失，作为loss，反向传播更新A和B值，优化模型参数。

> 交叉熵损失(Cross-Entrophy Loss)
> 给定输入token序列$x_1, \cdots ,x_{t-1}$，预测下一个token $x_t$，单个token的损失是$\mathcal{L}^{(t)} = -\sum_{i=1}^{V} y_i^{(t)} \log \hat{y}i^{(t)} = -\log \hat{y}{x_t}^{(t)}$

> 整条句子的平均交叉熵损失为$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log \hat{y}_{x_t}^{(t)}$

> 神经网络输出 logits $z = (z_1, …, z_C)$，经过 softmax变成概率：$\hat{y}i = \frac{\exp(z_i)}{\sum{j=1}^{C} \exp(z_j)}$

> 将其带入交叉熵中，可以写为：$\mathcal{L} = -\log\left( \frac{\exp(z_c)}{\sum_{j=1}^{C} \exp(z_j)} \right)
= -z_c + \log\left( \sum_{j=1}^{C} \exp(z_j) \right)$


# GRPO(Generative Reward-Preferring Optimization)
总结来说，GRPO就是在构造一个偏向高 reward 的目标分布 $q(x)$，然后让新模型 $\pi(x)$ 去模仿这个分布
## 原理
1.    采样输出
对于一个给定的 prompt，使用初始策略（旧模型）$\pi_{\text{old}}$ 生成多个候选输出 $x_i$，并记录每个输出的概率：$ \pi_{\text{old}}(x_i) = P(x_i \mid \text{prompt})$

2.    构造目标分布 $q(x)$
使用奖励模型reward model为每个输出 $x_i$ 打分，得到奖励 $r(x_i)$，构造一个“奖励偏好”的目标分布 $q(x)$：$ q(x_i) = \frac{1}{Z} \cdot \pi_{\text{old}}(x_i) \cdot \exp\left( \frac{r(x_i)}{\beta} \right)$
其中 $\beta$ 是温度系数，$Z$ 是归一化常数，使得 $\sum_i q(x_i) = 1$。
> 这个分布同时考虑到旧策略与奖励偏好，使其和原模型不会偏离太远

3. 计算Loss函数

GRPO 并不直接最小化 $\mathrm{KL}(q | \pi)$，而是通过如下 surrogate loss 进行优化：
$\mathcal{L}_{\text{GRPO}} = \sum{x_i} q(x_i) \cdot \left[ \log \frac{\pi_{\text{old}}(x_i)}{\pi(x_i)} + \frac{r(x_i)}{\beta} \right]$.
通过最小化该 loss，可以鼓励新策略 $\pi$ 更偏向生成高奖励输出 $x_i$。

# DAPO(Decouple Clip and Dynamic sAmpling Policy Optimization)
DAPO通过解耦梯度裁剪(clip)和动态采样策略来优化训练过程，相较于传统PRO等微调方法稳定性更好
> 解耦梯度裁剪：传统的梯度裁剪方法通常对所有参数使用统一的裁剪阈值，而DAPO对不同层或参数组采用自适应的裁剪策略。

> 动态采样策略：根据训练过程中的梯度统计信息动态调整样本采样策略，优先选择对模型改进最有帮助的样本。
## 原理
1. 对于一个prompt，用初始模型计算和标准答案的偏差，计算交叉熵损失$L_{basic}= -\sum y_i \log(p_i)$
2. 反向传播，计算每一个参数的梯度，调用每一层的裁剪阈值$clip\_params$,计算超出裁剪阈值的裁剪损失$L_{clip} = \sum_i (\max(0, |g_i| - \tau_i))^2$
3. 动态采样更新权重，我们把loss比较大的类型的权重增加，用$w_i = \frac{e^{s_i/\beta}}{\sum e^{s_k/\beta}}$更新，其中$\beta$=温度系数。更新采样器，计算KL正则，$
L_{sample} = KL(p_{sample} \parallel p_{uniform}) = \sum_{i=1}^N p_{sample}^{(i)} \log \left( \frac{p_{sample}^{(i)}}{p_{uniform}^{(i)}} \right)
$
> 用KL散度保持均衡，防止只练习错误高的部分
4. 计算总损失$ L_{total} = $L_{basic} + \lambda_1 \times L_{clip} + \lambda_2 \times L_{sample}$
5. 梯度下降更新参数