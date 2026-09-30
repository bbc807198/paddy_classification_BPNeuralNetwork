# **Model 1 — Paddy Yield Classification Baseline**

## 1\. 项目说明

本目录保存基于 Paddy Dataset 构建的第一个水稻单位面积产量分类神经网络模型。

该模型作为整个项目的 **Baseline Model（基线模型）** 保存，主要用于记录当前的数据处理方法、特征构造方式以及神经网络的基础实验结果。

当前模型并不是最终模型。后续如果对数据处理、标签定义、网络结构、训练方法或特征选择方法进行较大调整，将分别保存为后续迭代模型版本。

## 2\. 数据集

使用的数据集为：**paddydataset\.csv**，存放在原始数据集文件夹中。

**原始数据规模：**

样本数：**2789**

原始字段数：**45**

数据集中包含水稻种植过程中与土地面积、水稻品种、土壤、育苗、风向、肥料、农药以及其他农业管理因素相关的变量。

本实验不直接预测总产量，而是首先计算单位面积产量，然后将其转换为三分类任务。

## 3\. 模型任务

原始数据集中包含关键字段：

Paddy yield\(in Kg\)

Hectares

首先计算单位面积水稻产量：

Yield\_per\_hectare = Paddy yield\(in Kg\) / Hectares

公式表达如下：

$Yield\_per\_hectare = \frac{Paddy\ yield(in\ Kg)}{Hectares}$

随后根据整个数据集中 **Yield\_per\_hectare** 的 1/3 和 2/3 分位数，将预测任务转换为三分类问题。

**类别定义如下：**

```Plain Text
0 -> Low
1 -> Medium
2 -> High
```

**代码实现：**

```Plain Text
low_threshold = data["Yield_per_hectare"].quantile(1/3)
high_threshold = data["Yield_per_hectare"].quantile(2/3)
```

**分类规则：**

```Plain Text
Yield_per_hectare <= low_threshold
    -> Low

low_threshold < Yield_per_hectare <= high_threshold
    -> Medium

Yield_per_hectare > high_threshold
    -> High
```

这种划分方法可以使三个类别的样本数量保持相对均衡。

## 4\. 数据清洗

数据清洗在 **clean\_dataset\(\)** 函数中完成。

### 4\.1 删除与 Hectares 高度冗余的变量

数据分析发现，下列 10 个变量的取值完全由 **Hectares** 决定，因此在当前 Baseline 中将这些变量删除，避免信息冗余：

```Plain Text
Seedrate(in Kg)
LP_Mainfield(in Tonnes)
Nursery area (Cents)
LP_nurseryarea(in Tonnes)
DAP_20days
Weed28D_thiobencarb
Urea_40Days
Potassh_50Days
Micronutrients_70Days
Pest_60Day(in ml)
```

由于当前模型保留了 **Hectares**，继续保留上述变量会造成较强的信息冗余，影响模型训练。

## 5\. 防止目标泄漏

计算得到目标变量 **Yield\_per\_hectare** 后，需删除原始字段：**Paddy yield\(in Kg\)**。

原因：单位面积产量由总产量与种植面积计算得出，若保留总产量字段作为输入特征，会产生严重的 **目标泄漏（Target Leakage）** 问题，导致模型训练失真。

同时当前实验额外删除字段：**Trash\(in bundles\)**。

**本次实验总计删除字段：12个**

```Plain Text
10 个 Hectares 冗余变量
+
Paddy yield(in Kg)
+
Trash(in bundles)
```

## 6\. 清洗后的数据

原始数据字段总数：**45 个**

删除冗余及泄漏字段后：45 \- 12 = **33 个**

数据处理后新增字段：**Yield\_per\_hectare**（目标计算字段）

**清洗后数据集字段构成：**

33 个输入特征 \+ 1 个单位面积产量指标

**清洗结果保存文件：**rice\_cleaned\_dataset\_including\_repeated\.csv

**重要说明：**当前 Baseline **保留重复样本**。

代码中去重逻辑处于注释关闭状态，未生效：

```Plain Text
# df = df.drop_duplicates()
```

因此 Model 1 全程使用**包含重复样本**的数据集进行训练。

## 7\. 输入特征

模型训练使用的原始输入特征总数：**33 个**

特征构成分类：

数值变量：**25 个**

类别变量：**8 个**

**8个类别变量明细：**

```Plain Text
Agriblock
Variety
Soil Types
Nursery
Wind Direction_D1_D30
Wind Direction_D31_D60
Wind Direction_D61_D90
Wind Direction_D91_D120
```

除上述8个类别变量外，其余所有输入变量均按数值变量处理。

## 8\. One\-Hot Encoding

所有类别变量统一采用 **One\-Hot 独热编码** 处理，代码实现如下：

```Plain Text
X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    dtype=float
)
```

**编码示例（土壤类型）：**

原始类别：clay、alluvial

编码后格式：

```Plain Text
clay    alluvial
clay          1        0
alluvial      0        1
```

**特征维度变化：**

33 个原始输入特征 → 编码后 **60 个最终输入特征**

因此本次神经网络模型的**输入维度为60**。

**编码后数据保存文件：**x\.csv（特征）、y\.csv（标签），用于后续数据分析与复盘。

## 9\. 数据集划分

当前 Baseline 采用 sklearn 库的 **train\_test\_split\(\)** 完成数据集划分。

**划分比例：训练集80%、测试集20%**

**核心代码：**

```Plain Text
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    stratify=y
)
```

参数说明：

test\_size = 0\.2：指定20%数据作为测试集

stratify=y：分层抽样，保证训练集、测试集中 Low/Medium/High 三类样本比例与原始数据集一致

### 当前限制

Model 1 未设置 **random\_state** 固定随机种子，导致每次运行的数据集划分结果存在差异，最终模型准确率会有小幅波动。后续版本可固定随机种子，提升实验可复现性。

## 10\. 数据标准化

针对所有数值特征，使用 **StandardScaler** 进行标准化处理。

**标准化规则：严格区分训练集、测试集**

训练集：scaler\.fit\_transform\(\) —— 基于训练集计算均值、标准差并完成标准化

测试集：scaler\.transform\(\) —— 复用训练集的均值、标准差进行标准化，避免数据泄露

**标准化公式：**

$x'=\frac{x-\mu}{\sigma}$

参数说明：μ=训练集均值，σ=训练集标准差

**特殊规则：**One\-Hot Encoding 生成的类别特征不进行标准化处理。

## 11\. PyTorch 数据集封装

预处理后的数据按以下流程完成PyTorch格式封装，适配模型训练：

1\. 转换为 PyTorch Tensor 张量格式

```Plain Text
torch.tensor(x_train.values).float()
torch.tensor(y_train.values)
```

2\. 封装为 **TensorDataset** 数据集对象

3\. 通过 **DataLoader** 实现批量数据读取与迭代

**完整数据流转流程：**

Pandas DataFrame → PyTorch Tensor → TensorDataset → DataLoader → Neural Network

## 12\. 神经网络结构

模型基于 PyTorch 自定义实现，模型类名称：**PaddyYieldModel**。

网络结构包含：**2层全连接隐藏层 \+ 1层输出层**

**网络维度结构：60 → 128 → 64 → 3**

**网络层级流程：**

```Plain Text
Input（60维）
  │
  ▼
Linear（60→128）
  │
ReLU激活
  │
  ▼
Linear（128→64）
  │
ReLU激活
  │
  ▼
Linear（64→3）
  │
  ▼
Output Logits（3维分类输出）
```

**核心网络代码：**

```Plain Text
self.linear1 = nn.Linear(
    in_features=input_dim,
    out_features=128
)

self.linear2 = nn.Linear(
    in_features=128,
    out_features=64
)

self.output = nn.Linear(
    in_features=64,
    out_features=output_dim
)
```

**隐藏层激活函数：ReLU**

```Plain Text
x = torch.relu(self.linear1(x))
x = torch.relu(self.linear2(x))
```

## 13\. 为什么输出层没有使用 Softmax

模型输出层仅直接输出logits，未手动调用Softmax函数：

```Plain Text
x = self.output(x)
```

**核心原因：**模型训练使用 **nn\.CrossEntropyLoss\(\)** 作为损失函数。

PyTorch 中 CrossEntropyLoss 内部已自动整合 **LogSoftmax \+ 负对数似然损失（NLLLoss）**，因此无需手动添加Softmax，直接传入原始logits即可，是标准的分类模型训练写法。

## 14\. 模型训练参数

Model 1 完整训练配置如下表所示：

|参数|设置|
|---|---|
|Framework|PyTorch|
|Input Dimension|60|
|Hidden Layer 1|128|
|Hidden Layer 2|64|
|Output Dimension|3|
|Hidden Activation|ReLU|
|Loss Function|CrossEntropyLoss|
|Optimizer|Adam|
|Learning Rate|0\.001|
|Epochs|50|
|Training Batch Size|16|
|Train/Test Ratio|80% / 20%|

**核心参数代码：**

```Plain Text
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
criterion = nn.CrossEntropyLoss()
epochs = 50
batch_size = 16
```

训练参数补充：**shuffle=True**，每个Epoch自动打乱训练样本顺序，提升模型泛化能力。

## 15\. 模型训练过程

每个Batch的完整训练迭代流程：

输入数据 → 前向传播 → 计算交叉熵损失 → 清空梯度 → 反向传播 → Adam参数更新

**核心训练代码：**

```Plain Text
y_pred = model(x)
loss = criterion(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

每个 Epoch 结束后输出训练日志：**迭代轮数、平均损失、训练耗时**

日志输出示例：

```Plain Text
epoch:1,loss:1.xxxx,time:x.xx s
epoch:2,loss:1.xxxx,time:x.xx s
...
epoch:50,loss:x.xxxx,time:x.xx s
```

## 16\. 模型保存

训练完成后，仅保存模型**参数权重（state\_dict）**，不保存完整模型对象，节省存储空间、适配迁移部署。

**保存代码：**

```Plain Text
torch.save(
    model.state_dict(),
    "./水稻试验/model/modeldata.pth"
)
```

**保存文件：**modeldata\.pth

**模型加载说明：**测试或复现模型时，需先手动创建 PaddyYieldModel 网络实例，再加载权重参数。

## 17\. 模型测试

**测试集配置：**

Batch Size = 8，Shuffle = False

测试阶段开启模型评估模式：**model\.eval\(\)**，关闭Dropout、BN等训练专属机制，保证测试结果稳定。

模型输出为3维logits向量，通过 **torch\.argmax\(y\_pred, dim=1\)** 解析得到最终预测类别。

**预测示例：**

模型输出：\[1\.21, 0\.53, \-0\.41\]，最大值索引为0，对应预测类别Low。

最终评价指标：**准确率（正确预测样本数/总测试样本数）**

## 18\. 当前 Baseline 结果

Model 1 基准模型测试准确率：**约50%**

受未固定random\_state影响，每次运行的数据集划分不同，准确率存在小幅波动。

因此该结果定义为：**Approximately 50% Accuracy**（近似50%准确率），非固定精确值。

**当前指标局限：**仅计算准确率，未引入精确率、召回率、F1分数、混淆矩阵等精细化评价指标。

## 19\. 当前发现的数据问题

原始数据集共2789条样本，但在33个有效输入特征下，存在大量特征完全一致的重复样本。

数据统计结果：**有效唯一输入特征组合仅410种**

核心问题：**大量相同输入特征 X 对应不同的产量标签 Y**，存在严重标签冲突。

**标签冲突统计（410种唯一X）：**

73种 X：仅对应单一类别（无冲突）

255种 X：对应两个不同类别

82种 X：对应三个不同类别

**冲突样本占比：337/410**，绝大多数特征组合存在标签矛盾。

## 20\. 标签冲突对神经网络的影响

神经网络为确定性映射模型：$f(X)=Y$，相同输入必然输出相同结果。

但当前数据集存在「同一特征X对应Low/Medium/High多类标签」的矛盾标注，属于**数据集本身的监督信号缺陷**。

该问题无法通过优化网络结构、调整超参数彻底解决。

**数据理论上限：**对冲突样本采用「多数投票标签」修正后，数据集最高可实现准确率仅 **64\.5%**。

因此 Model 1 50% 的准确率，不仅受网络结构简单影响，**数据标签冲突是核心限制因素**。

## 21\. 当前模型的定位

Model 1 不追求最优性能，核心作用是**搭建一套完整、可运行的水稻产量分类基线实验流程**，为后续所有迭代模型提供对比基准。

**完整基线实验链路：**

Paddy Dataset → 数据清洗 → 单位面积产量计算 → 三分类标签构建 → 33维原始特征提取 → One\-Hot编码（60维）→ 标准化处理 → 8:2数据集划分 → 60\-128\-64\-3 MLP网络训练 → CrossEntropy损失\+Adam优化 → 50轮迭代 → 基准准确率50%

## 22\. 已知限制

### 22\.1 重复输入样本多

数据集存在大量特征完全一致的重复样本，数据冗余度高。

### 22\.2 特征与标签映射冲突

相同输入特征对应多类别标签，从数据层面限制模型分类精度上限。

### 22\.3 数据集划分不合理

普通随机划分导致相同特征样本同时出现在训练集、测试集，存在数据泄露风险，可改用分组划分策略。

### 22\.4 实验不可复现

未固定随机种子，数据划分、模型训练结果存在波动。

### 22\.5 评价指标单一

仅使用准确率，无法全面评估模型在各类别上的分类效果。

### 22\.6 标签离散化损失信息

将连续的单位面积产量人为离散为三分类，导致相近产量样本被划分为不同类别，丢失精细数值信息。

### 22\.7 无系统超参数优化

网络结构、学习率、批次大小等参数均为固定值，未开展参数搜索、正则化、早停等优化实验。

## 23\. 后续可能的改进方向

基于Model 1基线模型，后续迭代版本可从以下维度优化升级：

```Plain Text
1. 固定随机种子，提升实验可复现性
2. 新增精确率、召回率、F1分数、混淆矩阵等评价指标
3. 清洗重复样本、处理标签冲突问题
4. 引入多数标签修正策略优化数据集
5. 引入软标签优化分类监督信号
6. 采用分组式数据集划分，规避数据交叉泄露
7. 基于单位面积产量开展回归任务实验
8. 对比分类任务与回归任务的模型性能
9. 开展Hectares字段消融实验，验证特征有效性
10. 引入特征筛选机制，剔除无效冗余特征
11. 增加Dropout、权重衰减等正则化策略，防止过拟合
12. 引入早停机制，优化模型训练收敛效果
13. 调优网络深度、隐藏层维度，适配数据特征
14. 系统性超参数搜索与调优
15. 与传统机器学习模型（SVM、随机森林等）进行性能对比
```

## 24\. 运行环境

**核心依赖库：**Python、PyTorch、scikit\-learn、pandas、numpy、matplotlib、torchsummary

**建议补充：**新增requirements\.txt文件，记录各依赖库具体版本，进一步提升实验可复现性。

## 25\. 当前代码执行流程

**程序入口：**if \_\_name\_\_ == "\_\_main\_\_":

**默认执行顺序：**create\_dataset\(\) → train\(\) → evaluate\(\)

**注意事项：**clean\_dataset\(\) 函数默认处于注释关闭状态。首次运行代码需取消注释，生成清洗后的数据集文件，后续运行可保持注释状态，避免重复处理数据。

## 26\. Model 1 总结

Model 1 是本项目首个水稻单位面积产量三分类基线神经网络模型，搭建了完整的数据处理、模型训练、测试评估流水线，为后续迭代优化提供标准对比基准。

**模型核心参数汇总：**

```Plain Text
数据集总样本数：2789
原始字段数：45
训练输入特征数：33
编码后输入维度：60
任务类型：三分类（Low / Medium / High）
网络结构：60 → 128 → 64 → 3
激活函数：ReLU
损失函数：交叉熵损失
优化器：Adam
学习率：0.001
迭代轮数：50
训练批次：16
测试批次：8
数据集划分：80%训练、20%测试
基线准确率：约50%
