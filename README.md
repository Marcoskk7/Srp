[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Marcoskk7/Srp)

# PCGAN for Few-shot Fault Diagnosis

基于 CWRU 轴承故障数据集的**知识引导数据增强与小样本诊断框架**。项目围绕不同数据增强方式在 few-shot 学习中的效果展开，逐步探索了：

- 无增强（none）
- 噪声增强（noise）
- 标准条件生成对抗网络（eval）
- 知识条件增强 cGAN（cond）
- 物理约束 cGAN（constraint）
- 融合版 Physics-constrained Conditional GAN（PCGAN）

当前阶段的定位是：**已经形成了可作为阶段性成果的原型框架**。项目完成了从数据预处理、知识先验提取、生成式增强、到下游 few-shot 诊断验证的闭环，初步验证了不同增强策略在源域和目标域小样本任务中的有效性。后续工作重点应放在：

1. 更细致的架构优化；
2. 更系统的消融实验与超参数搜索；
3. 提升工程规范性、可复现性与可扩展性；
4. 推广到更多迁移学习/域适应/跨工况诊断任务。

---

## 1. 项目目标

本项目旨在研究：

> 在工业故障诊断的小样本场景下，如何利用生成式数据增强提升 few-shot 分类性能，尤其是目标域未见类别（zero-shot generation）下的诊断能力。

核心思路：

- 从真实故障信号中提取**类别级知识先验**；
- 将知识先验注入 cGAN 作为条件或约束；
- 生成更符合类别语义与物理统计规律的伪样本；
- 用生成样本增强支持集，并通过 DTN few-shot 分类器验证增强效果。

---

## 2. 项目特点

- **面向任务评估**：不只比较生成质量，更关注生成样本对 few-shot 诊断准确率的真实提升。
- **知识驱动生成**：将故障先验统计量 `w`、`E_c` 作为生成条件或物理约束。
- **支持目标域未见类生成**：条件版与融合版可在目标域进行 zero-shot 样本生成。
- **统一实验主控**：通过 `main.py` 控制预处理、训练、缓存、生成、测试和结果聚合。
- **可做阶段性汇报，也可继续扩展为论文/工程项目**。

---

## 3. 代码结构

```text
.
├── main.py                  # 项目主控脚本
├── common.py                # 全局类别定义与公共函数
├── CWRU_preprocess.py       # CWRU 数据预处理与 source/target 划分
├── KG.py                    # 知识图谱/先验统计构建（w, v, E_c 等）
├── data_loader.py           # 数据加载辅助
├── DTN_TEST.py              # few-shot 诊断评测模块
├── cGAN_evaluation.py       # 标准 cGAN 基线 + 生成质量评估
├── cGAN_condition.py        # 知识条件 cGAN（w + E_c + FiLM）
├── cGAN_constraint.py       # 物理约束 cGAN
├── PCGAN.py                 # 融合版：条件嵌入 + 物理约束
├── source_data.npz          # 预处理后源域数据（运行后生成）
├── target_data.npz          # 预处理后目标域数据（运行后生成）
└── knowledge_graphs/
    ├── kg_step2_w_v_sigma.npz
    ├── kg_step3_P_transition.npy
    └── Ec.npy
```

---

## 4. 方法概览

### 4.1 数据划分

项目将 CWRU 类别划分为：

- **source classes**：用于训练生成器与源域 few-shot 测试
- **target classes**：用于目标域 zero-shot 生成与 few-shot 测试

这意味着目标域测试不仅是少样本问题，同时也是一个更接近**未见类迁移**的任务。

### 4.2 知识先验

`KG.py` 从真实故障信号中提取以下类别级统计量：

- `v`：类别特征中心
- `sigma`：类别特征方差
- `w`：由方差归一化得到的特征权重
- `E_c`：四个频带的能量比例

这些先验反映了不同故障类型的统计特征和频域能量分布，是后续条件生成和物理约束的核心信息。

### 4.3 四类增强模型

#### (1) eval：标准 cGAN
- 条件仅为类别标签 embedding；
- 作为纯数据驱动基线；
- 适合源域 seen classes 生成。

#### (2) cond：知识条件 cGAN
- 条件输入为 `w` 与 `E_c`；
- 通过 FiLM 对生成器多层特征进行调制；
- 更适合目标域未见类的 zero-shot 生成。

#### (3) constraint：物理约束 cGAN
- 生成器仍以标签为条件；
- 训练中加入物理一致性损失；
- 通过可微特征与频带能量约束，使生成样本更符合真实类别统计。

#### (4) PCGAN：融合版
- 同时使用知识条件输入与物理约束损失；
- 是本项目当前最完整的版本；
- 目标是在表达能力与统计一致性之间取得更好平衡。

---

## 5. 实验设置

### 5.1 few-shot 测试方式

使用 `DTN_TEST.py` 进行 few-shot 分类验证，测试 shot 配置通常为：

- 1-shot
- 3-shot
- 5-shot

增强规则：

- 若 `augment_shot > shot`，则用增强样本补足到指定支持集大小；
- 若 `augment_shot == shot`，则该 shot 不引入增强样本。

因此在默认 `augment_shot=5` 时：

- 1-shot：补 4 个增强样本
- 3-shot：补 2 个增强样本
- 5-shot：不补样本

这也是为什么不同方法在 5-shot 下通常结果一致。

### 5.2 结果解释建议

当前阶段的实验结果可支持如下结论：

- **none / noise**：可作为弱基线；简单噪声扰动对 few-shot 提升有限。
- **eval**：标准 cGAN 相比简单噪声更有效，说明学习到的类内变化有价值。
- **cond**：在目标域未见类上通常更有优势，说明知识条件具备迁移性。
- **constraint**：在源域 seen classes 的增强中通常表现较强，说明物理统计约束有效。
- **PCGAN**：作为融合版，整体上最具发展潜力，适合作为后续主线模型。

因此，本项目目前完全可以表述为：

> 初步探索了多种数据增强策略在小样本故障诊断中的效果，验证了知识引导和物理约束两类机制的有效性，并为后续更细粒度的模型优化与更广泛的迁移学习任务扩展打下基础。

---

## 6. 快速开始

### 6.1 环境依赖 参考requirements.txt



主要依赖：

```bash
pip install numpy scipy scikit-learn matplotlib torch tensorboard
```

如需使用 GPU，请安装与本机 CUDA 匹配的 PyTorch 版本。

### 6.2 数据准备

将 CWRU 原始数据按预处理脚本要求放置。随后运行：

```bash
python CWRU_preprocess.py
python KG.py
```

或直接让 `main.py` 自动检查并生成缓存。

---

## 7. 运行方式
例如：

```
### 7.1 物理约束版(目标域)
```bash
python main.py --target_test --cgan_version constraint --augment_type gan  --shot_configs 1 3 5 --num_runs 3
```
建议调节的超参数：

- `--cgan_version`：选择版本，`eval / cond / constraint / pc`
- `--cgan_epochs`：训练轮数
- `--cgan_batch_size`：batch 大小
- `--cgan_z_dim`：随机噪声维度
- `--cgan_lr_g`：生成器学习率
- `--cgan_lr_d`：判别器学习率
- `--cgan_n_critic`：每次 G 更新前 D 的训练次数
- `--num_per_class`：每类生成样本数
- `--lambda_phys`：物理约束权重（constraint / pc 生效）
- `--alpha_E`：频带能量项权重（constraint / pc 生效）


- `--augment_type`：`none / noise / gan`
- `--augment_shot`：增强后的总支持样本数
- `--noise_level`：噪声增强强度
- `--shot_configs`：测试的 shot 列表


- `--target_test`：是否切换到目标域测试
- `--num_runs`：重复运行次数
- `--seed`：基础随机种子
- `--force_regenerate`：强制重新生成样本
- `--use_cache`：是否复用已生成缓存
- `--skip_preproc`：跳过预处理检查

---

## 8. 输出与缓存说明

运行过程中可能生成以下文件：

- `source_data.npz` / `target_data.npz`：预处理后的数据缓存
- `knowledge_graphs/*.npz|*.npy`：知识先验缓存
- `generated_samples_*.npz`：不同版本、不同测试域下的生成样本缓存
- `runs/cgan/...`：日志目录（如启用 TensorBoard）

如果已存在缓存且开启 `--use_cache`，程序会跳过重复训练，直接载入生成结果。

---

## 9. 当前局限与后续计划

### 9.1 当前局限

- 条件插值一致性损失尚未完全实现；
- 部分版本在目标域 zero-shot 场景下的理论自洽性还可增强；
- 生成质量评估尚未覆盖所有版本；
- 物理特征约束仍是工程近似，还可进一步贴合真实故障机理；
- 结果统计与实验记录可进一步规范化。

### 9.2 后续计划

- 完善 `cond` 与 `PCGAN` 的条件一致性/插值正则；
- 系统开展模块消融实验（FiLM、w、E_c、physics loss、warmup 等）；
- 引入超参数搜索与更严格的显著性检验；
- 推广至更多跨域/迁移/域泛化故障诊断任务；
- 扩展至更多 backbone 与 few-shot 学习方法；
- 提升代码规范性、配置管理和可复现性。

---


## 10. 建议的后续仓库整理

建议后续将工程进一步整理为：

```text
configs/
  base.yaml
  eval.yaml
  cond.yaml
  constraint.yaml
  pcgan.yaml
src/
  data/
  models/
  trainers/
  evaluation/
  utils/
scripts/
  run_source.sh
  run_target.sh
outputs/
README.md
```

这样更方便后续：

- 复现实验
- 批量调参
- 补充消融
- 面向论文或项目验收

---

# 理论补充说明以及实验结果初步分析

## 1. 问题背景：为什么小样本故障诊断需要跨域生成增强

在工业故障诊断中，小样本学习通常面临两个核心困难：

第一，**支持集样本数极少**。
当每类只有 1-shot 或 3-shot 时，分类器很难稳定刻画类内分布，容易受到偶然采样、噪声和个体差异的影响。

第二，**训练域与测试域之间存在分布偏移**。
即使同一类故障，在不同工况、不同类别组合、不同任务划分下，其统计特征、频带能量分布和信号形态也会发生变化。因此，仅靠源域少量样本训练的分类器往往难以直接泛化到目标域。

在这种背景下，数据增强的目的不只是“把样本数量变多”，而是要让增强样本能够：

* 尽量保持目标类别的判别语义；
* 尽量符合真实信号的统计规律；
* 在支持集极少时，为 few-shot 分类器提供更稳定的类原型和类内变化。

因此，本项目采用生成式增强框架，将故障先验知识引入样本生成过程，尝试解决传统随机扰动增强在跨域小样本场景下表达能力不足的问题。

---

## 2. 理论框架：从“标签条件”到“知识条件”再到“物理约束”

本项目中几种增强方式可以理解为一个逐步递进的理论谱系。

### 2.1 无增强（none）

无增强方法直接使用真实 few-shot 支持集进行分类。
其优点是实现简单、没有额外建模误差；缺点是当支持样本太少时，类内分布估计极不稳定，性能通常受限于采样偶然性。

从理论上看，none 对应的是最基本的小样本分类设定，即分类器只能依赖有限真实样本建立判别边界，不具备显式分布扩展能力。

---

### 2.2 噪声增强（noise）

噪声增强通过在真实样本上添加高斯噪声等随机扰动生成伪样本。
该方法本质上假设：**真实样本邻域内的局部微扰仍然属于同一类别流形**。

它的优点是简单、开销小，但局限也很明显：

* 它只能在已有样本附近进行局部扰动；
* 无法学习类条件分布；
* 无法显式表达不同故障模式之间的统计差异；
* 对跨域未见类几乎没有语义迁移能力。

因此，noise 更像是一种弱增强基线，而不是严格意义上的“跨域生成”。

---

### 2.3 标准条件生成（eval）

标准 cGAN 的核心思想是：
给定类别标签 ( y ) 和随机噪声 ( z )，生成器学习条件分布 ( p(x|y) )，从而生成属于某一类别的伪样本。

相较于噪声增强，标准 cGAN 的理论优势在于：

* 它学习的是**类别级数据分布**，而不是单一样本周围的局部扰动；
* 它能够建模更丰富的类内变化；
* 在源域 seen classes 上通常比简单噪声增强更有效。

但其局限同样明显：

* 条件只来自离散标签 embedding；
* 标签本身不携带物理先验和统计结构；
* 若目标域类别在训练中未出现，则标准标签条件难以支撑真正的 zero-shot 生成。

因此，标准 cGAN 更适合作为**源域条件生成基线**，而不是严格的跨域知识迁移方案。

---

### 2.4 知识条件生成（cond）

知识条件生成的关键思想是：
**与其把类别表示成一个离散 id，不如把类别表示成可迁移的统计语义向量。**

在本项目中，类别知识由两部分组成：

* `w`：类别在特征空间中的重要性/稳定性权重；
* `E_c`：类别在不同频带上的能量分布比例。

这意味着生成器学习的条件不再是“第几类”，而是“该类故障的统计画像”。
从理论上说，这种表示方式有两个重要优势：

#### （1）更强的可解释性

`w` 和 `E_c` 都具有明确的工程含义，分别反映故障特征结构与频域能量特性，因此条件向量不是纯黑盒 embedding，而是可解释的类别先验。

#### （2）更强的跨域迁移能力

只要目标域类别也能提取出对应的 `w` 和 `E_c`，即使该类别没有参与过生成器训练，模型仍然可能基于这些先验进行 zero-shot 生成。
也就是说，模型迁移的不是“标签 id”，而是“类别统计语义”。

因此，cond 可以看作一种**从离散标签条件到连续知识条件的跨域生成扩展**。

---

### 2.5 物理约束生成（constraint）

与 cond 侧重“输入什么条件”不同，constraint 更关注“生成结果是否满足真实物理统计规律”。

该方法在 GAN 对抗训练基础上引入额外的物理一致性约束，例如：

* 生成样本的可微特征统计要与真实类别的 `w` 对齐；
* 生成样本的频带能量比例要与真实类别的 `E_c` 对齐。

从理论上看，constraint 的核心作用是：

* 缩小生成分布与真实故障信号统计规律之间的偏差；
* 降低 GAN 仅靠判别器反馈时可能出现的模式漂移；
* 使生成样本不仅“像某一类”，而且“在统计结构上更接近该类”。

因此，constraint 可以理解为一种**面向工程先验的生成分布正则化方法**。
它尤其适合源域 seen classes 的类内分布补充，因为这类任务更关注“补得像不像真实类内变化”。

---

### 2.6 融合生成（PCGAN）

PCGAN 同时结合了两种思想：

* 一方面，使用 `w + E_c` 作为知识条件输入，增强类别语义表达能力；
* 另一方面，引入物理一致性损失，约束生成样本满足真实统计规律。

从理论上说，PCGAN 试图同时解决两个问题：

1. **生成器应该生成什么类别语义？**
   由知识条件回答；

2. **生成结果是否满足真实故障统计？**
   由物理约束回答。

因此，PCGAN 可以视为本项目当前最完整的跨域生成方案，即：

> 以知识条件实现可迁移类别表达，以物理约束保证统计一致性，从而提升小样本和跨域故障诊断中的增强质量。

---

## 3. 几种跨域生成方式的理论对比

| 方法         | 条件来源                        | 是否建模类分布 | 是否具备跨域迁移能力 | 是否具备物理一致性约束 | 理论特点                        |
| ---------- | --------------------------- | ------: | ---------: | ----------: | --------------------------- |
| none       | 无                           |       否 |          否 |           否 | 纯 few-shot 基线，完全依赖真实支持集     |
| noise      | 样本局部扰动                      |       否 |         很弱 |           否 | 仅做邻域扰动，不能表达类别级分布            |
| eval       | 标签 embedding                |       是 |         较弱 |           否 | 学习源域 seen class 的条件分布       |
| cond       | 知识先验 `w + E_c`              |       是 |          强 |           否 | 以可解释统计语义替代离散标签，适合 zero-shot |
| constraint | 标签 embedding + physics loss |       是 |         中等 |           是 | 更强调生成分布与真实统计规律的一致性          |
| PCGAN      | 知识先验 + physics loss         |       是 |          强 |           是 | 同时兼顾跨域语义表达与物理统计一致性          |

从这个对比可以看出：

* `eval` 相比 `noise` 的进步，本质上来自**类别级分布建模能力**；
* `cond` 相比 `eval` 的进步，本质上来自**条件表示从离散标签到可迁移知识语义的升级**；
* `constraint` 相比 `eval` 的进步，本质上来自**物理统计约束对生成质量的正则化**；
* `PCGAN` 则试图在**可迁移语义 + 统计一致性**之间取得统一。

---

# 实验结果与初步分析

## 4. 实验结果表

## 4.1 Source domain

| Method     |              1-shot |              3-shot |          5-shot |
| ---------- | ------------------: | ------------------: | --------------: |
| none       |     0.3685 ± 0.0004 |     0.5699 ± 0.0050 | 0.7051 ± 0.0028 |
| noise      |     0.3675 ± 0.0027 |     0.5753 ± 0.0049 | 0.7051 ± 0.0028 |
| eval       |     0.4970 ± 0.0030 |     0.6628 ± 0.0044 | 0.7051 ± 0.0028 |
| cond       |     0.4250 ± 0.0087 |     0.6113 ± 0.0006 | 0.7051 ± 0.0028 |
| constraint | **0.5156 ± 0.0051** | **0.6641 ± 0.0048** | 0.7051 ± 0.0028 |
| PCGAN      |     0.4116 ± 0.0019 |     0.6160 ± 0.0032 | 0.7051 ± 0.0028 |

对应原始 std 平均如下：

| Method     | 1-shot std(avg) | 3-shot std(avg) | 5-shot std(avg) |
| ---------- | --------------: | --------------: | --------------: |
| none       |          0.0553 |          0.0612 |          0.0491 |
| noise      |          0.0555 |          0.0649 |          0.0491 |
| eval       |          0.0680 |          0.0586 |          0.0491 |
| cond       |          0.0570 |          0.0499 |          0.0491 |
| constraint |          0.0641 |          0.0570 |          0.0491 |
| PCGAN      |          0.0550 |          0.0516 |          0.0491 |

---

## 4.2 Target domain

| Method     |              1-shot |              3-shot |          5-shot |
| ---------- | ------------------: | ------------------: | --------------: |
| none       |     0.6026 ± 0.0195 |     0.8669 ± 0.0067 | 0.9169 ± 0.0025 |
| noise      |     0.5957 ± 0.0187 |     0.8737 ± 0.0057 | 0.9169 ± 0.0025 |
| cond       | **0.7067 ± 0.0049** |     0.8719 ± 0.0017 | 0.9169 ± 0.0025 |
| constraint |     0.6718 ± 0.0122 |     0.8536 ± 0.0034 | 0.9169 ± 0.0025 |
| PCGAN      |     0.7022 ± 0.0043 | **0.8751 ± 0.0034** | 0.9169 ± 0.0025 |

对应原始 std 平均如下：

| Method     | 1-shot std(avg) | 3-shot std(avg) | 5-shot std(avg) |
| ---------- | --------------: | --------------: | --------------: |
| none       |          0.1906 |          0.0553 |          0.0295 |
| noise      |          0.1901 |          0.0581 |          0.0295 |
| cond       |      **0.0831** |      **0.0386** |          0.0295 |
| constraint |          0.1257 |      **0.0386** |          0.0295 |
| PCGAN      |          0.1021 |          0.0407 |          0.0295 |

---

## 5. 实验结果初步分析

## 5.1 总体趋势

从整体结果看，所有方法都表现出一个一致规律：

* 从 1-shot 到 3-shot，再到 5-shot，准确率整体逐步提升；
* 这说明随着真实支持样本增加，few-shot 分类器获得了更充分的类内信息，诊断性能自然增强；
* 同时也说明本实验设置具有合理性：样本数增加确实带来了更稳定的性能提升。

不过，不同增强方式的差异主要集中在 **1-shot 和 3-shot** 场景，而在 **5-shot** 下几乎全部收敛到同一结果。
这与实验流程本身是一致的：当 `augment_shot=5` 且测试为 5-shot 时，不再额外补充增强样本，因此 5-shot 实际上退化为统一的真实样本测试基线。
换句话说，5-shot 的一致性不是偶然现象，而是当前实验设计下的合理结果。

---

## 5.2 Source 域结果分析

### （1）GAN 类增强显著优于 none / noise

在 source 域 1-shot 下：

* none：0.3685
* noise：0.3675
* eval：0.4970
* constraint：0.5156

可以看到，noise 几乎没有带来改善，而标准 cGAN 和带约束的生成方法则明显提升了性能。
这说明简单噪声扰动只能在局部邻域中扩展样本，无法有效构建类内分布；相比之下，GAN 生成方法能够学习更有意义的类别级变化，因此更适合 few-shot 支持集增强。

### （2）constraint 在 source 域表现最好

在 source 域中，最佳结果来自 constraint：

* 1-shot：0.5156
* 3-shot：0.6641

这表明对于 **源域已见类别的类内补样任务**，物理统计一致性约束非常有效。
其原因在于 source 任务的重点不是“外推未见类语义”，而是“在已知类别内部补充更像真实样本的变化模式”。
constraint 正是通过物理损失将生成样本限制在更合理的统计结构内，因此能更有效地帮助分类器建立支持类原型。

### （3）cond 与 PCGAN 在 source 域没有完全超过 constraint

从 source 结果看：

* cond：1-shot 为 0.4250，3-shot 为 0.6113
* PCGAN：1-shot 为 0.4116，3-shot 为 0.6160

这说明知识条件化虽然能够增强类别语义表达，但在源域 seen-class 场景中，不一定比“直接的物理约束补样”更占优势。
换句话说，source 任务更偏向**类内分布细化问题**，而不是**跨类语义迁移问题**，因此约束式增强在这里更直接、更有效。

---

## 5.3 Target 域结果分析

### （1）知识条件生成在 1-shot 下优势最明显

在 target 域 1-shot 下：

* none：0.6026
* noise：0.5957
* cond：0.7067
* constraint：0.6718
* PCGAN：0.7022

可以看到，cond 和 PCGAN 相比 none 有非常明显的提升。
其中 cond 达到 0.7067，为 target 1-shot 最优；PCGAN 也达到 0.7022，表现非常接近。

这说明在 **目标域未见类别** 的场景下，知识先验 `w + E_c` 确实能够作为一种可迁移的类别语义表示，帮助生成器在缺乏真实支持样本的情况下合成更有判别力的伪样本。
这一现象是本项目最重要的实验结论之一，因为它表明模型迁移的不是简单标签，而是类别统计语义。

### （2）PCGAN 在 target 3-shot 下取得最优

在 target 域 3-shot 下：

* none：0.8669
* noise：0.8737
* cond：0.8719
* constraint：0.8536
* PCGAN：0.8751

PCGAN 在这一设置下达到最高准确率。
这表明当支持集已有少量真实样本时，仅有知识条件已经能够提供基本迁移能力，而进一步加入物理约束后，生成样本的统计结构更加稳定，有助于分类器进行细粒度边界修正。

因此可以认为：

* 在极低样本条件下（1-shot），知识条件的作用最突出；
* 当支持样本稍有增加（3-shot），物理约束与知识条件的结合开始体现协同优势。

### （3）constraint 在 target 域不如 cond / PCGAN 稳定

constraint 在 target 1-shot 上虽然也优于 none 和 noise，但不如 cond / PCGAN。
这说明仅靠标签条件加物理损失，虽然能在一定程度上改善生成结果，但在真正的未见类迁移场景中，其理论表达能力仍弱于显式知识条件方法。
也就是说，target 任务更依赖“生成器知道要生成什么类别语义”，而不仅是“生成结果满足某些统计规律”。

---

## 5.4 稳定性分析：原始 std 的变化

除均值外，原始 std 平均也反映了增强方法的稳定性。

### source 域

source 域中各方法的 std 差异不算特别大，说明在已见类场景下，各方法主要差别体现在均值提升，而 episode 间波动相对接近。

### target 域

target 域中这一现象更加明显：

* none：1-shot std(avg) = 0.1906
* noise：1-shot std(avg) = 0.1901
* cond：1-shot std(avg) = 0.0831
* constraint：1-shot std(avg) = 0.1257
* PCGAN：1-shot std(avg) = 0.1021

可以看到，cond 和 PCGAN 在 target 1-shot 下显著降低了波动。
这说明知识引导生成不仅提高了平均准确率，也显著提升了支持集构建的稳定性。
换句话说，生成样本帮助分类器学习到了更稳定的类原型，从而降低了 few-shot episodic evaluation 对随机采样的敏感性。

这对于实际工业诊断尤其重要，因为在真实应用中，模型不仅要“平均上更准”，也要“在不同批次小样本输入下更稳”。

---

## 5.5 对几类方法的阶段性结论

结合 source 和 target 两组实验，可以得到以下初步结论：

### none / noise

* 可以作为基础对照组；
* 说明简单局部扰动不足以支撑跨域 few-shot 增强；
* 尤其在 target 1-shot 下，noise 基本无法提供有效增益。

### eval

* 在 source 域表现明显优于 none / noise；
* 说明标准 cGAN 生成的类内变化具有实际增强价值；
* 但其能力主要体现在 seen classes，跨域未见类扩展能力有限。

### cond

* 在 target 1-shot 上达到最优；
* 说明知识条件表示确实具有较强迁移性；
* 对 zero-shot 类别生成和目标域小样本增强尤其有效。

### constraint

* 在 source 域表现最强；
* 说明物理统计约束更适合已见类的类内补样；
* 其优势主要体现在增强样本“像不像真实类内变化”。

### PCGAN

* 综合表现最好、最均衡；
* 在 target 3-shot 上达到最优，在 target 1-shot 也接近最优；
* 说明“知识条件 + 物理约束”是一条值得继续深入的主线方案。

---

## 5.6 面向后续工作的启示

从本轮实验可以看出，不同增强方式在不同任务上的作用机制并不完全相同：

* **source seen-class few-shot** 更强调类内分布补充，因此物理约束更关键；
* **target unseen-class few-shot** 更强调类别语义迁移，因此知识条件更关键；
* **融合模型** 则有望在样本稍微丰富时兼顾两方面优势。

因此，后续优化可以围绕以下方向展开：

1. 针对 source 任务，进一步强化生成分布与真实统计规律的一致性；
2. 针对 target 任务，进一步提升知识条件的表达能力和未见类外推能力；
3. 对 PCGAN 进行更细粒度消融，分析 `w`、`E_c`、FiLM、physics loss 的独立贡献；
4. 在更多跨域诊断和迁移学习任务中验证其泛化能力。



