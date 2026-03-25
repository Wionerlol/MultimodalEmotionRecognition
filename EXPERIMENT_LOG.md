# Experiment Log

用于记录每次模型结构、训练策略、推理流程的改动，以及对应实验结果、结论和原因分析。目标是为后续论文/报告中的 `Method`、`Ablation Study`、`Discussion`、`Conclusion` 提供可直接复用的素材。

## 使用规则

1. 每做一次可复现的改动，就新增一条记录。
2. 不只记录成功实验，失败实验也必须记录。
3. 结果至少写清：
   - 任务形态：`audio-only` / `video-only` / `multimodal`
   - 模型配置
   - 关键训练参数
   - 验证集 / 测试集指标
   - 与 baseline 的对比
4. 原因分析要区分：
   - 观察到的事实
   - 推测性的解释
5. 如果暂时没有结论，也要明确写 `待进一步验证`，不要留空。

---

## Baseline

### B0. 原始基线

- 日期：2026-03-24
- 任务：待补充
- 配置：原始项目默认配置
- 目的：作为所有后续结构改动和训练策略改动的对照组
- 结果：待补充
- 结论：待补充

---

## Experiment Entries

### E000. 历史优化总览（根据现有代码、项目文档与研究过程回顾补充）

- 日期：2026-03-24
- 状态：已整理
- 说明：
  - 本条不是单次实验，而是对项目早期到当前阶段的主要优化路径做结构化回顾。
  - 证据来源分为两类：
    - 代码/文档可验证：当前仓库中能直接看到实现痕迹
    - 历史实验回忆：根据你的补充保留，但不写成“当前代码事实”

### E000-A. Audio backbone: Mel-CNN / Mel-ResNet 向 WavLM 演进

- 日期：2026-03-24
- 状态：已完成
- 任务：`audio-only` 与 `multimodal`
- 模型：
  - 初始方案：`AudioCNN` / `AudioResNet18`
  - 后续方案：`WavLM`
- 改动内容：
  - 音频分支从基于 mel-spectrogram 的 CNN/ResNet，升级为预训练语音模型 `WavLM`
- 改动动机：
  - 传统卷积音频编码器对情感语音中的上下文依赖和语义信息建模有限
  - 希望利用大规模预训练语音模型获得更强的鲁棒性和表示能力
- 代码/文档依据：
  - [src/models/audio.py](/home/louis/projects/MultimodalEmotionRecognition/src/models/audio.py)
  - [src/models/wavlm_audio.py](/home/louis/projects/MultimodalEmotionRecognition/src/models/wavlm_audio.py)
  - [TECHNICAL_SUMMARY.md](/home/louis/projects/MultimodalEmotionRecognition/TECHNICAL_SUMMARY.md)
  - [README.md](/home/louis/projects/MultimodalEmotionRecognition/README.md)
- 当前结论：
  - `WavLM` 成为当前项目音频主干的关键升级方向
  - 该改动的核心价值不只是提升精度，也包括为后续跨模态融合提供更强的序列表示
- 可能原因分析：
  - 事实：
    - 当前代码中同时保留了 `AudioNet` 与 `WavLMAudioEncoder`
    - README 与技术总结均明确把 `WavLM pretrained speech model` 作为核心亮点
  - 推测：
    - WavLM 的预训练上下文表示比从头训练的 mel-CNN / mel-ResNet 更适合小规模情感数据集
- 是否纳入最终方案：是
- 论文可写结论：
  - “Replacing handcrafted mel-spectrogram encoders with a pretrained WavLM backbone substantially strengthened the audio branch and provided better sequence representations for downstream fusion.”

### E000-B. WavLM 两阶段训练

- 日期：2026-03-24
- 状态：已完成
- 任务：`audio-only` 与 `multimodal`
- 模型：
  - Stage 1：冻结 WavLM backbone，仅训练分类头/融合头
  - Stage 2：解冻最后若干层，使用更小 backbone learning rate 微调
- 改动内容：
  - 在 WavLM 上引入 two-stage finetuning
- 改动动机：
  - 直接全量微调容易破坏预训练表示，且在小数据集上训练不稳定
  - 需要先让任务头收敛，再对 backbone 做小步微调
- 代码/文档依据：
  - [src/models/wavlm_audio.py](/home/louis/projects/MultimodalEmotionRecognition/src/models/wavlm_audio.py)
  - [src/train.py](/home/louis/projects/MultimodalEmotionRecognition/src/train.py)
  - [TECHNICAL_SUMMARY.md](/home/louis/projects/MultimodalEmotionRecognition/TECHNICAL_SUMMARY.md)
  - [README.md](/home/louis/projects/MultimodalEmotionRecognition/README.md)
- 当前结论：
  - 两阶段训练是当前项目中 WavLM 微调的标准做法
  - 这类策略更强调稳定性与泛化，而不只是追求训练集拟合
- 可能原因分析：
  - 事实：
    - 当前代码已实现 `wavlm_stage=1/2`
    - 当前代码已支持按层解冻和分组学习率
  - 推测：
    - 该策略减少 catastrophic forgetting，并让预训练特征更平滑地迁移到情感识别任务
- 是否纳入最终方案：是
- 论文可写结论：
  - “A two-stage finetuning strategy improved optimization stability for WavLM by separating task-head learning from selective backbone adaptation.”

### E000-C. 数据增强与预处理优化

- 日期：2026-03-24
- 状态：已完成
- 任务：`audio-only`、`video-only`、`multimodal`
- 模型：数据层优化
- 改动内容：
  - Audio：
    - 加入真实酒吧背景噪声与高斯噪声回退
    - 使用不同权重/概率的噪声等级
    - mel 路径额外加入 `SpecAugment`
  - Video：
    - 加入随机模糊、亮度衰减、轻微高斯噪声等增强
    - 引入 `face-crop`，先做人脸检测与裁剪再输入视觉模型
- 改动动机：
  - 提高模型对真实环境干扰、低质量图像和人脸区域变化的鲁棒性
  - 让视觉分支更聚焦情绪相关区域，让音频分支更适应噪声环境
- 代码/文档依据：
  - [src/data/ravdess.py](/home/louis/projects/MultimodalEmotionRecognition/src/data/ravdess.py)
  - [src/utils/face_crop.py](/home/louis/projects/MultimodalEmotionRecognition/src/utils/face_crop.py)
  - [README.md](/home/louis/projects/MultimodalEmotionRecognition/README.md)
  - [TECHNICAL_SUMMARY.md](/home/louis/projects/MultimodalEmotionRecognition/TECHNICAL_SUMMARY.md)
- 当前结论：
  - 数据层增强是整个项目中非常关键的一类优化，因为它同时作用于单模态和融合模型
  - `face-crop` 对视频模态尤其重要，因为它减少了背景干扰并提高了表情区域占比
- 可能原因分析：
  - 事实：
    - 当前代码中音频增强概率分布为 `50% clean / 40% medium noise / 10% heavy noise`
    - 当前代码中视频增强包括模糊、暗化、轻噪声
    - 当前代码已支持 face crop 开关，且默认启用
  - 推测：
    - 这些增强共同提升了泛化能力，并减轻了 train/test domain gap
- 是否纳入最终方案：是
- 论文可写结论：
  - “Robust data preprocessing and augmentation, especially real-noise audio mixing and face-focused visual cropping, were necessary to stabilize both unimodal and multimodal performance.”

### E000-D. Fusion 路线：Concat -> Gated -> Cross-Attention

- 日期：2026-03-24
- 状态：已完成
- 任务：`multimodal`
- 模型：
  - `concat`
  - `gated`
  - `xattn`
- 改动内容：
  - 融合方法从最基础的特征拼接，逐步升级到门控融合，再到双向 cross-attention
- 改动动机：
  - `concat` 无法显式处理模态质量差异
  - `gated` 可以自适应控制音频/视频贡献
  - `cross-attention` 进一步建模音视频时序相关性
- 代码/文档依据：
  - [src/models/fusion.py](/home/louis/projects/MultimodalEmotionRecognition/src/models/fusion.py)
  - [TECHNICAL_SUMMARY.md](/home/louis/projects/MultimodalEmotionRecognition/TECHNICAL_SUMMARY.md)
  - [README.md](/home/louis/projects/MultimodalEmotionRecognition/README.md)
- 当前结论：
  - 这条优化路径构成了项目的核心研究主线
  - `concat` 更像强 baseline，`gated` 解决模态权重问题，`xattn` 代表更强表达能力但训练更难
- 可能原因分析：
  - 事实：
    - 当前代码完整保留了 `late / concat / gated / xattn`
    - README 中明确描述 `cross-attention` 优于简单 concat 或 late fusion
  - 推测：
    - 随着融合能力增强，模型表达能力提升，但也更容易带来过拟合和优化难度
- 是否纳入最终方案：是
- 论文可写结论：
  - “We progressively moved from simple concatenation to gated fusion and finally to bidirectional cross-attention, trading architectural simplicity for stronger cross-modal interaction modeling.”

### E000-E. 为抑制 xAttn 过拟合而加入训练稳定化策略

- 日期：2026-03-24
- 状态：已完成 / 部分历史项待确认
- 任务：`multimodal`
- 模型：`xattn`
- 改动内容：
  - 训练中加入：
    - `cosine annealing`
    - `early stopping`
    - `weight decay`
    - cross-attention 残差路径上的 `StochasticDepth`
    - warm-start 式分支初始化与两阶段训练
  - 历史实验回顾中提到：
    - `warmup`
- 改动动机：
  - `xattn` 表达能力强，但更容易过拟合或训练不稳定，需要更强的正则和更平滑的优化过程
- 代码/文档依据：
  - 代码/文档可验证项：
    - [src/train.py](/home/louis/projects/MultimodalEmotionRecognition/src/train.py)
    - [src/models/fusion.py](/home/louis/projects/MultimodalEmotionRecognition/src/models/fusion.py)
    - [TRAINING_GUIDE.md](/home/louis/projects/MultimodalEmotionRecognition/TRAINING_GUIDE.md)
    - [SYSTEM_SUMMARY.md](/home/louis/projects/MultimodalEmotionRecognition/SYSTEM_SUMMARY.md)
    - [TECHNICAL_SUMMARY.md](/home/louis/projects/MultimodalEmotionRecognition/TECHNICAL_SUMMARY.md)
  - 历史实验回忆项：
    - `warmup` 是你补充说明的优化尝试，但当前仓库代码里没有直接保留对应实现痕迹
- 当前结论：
  - 对 `xattn` 这类高容量融合模型，结构改进必须配套训练稳定化策略，否则容易出现过拟合和收益不稳定
  - `StochasticDepth + scheduler + early stopping` 属于当前代码层面可以确认的有效防护措施
- 可能原因分析：
  - 事实：
    - 当前训练器已实现 cosine annealing 与 early stopping
    - 当前 `xattn` 残差分支已实现 `StochasticDepth`
    - 当前训练流程已支持 warm-start 载入单模态 checkpoint
  - 推测：
    - `warmup` 若曾有效，主要作用应是降低高容量注意力模块在训练初期的梯度震荡
    - `xattn` 的收益往往依赖更精细的优化超参数，而不是单靠结构升级
- 是否纳入最终方案：是；其中 `warmup` 标记为历史经验，待你补充具体实现细节
- 论文可写结论：
  - “Cross-attention provided a stronger fusion mechanism but also introduced clear overfitting risk, which required additional stabilization strategies such as cosine annealing, early stopping, residual regularization, and staged optimization.”

### E000-F. 当前可以直接写进报告的方法演进摘要

- 日期：2026-03-24
- 状态：可直接复用
- 摘要：
  - 音频分支从传统 mel 编码器逐步升级到预训练 `WavLM`，并采用两阶段微调来提高稳定性
  - 数据层引入真实噪声混合、视频随机增强和人脸裁剪，以提升泛化能力
  - 融合层从 `concat` 演化到 `gated` 和 `cross-attention`，逐步增强模态交互能力
  - 随着模型容量提升，训练端同步加入 `cosine annealing`、`early stopping`、`weight decay`、`StochasticDepth` 等稳定化策略
  - `warmup` 可作为历史实验尝试写入报告，但建议标注为“曾尝试用于缓解训练初期不稳定”，除非你后面再补充更具体的实验记录

---

### E001. WavLM audio-only: Temporal Attention Pooling vs Mean Pooling vs Temporal Transformer

- 日期：2026-03-24
- 状态：已完成
- 任务：`audio-only`
- 模型：
  - Backbone: `WavLM`
  - Temporal pooling candidates: `mean`, `attn`, `transformer`
- 改动内容：
  - 在音频时序聚合位置，将原本的 `mean pooling` 替换为可配置的 `Temporal Attention Pooling` 或 `Temporal Transformer Pooling`
- 改动动机：
  - 期望保留并重加权时间维信息，减少简单均值池化带来的时序信息损失
- 实验现象：
  - `attn pooling` 与 `mean pooling` 效果基本一致
  - 二者都明显优于 `temporal transformer`
- 当前结论：
  - 对 `audio-only + WavLM` 而言，额外堆叠一个 temporal transformer 不是有效改进
  - 可学习 attention pooling 没有明显超过 mean pooling，但至少没有明显退化
- 可能原因分析：
  - 事实：
    - WavLM 主干本身已经包含多层 transformer，已经完成较充分的时序建模
    - 额外 temporal transformer 没有带来收益，反而性能下降
  - 推测：
    - 原因 1：WavLM 已具备强时序建模能力，后续再加 temporal transformer 会造成重复建模
    - 原因 2：新增模块扩大了参数量，但仍沿用原学习率和训练轮数，优化不足
    - 原因 3：数据规模较小，额外 transformer 更容易过拟合或训练不稳定
    - 原因 4：audio-only 任务的判别信息更依赖 backbone 已抽取出的全局语义，而非额外时序重编码
- 论文可写结论：
  - “For the WavLM-based audio-only setting, replacing temporal mean pooling with an additional temporal transformer did not improve performance. We conjecture that WavLM already performs sufficiently strong temporal modeling internally, making the extra transformer redundant under the current data scale and optimization setup.”
  - “A likely reason is that WavLM already provides strong contextualized temporal representations, and further temporal modeling introduces optimization difficulty and representation drift on a small dataset.”
- 后续建议：
  - 固定 `mean` 作为 audio-only 的默认基线
  - 若继续验证 `transformer pooling`，应单独调学习率、weight decay、dropout 和训练轮数
  - 优先在 `xattn` 多模态分支验证 temporal attention/transformer 是否更有价值

### E002. Video-only: Mean Pooling vs Temporal Attention Pooling vs Temporal Transformer

- 日期：2026-03-25
- 状态：已完成
- 任务：`video-only`
- 模型：
  - Backbone: `ResNet18`
  - Temporal pooling candidates: `mean`, `attn`, `transformer`
- 改动内容：
  - 在视频分支时间聚合位置，将原本的 `mean pooling` 替换为 `Temporal Attention Pooling` 或 `Temporal Transformer Pooling`
- 改动动机：
  - 视频模态天然包含帧间动态信息，理论上比音频 clip-level 聚合更可能受益于显式时序建模
- 实验现象：
  - `temporal transformer` 仍然是三者里效果最差的
  - 但它相对视频分支的表现，比在 audio 分支上的表现更好，说明视频模态确实更需要时序建模
  - `attention pooling` 的训练集表现与 `mean pooling` 基本一致
  - 但 `attention pooling` 在验证集和测试集上不如 `mean pooling`
- 当前结论：
  - 视频分支相比音频分支更有时序建模需求，但当前 temporal transformer 仍未带来收益
  - 对当前项目的数据规模和视频采样设置而言，`mean pooling` 仍然是视频分支最稳健的选择
  - `attention pooling` 虽然具备可学习加权能力，但泛化能力弱于 `mean pooling`
- 可能原因分析：
  - 事实：
    - `transformer pooling` 在 video 分支里仍然落后于 `mean pooling`
    - `attention pooling` 没有明显训练困难，训练集拟合能力正常
    - `attention pooling` 的退化主要发生在 validation / test，而不是 training
  - 推测：
    - 原因 1：video 分支确实需要一定时序信息，但当前 temporal 模型没有学到“关键帧”，反而引入了噪声或优化困难
    - 原因 2：数据集规模较小、每个样本抽样帧数也较少，temporal 模型不足以学到稳定的时间规律
    - 原因 3：`ResNet18 + mean pooling` 本身已经很强，因为每帧经过高层卷积特征提取后，表示已接近分类语义特征；此时 mean pooling 只是做稳定聚合
    - 原因 4：transformer 的信息路径更复杂，参数更多，更难在当前训练设置下充分优化
    - 原因 5：attention pooling 更容易过拟合训练集中的伪模式；一旦某几帧权重过高，pooled representation 就可能被带偏
    - 原因 6：mean pooling 天然具有更强正则化效果，对小数据集和小帧数采样更稳
- 当前建议：
  - 对 `audio` 和 `video` 单模态分支，都保留 `mean pooling` 作为默认方案
  - 若后续继续研究 temporal 模块，优先放在多模态 `xattn` 分支，而不是单独替换单模态 pooling
- 是否纳入最终方案：是；结论是保留 `mean pooling`
- 论文可写结论：
  - “Although the video branch showed a stronger need for temporal modeling than the audio branch, neither temporal transformer pooling nor attention pooling outperformed simple mean pooling. We conjecture that the ResNet18 frame encoder already produces high-level expression-aware features, making uniform averaging a strong and stable aggregator under small-data, low-frame-count settings.”
  - “Attention pooling preserved training performance but reduced validation and test performance, suggesting overfitting to spurious frame-level patterns rather than improved temporal reasoning.”

### E003. Multimodal xAttn: Cross-Attention 后 Mean Pooling vs Temporal Transformer

- 日期：2026-03-25
- 状态：已完成
- 任务：`multimodal`
- 模型：
  - Fusion backbone: `xattn`
  - Post-fusion temporal aggregation: `mean pooling`, `temporal transformer`
- 改动内容：
  - 在融合模块中，将 `cross-attention` 之后的 `mean pooling` 替换为 `temporal transformer`，测试更深时序建模是否能进一步提升多模态性能
- 改动动机：
  - 期望在跨模态交互之后继续进行时序建模，从融合后的序列表示中提取更强的动态判别信息
- 实验现象：
  - 将 `cross-attention` 后的 `mean pooling` 换成 `temporal transformer` 后，最终结果基本没有提升
  - 相比 `mean pooling`，训练过程中出现了更明显的过拟合
- 当前结论：
  - 对当前多模态融合设置而言，`cross-attention + mean pooling` 已经足够
  - 在 `cross-attention` 已显式建模跨模态时序交互后，额外叠加 `temporal transformer` 并没有进一步提升最终性能，反而增加了过拟合风险
  - 这说明当前数据规模和任务设置下，更深的序列建模主要增加了模型方差，而没有带来新的有效信息
- 可能原因分析：
  - 事实：
    - `temporal transformer` 版本与 `mean pooling` 版本最终性能基本一致
    - `temporal transformer` 版本在训练过程中表现出更明显的过拟合
  - 推测：
    - 原因 1：融合层本身已经“够强”，`cross-attention` 已经完成了关键的跨模态时序对齐与交互
    - 原因 2：后续再加一个 `temporal transformer`，对现有任务而言属于重复建模，新增表示能力有限
    - 原因 3：额外时序模块增加了模型容量和优化难度，在当前数据规模下更容易放大过拟合
    - 原因 4：`mean pooling` 在这里反而起到了更稳定的聚合作用，具备更强的隐式正则化效果
- 当前建议：
  - 多模态 `xattn` 默认保留 `cross-attention + mean pooling`
  - 后续若继续尝试更深时序模块，应优先同时加强正则化与数据规模，而不是直接堆叠模块
- 是否纳入最终方案：是；结论是保留 `mean pooling`
- 论文可写结论：
  - “Replacing mean pooling with an additional temporal transformer after cross-attention did not improve multimodal performance, while making overfitting more pronounced during training.”
  - “This suggests that once cross-attention has already explicitly modeled cross-modal temporal interactions, further temporal sequence modeling brings little additional information but substantially increases model variance and overfitting risk.”

### E004. Multimodal Gated Fusion + CLIP-style Semantic Alignment

- 日期：2026-03-25
- 状态：已完成
- 任务：`multimodal`
- 模型：
  - Fusion backbone: `gated`
  - Alignment module: `CLIP-style semantic alignment`
- 改动内容：
  - 在 `gated fusion` 之前加入共享语义空间对齐模块，并使用“分类损失 + 对比损失加权”的联合优化
- 改动动机：
  - 期望先对齐音频和视频表征，再进行门控融合，从而提升跨模态语义一致性
- 实验现象：
  - 加入 CLIP 后，训练结束时 `contrastive loss` 约为 `1.0`
  - 同时 `classification loss` 约为 `0.2`
  - 不加入 CLIP 时，对应分类损失约为 `0.15`
- 当前结论：
  - 对当前 `gated fusion` 设置而言，CLIP-style 对齐没有带来更好的分类训练结果
  - 从损失表现看，当前对比学习分支较难优化，并且会拖累主分类目标
- 可能原因分析：
  - 事实：
    - 带 CLIP 时，对比损失仍停留在较高水平（约 `1.0`）
    - 带 CLIP 时，分类损失高于不带 CLIP 的 baseline（`0.2` vs `0.15`）
  - 推测：
    - 原因 1：CLIP-style 对比学习在当前设置下不好训练，可能与 batch size 偏小、数据规模偏小有关，导致负样本不足、对比目标不稳定
    - 原因 2：对比学习目标与分类目标存在冲突，embedding 被 CLIP 拉向一个不利于分类的空间
    - 原因 3：分类任务更希望“同类样本聚得更近”，而 CLIP 式 instance-level 对比目标更强调“不同样本彼此分开”，两者优化方向不完全一致
- 当前建议：
  - 暂不将 CLIP-style 对齐作为 `gated fusion` 的默认方案
  - 若后续继续验证，应优先增大 batch size、增强负样本数量，或改成更贴近类别监督的 supervised contrastive / class-aware alignment 方案
- 是否纳入最终方案：否
- 论文可写结论：
  - “Adding CLIP-style semantic alignment before gated fusion did not improve optimization in our current setting. The contrastive branch remained relatively hard to train, and the classification loss was slightly worse than the non-CLIP baseline.”
  - “We conjecture that the instance-level contrastive objective may conflict with the class-discriminative objective under small-batch, small-data conditions, pulling the shared embedding space away from what is most beneficial for emotion classification.”

---

## Entry Template

复制下面模板新增实验记录。

```md
### EXXX. [实验标题]

- 日期：
- 状态：已完成 / 进行中 / 失败 / 待复现
- 任务：audio-only / video-only / multimodal
- 模型：
  - Backbone:
  - Fusion:
  - Temporal module:
- 改动内容：
- 改动动机：
- 实验配置：
  - 数据划分：
  - 训练轮数：
  - batch size：
  - learning rate：
  - 其他关键参数：
- 结果：
  - Val Acc:
  - Val F1:
  - Test Acc:
  - Test F1:
  - 相对 baseline 变化：
- 当前结论：
- 可能原因分析：
  - 事实：
  - 推测：
- 是否纳入最终方案：是 / 否 / 待定
- 论文可写结论：
- 后续动作：
```

---

## Recommended Dimensions To Record

后续做实验时，建议优先围绕这些维度持续记录：

- 时序聚合方式：`mean` / `attn` / `transformer`
- 音频 backbone：`Mel-CNN` / `Mel-ResNet` / `WavLM`
- 视频聚合方式：帧均值 / temporal attention / temporal transformer
- 融合方式：`late` / `concat` / `gated` / `xattn`
- 训练策略：单阶段 / 两阶段 / 冻结与解冻策略
- 优化器设置：学习率、backbone 学习率、weight decay、scheduler
- 正则化：dropout、stochastic depth、label smoothing
- 数据策略：face crop、噪声增强、SpecAugment、split 方式

---

## Writing Guidance For Final Report

最终写论文/报告时，可以按下面逻辑整理：

1. 先定义 baseline。
2. 再按模块做消融：
   - audio temporal modeling
   - video temporal modeling
   - fusion strategy
   - training strategy
3. 对每个实验都说明：
   - 为什么改
   - 怎么改
   - 结果如何
   - 为什么会这样
4. 失败实验也保留，用来体现研究过程和结论边界。
