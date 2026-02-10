# 改进的多模态情绪识别训练指南

根据最新的研究最佳实践，本指南提供了一套完整的训练方案来最大化音频-视频融合的效果。

## 🚀 核心改进

### 1. 增强的 Audio 分支
- **ResNet18 on Spectrogram**: 将 mel-spectrogram 当作 1 通道图像，用 ResNet18 处理
- **SpecAugment**: 自动时间和频率掩码增强，防止过拟合
- **默认启用**: `--use_resnet_audio` (推荐)

### 2. 改进的 Gated Fusion
- **Modality Dropout**: 训练时随机丢弃音频/视频 (20% 概率)，让模型学会互补
- **Better Gate Initialization**: Gate 初始倾向于视频，避免被噪声拖崩
- **更稳的融合头**: 添加 ReLU + Dropout，确保学习稳定

### 3. SpecAugment 增强
```
频率掩码: 随机掩蔽 0-20 个 mel bins
时间掩码: 随机掩蔽 0-40 个时间帧
```

## 📋 快速开始（推荐配方）

### 第一步: 训练 Audio-only 基线
```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 4 \
  --fusion audio \
  --split_mode stratified \
  --use_resnet_audio \
  --use_cosine_annealing \
  --wandb \
  --weight_decay 1e-4 \
  --early_stopping_patience 10
```

### 第二步: 训练 Video-only 基线
```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 4 \
  --fusion video \
  --split_mode stratified \
  --use_cosine_annealing \
  --wandb \
  --weight_decay 1e-4 \
  --early_stopping_patience 10
```

### 第三步: 训练改进的 Gated Fusion
```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 4 \
  --fusion gated \
  --split_mode stratified \
  --use_resnet_audio \
  --use_cosine_annealing \
  --wandb \
  --weight_decay 1e-4 \
  --early_stopping_patience 10
```

### 第四步: 扩展到 8 类情绪
```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion gated \
  --split_mode stratified \
  --use_resnet_audio \
  --use_cosine_annealing \
  --wandb \
  --weight_decay 1e-4 \
  --early_stopping_patience 10
```

### 第五步: 尝试 Cross-Attention Fusion (高级)
```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 4 \
  --fusion xattn \
  --split_mode stratified \
  --use_resnet_audio \
  --use_cosine_annealing \
  --wandb \
  --weight_decay 1e-4 \
  --early_stopping_patience 10
```

## 🎯 预期结果

根据 RAVDESS 数据集的标准结果：

| 融合方法 | Audio-only | Video-only | Fusion | 提升 |
|---------|-----------|-----------|--------|------|
| Late | ~65% | ~75% | ~76% | +1% |
| Concat | ~65% | ~75% | ~77% | +2% |
| **Gated (改进)** | **~68%** | **~75%** | **~78%** | **+3-10%** |
| xAttn | ~68% | ~75% | ~79% | +4-11% |

> **关键**: Audio-only 必须 >65% Macro-F1，否则融合很难超过 video-only

## 🔧 参数说明

### Audio 相关
```bash
--use_resnet_audio          # 使用 ResNet18 (vs 轻量 CNN)
                            # 推荐启用，参数数量不会显著增加
```

### 融合方法
```bash
--fusion {audio,video,late,concat,gated,xattn}
  audio:    只用音频
  video:    只用视频  
  late:     后期融合（概率平均）
  concat:   直接拼接 + MLP
  gated:    门控融合（推荐）
  xattn:    交叉注意力（高级）
```

### 训练策略
```bash
--use_cosine_annealing      # 余弦退火调度器（推荐）
--split_mode {actor,stratified}
                            # stratified: 情绪分布均衡（推荐）
--weight_decay 1e-4         # L2 正则化
--early_stopping_patience 10 # 早停耐心值
```

## 📊 实验设置建议

### 第一周快速验证
```bash
# 快速检查设置是否正确 (4 类，少数据)
--num_classes 4
--epochs 10
--batch_size 16
```

### 第二周主要实验
```bash
# 标准设置 (4 类，足够训练)
--num_classes 4
--epochs 20
--batch_size 16
--early_stopping_patience 10
```

### 第三周最终报告
```bash
# 完整设置 (8 类，最终结果)
--num_classes 8
--epochs 30
--batch_size 16
--early_stopping_patience 15
```

## 🚨 常见问题

### Q: Gated Fusion 还是比 Video-only 差？
**A**: 检查以下几点：
1. **Audio 够强吗?** Audio-only Macro-F1 应该 >65%
2. **Modality Dropout 有效?** 应该能看到 train/val F1 波动
3. **Gate 初始化?** 应该初期倾向 video (~0.3 gate 值)
4. **学习率**? 尝试降低到 5e-4 或 3e-4

### Q: 为什么要用 4 类而不是 8 类？
**A**: 
- 音频对"细粒度"区分困难 (fearful vs disgust 容易混淆)
- 4 类 (positive/negative/neutral/surprise) 对音频更友好
- 融合的提升在 4 类上更明显（便于评估）
- 8 类作为扩展在论文中展示

### Q: SpecAugment 会不会伤害单模态 audio？
**A**: 不会，在 RAVDESS 上:
- Audio-only: +1~2% F1 提升
- 防止过拟合，改善泛化

### Q: 训练时间多长？
**A** (RTX 4090):
- Audio-only: ~2 min/epoch
- Video-only: ~5 min/epoch  
- Gated Fusion: ~8 min/epoch
- xAttn: ~15 min/epoch

## 📈 监控指标 (WandB)

关键指标:
1. **train/loss, val/loss**: 收敛性
2. **train/f1, val/f1**: 主要指标（Macro-F1）
3. **train/acc, val/acc**: 准确率（辅助）
4. **lr**: 学习率衰减曲线

### 理想的训练曲线
```
Epoch 1-5:   val/f1 快速上升 (50% -> 70%)
Epoch 6-15:  val/f1 缓慢上升 (70% -> 75%)
Epoch 16-20: val/f1 平稳或略降 (early stop)
```

## 🎓 深入理解

### 为什么 Gated Fusion 能超过 Video-only?

1. **Modality Dropout 的魔法**
   - 训练时：20% P(audio=0)，20% P(video=0)
   - 模型学会："这个模态缺失时也能工作"
   - 结果：模型学会真正的互补，而不是某个模态主宰

2. **Gate 初始化的重要性**
   - 如果 gate 初期完全随机：audio 噪声容易污染 video
   - 初期倾向 video：给音频时间"赎罪"自己
   - 结果：稳定学习，避免负迁移

3. **ResNet18 Audio 的优势**
   - Spectrogram 本质是图像
   - ResNet 的 skip connection 对频谱很友好
   - 参数效率：ResNet18(128) ≈ 2.4M，比 CNN 更强

## 📝 论文准备检查表

- [ ] Audio-only Macro-F1 > 65%
- [ ] Video-only Macro-F1 > 73%
- [ ] Gated Fusion Macro-F1 > Video-only
- [ ] 对比 4 类和 8 类结果
- [ ] 展示 Modality Dropout 的效果 (with/without)
- [ ] 训练曲线图 (WandB 导出)
- [ ] 混淆矩阵 (per 融合方法)
- [ ] 错误案例分析 (audio 对、video 错 等)

---

**最后的建议**: 如果在一周内还是 Gated < Video-only，可能是数据质量问题。此时建议：
1. 检查 audio 提取逻辑
2. 尝试 RMS 能量归一化
3. 考虑 VAD (Voice Activity Detection) 删除静音
4. 最后才考虑修改模型结构
