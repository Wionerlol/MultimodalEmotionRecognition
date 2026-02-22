# Multimodal Emotion Recognition System

Audio-Visual Emotion Recognition using Deep Learning and Cross-Modal Fusion

## 项目简介

本项目实现了一个完整的音视频多模态情感识别系统，能够从 3 秒视频中联合分析：

- 人脸表情（视觉模态）
- 语音情感特征（音频模态）

并预测以下 8 类情绪：

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

系统支持：

- 单模态训练（audio / video）
- 多模态融合训练
- 实时推理 API
- Web 端实时预测
- Docker 部署

## 项目亮点（面试核心）

### 1. 多模态深度学习架构设计

实现完整双模态 emotion recognition pipeline：

视觉模态：

- ResNet18 backbone（ImageNet pretrained）
- MediaPipe 人脸检测与裁剪
- 时间维特征聚合

音频模态：

- WavLM pretrained speech model
- 两阶段 finetuning 策略：
  - Stage 1: freeze backbone
  - Stage 2: selective unfreeze

输出 embedding：

- Video embedding: `[B, T, 512]`
- Audio embedding: `[B, Ta, 768]`

### 2. Cross-Attention 跨模态融合（核心创新）

实现 bidirectional cross-attention fusion：

- Video -> attends -> Audio
- Audio -> attends -> Video

结构：

- Video features -> MHA(query=video, key=audio)
- Audio features -> MHA(query=audio, key=video)

实现包含：

- Multi-Head Attention
- residual connection
- layer normalization
- stochastic depth regularization

该结构允许模型：

- 学习 audio-visual correlation
- 在实验中优于简单 concat 或 late fusion

### 3. Curriculum Learning 噪声增强策略（Research 级）

实现真实环境噪声训练：

概率分布：

- 50% clean audio
- 40% medium noise
- 10% heavy noise

使用真实 bar noise：

- `y(t) = s(t) + alpha * n(t)`

效果：

- 提升 noisy environment robustness

### 4. 两阶段训练策略（Fine-tuning Optimization）

训练流程：

Stage 1:

- Freeze backbone
- Train classification/fusion head

Stage 2:

- Unfreeze last N layers
- Fine-tune with lower LR

优点：

- 防止 catastrophic forgetting
- 提高训练稳定性
- 提升最终 accuracy / F1

### 5. 多种融合方法对比实验

实现并对比：

- Late Fusion
- Concat Fusion
- Gated Fusion
- Cross-Attention Fusion

用于研究：

- 不同 fusion strategy 的性能差异

### 6. 完整推理与部署系统

实现 production-ready inference stack：

Backend:

- FastAPI inference server

Frontend:

- Web video recorder

Deployment:

- Docker containerization

API：

- `POST /predict`

返回示例：

```json
{
  "top1": "angry",
  "prob": 0.82
}
```

## 系统架构图

![Model Architecture Diagram](assets/diagrams/model_architecture_overview.png)

![Model Architecture Diagram](assets/diagrams/model_architecture_overview_2.png)

## 可视化展示（PPT/面试可直接使用）

### 1) 处理后数据样例（Audio + Video）

> 保留当前 assets 路径，不改动文件。

<video controls width="720" src="assets/examples/processed_video_example.mp4">
  Your browser does not support the video tag.
</video>

<audio controls src="assets/examples/processed_audio_example.wav">
  Your browser does not support the audio element.
</audio>

备用链接：

- [Processed Video Example](assets/examples/processed_video_example.mp4)
- [Processed Audio Example](assets/examples/processed_audio_example.wav)

### 2) W&B Charts

每张图为 3 次训练曲线叠加（含单模态/多模态对比）。

Train metrics:

| Loss | Accuracy | Macro-F1 |
|---|---|---|
| ![Train Loss](assets/wandb_chart/train_loss.png) | ![Train Accuracy](assets/wandb_chart/train_acc.png) | ![Train F1](assets/wandb_chart/train_f1.png) |

Validation metrics:

| Loss | Accuracy | Macro-F1 |
|---|---|---|
| ![Val Loss](assets/wandb_chart/val_loss.png) | ![Val Accuracy](assets/wandb_chart/val_acc.png) | ![Val F1](assets/wandb_chart/val_f1.png) |

Test metrics:

| Loss | Accuracy | Macro-F1 |
|---|---|---|
| ![Test Loss](assets/wandb_chart/test_loss.png) | ![Test Accuracy](assets/wandb_chart/test_acc.png) | ![Test F1](assets/wandb_chart/test_f1.png) |

Confusion matrices:

| Audio | Video | Gated | Xatten
|---|---|---|---|
| ![Audio CM](assets/wandb_chart/audio_confusion_matrix.png) | ![Video CM](assets/wandb_chart/video_confusion_matrix.png) | ![Gated CM](assets/wandb_chart/gated_confusion_matrix.png) | ![Xatten CM](assets/wandb_chart/xattn_confusion_matrix.png)


## 数据集

使用：

- RAVDESS dataset

包含：

- 24 actors
- 8 emotion classes
- audio + video paired samples

## 实验结果

Typical performance（示例区间）：

| Model | Test Accuracy |
|---|---:|
| Audio only | ~81% |
| Video only | ~77% |
| Multimodal (gated) | ~93% |
| Multimodal (cross-attention) | ~92% |

Gated fusion 在多次实验中通常达到最佳或并列最佳性能（具体以你的 W&B 结果为准）。

## 技术栈

Framework:

- PyTorch
- FastAPI
- Docker

Models:

- WavLM
- ResNet18

Tools:

- MediaPipe
- Weights & Biases
- ffmpeg

## 项目结构

```text
src/
  models/
  data/
  train.py

backend/
frontend/
checkpoints/
assets/
  examples/
  wandb/
  diagrams/
```

## 推理演示

Web UI:

- Record video -> Predict emotion -> Show probability

API:

```bash
curl -X POST http://localhost:8000/predict
```

## 项目展示能力（面试重点）

本项目展示以下核心能力：

- Multimodal Deep Learning
  - audio-visual fusion
  - cross-attention design
- Deep Learning Engineering
  - model training
  - fine-tuning strategies
- Research Capability
  - architecture design
  - fusion comparison experiments
- Production Deployment
  - inference system
  - Docker deployment

## 作者

Zheyi Liu  
National University of Singapore  
Master of Artificial Intelligence Systems
