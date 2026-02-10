# 技术总结：当前训练模型架构（与代码一致）

更新时间：2026-02-10  
覆盖代码：`src/train.py`、`src/models/*.py`、`src/data/ravdess.py`

## 1. 系统目标与总体流程

本项目是一个基于 RAVDESS 的多模态情感识别系统，支持以下训练形态：

- 单模态：`audio`、`video`
- 融合：`late`、`concat`、`gated`、`xattn`（以及别名 `xattn_concat`、`xattn_gated`）
- 分类任务：`8` 类或 `4` 类（通过标签映射）

训练主链路：

1. 从 RAVDESS 文件名解析元数据，构造音视频配对。
2. 做数据切分（`actor` 或 `stratified`）。
3. 数据集输出 `video/audio/label/meta`。
4. 按 `--fusion` 构建模型。
5. 按模式执行前向、损失、优化、验证、早停与保存。

---

## 2. 数据与预处理（训练输入真实形态）

### 2.1 配对规则（`src/data/ravdess.py`）

- 视频来源：`modality=02` `.mp4`
- 音频来源：`modality=03` `.wav`
- 用键 `(vocal_channel, emotion, intensity, statement, repetition, actor)` 做一一配对
- 默认 `vocal_channel=1`

输出记录结构：`PairRecord(video_path, audio_path, emotion, intensity, statement, repetition, actor)`

### 2.2 标签映射

- `num_classes=8`：`emotion_id - 1`（0~7）
- `num_classes=4`：
  - `{1,2}` -> `0`（neutral/calm）
  - `{3}` -> `1`（positive）
  - `{4,5,6,7}` -> `2`（negative）
  - `{8}` -> `3`（surprise）

### 2.3 切分策略

- `actor`：按演员 ID 显式划分 train/val/test
- `stratified`：按情感分组后分别随机切分（默认 0.7/0.15/0.15）

### 2.4 视频张量（`load_video_frames`）

- 采样：均匀采样 `num_frames`（默认 8 帧）
- 尺寸：`112x112`
- 归一化：ImageNet mean/std
- 形状：`[T, 3, H, W]`，训练中再组成 `[B, T, 3, H, W]`

可选增强（训练集 `augment=True`）：

- 人脸检测与裁剪（首帧检测 bbox，后续帧复用）
- 高斯模糊（核随机 `3/5/7`）
- 亮度衰减（系数 `0.2~0.6`）
- 轻微高斯噪声（`noise_scale` 随机）

### 2.5 音频张量（Mel 分支）

- 采样率：`16k`
- 时长：`3s`（不足补零，超长截断）
- 特征：`MelSpectrogram(n_mels=64, win_length=400, hop_length=160)` + `AmplitudeToDB`
- 形状：`[1, 64, 301]`，训练中为 `[B, 1, 64, 301]`

可选增强（训练集 `augment=True`）：

- 50% 保持干净音频
- 40% 按 `20/15/10 dB` SNR 混入酒吧噪声（若存在 `data/Noise/noise.wav`）
- 10% 按 `5 dB` 重噪声
- 若无真实噪声文件，回退高斯噪声

### 2.6 音频张量（WavLM 分支）

- 采样率：`16k`
- 时长：`3s`
- 形状：`[1, 48000]`，训练中为 `[B, 1, 48000]`
- 使用与 Mel 分支同策略的时域噪声增强

---

## 3. 模型架构细节

## 3.1 AudioNet（`src/models/audio.py`）

`AudioNet` 支持两种编码器：

1. `AudioCNN`（轻量）
2. `AudioResNet18`（“ResNet18 风格”卷积堆叠）

默认分类头：`Linear(embedding_dim -> num_classes)`，`embedding_dim=128`。

`SpecAugment` 只在训练期启用：

- `freq_mask_param=20`
- `time_mask_param=40`
- `num_masks=2`
- `p=0.5`

说明：

- CLI 中 `--use_resnet_audio` 是开关参数（默认不加时为 `False`，即轻量 CNN）。
- 函数 `build_model()` 的默认值虽然是 `True`，但实际训练以 CLI 解析结果为准。

## 3.2 VideoNet（`src/models/video.py`）

- 主干：`torchvision resnet18` 去掉最后 FC
- 每帧提取 512 维特征，然后时间维求均值
- 输出：
  - `encode`: `[B, 512]`
  - `forward`: `[B, num_classes]`

## 3.3 WavLMAudioEncoder（`src/models/wavlm_audio.py`）

- 主干：`microsoft/wavlm-base`
- 分类头：`Linear(hidden_size -> 768) -> ReLU -> Dropout(0.2) -> Linear(768 -> num_classes)`
- 默认初始化后冻结 backbone（仅 head 可训练）

提供两阶段接口：

- Stage 1：`get_stage1_params()`（仅分类头）
- Stage 2：`unfreeze_backbone(num_last_layers)` + `get_stage2_params()`（backbone 与 head 分组）

前向输入：`[B,1,T]` 或 `[B,T]`  
前向输出：`[B,num_classes]`

## 3.4 FusionModel（`src/models/fusion.py`）

### A) `late`

- 分别得到 `a_logits` 与 `v_logits`
- `softmax` 后做平均：
  - `p = (softmax(a_logits) + softmax(v_logits)) / 2`
- 返回概率（不是 logits）

### B) `concat`

- 分别编码 `a_emb`、`v_emb`
- 投影到共同维度 `common_dim=256`
- 拼接后 MLP：
  - `Linear(512 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> num_classes)`

### C) `gated`

- 与 `concat` 一样先投影
- `ModalityDropout`（音频/视频各 `p=0.2`）
- 门控网络：
  - `Linear(512 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 1) -> Sigmoid`
- 融合：
  - `fused = g * a + (1 - g) * v`
- 分类：
  - `Linear(256 -> num_classes)`

### D) `xattn`

输入与时序处理：

- 视频：`[B,T,3,H,W] -> backbone -> [B,T,512] -> Linear(512->d_model)`
- 音频主路径（warm-start 友好）：
  - 若 `audio_model` 提供 `encode_sequence()`（WavLM/AudioNet 已提供），则走：
  - `audio_model.encode_sequence(audio) -> [B,Ta,H] -> Linear(H->d_model)`
- 音频回退路径（兼容旧 mel 编码器）：
  - `[B,1,n_mels,Ta] -> squeeze -> [B,n_mels,Ta] -> Conv1d(n_mels->d_model) -> [B,Ta,d_model] -> Linear(d_model->d_model)`

跨模态注意力：

- `v2a_attn(query=v, key=a, value=a, dropout=xattn_attn_dropout)` + 残差层归一化
- `a2v_attn(query=a, key=v, value=v, dropout=xattn_attn_dropout)` + 残差层归一化
- 残差分支支持 `StochasticDepth(drop_prob=xattn_stochastic_depth)`

池化与头部：

- 时间维平均得到 `v_emb/a_emb`
- `xattn_head=concat`：
  - `Linear(2*d_model -> 256) -> ReLU -> Linear(256 -> num_classes)`
- `xattn_head=gated`：
  - 门控后 `Linear(d_model -> num_classes)`

---

## 4. 训练器与优化逻辑（`EmotionTrainer`）

### 4.1 模型构建映射

- `audio`: `AudioNet` 或 `WavLMAudioEncoder`
- `video`: `VideoNet`
- `late/concat/gated`: `FusionModel(mode=...)`
- `xattn/xattn_concat/xattn_gated`: `FusionModel(mode="xattn", xattn_head=...)`

### 4.2 损失函数

- `fusion == "late"`：`NLLLoss`（因为模型输出概率，训练前取 `log`）
- 其他模式：`CrossEntropyLoss(label_smoothing=label_smoothing)`

### 4.3 优化器与调度器

- 单模态 `audio + use_wavlm`：
  - `wavlm_stage=1`：仅训练 WavLM 分类头
  - `wavlm_stage=2`：分组 LR（`backbone_lr` + `lr`）
- 融合模式默认（不启用 `--two_stage_training`）：
  - `Adam` 优化所有 `requires_grad=True` 参数
- 融合模式启用 `--two_stage_training`：
  - Stage 1：冻结 `audio_model` + `video_model`，只训练融合头
  - Stage 2：按配置解冻 encoder，并用分层学习率
    - WavLM 音频：冻结后按 `fusion_unfreeze_wavlm_layers` 解冻最后 N 层（分类头始终可训练）
    - 非 WavLM 音频：由 `fusion_unfreeze_audio` / `--no_fusion_unfreeze_audio` 控制是否解冻
    - 融合头：`lr`
    - 音频分支：`audio_backbone_lr`
    - 视频分支：`video_backbone_lr`
- 调度器：可选 `CosineAnnealingLR(T_max=当前阶段epochs, eta_min=1e-5)`
- 早停：`early_stopping_patience`（默认 10）

### 4.4 指标与保存

- 指标：`accuracy`、`macro_f1`
- 最优保存：`outputs/best_{fusion}.pt`
- 保存内容：`{"model": state_dict, "val_f1": best_f1}`

### 4.5 融合分支 Warm-Start

- 新增 CLI：
  - `--audio_ckpt`
  - `--video_ckpt`
  - `--fusion_unfreeze_audio`
  - `--no_fusion_unfreeze_audio`
  - `--xattn_attn_dropout`
  - `--xattn_stochastic_depth`
  - `--label_smoothing`
- 行为：
  - 在构建融合模型后，将 checkpoint 分别加载到 `audio_model` / `video_model`
  - 支持 `{"model": state_dict}` 或原始 `state_dict` 两种格式

### 4.6 WavLM `encode()` 梯度策略

- 当 WavLM backbone 全冻结时，`encode()` 采用 `no_grad`（节省显存/计算）
- 当融合 Stage 2 解冻了 WavLM 层时，`encode()` 会允许梯度反传（确保解冻生效）

### 4.7 WSL 相关 DataLoader 策略

`num_workers=-1` 自动策略：

- Windows：`0`
- WSL 且数据在 `/mnt/*`：`0`
- WSL 且数据在 Linux 文件系统：`2`
- 原生 Linux：`min(8, max(2, cpu_count//2))`

若 `num_workers > 0`，启用：

- `persistent_workers=True`
- `prefetch_factor=2`

---

## 5. 参数规模（按当前代码实测）

统计方式：`sum(p.numel())`。  
下表为 `num_classes=8`，`pretrained_video=False`（仅影响初始化，不影响参数量）。

### 5.1 非 WavLM（音频编码器为 AudioResNet18）

| 模式 | 总参数量 | 可训练参数 |
|---|---:|---:|
| audio | 12,785,224 | 12,785,224 |
| video | 11,180,616 | 11,180,616 |
| late | 23,965,840 | 23,965,840 |
| concat | 24,263,576 | 24,263,576 |
| gated | 24,263,833 | 24,263,833 |
| xattn (concat head) | 24,273,176 | 24,273,176 |
| xattn (gated head) | 24,239,385 | 24,239,385 |

### 5.2 非 WavLM（音频编码器为轻量 AudioCNN）

| 模式 | 总参数量 |
|---|---:|
| audio | 155,752 |
| late | 11,336,368 |
| concat | 11,634,104 |
| gated | 11,634,361 |

### 5.3 WavLM 路径（默认 backbone 冻结）

| 模式 | 总参数量 | 可训练参数（当前默认） |
|---|---:|---:|
| audio | 94,978,680 | 596,744 |
| late | 106,159,296 | 11,777,360 |
| concat | 106,620,872 | 12,238,936 |
| gated | 106,621,129 | 12,239,193 |
| xattn (concat head) | 106,736,968 | 12,355,032 |

WavLM 音频单模态两阶段可训练参数（`num_classes=8`）：

- Stage 1（冻结 backbone）：`596,744`
- Stage 2（解冻最后 2 层）：`14,773,552`
- Stage 2（解冻最后 4 层）：`28,950,360`

---

## 6. 当前推荐训练命令模板

### 6.1 纯音频（Mel + ResNet 风格编码器）

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion audio \
  --use_resnet_audio \
  --epochs 20 \
  --batch_size 16 \
  --split_mode stratified
```

### 6.2 音视频门控融合（推荐基线）

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion gated \
  --use_wavlm \
  --use_cosine_annealing \
  --weight_decay 1e-4 \
  --early_stopping_patience 8
```

### 6.3 WavLM 单模态（Stage 1 / Stage 2）

```bash
# Stage 1: 仅训练分类头
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion audio \
  --use_wavlm \
  --wavlm_stage 1

# Stage 2: 解冻最后若干层（当前实现参数为 2 层）
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion audio \
  --use_wavlm \
  --wavlm_stage 2 \
  --backbone_lr 3e-5 \
  --lr 3e-4
```

### 6.4 Warm-Start Gated 融合（两阶段 + 分层 LR）

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion gated \
  --use_wavlm \
  --audio_ckpt outputs/best_audio.pt \
  --video_ckpt outputs/best_video.pt \
  --two_stage_training \
  --stage1_epochs 5 \
  --lr 3e-4 \
  --audio_backbone_lr 1e-5 \
  --video_backbone_lr 1e-5 \
  --fusion_unfreeze_wavlm_layers 2 \
  --fusion_unfreeze_video_blocks 1 \
  --epochs 30 \
  --batch_size 8 \
  --weight_decay 1e-4 \
  --use_cosine_annealing \
  --early_stopping_patience 8
```

---

### 6.5 Warm-Start xAttn 融合（两阶段 + 正则化）

```bash
uv run python src/train.py \
  --data_root data \
  --num_classes 8 \
  --fusion xattn \
  --xattn_head gated \
  --xattn_d_model 96 \
  --xattn_heads 4 \
  --xattn_attn_dropout 0.1 \
  --xattn_stochastic_depth 0.1 \
  --label_smoothing 0.05 \
  --use_wavlm \
  --audio_ckpt outputs/best_audio.pt \
  --video_ckpt outputs/best_video.pt \
  --two_stage_training \
  --stage1_epochs 6 \
  --lr 2e-4 \
  --audio_backbone_lr 8e-6 \
  --video_backbone_lr 8e-6 \
  --fusion_unfreeze_wavlm_layers 2 \
  --fusion_unfreeze_video_blocks 1 \
  --split_mode stratified \
  --train_ratio 0.75 \
  --val_ratio 0.15 \
  --epochs 35 \
  --batch_size 8 \
  --weight_decay 2e-4 \
  --use_cosine_annealing \
  --early_stopping_patience 10
```

---

## 7. 代码现状中的注意点（与训练架构强相关）

以下是当前仍需注意的行为：

1. `--train_ratio`、`--val_ratio` 已接入 `stratified` 主流程，`test_ratio` 由 `1-train_ratio-val_ratio` 自动计算；建议保持三者和为 1.0。
2. `--use_wavlm --fusion xattn` 已支持序列特征路径（`encode_sequence()`）；若自定义音频编码器未实现该接口，会自动回退到 mel-conv 路径。

---

## 8. 结论

当前项目训练侧已经形成可扩展的多模态架构族：

- 数据层：统一 PairRecord + 两套数据集（Mel / WavLM）
- 模型层：音频/视频/多种融合（late/concat/gated/xattn）
- 训练层：统一 `EmotionTrainer`，支持 WSL 友好 DataLoader 策略、早停、Cosine、W&B

如果后续要继续扩展，优先建议围绕两点推进：

1. 为 xAttn 增加 attention mask（对无效时间步/补零区域更稳）。
2. 在多 seed + 多 split 上做系统超参搜索（`d_model/heads/dropout/drop_path/label_smoothing`）。
