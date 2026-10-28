# 独立 Sparse Structure VAE + 图片条件 Flow

本目录是可整体复制到其他项目的独立子项目，提供网络、训练、数据缓存、推理、预训练权重和本地 Web 界面。核心代码不导入 `trellis`，不依赖原仓库目录、spconv、flash-attn、xformers、utils3d、nvdiffrast 或 SLAT 模块。注意力使用 PyTorch 原生 SDPA。

模型架构详解见 [中文 PDF（20 页）](docs/SS_VAE_SS_Flow_Architecture.pdf)，包含逐层尺寸、结构图、公式、参数量与代码索引。

## 目录

```text
ss_module/
├── __init__.py               # 独立包：ss_module
├── datasets/                 # 占据网格与图片条件潜变量数据集
│   ├── components.py
│   ├── sparse_structure.py
│   └── sparse_structure_latent.py
├── models/                   # SS VAE Encoder / Decoder 与 SS Flow
├── modules/                  # Attention、Transformer、Norm、空间操作
├── pipelines/
│   ├── sparse_structure.py   # OccupancyPipeline
│   └── samplers/flow_euler.py
├── trainers/
│   ├── basic.py              # 通用训练、AMP、EMA、梯度累积、恢复与验证
│   ├── vae/sparse_structure_vae.py
│   └── flow_matching/flow_matching.py
├── utils/                    # 数据 I/O、图片编码、权重定位、RNG、占据结果
├── train.py                  # ss-train / python train.py
├── infer.py                  # ss-infer / python infer.py
├── prepare.py                # ss-prepare：缓存潜变量与图片特征
├── export_weights.py         # ss-export：导出独立权重
├── demo_data.py              # 人工数据生成，仅用于流程测试
├── configs/                  # 正式及 tiny 配置
├── weights/                  # 预训练权重，非符号链接
├── webapp/                   # 3D Web 界面
├── docs/                     # 架构 PDF 与可重建源文件
├── tests/
├── example_sparse_structure.py
├── pyproject.toml
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

包内相对导入保持模块间的依赖清晰，对外统一使用：

```python
from ss_module.models import SparseStructureEncoder, SparseStructureDecoder, SparseStructureFlowModel
from ss_module.datasets import SparseStructureDataset, ImageConditionedSparseStructureLatentDataset
from ss_module.pipelines import OccupancyPipeline
from ss_module.trainers import train
```

## 迁移与安装

将整个 `ss_module` 目录复制到新项目，不需要复制 `trellis/`。`weights/` 大约 1.3 GB，已从本机原始 checkpoint 复制，迁移时必须一并复制；它被 `.gitignore` 排除，单独提交 Git 代码不会携带权重。

先在目标环境安装适配显卡驱动的 PyTorch，再运行：

```bash
cd /新项目/ss_module
python -m pip install -e .
# 需要直接读照片和去背景时：
python -m pip install -e '.[image]'
# 需要 Web 界面时：
python -m pip install -e '.[web]'
# 运行测试时：
python -m pip install -e '.[test]'
```

以下命令均在复制后的 `ss_module` 目录执行。当前机器可以先激活现有环境：

```bash
source /home/magengyu/TRELLIS.2/.venv/bin/activate
cd /home/magengyu/TRELLIS/ss_module
```

仅使用缓存特征训练或推理时不需要 DINOv2、torchvision 或 rembg。直接读照片时，DINOv2 首次通过 Torch Hub 下载源码和权重；rembg 首次可能下载去背景模型。离线直接读照片需要提前准备这些缓存，`--dino-repo /本地/dinov2` 可指定本地源码，但其预训练权重仍需提前放入 Torch Hub 缓存。`--no-rembg` 可跳过去背景，已有透明前景的图片仍会按 alpha 裁剪。

如需从原始本地模型再次导出权重：

```bash
ss-export --source /原项目/ckpts/TRELLIS-image-large --output weights
```

生成的 `encoder.{json,safetensors}`、`decoder.{json,safetensors}`、`denoiser.{json,safetensors}` 为普通文件，可独立加载。推理也兼容原始 `ss_enc_*`、`ss_dec_*`、`ss_flow_img_*` 文件名。

## 1. 准备 VAE 数据

输入为二值占据网格，不是照片。每个对象保存 `.npy` 或 `.npz`：

```python
import numpy as np

# occupancy.shape == (64, 64, 64)，值为 0 或 1
np.savez_compressed('object_001.npz', occupancy=occupancy.astype(bool))
```

`train.jsonl` 每行一个对象，路径相对于 **JSONL 文件所在目录**；也支持绝对路径：

```jsonl
{"id":"001","occupancy":"object_001.npz","images":["images/001_front.png","images/001_side.png"]}
{"id":"002","occupancy":"object_002.npz","images":["images/002.png"]}
```

VAE 阶段只读取 `occupancy`，可省略 `images`。图片在后续 Flow 缓存阶段使用。训练、验证使用不同 manifest，避免对象泄漏。

保持模型中心、尺度、坐标轴及图片与目标网格关系一致。官方数据体素化标记的是与网格表面相交的体素，不是填满内部的实体；训练自己的实体占据数据也可以，但与官方预训练分布不同。这里不包含网格体素化、相机估计或物体朝向对齐。

## 2. 训练或微调 VAE

```bash
ss-train --config configs/vae.json \
  --manifest /数据/train.jsonl --val-manifest /数据/val.jsonl \
  --output outputs/vae --device cuda
```

微调已有预训练 VAE：

```bash
ss-train --config configs/vae.json \
  --manifest /数据/train.jsonl --output outputs/vae \
  --init weights --device cuda
```

网络和损失：

```text
[B,1,64,64,64] 占据网格
  → Encoder → mean、logvar，采样 z [B,8,16,16,16]
  → Decoder → [B,1,64,64,64] logits
loss = Dice 重建损失 + 0.001 × KL
```

`loss_type` 也支持 `bce`。Encoder 推理默认使用后验均值。

## 3. 为 Flow 缓存数据

用选定的 VAE Encoder 缓存目标潜变量；用 DINOv2 缓存输入照片特征：

```bash
ss-prepare --manifest /数据/train.jsonl \
  --encoder-weights outputs/vae/ema_weights \
  --output /数据/flow_cache --device cuda
```

也可用 `--encoder-weights weights` 直接使用官方 VAE。若要使用现有 Flow 预训练权重，先使用与其匹配的官方 VAE。

每个缓存对象的 `.npz` 包含：

| 字段 | 形状 | 含义 |
|---|---|---|
| `latent` | `[8,16,16,16]` | VAE 后验均值，不做额外缩放或标准化 |
| `cond` | `[V,T,1024]` | 每个视角的 DINOv2 tokens，包括模型返回的前缀 tokens，按最后一维 layer norm |

同时输出 `manifest.jsonl` 和 `cache_metadata.json`，后者记录编码器权重哈希、图片编码器和预处理选项。多图每个对象可有不同数量视角；训练时每个 epoch 为对象选择一个视角。这与推理时的多视角融合是两件事。

已有图片特征时，源 manifest 可用 `"cond":"features.npy"` 替代 `images`，形状为 `[T,C]` 或 `[V,T,C]`。这样不会加载 DINOv2，适合接入其他编码器；更换 C 必须同时修改 Flow 的 `cond_channels`，与官方权重结构不同时不能直接微调原权重。训练同一 batch 的 tokens 数 T 必须一致。

**VAE 改变后必须重新缓存潜变量，并训练或微调与之匹配的 Flow。推理时 Decoder 必须与生成潜变量所用的 VAE 对应。** 不能任意混用新 VAE 与旧 Flow。

## 4. 训练或微调图片条件 Flow

```bash
ss-train --config configs/flow.json \
  --manifest /数据/flow_cache/manifest.jsonl \
  --output outputs/flow --device cuda
```

微调时添加 `--init weights`。验证集可以添加 `--val-manifest`，格式相同。

Flow Matching 使用与原实现相同的时间方向和目标：

```text
t = sigmoid(N(1,1))
x_t = (1-t) × z + (sigma_min + (1-sigma_min) × t) × noise
velocity_target = (1-sigma_min) × noise - z
loss = MSE(model(x_t, 1000t, cond), velocity_target)
```

`p_uncond=0.1` 表示随机将图片条件置零，用于 CFG。推理 Euler 从 `t=1` 积分到 `t=0`。

### 通用训练配置和输出

`configs/vae.json` 和 `configs/flow.json` 保留官方网络架构，但使用独立训练循环。训练参数为 FP32 master parameters，支持 `precision=fp32/fp16/bf16`、梯度裁剪、梯度累积、EMA 和随机状态恢复。默认 batch 为 1，Flow 开启梯度检查点。

- `latest.pt`：模型、优化器、AMP scaler、EMA、步数、epoch、batch 位置和 RNG，用于继续训练。
- `weights/`：当前模型的 JSON + safetensors，用于推理或微调。
- `ema_weights/`：EMA 模型的 JSON + safetensors，建议用于推理；将 `ema_decay` 设为 0 时不生成。
- `metrics.jsonl`：训练损失和可选验证损失。
- `config.json`：本次配置。

恢复训练（保持数据内容、batch、模型、精度和训练超参数不变）：

```bash
ss-train --config configs/flow.json \
  --manifest /数据/flow_cache/manifest.jsonl \
  --output outputs/flow --resume outputs/flow/latest.pt --device cuda
```

`--max-steps` 可覆盖配置，表示总目标步数，不是追加步数。改变网络或训练设定时使用 `--init` 开始新的微调，而非 `--resume`。完整 `.pt` 训练状态仅应加载可信文件；推理使用 safetensors。

这是单进程 CPU / 单 GPU 训练实现，不包含 DDP、原仓库自适应梯度裁剪和渲染评估，因此不是原训练器的逐项复制。正式模型训练的显存需求远高于推理，8 GB 显卡不能据此保证可训练完整 L 模型；可减小网络、关闭 EMA、使用梯度累积/检查点。CPU 训练请设 `precision=fp32`。

## 5. 推理

### 官方权重：单图、多图生成

```bash
ss-infer generate --weights weights \
  --images /照片/front.png --steps 12 --output outputs/single.npz

ss-infer generate --weights weights \
  --images /照片/front.png /照片/side.png /照片/back.png \
  --mode stochastic --steps 12 --output outputs/multi.npz
```

`stochastic` 每步按顺序轮流使用视角，步数须不少于图片数；`multidiffusion` 每步平均各视角预测。无相机位姿输入，不显式知道正面、侧面标签。CFG 公式为 `(1 + cfg_strength) × conditional - cfg_strength × unconditional`，默认只在 `t∈[0.5,1]` 启用，时间重标定 `rescale_t=3`。

使用自己训练的 Flow 与对应 Decoder：

```bash
ss-infer generate --weights outputs/flow/ema_weights \
  --decoder-weights outputs/vae/ema_weights \
  --images /照片/front.png --output outputs/custom.npz
```

仅输入缓存条件，可完全跳过图片编码器和去背景模块：

```bash
ss-infer generate --weights weights --cond /数据/features.npy \
  --steps 12 --output outputs/from_features.npz
```

### VAE 重建

```bash
ss-infer reconstruct --weights weights \
  --occupancy /数据/object_001.npz --output outputs/reconstruction.npz
```

### Python API

```python
from PIL import Image
from ss_module.pipelines import OccupancyPipeline
from ss_module.utils.occupancy import save_result

pipeline = OccupancyPipeline('weights', device='cuda')
result = pipeline.generate(images=[Image.open('front.png'), Image.open('side.png')], steps=12)
save_result(result, 'occupancy.npz')
```

输出包含 `occupancy: bool[64,64,64]`、`coords: int32[N,3]` 和 `logits: float32[64,64,64]`；占据判据为 `logits > 0`。体素中心坐标为 `(coords + 0.5) / 64 - 0.5`。

## 6. Web 界面

```bash
python webapp/app.py
```

打开 http://127.0.0.1:7860，支持单图、多图、中心点 + 体素边线的三维预览及下载。默认使用本目录的 `weights/`。使用自己训练的权重：

```bash
SS_WEIGHTS=outputs/flow/ema_weights \
SS_DECODER_WEIGHTS=outputs/vae/ema_weights python webapp/app.py
```

更详细的界面说明见 [webapp/README.md](webapp/README.md)。

## 7. 不下载模型的完整小规模检查

以下 tiny 网络只检查流程，输入为 `8³` 人工方块及随机条件，不代表真实训练效果，也不能加载官方权重：

```bash
python -m ss_module.demo_data --output outputs/demo
ss-train --config configs/vae_tiny.json \
  --manifest outputs/demo/manifest.jsonl --output outputs/demo_vae --device cpu
ss-prepare --manifest outputs/demo/manifest.jsonl \
  --encoder-weights outputs/demo_vae/weights --resolution 8 \
  --output outputs/demo_cache --device cpu
ss-train --config configs/flow_tiny.json \
  --manifest outputs/demo_cache/manifest.jsonl --output outputs/demo_flow --device cpu
ss-infer reconstruct --weights outputs/demo_vae/weights \
  --occupancy outputs/demo/0.npz --resolution 8 --device cpu --output outputs/demo_recon.npz
ss-infer generate --weights outputs/demo_flow/weights \
  --decoder-weights outputs/demo_vae/weights --cond outputs/demo_cache/0000000.npz \
  --steps 2 --device cpu --output outputs/demo_generated.npz
python -m pytest tests -q
```

代码与测试可迁移运行不等于完整 L 模型已训练收敛。请用真实数据和验证集评估形状重建与图片条件生成质量。

本次已完成的检查见 [VALIDATION.md](VALIDATION.md)。
