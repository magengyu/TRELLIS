# TRELLIS 中文快速上手

TRELLIS 可以将图片或文本转换为 3D 资产，支持导出带纹理的网格（GLB）和 3D Gaussian（PLY）。建议优先使用图片生成，效果通常更好。

## 1. 环境准备

- Linux，NVIDIA GPU，至少 16 GB 显存（项目 README 的要求，实际占用随模型和任务变化）。
- 安装 Conda、CUDA Toolkit；默认安装方案使用 Python 3.10、PyTorch 2.4.0 和 CUDA 11.8。
- 安装和首次运行需要联网下载依赖及模型。

## 2. 安装

在已初始化 Conda 的 Bash 终端执行：

```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS

# 已有仓库时，从这里开始
git submodule update --init --recursive
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
conda activate trellis
```

如果使用已有环境，去掉 `--new-env`，并自行准备兼容的 PyTorch/CUDA。安装了多个 CUDA Toolkit 时，先将对应版本的 `bin` 加入 `PATH`，例如：

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
```

## 3. 运行图片生成示例

在项目根目录执行：

```bash
conda activate trellis
python example.py
```

脚本会加载 `microsoft/TRELLIS-image-large`，使用示例图片 `assets/example_image/T.png`，并将结果保存到当前目录：

| 文件 | 内容 |
| --- | --- |
| `sample.glb` | 带纹理的网格，可导入 Blender 等软件 |
| `sample.ply` | 3D Gaussian 数据 |
| `sample_gs.mp4` | Gaussian 旋转预览 |
| `sample_rf.mp4` | 辐射场旋转预览 |
| `sample_mesh.mp4` | 网格法线旋转预览 |

**使用自己的图片：**修改 `example.py` 中的图片路径后重新运行：

```python
image = Image.open("/path/to/your_image.png")
```

可以调整 `pipeline.run(..., seed=1)` 的随机种子生成不同结果。导出 GLB 时，`texture_size=1024` 控制纹理尺寸，`simplify=0.95` 表示移除约 95% 的三角面。

## 4. 使用网页界面

```bash
conda activate trellis
. ./setup.sh --demo
python app.py
```

打开终端显示的地址，上传图片，点击 **Generate** 生成并预览，再使用界面中的导出功能下载资产。界面也支持多图输入。

## 5. 其他入口

以下命令任选其一执行：

```bash
python example_multi_image.py  # 同一物体的多张图片生成 3D
python example_text.py         # 文本生成 3D
python example_variant.py      # 生成资产变体
python app_text.py             # 文本生成的网页界面（需先安装 --demo）
```

文本示例和文本网页默认使用 `TRELLIS-text-xlarge`，显存需求可能更高。具体输入和参数在对应脚本中修改。

## 常见问题

- **GPU 不支持 Flash Attention（如 V100）：**安装时去掉 `--flash-attn`，保留 `--xformers`，运行前执行 `export ATTN_BACKEND=xformers`。
- **CUDA 扩展编译失败：**检查 `nvcc --version`、PyTorch 的 CUDA 版本和 `PATH`；可将安装命令拆成单个选项逐项排查。
- **模型下载失败：**检查 Hugging Face 网络访问。主模型也可下载后保留目录结构，将 `from_pretrained("microsoft/TRELLIS-image-large")` 改为本地模型目录；辅助模型仍可能需要下载。

更多安装说明和训练方法见 [README.md](README.md)。
