# 模型架构文档

- [SS VAE 与 SS Flow 模型架构详解（PDF，20 页）](SS_VAE_SS_Flow_Architecture.pdf)
- [HTML 版本](SS_VAE_SS_Flow_Architecture.html)
- `architecture_stats.json`：依据正式配置统计的网络参数量。
- `architecture_diagrams.py`：六张可缩放的 SVG block 图，标注模块名、输入输出维数、主数据流及残差旁路。
- `block_shapes.json`：通过实际模型的 meta-tensor 前向 hook 核对的 VAE 逐 block 输入输出形状。

新增结构图页码：Encoder 第 3 页；Decoder 第 5 页；ResBlock3d 第 7 页；上下采样与归一化第 8 页；Flow 总图第 11 页；Transformer CrossBlock 第 13 页。
- `build_architecture_pdf.py`：生成文档的脚本，使用本地 Chrome/Chromium 打印 PDF，无外部网页或字体下载。

在已安装独立包依赖和中文字体的环境中重新生成：

```bash
python docs/build_architecture_pdf.py
```

脚本重新计算网络参数量，文字和逐层表格对应本次正式配置快照。修改网络结构后，除重新生成外，也需要更新脚本中的说明与表格。
