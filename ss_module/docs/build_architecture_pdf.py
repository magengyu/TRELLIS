"""Build an HTML/PDF architecture reference from the portable package's config."""
import html
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys
sys.path.insert(0, str(ROOT.parent))
import torch
from ss_module.models import build_model
STATS = {}
for stage in ('vae', 'flow'):
    cfg = json.loads((ROOT/'configs'/f'{stage}.json').read_text())
    for role, spec in cfg['models'].items():
        with torch.device('meta'):
            model = build_model(spec)
        STATS[role] = {'total': sum(p.numel() for p in model.parameters()),
                       'stages': {n: sum(p.numel() for p in c.parameters()) for n,c in model.named_children()}}


def p(text): return '<p>'+text+'</p>'
def h(text): return '<h2>'+text+'</h2>'
def note(text): return '<div class="note">'+text+'</div>'
def eq(text): return '<pre class="equation">'+html.escape(text)+'</pre>'
def code(text): return '<pre class="code">'+html.escape(text)+'</pre>'
def table(headers, rows, widths=None):
    cols = '' if widths is None else '<colgroup>'+''.join(f'<col style="width:{v}%">' for v in widths)+'</colgroup>'
    return '<table>'+cols+'<thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def bullets(items): return '<ul>'+''.join('<li>'+i+'</li>' for i in items)+'</ul>'

def diagram(labels, subtitle=''):
    width=680; gap=18; count=len(labels); box=(width-(count-1)*gap)/count
    out=['<svg viewBox="0 0 680 115" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#42786b"/></marker></defs>']
    for i,(title,detail) in enumerate(labels):
        x=i*(box+gap)
        out.append(f'<rect x="{x}" y="8" width="{box}" height="69" rx="8" fill="{ "#e6f0eb" if i%2==0 else "#f0f4f0"}" stroke="#abc5b8"/>')
        out.append(f'<text x="{x+box/2}" y="35" text-anchor="middle" font-size="13" fill="#214c40">{html.escape(title)}</text>')
        out.append(f'<text x="{x+box/2}" y="58" text-anchor="middle" font-size="10" fill="#63796d">{html.escape(detail)}</text>')
        if i<count-1:out.append(f'<path d="M{x+box+2} 43 H{x+box+gap-3}" stroke="#42786b" marker-end="url(#arrow)"/>')
    out.append(f'<text x="340" y="103" text-anchor="middle" font-size="10" fill="#63796d">{html.escape(subtitle)}</text></svg>')
    return ''.join(out)

pages=[]
def page(title,body,kicker): pages.append((title,body,kicker))

page('SS VAE 与 SS Flow\n模型架构详解',
'<div class="subtitle">基于本项目独立实现的逐层结构、张量形状与算法说明</div>'
+diagram([('SS Encoder','64³ → 16³'),('SS Flow','图片条件潜变量生成'),('SS Decoder','16³ → 64³')], 'Encoder 用于训练缓存；生成时由图片条件 Flow 生成潜变量，再经 Decoder 解码')
+p('本说明以 <b>ss_module/configs/vae.json</b> 与 <b>flow.json</b> 的正式架构为基准，并对照 <b>ss_module</b> 的实际 Python 实现。所有尺寸均针对 64³ 占据网格、8 通道潜变量和 L 规格图片条件 Flow。tiny 测试配置不属于本文架构。')
+table(['模块','功能','训练输入'],[
['SS VAE','学习占据网格的可重建潜表示','二值占据网格'],
['SS Flow','从噪声生成图片条件潜变量','VAE 潜变量 + 图片特征'],
['DINOv2','提供外部图片条件；默认冻结','前景图片']])
+h('阅读导航')
+table(['页','主题','页','主题'],[
['02–08','VAE 逐层结构与 block 图 A–D','09','VAE 训练目标与数据'],
['10–15','Flow 结构图 E–F、注意力与编码','16','图片条件与多视角'],
['17','Flow Matching 训练','18','Euler 采样、CFG、多图融合'],
['19–20','参数规模、显存与代码索引','—','结构图含逐 block 名称和维数']])
+note('张量约定：体素张量为 [B,C,X,Y,Z]，序列张量为 [B,L,D]。X/Y/Z 表示数组的三个空间轴；“正面”和“向上”需要数据集自行约定。本文不假设额外的相机位姿输入。')
+'<div class="edition">工程架构文档 · 2026-10-28 · 中文版</div>', 'ARCHITECTURE REFERENCE')

page('01 / SS VAE：编码器逐层结构',
p('<b>SparseStructureEncoder</b> 使用稠密 3D 卷积，将二值网格映射到对角高斯后验 q(z|S)。虽然名称包含 SparseStructure，网络内部张量仍是稠密张量；“稀疏”描述占据结构，而不是 spconv 运算。')
+table(['顺序 / 对应模块','运算','输出形状'],[
['输入 S','0/1 占据网格','[B,1,64,64,64]'],
['input_layer','Conv3d 1→32，k=3,s=1,p=1','[B,32,64,64,64]'],
['blocks：64³ 层','ResBlock3d(32) ×2','[B,32,64,64,64]'],
['下采样 1','Conv3d 32→128，k=2,s=2,p=0','[B,128,32,32,32]'],
['blocks：32³ 层','ResBlock3d(128) ×2','[B,128,32,32,32]'],
['下采样 2','Conv3d 128→512，k=2,s=2,p=0','[B,512,16,16,16]'],
['blocks：16³ 层','ResBlock3d(512) ×2','[B,512,16,16,16]'],
['middle_block','ResBlock3d(512) ×2','[B,512,16,16,16]'],
['out_layer','ChannelLN → SiLU → Conv3d 512→16','[B,16,16,16,16]'],
['chunk(2,dim=1)','前 8 通道为 μ，后 8 通道为 log σ²','各 [B,8,16,16,16]']], [26,43,31])
+h('后验参数与重参数化')
+eq('q(z|S) = Normal(μ(S), diag(exp(logvar(S))))\nz = μ + exp(0.5 × logvar) ⊙ ε,    ε ~ Normal(0,I)')
+p('训练时 <b>sample_posterior=True</b>，通过重参数化采样 z；推理或缓存时默认 False，直接使用 μ。<b>return_raw=True</b> 返回 (z, μ, logvar)，否则只返回 z。Encoder 共 8 个残差块，残差块内部合计 16 个 3×3×3 卷积。')
+note('空间分辨率每轴压缩 4 倍：64³→16³。考虑通道后，单个样本从 262,144 个输入标量变为 32,768 个潜标量，是标量数 8 倍压缩；不是按实际字节数定义的文件压缩率。'), 'VAE / ENCODER')

page('02 / SS VAE：解码器逐层结构',
p('<b>SparseStructureDecoder</b> 将 8 通道潜变量恢复为一个通道的占据 logits。它是前馈卷积网络，不需要图片条件，也没有编码器到解码器的 U-Net 跳连。')
+table(['顺序 / 对应模块','运算','输出形状'],[
['输入 z','来自 Encoder 或 Flow','[B,8,16,16,16]'],
['input_layer','Conv3d 8→512，k=3,s=1,p=1','[B,512,16,16,16]'],
['middle_block','ResBlock3d(512) ×2','[B,512,16,16,16]'],
['blocks：16³ 层','ResBlock3d(512) ×2','[B,512,16,16,16]'],
['上采样 1：卷积','Conv3d 512→1024，k=3,p=1','[B,1024,16,16,16]'],
['上采样 1：重排','PixelShuffle3D(r=2)，1024/8=128','[B,128,32,32,32]'],
['blocks：32³ 层','ResBlock3d(128) ×2','[B,128,32,32,32]'],
['上采样 2：卷积','Conv3d 128→256，k=3,p=1','[B,256,32,32,32]'],
['上采样 2：重排','PixelShuffle3D(r=2)，256/8=32','[B,32,64,64,64]'],
['blocks：64³ 层','ResBlock3d(32) ×2','[B,32,64,64,64]'],
['out_layer','ChannelLN → SiLU → Conv3d 32→1','[B,1,64,64,64]']], [26,43,31])
+h('输出是 logits，不是已阈值化的网格')
+eq('a = Decoder(z)\np(occupied) = sigmoid(a)\nS_hat = (a > 0)    等价于    (p > 0.5)')
+p('输出头不包含 sigmoid。Dice 损失中再做 sigmoid；BCE 模式直接使用 binary_cross_entropy_with_logits。推理时以零为阈值即可，无需为二值输出额外运行 sigmoid。')
+note('Encoder 与 Decoder 必须配套。若更换 VAE，原 Flow 生成的潜变量不一定落在新 Decoder 能正确解释的潜空间中；不能仅凭潜变量形状相同就混用权重。'), 'VAE / DECODER')

page('03 / VAE 基本算子与初始化',
h('预激活残差块 ResBlock3d')
+diagram([('输入 x','[B,C,R,R,R]'),('归一化 + SiLU','Conv3d 3×3×3'),('归一化 + SiLU','Conv3d 3×3×3'),('残差相加','输出 x + F(x)')], '旁路为 Identity；通道变化时类支持 1×1×1 卷积，正式层级中的残差块不改变通道')
+eq('F(x) = Conv2(SiLU(Norm2(Conv1(SiLU(Norm1(x))))))\ny = F(x) + skip(x)')
+p('conv2 的权重和偏置初始化为 0，所以同通道残差块初始化时接近恒等映射。正式配置中的每个残差块两次卷积都使用 k=3、padding=1、stride=1；空间尺寸保持不变。')
+h('ChannelLayerNorm32 的归一化维度')
+p('先把 [B,C,X,Y,Z] 置换为 [B,X,Y,Z,C]，然后对最后一维 C 做 LayerNorm，再置换回来。每个体素位置单独归一化其通道，既不是对整个三维网格归一化，也不是 BatchNorm。默认带可学习仿射参数，eps=1e-5；归一化计算转为 FP32 后再恢复输入 dtype。')
+h('降采样与升采样')
+table(['算子','默认实现','尺寸变化'],[
['DownsampleBlock3d','Conv3d(k=2,stride=2)','[B,Cin,R,R,R]→[B,Cout,R/2,R/2,R/2]'],
['UpsampleBlock3d','Conv3d(Cin→8Cout,k=3) + PixelShuffle3D(2)','[B,Cin,R,R,R]→[B,Cout,2R,2R,2R]']],[24,35,41])
+eq('PixelShuffle3D(r):\n[B, C × r³, X, Y, Z] → [B, C, rX, rY, rZ]')
+p('PixelShuffle 本身没有参数，只把通道分组重排到空间子体素；用于学习上采样的参数来自前面的卷积。代码还提供 avgpool / nearest 分支，但正式 Encoder / Decoder 使用 conv 分支。')
+note('实现细节：norm_type 传给输入/输出头的 norm_layer；当前构造残差块时未向其透传 norm_type，残差块保持默认 layer。若未来要整体切换 GroupNorm，需要同时修改构造逻辑，并检查权重兼容性。'), 'VAE / BUILDING BLOCKS')

page('04 / VAE 训练目标与样本语义',
p('VAE 联合更新 Encoder 与 Decoder，不输入照片。数据集提供 S∈{0,1}<sup>B×1×64×64×64</sup>，训练目标是重建输入网格，同时使后验接近标准正态。')
+h('默认 Dice + KL')
+eq('p = sigmoid(Decoder(z))\nL_dice = 1 - (2 × Σ(p ⊙ S) + 1) / (Σp + ΣS + 1)\nL_KL = 0.5 × mean(μ² + exp(logvar) - logvar - 1)\nL_VAE = L_dice + 0.001 × L_KL')
+p('Σ 在当前实现中跨整个 batch 和全部体素求和，末尾 +1 是平滑项。KL 对 batch、潜通道和空间位置求均值。Dice 关注重建占据区域的重叠；KL 约束潜表示的分布。二者的权重与归约方式必须同时考虑，不能只比较公式中的数值系数。')
+h('可选 BCE 重建项')
+eq('L_BCE = mean(BCEWithLogits(a, S))\nL_VAE = L_BCE + λ_KL × L_KL')
+table(['训练设定','独立正式配置'],[
['优化器','AdamW；学习率 1e-4；weight_decay=0'],
['精度','FP32 master parameters + bf16 autocast'],
['梯度与 EMA','全局梯度范数裁剪 1.0；EMA 衰减 0.9999'],
['batch / 累积','batch_size=1；accumulation_steps=1，可调整'],
['步数','max_steps=100000；这是当前独立配置，不是原论文训练预算']])
+h('占据网格需要保持一致的定义')
+p('原体素化工具记录与三角网格表面相交的体素，没有执行封闭实体内部填充。自行使用实体占据训练也可行，但目标分布会改变。应统一中心、尺度、坐标轴；若统一正面，则在整个数据集维持同一约定。')
+note('Flow 缓存使用 Encoder 的 μ，不对 SS 潜变量额外缩放或标准化。重新训练 Encoder 后应重新缓存；cache_metadata.json 会记录编码器权重 SHA-256。'), 'VAE / OBJECTIVE')

page('05 / SS Flow：总体网络结构',
p('<b>SparseStructureFlowModel</b> 是带图片 Cross-Attention 的三维潜空间 Transformer。输入为带噪潜变量 x<sub>t</sub>、时间 t 和条件序列 c，输出与 x<sub>t</sub> 同形状的速度场 v<sub>θ</sub>，不是直接输出 64³ 占据概率。')
+diagram([('带噪潜变量','8 × 16³'),('空间序列化','4096 × 8'),('Transformer ×24','4096 × 1024'),('速度场','8 × 16³')], '图片 tokens 进入每个块的 Cross-Attention；时间嵌入调制每个块')
+table(['步骤','运算','输出形状'],[
['输入 x_t','潜变量网格','[B,8,16,16,16]'],
['patchify，p=1','不合并邻域，只重排维度','[B,8,16,16,16]'],
['flatten + transpose','L=(16/p)³=4096','[B,4096,8]'],
['input_layer','Linear(8→1024)','[B,4096,1024]'],
['加位置编码','固定 APE buffer [4096,1024]','[B,4096,1024]'],
['blocks','ModulatedTransformerCrossBlock ×24','[B,4096,1024]'],
['输出归一化','F.layer_norm，末维 1024，无仿射参数','[B,4096,1024]'],
['out_layer','Linear(1024→8)','[B,4096,8]'],
['reshape + unpatchify','恢复规则三维网格','[B,8,16,16,16]']],[27,42,31])
+h('正式超参数')
+p('model_channels=1024；num_heads=16；每头 64 维；num_blocks=24；MLP ratio=4（隐藏维度 4096）；patch_size=1；pe_mode=ape；share_mod=False；自注意力 qk_rms_norm=True，交叉注意力 qk_rms_norm_cross=False。')
+note('这不是多尺度 U-Net：所有 24 个块都在相同 16³ 空间尺度上处理序列，没有 Transformer 内部的空间上下采样，也没有跨层 Encoder–Decoder 拼接。'), 'FLOW / OVERVIEW')

page('06 / Flow Transformer 块内部',
p('每块依次执行时间调制的 Self-Attention、图片 Cross-Attention、时间调制的 MLP，包含三条残差路径。h 始终为 [B,4096,1024]；c 为 [B,T,1024]；时间嵌入 e 为 [B,1024]。')
+h('时间条件产生六组调制向量')
+eq('[shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff]\n    = chunk(Linear_1024→6144(SiLU(e)), 6)')
+p('每一组形状为 [B,1024]，沿 token 维广播为 [B,1,1024]。share_mod=False 表示各块分别持有调制 MLP，不在 24 层之间共享参数。')
+h('三条残差路径的实际顺序')
+eq('u = LN1(h) × (1 + scale_sa) + shift_sa\nh1 = h  + gate_sa × SelfAttention(u)\n\nh2 = h1 + CrossAttention(LN2(h1), c)\n\nw = LN3(h2) × (1 + scale_ff) + shift_ff\nh3 = h2 + gate_ff × MLP(w)')
+table(['组件','实现细节'],[
['LN1 / LN3','elementwise_affine=False，eps=1e-6'],
['LN2','elementwise_affine=True，eps=1e-6；共 2048 个仿射参数'],
['MLP','Linear 1024→4096 → GELU(approximate=tanh) → Linear 4096→1024'],
['Cross-Attention','Q 来自网格 tokens；K/V 来自图片 tokens；没有额外时间 gate'],
['注意力 Dropout','独立 SDPA 调用使用默认 dropout_p=0']],[28,72])
+h('初始化与梯度检查点')
+p('一般 Linear 用 Xavier uniform，偏置置零；时间 MLP 权重用标准差 0.02 的正态分布。各块调制输出层和整个 Flow 的 out_layer 置零。初始化时自注意力和 MLP gate 为零，但 Cross-Attention 路径未被该 gate 关闭。正式训练启用 use_checkpoint=True，通过重算块前向降低激活保存量。'), 'FLOW / BLOCK')

page('07 / Self-Attention 与 Cross-Attention',
h('自注意力：建立三维网格位置之间的关系')
+eq('h: [B,L,1024] → Linear_QKV → [B,L,3,16,64]\nQ,K,V: [B,16,L,64]    (L=4096)\nAttention(Q,K,V) = softmax(Q Kᵀ / √64) V\n拼接各头 → [B,L,1024] → Linear_out')
+p('Q、K 经过每头的 MultiHeadRMSNorm，V 不归一化。实现将向量按最后一维做 L2 normalize，再乘 √64 和可学习 γ。γ 的形状为 [16,64]，Q 与 K 各自有一组参数。')
+eq('Q_norm = normalize_L2(Q.float(), dim=-1) ⊙ γ_Q × √64\nK_norm = normalize_L2(K.float(), dim=-1) ⊙ γ_K × √64')
+h('交叉注意力：把图片信息注入网格')
+eq('Q = Linear_Q(h),         h: [B,4096,1024]\nK,V = split(Linear_KV(c)), c: [B,T,1024]\nQ: [B,16,4096,64]    K,V: [B,16,T,64]')
+p('每个三维位置可以查询图片中所有 token，学习可见轮廓、部件和外观线索与目标形状的关系。正式配置未开启交叉注意力的 Q/K RMSNorm。模型没有把某个图片 token 通过已知相机投影，硬性对应到某个三维位置。')
+table(['项目','Self-Attention','Cross-Attention'],[
['Query 来源','网格 tokens','网格 tokens'],
['Key / Value 来源','同一组网格 tokens','图片 tokens'],
['注意力矩阵概念尺寸','[B,16,4096,4096]','[B,16,4096,T]'],
['Q/K RMSNorm','开启','默认关闭'],
['空间 / 条件联系','网格位置之间','网格位置与图片之间']])
+note('独立版本使用 PyTorch SDPA；参数布局与原网络兼容。SDPA 可选高效内核，实际运行不一定显式保存完整注意力矩阵；上表描述数学上的尺寸。'), 'FLOW / ATTENTION')

page('08 / 三维位置编码与时间编码',
h('固定三维正弦位置编码 APE')
+p('对 16×16×16 网格构造 meshgrid(indexing=ij)，坐标范围为每轴 0～15，按与潜网格相同的顺序展平成 4096 个位置。位置编码无可训练参数，作为 pos_emb buffer 保存到 state dict。')
+eq('F = floor(1024 / 3 / 2) = 170\nω_j = 10000^(-j/F),    j=0,...,169\nPE_axis(a) = concat(sin(aω), cos(aω))  # 340 维\nPE(x,y,z) = concat(PE_axis(x), PE_axis(y), PE_axis(z), 0,0,0,0)')
+p('三个轴共产生 1020 维，再补 4 个零得到 1024 维；位置向量在进入第一个 Transformer 块之前相加。当前独立 Flow 仅支持 APE，不能通过切换配置把现有权重直接变成 RoPE 架构。')
+h('时间编码 TimestepEmbedder')
+eq('τ = 1000 × t\nω_j = exp(-log(10000) × j/128),    j=0,...,127\nf(τ) = concat(cos(τω), sin(τω))     # [B,256]\ne = Linear_1024→1024(SiLU(Linear_256→1024(f(τ))))')
+p('训练和推理都先将归一化时间 t∈[0,1] 乘以 1000，再送给模型。若在外部调用 forward 时遗漏该倍率，会改变网络看到的时间条件。时间 MLP 输出 [B,1024]，再由每个块的调制 MLP 产生六组 scale、shift、gate。')
+h('patch_size 如何影响 token 数')
+table(['patch_size','token 长度 L','每 token 的输入维度'],[
['1（正式权重）','16³=4096','8'],
['2（架构可配置）','8³=512','8×2³=64']])
+note('改变 patch_size 会改变输入/输出 Linear 权重形状与 pos_emb 尺寸，通常不能严格加载现有权重。扩展分辨率也必须同步检查 VAE、位置编码、训练潜变量和计算开销。'), 'FLOW / EMBEDDINGS')

page('09 / 图片编码与多视角条件',
p('默认外部图片编码器为 <b>dinov2_vitl14_reg</b>。它不属于 555.54M 参数的 SS Flow；独立训练默认读取已缓存图片特征，DINOv2 不参与 Flow 的优化器更新。')
+h('前处理与特征提取')
+diagram([('图片 / Alpha','EXIF 朝向修正'),('前景裁剪','去背景 + 1.2×包围框'),('缩放与归一化','518×518 RGB'),('DINOv2 tokens','LayerNorm → 条件 c')])
+bullets([
'已有非全不透明 alpha 时直接使用；否则默认调用 rembg。去背景前先把长边限制到 1024。',
'以 alpha&gt;0.8×255 的区域确定包围框，围绕中心按较长边扩大约 1.2 倍，裁剪为方形并缩放到 518×518。RGB 与 alpha 相乘，透明区域对应黑色。',
'RGB 归一化到 [0,1]，再按 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225] 标准化。',
'取 DINOv2 的 x_prenorm，随后对最后一维执行无仿射 LayerNorm；不是只取 CLS，也不是只取 patch tokens。'])
+table(['条件维度','默认值及来源'],[
['图像 patch 网格','518/14=37，因此 37²=1369 个 patch tokens'],
['序列长度 T','1369 patch + 1 CLS + 4 register = 1374'],
['token 特征维度','1024，与 Flow cond_channels 匹配'],
['单视角 / 多视角缓存','[1,1374,1024] / [V,1374,1024]']])
+h('训练视角与推理融合')
+p('缓存以一个对象的多个视角为一条记录。训练时数据集每个 epoch 为对象选取一个视角；推理时可轮流引导或逐步平均预测。没有正面/侧面标签、相机内外参、相机位姿估计或显式多视角三角测量。')
+note('上表 T=1374 针对默认 DINOv2 版本的前缀 token 约定。外接条件允许可变 T，但同一训练 batch 需要一致 T；更换条件通道 C 必须同步修改网络并重新训练匹配权重。'), 'CONDITIONING / IMAGES')

page('10 / Flow Matching 的训练定义',
p('训练目标 z 使用配套 VAE Encoder 的后验均值。Flow 不通过二值网格上的重建损失来训练，而是在带噪潜空间回归速度。VAE 与 Flow 在当前工程中分阶段训练，不进行端到端联合更新。')
+h('带噪路径与目标速度')
+eq('ε ~ Normal(0,I)\nt = sigmoid(η),  η ~ Normal(mean=1,std=1)\nσ_min = 1e-5\nx_t = (1-t)z + [σ_min + (1-σ_min)t] ε\nv_target = d(x_t)/dt = (1-σ_min)ε - z\nL_flow = mean((v_θ(x_t,1000t,c) - v_target)²)')
+p('t=1 对应纯标准高斯噪声；t=0 对应 z+σ_min ε。因此这条路径在零时刻仍有极小噪声项，采样实现直接返回积分终点，不额外做解析去噪。不要把这里的时间方向与其他 Flow 实现混用。')
+h('Classifier-Free Guidance 的训练准备')
+eq('m ~ Bernoulli(p_uncond=0.1), 每个样本独立\nc_train = 0 if m=1 else c')
+p('置零的是整条图片条件序列。这样同一个模型学到有条件与无条件速度，推理时可以组合二者。条件置零不等于传入一张黑色图片：黑色图片经过 DINOv2 仍可能产生非零特征。')
+table(['阶段','训练哪些参数','固定 / 缓存部分'],[
['VAE 重建','Encoder + Decoder','不使用图片编码器'],
['缓存数据','不更新参数','Encoder 均值 + 冻结 DINOv2'],
['Flow Matching','仅 SS Flow','缓存潜变量、图片特征；不更新 VAE / DINOv2']])
+note('同一对象的照片与目标网格必须遵循一致的坐标语义。若改用随机旋转目标，又没有同步图片关系或额外相机条件，训练目标可能产生歧义。模型并未显式获知“此图是正面”。'), 'FLOW / TRAINING')

page('11 / Euler 采样、CFG 与多图融合',
h('时间序列与 Euler 更新')
+eq('u_i = 1 - i/N,    i=0,...,N\nt_i = r×u_i / [1+(r-1)u_i],    r=3\nx_1 ~ Normal(0,I)\nx_(t_next) = x_t - (t - t_next) × v_guided(x_t,t,c)')
+p('默认 N=12，从 t=1 向 0 积分；重标定 r=3 改变各阶段的步长分布。Flow 预测潜空间速度，每次只更新同一个潜网格。最后运行一次 VAE Decoder，再以 logits&gt;0 得到 64³ 占据网格。')
+h('引导区间与强度的精确定义')
+eq('t∈[0.5,1]: v_guided = (1+g) v_cond - g v_uncond\n其他时间:  v_guided = v_cond\n默认 g=7.5；v_uncond 使用全零条件')
+p('这里的 g 与有些实现中 v_uncond+w(v_cond-v_uncond) 的 w 相差 1：w=1+g。g=0 仍使用条件预测，并不变为无条件采样。')
+h('多视角如何共同约束一个网格')
+table(['模式','每一步的条件预测','计算特性'],[
['stochastic','第 i 步用第 i mod V 张图；实际为确定性轮换','每步一张条件图；步数须≥图片数'],
['multidiffusion','对同一个 x_t 分别预测各视角速度，再算术平均','每步 V 次条件预测，CFG 区间再加一次无条件预测']],[22,48,30])
+p('对于 3 张图和 12 步，轮流模式按 0、1、2 循环，每张图使用 4 次。两种模式都不是先生成多个模型再拼接；也不是将所有视角特征拼成长序列后一次送入网络。随机种子控制初始噪声，轮流模式的视角顺序由输入列表决定。')
+note('局限：训练数据提供形状先验，图片条件提供可见线索，但没有显式相机几何约束；多视图冲突、遮挡或外观不一致时，不能保证严格几何对齐。'), 'INFERENCE / SAMPLING')

enc=STATS['encoder']['total'];dec=STATS['decoder']['total'];flow=STATS['denoiser']['total']
page('12 / 参数规模与计算成本',
p('下表由当前正式配置实例化网络并汇总 Parameter.numel() 得到，包含可训练权重与偏置，不包含 DINOv2，也不把 pos_emb buffer 计为参数。')
+table(['模块','精确参数量','约百万参数'],[
['SS Encoder',f'{enc:,}',f'{enc/1e6:.2f} M'],
['SS Decoder',f'{dec:,}',f'{dec/1e6:.2f} M'],
['SS VAE 合计',f'{enc+dec:,}',f'{(enc+dec)/1e6:.2f} M'],
['SS Flow',f'{flow:,}',f'{flow/1e6:.2f} M']])
+h('Flow 的参数主要集中于 24 个 Transformer 块')
+table(['单块组件','参数量'],[
['Self-Attention，含 Q/K RMSNorm','4,200,448'],
['Cross-Attention','4,198,400'],
['两层 MLP','8,393,728'],
['时间调制 SiLU + Linear','6,297,600'],
['LN2 仿射参数','2,048'],
['每块总计 / 24 块总计','23,092,224 / 554,213,376']])
+h('显存：参数量之外还要计算训练状态与激活')
+p('Flow 的 FP32 参数约 2.07 GiB；若同时保存参数、梯度和 Adam 的两个 FP32 动量，仅这四项就约 8.28 GiB。再保存一份 FP32 EMA 参数约增加 2.07 GiB，尚未包含激活、CUDA 工作区、位置 buffer、加载峰值等。混合精度不代表这些 FP32 master 状态全部减半。')
+p('单样本隐藏序列 [4096,1024] 在 FP16 下约 8 MiB；理论自注意力分数 [16,4096,4096] 在 FP16 下约 512 MiB，FP32 下约 1 GiB。SDPA 高效内核可减少显式矩阵保存；梯度检查点用重算换取更低激活存储。')
+note('这些是布局和状态的数量级估算，不是整机实测峰值。8 GB 显卡可以完成此项目预训练推理，不意味着能训练完整 L 模型。减少宽度/层数/分辨率、关闭 EMA 都会改变成本，其中部分修改也会破坏现有权重的结构兼容性。'), 'SCALE / MEMORY')

page('13 / 代码索引与迁移时的检查点',
table(['主题','ss_module 下的文件 / 类'],[
['VAE 网络','models/sparse_structure_vae.py\nSparseStructureEncoder / SparseStructureDecoder'],
['Flow 网络','models/sparse_structure_flow.py\nSparseStructureFlowModel / TimestepEmbedder'],
['时间调制块','modules/transformer/modulated.py\nModulatedTransformerCrossBlock'],
['位置编码 / MLP','modules/transformer/blocks.py'],
['QKV 投影 / RMSNorm','modules/attention/modules.py'],
['SDPA 适配 / 空间算子','modules/attention/full_attn.py / modules/spatial.py'],
['训练目标','trainers/vae/sparse_structure_vae.py\ntrainers/flow_matching/flow_matching.py'],
['采样器','pipelines/samplers/flow_euler.py'],
['数据 / 缓存 / 图片','datasets/ / prepare.py / utils/image.py'],
['训练 / 推理 API','trainers/basic.py / train.py\npipelines/sparse_structure.py / infer.py'],
['正式配置','configs/vae.json / configs/flow.json']],[27,73])
+h('三个独立权重的角色')
+p('<b>weights/encoder</b>：网格编码与训练缓存；<b>weights/decoder</b>：潜变量解码；<b>weights/denoiser</b>：图片条件速度预测。每个前缀对应一个 .json 和一个 .safetensors。权重加载保留原模型 state-dict 名称；骨干结构兼容，不代表任意组合具有语义兼容性。')
+h('迁移和修改的关键检查')
+bullets([
'改变 VAE 时同步 Encoder、Decoder、潜变量缓存及 Flow；改变图片编码器时检查 token 约定与 cond_channels。',
'不要将 SS 潜变量与后续 SLAT 潜变量混淆：本文只覆盖 SS 两阶段，未讨论 Mesh、Gaussian 或外观解码。',
'独立训练采用 AMP、全局范数裁剪与 EMA；未复刻原训练器的全部分布式与自适应裁剪策略。网络架构兼容不等于复现原训练收敛结果。',
'当前已验证 tiny 网络训练/恢复、GPU 混合精度与预训练 64³ 推理；没有进行完整规模长时间收敛实验。'])
+note('本报告依据本地代码和配置生成；DINOv2 token 数另对照本机缓存的上游源码。它是工程实现说明，不把尚未实测的训练效果或显存峰值当作结论。使用与验证命令见 ../README.md 和 ../VALIDATION.md。'), 'REFERENCE / IMPLEMENTATION')

# Insert detailed block diagrams immediately after their corresponding explanation.
from architecture_diagrams import extra_pages
expanded = []
extras = extra_pages()
for title, body, kicker in pages:
    expanded.append((title, body, kicker))
    expanded.extend(extras.get(kicker, []))
pages = expanded

css='''
@page{size:A4;margin:15mm 14mm}
*{box-sizing:border-box}body{margin:0;color:#233c36;font-family:"Noto Sans CJK SC","Droid Sans Fallback",sans-serif;font-size:10.1pt;line-height:1.65;background:white}
.sheet{height:265mm;position:relative;break-after:page;padding:0 0 13mm;overflow:hidden}.sheet:last-child{break-after:auto}.sheet:last-child table{font-size:8.5pt}.sheet:last-child td,.sheet:last-child th{padding:5px 8px}.kicker{font-size:8pt;letter-spacing:2px;color:#678778;padding:0 0 9px;border-bottom:1px solid #cfded5;margin-bottom:16px}
h1{font-size:21pt;line-height:1.35;font-weight:700;letter-spacing:-.6px;margin:0 0 17px;white-space:pre-line}h2{font-size:12.3pt;line-height:1.4;color:#2e6754;margin:18px 0 7px}p{margin:7px 0 10px}b{font-weight:700}svg{width:100%;display:block;margin:15px 0 5px;font-family:inherit}table{width:100%;border-collapse:collapse;table-layout:fixed;margin:12px 0 15px;font-size:9pt;line-height:1.6}th{background:#e8f0eb;color:#2c5848;text-align:left;font-weight:600}td,th{padding:7px 8px;border-bottom:1px solid #dfe7e2;vertical-align:top;overflow-wrap:anywhere;white-space:pre-line}tbody tr:nth-child(even){background:#f7f9f7}.note{background:#eef4ef;border-left:3px solid #588b70;padding:10px 13px;font-size:9pt;line-height:1.7;margin:15px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:"DejaVu Sans Mono","Noto Sans CJK SC",monospace;font-size:9pt;line-height:1.8;margin:10px 0 12px;padding:12px 14px;background:#f1f4f2;border-radius:5px}.equation{color:#24503f}ul{padding-left:20px;margin:10px 0}li{margin:7px 0}.subtitle{font-size:13pt;color:#678477;margin:0 0 23px}.edition{margin-top:25px;color:#7d9386;font-size:9pt}.footer{position:absolute;bottom:0;left:0;right:0;display:flex;justify-content:space-between;border-top:1px solid #d7e2da;padding-top:7px;font-size:8pt;color:#7b9185}
'''
sections=[]
for i,(title,body,kicker) in enumerate(pages,1):
    sections.append(f'<section class="sheet"><div class="kicker">{kicker}</div><h1>{title}</h1>{body}<div class="footer"><span>SS VAE / SS FLOW · 架构详解</span><span>{i:02d} / {len(pages):02d}</span></div></section>')
output=HERE/'SS_VAE_SS_Flow_Architecture.html'
output.write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>SS VAE 与 SS Flow 模型架构详解</title><style>'+css+'</style><body>'+''.join(sections)+'</body></html>')
(HERE/'architecture_stats.json').write_text(json.dumps(STATS,indent=2)+'\n')
chrome=shutil.which('google-chrome') or shutil.which('chromium')
if not chrome:raise SystemExit('需要 Chrome/Chromium 生成 PDF；HTML 已生成。')
subprocess.run([chrome,'--headless','--no-sandbox','--disable-dev-shm-usage','--no-pdf-header-footer',
                '--print-to-pdf='+str(HERE/'SS_VAE_SS_Flow_Architecture.pdf'),output.as_uri()],check=True)
print(HERE/'SS_VAE_SS_Flow_Architecture.pdf')
