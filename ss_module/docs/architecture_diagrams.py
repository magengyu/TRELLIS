"""Vector network block diagrams for the architecture PDF; all labels stay selectable."""
from html import escape as esc


class Canvas:
    def __init__(self, height):
        self.parts = [f'<svg viewBox="0 0 680 {height}" xmlns="http://www.w3.org/2000/svg" role="img">',
                      '<defs><marker id="block-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#487b69"/></marker></defs>']

    def text(self, x, y, text, size=11, anchor='middle', color='#466d5c'):
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}">{esc(text)}</text>')

    def box(self, x, y, width, height, title, detail='', fill='#edf4ef'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}" stroke="#a8c1b2"/>')
        self.text(x+width/2, y+17, title, 12, color='#234e3e')
        if detail:
            self.text(x+width/2, y+33, detail, 10)

    def arrow(self, points, label=None, lx=None, ly=None, dashed=False):
        dash='stroke-dasharray="5 4"' if dashed else ''
        self.parts.append(f'<path d="{points}" fill="none" stroke="#487b69" stroke-width="1.4" marker-end="url(#block-arrow)" {dash}/>' )
        if label:self.text(lx, ly, label, 10)

    def done(self):return ''.join(self.parts)+'</svg>'


def chain(rows, input_label, step=51):
    c=Canvas(29+len(rows)*step)
    c.text(340,14,input_label,11)
    c.arrow('M340 19 V28')
    for i,(name,dims) in enumerate(rows):
        y=29+i*step
        c.box(37,y,606,41,name,dims)
        if i<len(rows)-1:c.arrow(f'M340 {y+41} V{y+step-1}')
    return c.done()


def encoder():
    rows=[('input_layer · Conv3d(1→32, k3/s1/p1)','in [B,1,64,64,64] → out [B,32,64,64,64]')]
    for index,c,r in [(0,32,64),(1,32,64)]:
        rows.append((f'blocks[{index}] · ResBlock3d({c})',f'in [B,{c},{r},{r},{r}] → out [B,{c},{r},{r},{r}]'))
    rows.append(('blocks[2] · DownsampleBlock3d(32→128)','in [B,32,64,64,64] → out [B,128,32,32,32]'))
    for index in (3,4):rows.append((f'blocks[{index}] · ResBlock3d(128)','in [B,128,32,32,32] → out [B,128,32,32,32]'))
    rows.append(('blocks[5] · DownsampleBlock3d(128→512)','in [B,128,32,32,32] → out [B,512,16,16,16]'))
    for name in ('blocks[6]','blocks[7]','middle_block[0]','middle_block[1]'):
        rows.append((f'{name} · ResBlock3d(512)','in [B,512,16,16,16] → out [B,512,16,16,16]'))
    rows += [('out_layer · ChannelLN(512) → SiLU → Conv3d(512→16, k3/s1/p1)',
              'in [B,512,16,16,16] → out [B,16,16,16,16]'),
             ('chunk(2, dim=1) · 后验参数分支',
              'in [B,16,16,16,16] → μ [B,8,16,16,16] + logvar [B,8,16,16,16]'),
             ('训练：重参数化采样 z；推理 / 缓存：z=μ',
              'μ、logvar、ε 的形状均为 [B,8,16,16,16] → z [B,8,16,16,16]')]
    return chain(rows,'输入 S：[B,1,64,64,64]；实线箭头表示前向数据流')


def decoder():
    rows=[('input_layer · Conv3d(8→512, k3/s1/p1)','in [B,8,16,16,16] → out [B,512,16,16,16]')]
    for name in ('middle_block[0]','middle_block[1]','blocks[0]','blocks[1]'):
        rows.append((f'{name} · ResBlock3d(512)','in [B,512,16,16,16] → out [B,512,16,16,16]'))
    rows.append(('blocks[2] · UpsampleBlock3d(512→128)','in [B,512,16,16,16] → out [B,128,32,32,32]'))
    for index in (3,4):rows.append((f'blocks[{index}] · ResBlock3d(128)','in [B,128,32,32,32] → out [B,128,32,32,32]'))
    rows.append(('blocks[5] · UpsampleBlock3d(128→32)','in [B,128,32,32,32] → out [B,32,64,64,64]'))
    for index in (6,7):rows.append((f'blocks[{index}] · ResBlock3d(32)','in [B,32,64,64,64] → out [B,32,64,64,64]'))
    rows += [('out_layer · ChannelLN(32) → SiLU → Conv3d(32→1, k3/s1/p1)',
              'in [B,32,64,64,64] → logits [B,1,64,64,64]'),
             ('占据判定：logits > 0 （Decoder 外部操作）',
              'in [B,1,64,64,64] float → out [B,1,64,64,64] bool')]
    return chain(rows,'输入 z：[B,8,16,16,16]；Decoder 中没有 Encoder 特征跳连')


def residual():
    c=Canvas(551)
    c.text(340,17,'输入 x：[B,C,R,R,R]；正式模型中 Cin=Cout=C',12)
    c.arrow('M340 22 V40')
    rows=[('norm1 · ChannelLayerNorm32(C)','[B,C,R,R,R] → [B,C,R,R,R]'),
          ('F.silu · SiLU','[B,C,R,R,R] → [B,C,R,R,R]'),
          ('conv1 · Conv3d(C→C, k3/s1/p1)','[B,C,R,R,R] → [B,C,R,R,R]'),
          ('norm2 · ChannelLayerNorm32(C)','[B,C,R,R,R] → [B,C,R,R,R]'),
          ('F.silu · SiLU','[B,C,R,R,R] → [B,C,R,R,R]'),
          ('conv2 · Conv3d(C→C, k3/s1/p1)，零初始化','[B,C,R,R,R] → [B,C,R,R,R]'),
          ('Add · 残差相加','F(x) + skip(x) → [B,C,R,R,R]')]
    for i,(name,dims) in enumerate(rows):
        y=40+i*64;c.box(145,y,480,43,name,dims)
        if i<6:c.arrow(f'M340 {y+43} V{y+63}')
    c.arrow('M340 30 H74 V445 H144',dashed=True)
    c.text(77,257,'skip(x)',11)
    c.text(77,275,'Identity',10)
    c.text(77,293,'[B,C,R,R,R]',9)
    c.arrow('M340 467 V495')
    c.text(340,516,'输出 y：[B,C,R,R,R]',12)
    c.text(340,539,'虚线为跳连数据流；通道变化时类支持 Conv3d(k1)，正式残差块没有通道变化',10)
    return c.done()


def primitives():
    c=Canvas(565)
    c.text(340,18,'DownsampleBlock3d · 默认 mode=conv',13)
    c.text(340,44,'输入 [B,Cin,R,R,R]',11);c.arrow('M340 51 V70')
    c.box(75,70,530,44,'conv · Conv3d(Cin→Cout, kernel=2, stride=2, padding=0)',
          '[B,Cin,R,R,R] → [B,Cout,R/2,R/2,R/2]')
    c.arrow('M340 114 V133');c.text(340,153,'输出 [B,Cout,R/2,R/2,R/2]',11)
    c.text(340,194,'UpsampleBlock3d · 默认 mode=conv',13)
    c.text(340,220,'输入 [B,Cin,R,R,R]',11);c.arrow('M340 227 V245')
    c.box(75,245,530,44,'conv · Conv3d(Cin→8Cout, kernel=3, stride=1, padding=1)',
          '[B,Cin,R,R,R] → [B,8Cout,R,R,R]')
    c.arrow('M340 289 V309')
    c.box(75,309,530,44,'pixel_shuffle_3d(scale_factor=2) · 无可训练参数',
          '[B,8Cout,R,R,R] → [B,Cout,2R,2R,2R]')
    c.arrow('M340 353 V373');c.text(340,393,'输出 [B,Cout,2R,2R,2R]',11)
    c.text(340,435,'ChannelLayerNorm32 · 每个体素独立归一化通道',13)
    c.box(22,458,194,44,'permute','[B,C,X,Y,Z] → [B,X,Y,Z,C]')
    c.box(243,458,194,44,'LayerNorm(C), FP32','[B,X,Y,Z,C] → 同形状')
    c.box(464,458,194,44,'permute 回原轴序','[B,X,Y,Z,C] → [B,C,X,Y,Z]')
    c.arrow('M216 480 H240');c.arrow('M437 480 H461')
    c.text(340,541,'维度符号：B=batch；C=通道；R=空间边长；k/s/p=kernel/stride/padding',11)
    return c.done()


def flow_overview():
    c=Canvas(717)
    c.text(232,18,'主数据 x_t：[B,8,16,16,16]',12)
    main=[('patchify(p=1) → flatten → transpose','[B,8,16,16,16] → [B,4096,8]'),
          ('input_layer · Linear(8→1024)','[B,4096,8] → [B,4096,1024]'),
          ('Add · 固定 3D APE 位置编码','h [B,4096,1024] + pos_emb [4096,1024]'),
          ('blocks[0] … blocks[23] · CrossBlock ×24','每块 in/out：[B,4096,1024]'),
          ('F.layer_norm · 无仿射参数','[B,4096,1024] → [B,4096,1024]'),
          ('out_layer · Linear(1024→8)','[B,4096,1024] → [B,4096,8]'),
          ('transpose → reshape → unpatchify(p=1)','[B,4096,8] → vθ [B,8,16,16,16]')]
    for i,(name,dims) in enumerate(main):
        y=43+i*67;c.box(15,y,430,44,name,dims)
        if i<6:c.arrow(f'M230 {y+44} V{y+66}')
    c.arrow('M230 23 V41')
    c.box(470,45,198,44,'时间 t → 1000t','in [B] → [B]')
    c.arrow('M569 89 V111')
    c.box(470,112,198,44,'t_embedder · Time MLP','[B] → e [B,1024]')
    c.arrow('M569 156 V257 H447',dashed=True)
    c.text(554,238,'传入每个块',10)
    c.box(470,333,198,44,'图片 tokens c','[B,T,1024]；默认 T=1374')
    c.arrow('M470 355 H456 V277 H447',dashed=True)
    c.arrow('M230 489 V509')
    c.box(15,510,653,44,'外部 Euler 采样循环 · 使用 vθ 更新 x_t，重复 N 步',
          'x_next = x_t − Δt × vθ；两者均为 [B,8,16,16,16] → 最终潜变量 z')
    c.arrow('M230 554 V574')
    c.box(15,575,653,44,'SparseStructureDecoder · 独立 VAE 解码器',
          '[B,8,16,16,16] → logits [B,1,64,64,64] → threshold >0 → 占据网格')
    c.text(340,648,'主干所有 Transformer 块都保持 L=4096、D=1024，不改变三维潜网格分辨率',11)
    c.text(340,673,'c 是条件序列；e 是时间条件。它们不是主干空间 tokens 的直接拼接。',11)
    c.text(340,698,'先完成 N 步 Flow 迭代得到最终潜变量，再调用一次 Decoder；不是每步都解码网格。',10)
    return c.done()


def transformer():
    c=Canvas(705)
    c.text(245,18,'输入 h：[B,L,1024]，L=4096',12)
    rows=[('norm1 · LayerNorm，无仿射','[B,L,1024] → [B,L,1024]'),
          ('时间调制：u=LN(h)×(1+scale_sa)+shift_sa','[B,L,1024]；scale / shift [B,1,1024]'),
          ('self_attn · MultiHeadAttention，16 heads','[B,L,1024] → [B,L,1024]'),
          ('gate_sa × SelfAttn(u) + h · 第 1 次残差相加','gate [B,1,1024] → h1 [B,L,1024]'),
          ('norm2 + cross_attn · 图片交叉注意力','Q [B,L,1024]；K/V 条件 [B,T,1024] → [B,L,1024]'),
          ('CrossAttn(...) + h1 · 第 2 次残差相加','[B,L,1024] → h2 [B,L,1024]'),
          ('norm3 + 时间调制','[B,L,1024]；scale_ff / shift_ff [B,1,1024]'),
          ('mlp · Linear(1024→4096) → GELU → Linear(4096→1024)',
           '[B,L,1024] → [B,L,4096] → [B,L,1024]'),
          ('gate_ff × MLP(...) + h2 · 第 3 次残差相加','gate [B,1,1024] → 输出 [B,L,1024]')]
    x=103;width=525
    for i,(name,dims) in enumerate(rows):
        y=37+i*63;c.box(x,y,width,43,name,dims)
        if i<8:c.arrow(f'M365 {y+43} V{y+62}')
    c.arrow('M245 23 V35')
    # Residual paths explicitly branch before each residual sublayer.
    c.arrow('M245 28 H42 V248 H101',dashed=True)
    c.arrow('M365 280 H66 V374 H101',dashed=True)
    c.arrow('M365 406 H42 V563 H101',dashed=True)
    c.text(46,145,'h',11);c.text(65,325,'h1',11);c.text(44,470,'h2',11)
    c.text(340,645,'额外输入：e [B,1024] → SiLU + Linear(1024→6144) → 六组 [B,1024]',11)
    c.text(340,668,'e 调制 SelfAttn / MLP；图片 c [B,T,1024] 只进入 cross_attn 的 K/V 投影。',11)
    c.text(340,691,'每一个 blocks[i]（i=0…23）独立持有以上结构；虚线为残差旁路。',10)
    return c.done()


def extra_pages():
    return {
      'VAE / ENCODER': [('结构图 A / SparseStructureEncoder',
        '<p>每个 ResBlock3d 和下采样 block 均单独展开；名称与 Python ModuleList 的索引一致。下图所有输入输出维数都包含 batch 轴。</p>'+encoder(), 'BLOCK DIAGRAM / ENCODER')],
      'VAE / DECODER': [('结构图 B / SparseStructureDecoder',
        '<p>解码器与编码器具有不同的执行顺序：先 middle_block，再执行 blocks。上采样 block 的内部算子在结构图 D 展开。</p>'+decoder(), 'BLOCK DIAGRAM / DECODER')],
      'VAE / BUILDING BLOCKS': [
        ('结构图 C / ResBlock3d 内部数据流',
         '<p>正式 Encoder / Decoder 的残差块中，C∈{32,128,512}；R 对应 64、32 或 16。两条路径在 Add 节点逐元素相加。</p>'+residual()+
         '<div class="note">例：blocks[0] 的输入输出均为 [B,32,64,64,64]；Encoder middle_block[0] 的输入输出均为 [B,512,16,16,16]。norm1/norm2 的具体轴变换见结构图 D。</div>', 'BLOCK DIAGRAM / RESBLOCK3D'),
        ('结构图 D / 上下采样与归一化',
         primitives()+'<div class="note">Encoder 实例：Downsample(32→128,R=64)、Downsample(128→512,R=32)。Decoder 实例：Upsample(512→128,R=16)、Upsample(128→32,R=32)。通道到空间重排的尺寸变化由 r³=8 决定。</div>', 'BLOCK DIAGRAM / PRIMITIVES')],
      'FLOW / OVERVIEW': [('结构图 E / SS Flow 与条件分支',
        '<p>虚线表示额外条件输入，实线表示主数据流。循环迭代发生在 Flow 网络之外；图末尾的 Decoder 用于采样完成后的解码。</p>'+flow_overview(), 'BLOCK DIAGRAM / FLOW')],
      'FLOW / BLOCK': [('结构图 F / 单个 Transformer CrossBlock',
        '<p>与 ModulatedTransformerCrossBlock._forward 的运算顺序一致。所有残差相加两侧张量都为 [B,L,1024]，时间 gate 在 token 轴广播。</p>'+transformer(), 'BLOCK DIAGRAM / CROSSBLOCK')]
    }
