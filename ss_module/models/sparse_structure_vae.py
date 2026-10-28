from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    归一化层工厂函数（Return a normalization layer）。

    根据 norm_type 字符串返回对应的归一化模块：
        - "group"：GroupNorm32（32 组，常见于扩散模型，内部会做类型/精度适配）
        - "layer"：ChannelLayerNorm32（按通道做 LayerNorm，适合 3D 特征）

    参数：
        norm_type : 归一化类型，只支持 "group" 或 "layer"
        *args / **kwargs : 透传给具体归一化层的参数（如通道数 channels）

    返回：
        对应的归一化层实例
    """
    if norm_type == "group":
        # 分组归一化，固定 32 组；输入通道数由 args 传入
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        # 通道维 LayerNorm，输入通道数由 args 传入
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        # 非法类型直接报错，避免拼写错误被静默忽略
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    """
    3D 残差卷积块（pre-norm 结构）。

    结构顺序：
        x -> Norm -> SiLU -> Conv1 -> Norm -> SiLU -> Conv2(zero-init) -> + skip(x)

    说明：
        - 使用 pre-norm（先归一化再卷积），训练更稳定；
        - 第二个卷积 conv2 通过 zero_module 把权重和偏置初始化为 0，
          因此训练初始阶段该残差分支输出为 0，整个块等价于恒等映射，
          有利于深层网络稳定收敛（类似 DiT / LDM 的做法）；
        - 当输入输出通道不一致时，用 1x1 卷积把 skip 分支对齐到 out_channels，
          否则直接使用恒等映射。
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels  # 未指定 out_channels 时与输入通道相同

        # 两个归一化层分别位于 conv1 与 conv2 之前
        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        # 第一个卷积：可改变通道数
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        # 第二个卷积：零初始化，使残差分支初始输出为 0
        self.conv2 = zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        # skip 连接：通道一致走 Identity，否则用 1x1 卷积做通道对齐
        self.skip_connection = nn.Conv3d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)          # SiLU (swish) 激活
        h = self.conv1(h)      # 3D 卷积，保持空间尺寸（padding=1）
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)      # 零初始化的残差分支
        h = h + self.skip_connection(x)  # 残差求和
        return h


class DownsampleBlock3d(nn.Module):
    """
    3D 下采样块：把空间分辨率缩小为原来的一半（stride=2）。

    支持两种模式：
        - "conv"   ：用 kernel=2、stride=2 的 3D 卷积同时完成下采样与通道变换；
        - "avgpool"：用 avg_pool3d 做 2x2x2 平均池化，要求输入输出通道相同。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            # kernel=2 + stride=2：每个 2x2x2 区域被压缩成 1 个体素
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            # 池化不改变通道数，因此要求 in_channels == out_channels
            assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            # 平均池化：2x2x2 窗口，stride 默认等于 kernel size，等价于减半
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    """
    3D 上采样块：把空间分辨率放大为原来的两倍。

    支持两种模式：
        - "conv"   ：先做 3D 卷积得到 out_channels * 8 个通道，
                     再用 pixel_shuffle_3d 做 2x2x2 的亚像素重排（3D 版 PixelShuffle），
                     这与转置卷积相比能避免棋盘伪影，也是 LDM/DiT 常用做法；
        - "nearest"：直接用最近邻插值放大 2 倍，要求输入输出通道相同。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            # 每个空间位置生成 8 组特征，对应 2x2x2 = 8 个子体素
            self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        elif mode == "nearest":
            # 最近邻插值不改变通道数
            assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            # 把通道维的 8 倍拆分到 (D, H, W) 三个空间维，实现 2 倍上采样
            return pixel_shuffle_3d(x, 2)
        else:
            # 三线性/最近邻插值放大 2 倍
            return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureEncoder(nn.Module):
    """
    稀疏结构编码器（论文 Sec. 3.3 中的 \mathcal{E}_S）。

    作用：把输入的二值/占据场（sparse structure）编码为 VAE 的潜空间表示。
    整体流程：
        Input Conv3d
          -> 多个阶段的 [ResBlock3d x num_res_blocks + DownsampleBlock3d]（逐级下采样）
          -> middle_block（多个 ResBlock3d，保持分辨率）
          -> out_layer（Norm + SiLU + Conv3d 输出 2 * latent_channels 通道）
          -> 把输出通道一分为二得到 mean 与 logvar
          -> 重参数化采样得到 z（可选的 sample_posterior）

    Args:
        in_channels (int): 输入通道数。
        latent_channels (int): 潜空间通道数（输出会翻倍用于 mean/logvar）。
        num_res_blocks (int): 每个分辨率阶段的残差块数量。
        channels (List[int]): 各阶段的通道数列表，长度决定下采样次数。
        num_res_blocks_middle (int): 中间瓶颈处的残差块数量。
        norm_type (Literal["group", "layer"]): 归一化层类型。
        use_fp16 (bool): 是否用 FP16 构建/运行。
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        # 保存结构超参数，便于序列化/调试
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        # 主体部分（blocks/middle_block）使用的计算精度
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # 输入卷积：把原始通道映射到第一个阶段的通道数，空间尺寸不变
        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        # 逐阶段堆叠：每个阶段先放 num_res_blocks 个残差块，再（除最后一阶段外）下采样
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                # 阶段之间插入下采样，把分辨率减半并切到下一阶段通道数
                self.blocks.append(
                    DownsampleBlock3d(ch, channels[i+1])
                )

        # 瓶颈（middle）部分：保持最低分辨率与最大通道数，进一步加深网络
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        # 输出层：归一化 + 激活 + 卷积，输出 2*latent_channels 通道
        # （一半作为 mean，一半作为 logvar）
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1)
        )

        # 若启用 fp16，则把主体转换到半精度（输入/输出层保持 fp32 以兼容外部）
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        返回模型参数所在的设备（Return the device of the model）。
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        把模型主体（blocks 与 middle_block）转换为 float16。
        注意：input_layer / out_layer 不在转换范围内，避免输入输出精度不一致。
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        把模型主体（blocks 与 middle_block）转换回 float32。
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
        """
        前向传播。

        参数：
            x              : 输入张量，形状 [B, in_channels, D, H, W]
            sample_posterior: 是否用重参数化技巧从后验分布中采样 z；
                              为 False 时直接返回均值 mean（推理常用）
            return_raw     : 是否额外返回 (z, mean, logvar)，训练 VAE 时需要

        返回：
            - return_raw=False 时返回 z
            - return_raw=True  时返回 (z, mean, logvar)
        """
        # 输入卷积保持 fp32（或外部 dtype）
        h = self.input_layer(x)
        # 主体部分切换到内部 dtype（fp16/fp32）
        h = h.type(self.dtype)

        # 逐阶段：残差块 + 下采样
        for block in self.blocks:
            h = block(h)
        # 瓶颈残差块
        h = self.middle_block(h)

        # 输出层前切回输入 dtype，保证输出与调用方精度一致
        h = h.type(x.dtype)
        h = self.out_layer(h)

        # 沿通道维一分为二：前一半是后验均值，后一半是后验对数方差
        mean, logvar = h.chunk(2, dim=1)

        if sample_posterior:
            # 重参数化技巧：z = mean + std * eps, eps ~ N(0, I)
            # std = exp(0.5 * logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            # 不采样时直接使用均值，得到确定性的潜表示
            z = mean

        if return_raw:
            # 训练 VAE 时需要 mean/logvar 来算 KL 损失
            return z, mean, logvar
        return z


class SparseStructureDecoder(nn.Module):
    """
    稀疏结构解码器（论文 Sec. 3.3 中的 \mathcal{D}_S）。

    作用：把潜空间表示 z 解码回稀疏结构（占据场 logits）。
    整体流程：
        Input Conv3d
          -> middle_block（多个 ResBlock3d，保持最低分辨率）
          -> 多个阶段的 [ResBlock3d x num_res_blocks + UpsampleBlock3d]（逐级上采样）
          -> out_layer（Norm + SiLU + Conv3d 输出 out_channels）

    Args:
        out_channels (int): 输出通道数（例如 1，表示二值占据场）。
        latent_channels (int): 输入潜空间通道数。
        num_res_blocks (int): 每个分辨率阶段的残差块数量。
        channels (List[int]): 各阶段的通道数列表，长度决定上采样次数。
        num_res_blocks_middle (int): 中间瓶颈处的残差块数量。
        norm_type (Literal["group", "layer"]): 归一化层类型。
        use_fp16 (bool): 是否用 FP16 构建/运行。
    """
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        # 保存结构超参数
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # 输入卷积：把 latent_channels 映射到第一个阶段的通道数
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        # 瓶颈部分：位于最低分辨率，先处理 latent 再逐级上采样
        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        # 逐阶段堆叠：每个阶段先放 num_res_blocks 个残差块，再（除最后一阶段外）上采样
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                # 阶段之间插入上采样，把分辨率翻倍并切到下一阶段通道数
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        # 输出层：归一化 + 激活 + 卷积，映射到最终输出通道数
        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        返回模型参数所在的设备（Return the device of the model）。
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        把模型主体（blocks 与 middle_block）转换为 float16。
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        把模型主体（blocks 与 middle_block）转换回 float32。
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x : 潜空间表示，形状 [B, latent_channels, D, H, W]

        返回：
            重建的稀疏结构 logits，形状 [B, out_channels, D*2^n, H*2^n, W*2^n]
            （n 为 channels 长度 - 1，即上采样次数）
        """
        # 输入卷积
        h = self.input_layer(x)

        # 主体计算精度切换
        h = h.type(self.dtype)

        # 先过瓶颈残差块，再逐阶段：残差块 + 上采样
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        # 输出层前切回输入 dtype
        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h
