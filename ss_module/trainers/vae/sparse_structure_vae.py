import torch
from torch.nn import functional as F


def vae_loss(encoder, decoder, occupancy, loss_type='dice', lambda_kl=0.001):
    """
    Sparse Structure VAE 的训练损失函数。

    本函数针对稀疏结构（如 3D 体素占据场）的二值重建任务构造 VAE 训练目标，
    由两部分组成：
        reconstruction（重建损失）：衡量 decoder 重建 occupancy 的精度；
        kl（KL 散度）           ：约束 encoder 输出的后验分布接近标准正态 N(0, I)。

    参数说明：
        encoder    : VAE 编码器，调用形式为 encoder(occupancy, sample_posterior=True, return_raw=True)
                     返回 z（采样后的潜变量）、mean（后验均值）、logvar（后验对数方差）
        decoder    : VAE 解码器，调用形式为 decoder(z)，返回每个体素的 logits
        occupancy  : 真实的二值占据场，形状通常为 [B, 1, X, Y, Z] 或 [B, C, X, Y, Z]，
                     元素取值 {0, 1}
        loss_type  : 重建损失类型，可选 'dice'（Dice 损失）或 'bce'（二值交叉熵）
        lambda_kl  : KL 散度项的权重系数，用于在重建质量与潜空间正则之间做权衡

    返回：
        一个 dict，包含：
            'loss'          : 总损失 = reconstruction + lambda_kl * kl，用于反向传播
            'reconstruction' : 仅重建损失，用于日志/分析
            'kl'            : 仅 KL 散度项，用于日志/分析
    """
    # ============ 1. 编码阶段：从 occupancy 得到潜变量与后验分布参数 ============
    # sample_posterior=True  使用重参数化技巧 z = mean + std * eps 采样潜变量
    # return_raw=True       一次性返回 (z, mean, logvar)，方便同时计算重建损失与 KL
    z, mean, logvar = encoder(occupancy, sample_posterior=True, return_raw=True)

    # ============ 2. 解码阶段：从 latent 重建 occupancy，得到逐体素 logits ============
    # logits 是 decoder 的原始输出（未经过 sigmoid），转换为 float 是为了在混合精度训练中
    # 避免低精度下的数值不稳定（NaN / 溢出）问题
    logits = decoder(z).float()

    # ============ 3. 重建损失分支 ============
    if loss_type == 'dice':
        # 先把 logits 通过 sigmoid 转为 [0, 1] 的预测概率
        prob = logits.sigmoid()

        # Dice 系数：衡量预测占据体素与真实占据体素之间的重叠度
        #   Dice = 2 * |prob ∩ occupancy| / (|prob| + |occupancy|)
        # 分母与分子同时 +1 是 Laplace / Additive 平滑，防止全空 batch 时分母为 0，
        # 同时也避免训练初期 prob 接近 0 时梯度消失。
        # 这里取 1 - Dice 作为"损失"：预测越准，损失越接近 0。
        reconstruction = 1 - (2 * (prob * occupancy).sum() + 1) / (prob.sum() + occupancy.sum() + 1)
    elif loss_type == 'bce':
        # 二值交叉熵损失：等价于 sigmoid(z) 与 occupancy 之间的逐元素交叉熵
        # PyTorch 内部已在数值稳定形式下计算（自带 log-sum-exp 技巧），
        # 与直接 F.binary_cross_entropy(logits.sigmoid(), occupancy) 等价但更稳定。
        reconstruction = F.binary_cross_entropy_with_logits(logits, occupancy)
    else:
        # 未知 loss_type 时直接报错，避免后续 None/0 损失静默失败
        raise ValueError('loss_type must be dice or bce')

    # ============ 4. KL 散度：约束后验分布 q(z|x) 接近标准正态先验 p(z) = N(0, I) ============
    # 假设后验分布 q(z|x) = N(mean, exp(logvar))，则与 N(0, I) 的解析 KL 为：
    #     KL = 0.5 * Σ ( mean^2 + exp(logvar) - logvar - 1 )
    # 最后在 batch 维度取 .mean()，得到一个标量；float() 用于在混合精度下安全做指数运算。
    kl = 0.5 * (mean.float().square() + logvar.float().exp() - logvar.float() - 1).mean()

    # ============ 5. 汇总并返回损失字典 ============
    # 'loss' 是反向传播用到的总目标；
    # 'reconstruction' 与 'kl' 仅作为日志指标，方便训练时观察两者的相对量级，
    # 进而动态调整 lambda_kl。
    return {'loss': reconstruction + lambda_kl * kl, 'reconstruction': reconstruction, 'kl': kl}
