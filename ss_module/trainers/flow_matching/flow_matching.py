import torch
from torch.nn import functional as F


def flow_loss(model, latent, cond, sigma_min=1e-5, p_uncond=0.1, t_mean=1., t_std=1.):
    """
    Flow Matching 损失函数。

    本函数实现 Rectified / Conditional Flow Matching 的完整训练步骤：
        1. 在 (0, 1) 区间内为每个样本随机采样一个时间步 t；
        2. 用 t 在 latent 与标准高斯噪声之间做线性插值，构造带噪样本 noisy；
        3. 以 p_uncond 的概率对条件 cond 进行置零（dropout），
           使模型同时学习"无条件分支"与"有条件分支"，
           便于推理阶段使用 classifier-free guidance；
        4. 调用 model 预测速度场；
        5. 构造真实速度场作为回归 target，计算 MSE 损失。

    参数说明：
        model     : 待训练的流匹配模型，调用形式为 model(noisy, t * 1000, cond)
                    （乘 1000 是为了让时间步落在常见扩散式位置编码设计的 [0, 1000] 范围内）
        latent    : 真实的潜空间样本，形状通常为 [B, C, F, H, W]
        cond      : 条件信息（如文本特征、类别标签等），形状由 model 自行约定
        sigma_min : 最小噪声尺度，避免 t -> 0 时噪声权重为 0 带来的数值不稳定
        p_uncond  : 条件 cond 被置零的概率，用于无条件训练
        t_mean    : 采样 t 时所用正态分布的均值
        t_std     : 采样 t 时所用正态分布的标准差

    返回：
        一个 dict，包含：
            'loss' : 标量损失，可直接用于反向传播
            'mse'  : 与 'loss' 同值的 MSE 值，仅用于日志记录
    """
    # ============ 1. 生成与 latent 同形状的标准高斯噪声 ============
    # 该噪声对应 flow matching 中的"端点"之一（即 t = 1 时的目标分布）
    noise = torch.randn_like(latent)

    # ============ 2. 为每个样本独立采样一个时间步标量 t ============
    # 流程：先从 N(t_mean, t_std^2) 中采样得到 u，再经过 sigmoid(u) 映射到 (0, 1)
    #   - sigmoid 保证 t 始终为正且不会越界
    #   - 当 t_mean > 0 时，t 的分布偏向 1 一侧，这也是 Rectified Flow 的常见采样策略
    t = (torch.randn(len(latent), device=latent.device) * t_std + t_mean).sigmoid()

    # ============ 3. 将 t 扩展为与 latent 可广播的形状 ============
    # 假设 latent 的形状为 [B, C, F, H, W]，则 t_grid 形状为 [B, 1, 1, 1, 1]
    # 借助广播机制，t_grid 可以与 latent / noise 在每个空间位置上一一对应
    t_grid = t[:, None, None, None, None]

    # ============ 4. 在 latent 与 noise 之间做线性插值，构造带噪样本 noisy ============
    # 公式：noisy = (1 - t_grid) * latent + (sigma_min + (1 - sigma_min) * t_grid) * noise
    # 含义：
    #   - 当 t -> 0 时，noisy ≈ latent + sigma_min * noise（接近干净样本）
    #   - 当 t -> 1 时，noisy ≈ latent + (1 - sigma_min) * noise（接近纯噪声）
    #   - 引入 sigma_min 是为了在 t = 0 时仍保留极小噪声，
    #     避免后端 ODE / SDE 求解时出现奇异点
    noisy = (1 - t_grid) * latent + (sigma_min + (1 - sigma_min) * t_grid) * noise

    # ============ 5. 构造"无条件训练"的 dropout 掩码 ============
    # 以 p_uncond 的概率随机选出需要将 cond 丢弃的样本
    # drop 形状为 [B, 1, 1]，便于与 cond 在最后两维上做元素级条件替换
    drop = torch.rand(len(latent), 1, 1, device=latent.device) < p_uncond

    # ============ 6. 对被选中的样本把 cond 置零 ============
    # torch.where(condition, x, y) 会按 condition 在元素级从 x 和 y 之间取值
    # 这样模型在训练阶段就会同时见到"有条件"和"无条件"两种输入，
    # 推理时可通过 classifier-free guidance 提升生成质量与可控性
    cond = torch.where(drop, torch.zeros_like(cond), cond)

    # ============ 7. 模型前向：预测速度场 ============
    # 将 t 乘以 1000 后送入 model，是为了映射到很多扩散模型常用的 [0, 1000] 时间步范围，
    # 使 model 内部的位置编码/时间嵌入分量与训练时的尺度一致。
    # 输出 prediction 是模型预测的"瞬时速度场" v_theta(noisy, t)
    prediction = model(noisy, t * 1000, cond)

    # ============ 8. 构造回归目标 target ============
    # 对 noisy 关于 t 求导，可得理论速度场：
    #     d(noisy)/dt = (1 - sigma_min) * noise - latent
    # 这个 target 与 model 的预测值维度完全一致，可直接用 MSE 回归。
    # 直观含义：从当前带噪状态 noisy 沿着"时间增大的方向"应当如何移动才能到达纯噪声端点。
    target = (1 - sigma_min) * noise - latent

    # ============ 9. 计算 MSE 损失 ============
    # prediction.float() 是为了在混合精度训练中把预测值转成 fp32，
    # 避免低精度下损失可能为 NaN/0 等数值问题。
    loss = F.mse_loss(prediction.float(), target)

    # ============ 10. 返回损失字典 ============
    # 'loss' 用于反向传播，'mse' 仅作为日志指标使用，方便训练时监控曲线
    return {'loss': loss, 'mse': loss}
