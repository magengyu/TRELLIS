"""单进程 CPU/CUDA 训练脚本，支持 AMP（自动混合精度）、EMA（指数滑动平均）、
梯度累积、断点续训（resume）以及验证集评估。"""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..datasets import build_dataset
from .vae import vae_loss
from .flow_matching import flow_loss
from ..models import build_model, save_model
from ..utils.random import rng_state, restore_rng


def train(config, manifest, output, device='cuda', resume=None, init=None, val_manifest=None):
    """
    执行训练主流程。

    参数：
        config       : 训练配置字典，至少需要包含 'stage'（'vae' 或 'flow'）与 'models'；
                       其余字段会与下面的 defaults 合并，缺省值由 defaults 补齐。
        manifest     : 训练集清单文件路径（由 build_dataset 解析）。
        output       : 输出目录，用于保存 config.json、metrics.jsonl、latest.pt 以及权重。
        device       : 设备字符串，如 'cuda' 或 'cpu'。
        resume       : 断点续训用的 checkpoint 路径（latest.pt），会恢复模型/优化器/RNG 状态。
        init         : 预训练权重目录，用于“热启动”微调（与 resume 互斥）。
        val_manifest : 验证集清单路径，若为 None 则跳过验证。

    返回：
        output / 'latest.pt'，即最新 checkpoint 的路径。
    """
    # 深拷贝配置，避免训练过程中修改调用方传入的原始 dict
    cfg = copy.deepcopy(config)
    # 默认超参数，使用 dict 合并保证用户传入的值优先
    defaults = dict(resolution=64, batch_size=1, accumulation_steps=1, max_steps=100000,
                    lr=1e-4, weight_decay=0., precision='fp32', grad_clip=1., ema_decay=0.,
                    save_every=1000, log_every=10, seed=42, num_workers=0)
    cfg = {**defaults, **cfg}

    # ---------- 1. 配置合法性校验 ----------
    # stage 决定训练 VAE 还是 Flow Matching；precision 决定数值精度
    if cfg['stage'] not in ('vae', 'flow') or cfg['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('Invalid stage or precision')
    # 不同 stage 需要不同的模型角色：VAE 需要 encoder/decoder，flow 需要 denoiser
    expected_roles = {'encoder', 'decoder'} if cfg['stage'] == 'vae' else {'denoiser'}
    if set(cfg['models']) != expected_roles:
        raise ValueError(f'{cfg["stage"]} requires model roles {sorted(expected_roles)}')
    # 这类计数型参数必须 >= 1，否则训练循环无法推进或日志/保存会异常
    for key in ('batch_size', 'accumulation_steps', 'max_steps', 'save_every', 'log_every'):
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    # EMA 衰减系数必须在 [0, 1) 内，1 意味着完全不更新
    if not 0 <= cfg['ema_decay'] < 1:
        raise ValueError('ema_decay must be in [0,1)')
    # 学习率、梯度裁剪阈值必须为正，worker 数不能为负
    if cfg['lr'] <= 0 or cfg['grad_clip'] <= 0 or cfg['num_workers'] < 0:
        raise ValueError('Invalid learning rate, gradient clipping or worker count')
    # flow 阶段额外要求 p_uncond（无条件概率）与 sigma_min（最小噪声尺度）在合法区间
    if cfg['stage'] == 'flow' and not (0 <= cfg['p_uncond'] <= 1 and 0 <= cfg['sigma_min'] < 1):
        raise ValueError('Invalid unconditional probability or minimum noise')

    # ---------- 2. 设备与精度校验 ----------
    device = torch.device(device)
    # CPU 上不支持 fp16/bf16 的 autograd，因此强制要求 fp32
    if device.type != 'cuda' and cfg['precision'] != 'fp32':
        raise ValueError('Use precision=fp32 for CPU training')
    # 只有支持 bf16 的 GPU 才能用 bf16，否则退回 fp16/fp32
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use fp16 or fp32')

    # ---------- 3. 固定随机种子，保证可复现 ----------
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])

    # ---------- 4. 构建并初始化网络 ----------
    # 根据配置里的 models 规范（spec）实例化每个角色对应的网络，并放到目标设备
    # training=True 表示按训练模式构建（可能影响精度相关设置）
    networks = {name: build_model(spec, training=True).to(device).train()
                for name, spec in cfg['models'].items()}

    # resume 与 init 语义不同：resume 是完整恢复训练，init 只是加载权重做微调，二者互斥
    if resume and init:
        raise ValueError('Use either --resume or --init, not both')

    if init:
        # 从预训练目录加载权重（--init 微调流程）
        from ..models import load_model
        # 每个角色提供默认权重名，当 init 目录下找不到对应文件时作为回退
        names = {'encoder': 'ss_enc_conv3d_16l8_fp16', 'decoder': 'ss_dec_conv3d_16l8_fp16',
                 'denoiser': 'ss_flow_img_dit_L_16l8_fp16'}
        for name, model in networks.items():
            # 优先尝试 <init>/<role> 路径；若没有同名 .json 配置，则回退到默认权重名
            path = Path(init) / name
            if not path.with_suffix('.json').exists():
                path = Path(init) / names[name]
            pretrained = load_model(path, training=True)
            # strict=True 保证权重 key 完全匹配，避免静默错配
            model.load_state_dict(pretrained.state_dict(), strict=True)
            del pretrained  # 及时释放预训练模型内存

    # ---------- 5. 优化器、AMP scaler、EMA ----------
    # 收集所有网络的可训练参数，统一交给一个优化器管理
    parameters = [p for model in networks.values() for p in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # fp16 训练需要 GradScaler 做损失缩放；bf16 与 fp32 不需要
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    # EMA 保存每个网络的一份“影子权重”（detach + clone），不参与反向传播
    ema = {name: {k: v.detach().clone() for k, v in model.state_dict().items()}
           for name, model in networks.items()} if cfg['ema_decay'] else None

    # 训练进度：step=全局步数，epoch=当前 epoch，batch_offset=当前 epoch 内已消费的 batch 数
    step, epoch, batch_offset = 0, 0, 0

    # ---------- 6. 断点续训 ----------
    if resume:
        # 完整训练 checkpoint 中包含优化器状态以及 Python/NumPy 的 RNG 状态，
        # 只应恢复可信来源的 checkpoint；推理权重则用安全的 safetensors 格式。
        state = torch.load(resume, map_location='cpu', weights_only=False)
        # 恢复前先校验关键配置是否一致，避免“参数不匹配却静默继续”的问题
        for key in ('stage', 'models', 'batch_size', 'accumulation_steps', 'seed', 'precision',
                    'resolution', 'lr', 'weight_decay', 'ema_decay', 'sigma_min', 'p_uncond',
                    't_mean', 't_std', 'lambda_kl', 'loss_type'):
            if state['config'].get(key) != cfg.get(key):
                raise ValueError(f'Resume config mismatch: {key}; use --init for fine-tuning')
        # 恢复模型权重
        for name, model in networks.items():
            model.load_state_dict(state['models'][name], strict=True)
        # 恢复优化器与 GradScaler 状态
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        # 恢复 EMA 影子权重并搬到目标设备
        if ema is not None:
            ema = {name: {k: v.to(device) for k, v in weights.items()}
                   for name, weights in state['ema'].items()}
        # 恢复训练进度与 RNG 状态，保证续训后的随机性与原训练等价
        step, epoch, batch_offset = state['step'], state['epoch'], state['batch_offset']
        restore_rng(state['rng'])
        del state  # 释放 checkpoint 占用的内存

    # ---------- 7. 准备输出目录与数据集 ----------
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # 把最终生效的配置写入 config.json，便于复现与排查
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    dataset = build_dataset(manifest, cfg['stage'], cfg)
    # drop_last=True 会丢弃最后一个不完整 batch，因此要求数据量至少 >= batch_size
    if len(dataset) < cfg['batch_size']:
        raise ValueError('Dataset must contain at least batch_size records')
    # 验证集为可选项，只有在 val_manifest 不为 None 时才构建
    validation = build_dataset(val_manifest, cfg['stage'], cfg) if val_manifest else None
    # 把精度字符串映射为对应的 torch dtype，供 autocast 使用
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[cfg['precision']]

    def loader_for_epoch():
        """
        为当前 epoch 构造 DataLoader。
        注意：使用独立于全局 RNG 的 generator，使断点续训时的数据顺序
        只由 (seed + epoch) 决定，从而与模型 RNG 解耦。
        """
        dataset.epoch = epoch
        # 每个 epoch 用 seed+epoch 作种子，保证可复现且不同 epoch 打乱顺序不同
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
        return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                          num_workers=cfg['num_workers'], generator=generator,
                          pin_memory=device.type == 'cuda')

    # 初始化 iterator，并快进 batch_offset 个 batch（续训时跳过已消费的数据）
    iterator = iter(loader_for_epoch())
    for _ in range(batch_offset):
        next(iterator)

    def next_batch():
        """
        获取下一个 batch；当当前 epoch 迭代结束时自动开启下一个 epoch。
        返回时把所有张量搬到目标设备，non_blocking 配合 pin_memory 提升拷贝效率。
        """
        nonlocal iterator, epoch, batch_offset
        try:
            batch = next(iterator)
        except StopIteration:
            # 当前 epoch 结束：epoch+1、重置 batch_offset、重建 DataLoader
            epoch += 1; batch_offset = 0
            iterator = iter(loader_for_epoch()); batch = next(iterator)
        batch_offset += 1
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def losses(batch):
        """
        根据 stage 计算一个 batch 的损失字典。
        在 autocast 上下文中执行前向，实现 AMP；fp32 时 enabled=False 等价于无操作。
        """
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=cfg['precision'] != 'fp32'):
            if cfg['stage'] == 'vae':
                # VAE 阶段：用稀疏结构样本 'ss' 计算重建 + KL 损失
                return vae_loss(networks['encoder'], networks['decoder'], batch['ss'],
                                cfg.get('loss_type', 'dice'), cfg.get('lambda_kl', .001))
            # Flow 阶段：用 latent 与 cond 计算 flow matching 损失
            return flow_loss(networks['denoiser'], batch['latent'], batch['cond'],
                             **{k: cfg[k] for k in ('sigma_min', 'p_uncond', 't_mean', 't_std')})

    def save():
        """
        保存 checkpoint 与推理用权重：
            latest.pt     : 完整训练状态（模型/优化器/scaler/EMA/RNG/进度）
            weights/*     : 原始权重（供推理 CLI 直接使用）
            ema_weights/* : EMA 权重
        """
        state = dict(config=cfg, step=step, epoch=epoch, batch_offset=batch_offset,
                     models={name: model.state_dict() for name, model in networks.items()},
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), ema=ema, rng=rng_state())
        # 先写入临时文件再原子替换，避免保存过程中被中断产生损坏的 checkpoint
        temporary = output / 'latest.tmp'
        torch.save(state, temporary); temporary.replace(output / 'latest.pt')
        # 原始权重与 EMA 权重都可直接被推理 CLI 读取
        for name, model in networks.items():
            save_model(model, cfg['models'][name], output / 'weights' / name)
            if ema is not None:
                # 临时交换模型参数为 EMA 权重以导出，随后再恢复原始参数
                raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema[name])
                save_model(model, cfg['models'][name], output / 'ema_weights' / name)
                model.load_state_dict(raw)

    # ---------- 8. 训练主循环 ----------
    while step < cfg['max_steps']:
        # 每个优化步开始时清空梯度（set_to_none=True 更省内存/更快）
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        # 梯度累积：累加 accumulation_steps 个 micro-batch 的梯度后再更新一次参数
        for _ in range(cfg['accumulation_steps']):
            terms = losses(next_batch())
            # 出现非有限损失（NaN/Inf）立即报错，避免污染模型权重
            if not torch.isfinite(terms['loss']):
                raise FloatingPointError(f'Non-finite loss at step {step + 1}')
            # AMP 缩放后反向传播；除以累积步数实现梯度平均
            scaler.scale(terms['loss'] / cfg['accumulation_steps']).backward()
            # 对每个损失分量做累积平均，方便后续日志
            for key, value in terms.items():
                metrics[key] = metrics.get(key, 0.) + value.detach().item() / cfg['accumulation_steps']

        # 梯度裁剪（先 unscale 还原真实梯度，再裁剪）
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'])
        if not torch.isfinite(norm):
            # fp16 下 GradScaler 会自动跳过这一步；fp32 下直接报错
            if not scaler.is_enabled():
                raise FloatingPointError('Non-finite gradients; lower the learning rate or use fp32')
        # 记录缩放因子，用于判断本次 optimizer.step 是否真的执行了
        old_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        # 若缩放因子没有下降，说明本次 step 没有被跳过（GradScaler 检测到 inf 时会降低 scale 并跳过）
        updated = scaler.get_scale() >= old_scale
        # 只有真正更新了参数时才更新 EMA，避免 EMA 在跳过步上被错误地“平均”
        if ema is not None and updated:
            with torch.no_grad():
                for name, model in networks.items():
                    for key, value in model.state_dict().items():
                        if value.is_floating_point():
                            # 浮点参数：ema = ema * decay + value * (1 - decay)
                            ema[name][key].lerp_(value, 1 - cfg['ema_decay'])
                        else:
                            # 整型 buffer（如 num_batches_tracked）：直接复制
                            ema[name][key].copy_(value)
        step += 1

        # ---------- 9. 日志记录 ----------
        if step % cfg['log_every'] == 0:
            record = dict(step=step, epoch=epoch, optimizer_updated=updated, **metrics)
            print(json.dumps(record), flush=True)
            # 追加写入 metrics.jsonl，便于用曲线工具可视化
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')

        # ---------- 10. 定期保存 + 验证 ----------
        if step % cfg['save_every'] == 0 or step == cfg['max_steps']:
            if validation:
                # 验证会消耗 RNG（如 dropout/数据打乱），先保存 RNG 状态，结束后恢复
                saved_rng = rng_state()
                # 切换到 eval 模式（关闭 dropout、使用 BN 滑动统计等）
                for model in networks.values():model.eval()
                totals, count = {}, 0
                with torch.no_grad():
                    for batch in DataLoader(validation, batch_size=cfg['batch_size']):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        # 用 batch 中任意一个张量的第一维大小作为当前 batch 的样本数
                        size = len(next(iter(batch.values())))
                        # 按样本数加权累加各损失分量
                        for key, value in losses(batch).items():
                            totals[key] = totals.get(key, 0.) + value.item() * size
                        count += size
                # 求验证集上的加权平均损失
                record = dict(step=step, validation={k: v/count for k,v in totals.items()})
                print(json.dumps(record), flush=True)
                with (output / 'metrics.jsonl').open('a') as f:f.write(json.dumps(record) + '\n')
                # 恢复训练模式与 RNG 状态，保证验证不影响后续训练
                for model in networks.values():model.train()
                restore_rng(saved_rng)
            # 保存 checkpoint（latest.pt + weights/ + ema_weights/）
            save()
    return output / 'latest.pt'
