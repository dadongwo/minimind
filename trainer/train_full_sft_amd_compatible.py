"""
MiniMind SFT训练脚本 - AMD GPU兼容版本
支持 NVIDIA CUDA 和 AMD ROCm GPU 的统一训练
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径（仅在需要时）
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from utils.device_utils import device_manager, get_optimal_device, get_device_specific_config, get_device_object

warnings.filterwarnings('ignore')


def logger(content):
    """日志输出函数，支持分布式训练"""
    try:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except (RuntimeError, AttributeError):
        pass
    print(content)


def get_lr(it, max_iters, learning_rate):
    min_lr = learning_rate * 0.1
    if it < max_iters * 0.05:
        return learning_rate * (it / (max_iters * 0.05))
    elif it > max_iters * 0.95:
        decay_ratio = (it - max_iters * 0.95) / (max_iters * 0.05)
        return min_lr + (learning_rate - min_lr) * (1 - decay_ratio)
    else:
        return learning_rate


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            logits = res.logits
            loss = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
            
            # 添加MoE辅助损失（如果存在）
            if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                loss += res.aux_loss
            
            # 梯度累积
            loss = loss / args.accumulation_steps

        # 使用设备兼容的梯度缩放和正确的梯度处理
        if scaler is not None:
            scaler.scale(loss).backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                # 在梯度裁剪前先unscale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 显示累积前的损失值
            display_loss = loss.item() * args.accumulation_steps
            logger(
                f'Epoch:[{epoch}/{args.epochs}]({step}/{iter_per_epoch}) '
                f'loss:{display_loss:.3f} lr:{lr:.7f} '
                f'epoch_Time:{spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60:.0f}min:'
            )

            if (wandb is not None) and (not ddp or ddp_local_rank == 0):
                wandb.log({
                    "loss": display_loss,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch": epoch,
                    "step": step
                })

        if step % args.save_interval == 0:
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.out_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, ckp)
            logger(f'模型已保存: {ckp}')
            model.train()


def init_model(lm_config):
    """初始化模型和tokenizer"""
    # 定义tokenizer路径优先级
    tokenizer_paths = ['../model', 'model/', './model']
    tokenizer = None

    # 按优先级尝试加载tokenizer
    for path in tokenizer_paths:
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            logger(f"成功加载tokenizer: {path}")
            break
        except (OSError, ValueError, RuntimeError) as e:
            logger(f"加载tokenizer失败 {path}: {e}")
            continue

    # 如果所有路径都失败，使用GPT2作为后备
    if tokenizer is None:
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            logger("使用GPT2 tokenizer作为后备")
        except Exception as e:
            raise RuntimeError(f"无法加载任何tokenizer: {e}")

    # 初始化模型
    model = MiniMindForCausalLM(lm_config)
    
    # 加载预训练模型
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.out_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
    
    # 处理设备字符串
    if args.device == 'auto':
        device = device_manager.get_default_device()
        if device_manager.device_type == 'directml':
            map_location = 'cpu'  # DirectML需要先加载到CPU
        else:
            map_location = device
    else:
        map_location = args.device

    state_dict = torch.load(ckp, map_location=map_location)
    model.load_state_dict(state_dict, strict=False)

    # 确保所有参数都需要梯度
    for param in model.parameters():
        param.requires_grad = True

    logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # 处理设备转移
    if args.device == 'auto':
        if device_manager.device_type == 'directml':
            import torch_directml
            model = model.to(torch_directml.device())
        else:
            device = device_manager.get_default_device()
            model = model.to(device)
    else:
        model = model.to(args.device)

    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 使用设备管理器获取合适的后端
    backend = device_manager.get_distributed_backend()
    logger(f"使用分布式后端: {backend}")

    dist.init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    # 使用设备管理器设置设备
    DEVICE = device_manager.get_default_device(ddp_local_rank)
    device_manager.set_device(ddp_local_rank)

    logger(f"分布式训练初始化完成: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind SFT Training - AMD Compatible")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="auto", help="设备类型: auto, cuda:0, cpu 等")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    # 添加gradient clipping参数到参数解析器中
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_tiny.jsonl")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔步数")
    parser.add_argument("--log_interval", type=int, default=50, help="日志输出间隔步数")
    args = parser.parse_args()

    # 验证数据路径
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据文件不存在: {args.data_path}")

    # 打印设备信息和配置
    device_manager.print_device_info()
    device_config = get_device_specific_config()
    logger(f"设备配置: {device_config}")

    # 获取最优设备配置
    device_str = get_optimal_device(args.device)
    args.device = device_manager.device_from_string(device_str)
    logger(f"使用设备: {device_str}")
    
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    
    # 使用设备管理器判断设备类型
    device_type = device_manager.device_type if device_manager.is_gpu_available() else "cpu"
    logger(f"设备类型: {device_type}")

    args.wandb_run_name = f"MiniMind-SFT-{device_type.upper()}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 使用设备管理器获取AMP上下文
    ctx = device_manager.get_amp_context(enabled=(device_type != "cpu" and args.dtype in ['float16', 'bfloat16']))
    logger(f"混合精度训练: {'启用' if ctx != nullcontext() else '禁用'}")

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, device_manager.get_default_device(0)

    base_seed = 1337
    # 使用设备管理器设置随机种子
    device_manager.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = device_manager.device_from_string(DEVICE)
        rank = dist.get_rank()
        # 使用设备管理器设置分布式随机种子
        device_manager.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    model, tokenizer = init_model(lm_config)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 使用设备管理器获取梯度缩放器
    scaler = device_manager.get_grad_scaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    logger(f"梯度缩放器: {'启用' if scaler is not None else '禁用'}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    logger(f"开始SFT训练: {args.epochs} epochs, {iter_per_epoch} steps/epoch")
    
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb if args.use_wandb else None)

    # 训练结束后保存最终模型
    if not ddp or dist.get_rank() == 0:
        model.eval()
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.out_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, ckp)
        logger(f'SFT训练完成，最终模型已保存: {ckp}')
