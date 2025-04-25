"""
Train a diffusion model on images.
"""
import argparse
import os,sys

# 获取当前脚本所在目录（scripts/）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录（上层目录）
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import torch
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop



def main():
    args = create_argparser().parse_args()

    # 设置日志目录，默认为当前目录下的 log
    if not args.log_dir:
        args.log_dir = os.environ.get("OPENAI_LOGDIR", "./log")
    logger.configure(dir=args.log_dir)

    # 检测 GPU 数量并设置设备
    args.num_gpus = torch.cuda.device_count()
    args.use_multi_gpu = args.num_gpus > 1 and args.multi_gpu
    if args.use_multi_gpu:
        # 多 GPU 模式，初始化分布式训练
        dist_util.setup_dist(args.local_rank)
        logger.log(f"Running on {args.num_gpus} GPUs, rank {dist.get_rank()}")
    else:
        # 单 GPU 模式
        logger.log("Running on single GPU or CPU")
        dist_util.setup_single()

    # 创建模型和扩散过程
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 将模型移到适当设备
    if args.use_multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev()
        )
    else:
        model.to(dist_util.dev())

    # 创建调度采样器
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # 创建数据加载器
    logger.log("creating data loader...")
    data = load_data(
        data_dir_sar=args.data_dir_sar,
        data_dir_opt=args.data_dir_opt,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    # 开始训练
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()



def create_argparser():
    defaults = dict(
        # 数据集路径
        data_dir_sar="./dataset/sar_images/train_png",  # SAR 图像数据集目录路径，默认指向训练集的 PNG 格式图像
        data_dir_opt="./dataset/opt_images/train_png",  # 光学（OPT）图像数据集目录路径，默认指向训练集的 PNG 格式图像

        # 训练调度参数
        schedule_sampler="uniform",  # 调度采样器类型，决定如何选择扩散时间步，"uniform" 表示均匀采样
        lr=1e-4,  # 学习率，控制优化器更新参数的步长，默认 1e-4
        weight_decay=0.0,  # 权重衰减（L2 正则化系数），用于防止过拟合，默认 0.0 表示不使用
        lr_anneal_steps=0,  # 学习率退火步数，0 表示不进行退火（学习率保持不变）

        # 批处理参数
        batch_size=2,  # 每个训练批次的样本数量，默认 2，影响显存占用和训练稳定性
        microbatch=-1,  # 微批次大小，-1 表示禁用微批次（使用完整批次），用于显存不足时分块处理

        # 模型参数
        ema_rate="0.9999",  # 指数移动平均（EMA）的衰减率，用于平滑模型参数，默认 0.9999
        log_interval=10,  # 日志记录间隔（步数），每 10 步记录一次训练指标（如损失）
        save_interval=10000,  # 模型保存间隔（步数），每 10000 步保存一次模型检查点
        resume_checkpoint="",  # 恢复训练的检查点路径，空字符串表示从头训练
        use_fp16=False,  # 是否使用混合精度训练，False 表示使用 FP32，启用可减少显存占用
        fp16_scale_growth=1e-3,  # 混合精度训练的损失缩放增长率，用于防止 FP16 数值溢出

        # 日志和分布式训练参数
        log_dir="./logs",  # 日志和检查点保存目录，默认保存到 ./logs
        multi_gpu=False,  # 是否启用多 GPU 训练，False 表示单 GPU 或 CPU 训练
        local_rank=0,  # 分布式训练的本地 rank，用于多 GPU 训练时区分进程

        # 模型和扩散过程参数（从 train_ori.sh 迁移）
        image_size=256,  # 输入图像的分辨率（宽高均为 256），影响显存占用和生成质量
        num_channels=128,  # U-Net 模型的基础通道数，控制模型容量，默认 128
        num_res_blocks=3,  # 每个分辨率级别的残差块数量，控制模型深度，默认 3
        learn_sigma=False,  # 是否学习噪声方差，False 表示使用固定方差（简化扩散过程）
        diffusion_steps=2000,  # 扩散过程的总步数，控制去噪过程的长度
        noise_schedule="linear",  # 噪声调度类型，"linear" 表示线性噪声スケジュール
    )
    defaults.update(model_and_diffusion_defaults())  # 更新默认参数，添加模型和扩散的额外配置
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    add_dict_to_argparser(parser, defaults)  # 将默认参数添加到解析器
    return parser  # 返回配置好的参数解析器

if __name__ == "__main__":
    main()