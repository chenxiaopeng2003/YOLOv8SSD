import os

import torch

from ultralytics import YOLO


def run_finetuning():
    # ================= 用户配置区域 =================
    # 1. 权重路径：确保这里是你上一轮训练最好的权重
    best_weights_path = "best.pt"

    # 2. 数据集路径
    dataset_yaml = (
        "L:\dasktop\A_Graduation_Project/ultralytics-main/ultralytics\datasets\data.yaml"  # 你的数据集yaml文件路径
    )

    # 3. 超参数文件
    hyp_yaml = "steel_hyp.yaml"

    # 4. 显存安全设置 (RTX 3050 4G 专用)
    # 建议从 8 开始尝试。如果报错 CUDA out of memory，请改为 4。
    # YOLOv8 会自动处理梯度累积，所以 Batch 小也能训练。
    BATCH_SIZE = 8
    # ==============================================

    # 简单检查权重文件是否存在
    if not os.path.exists(best_weights_path):
        print(f"错误: 找不到文件 {best_weights_path}")
        print("请将 best.pt 复制到当前目录，或修改脚本中的路径。")
        return

    # 显存检查提示
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)} (显存: {vram:.2f} GB)")
    else:
        print("警告: 未检测到 GPU，将在 CPU 上运行 (极慢)。")

    print(f"正在加载权重: {best_weights_path} ...")
    model = YOLO(best_weights_path)

    print("开始微调 (针对 4GB VRAM 优化策略)...")
    print(f"分辨率: 640 | Batch: {BATCH_SIZE} | 增强: FlipUD+CopyPaste")

    try:
        model.train(
            data=dataset_yaml,
            cfg=hyp_yaml,  # 加载定制的超参数
            # --- 针对 3050 Laptop 的核心调整 ---
            epochs=80,  # 微调轮次，配合早停
            patience=20,  # 20轮 loss 不下降则停止
            imgsz=640,  # 【重要】保持640，严禁上1024，否则必爆显存
            batch=BATCH_SIZE,  # 安全的批次大小
            workers=2,  # Laptop CPU 核心通常较少，设小一点防止卡顿
            # --- 优化策略 ---
            optimizer="SGD",  # 保持 SGD，对于微调更稳
            cos_lr=True,  # 【开启】使用余弦退火，让收敛曲线更平滑
            close_mosaic=10,  # 最后10轮关闭Mosaic，让模型适应真实分布
            project="runs/detect",
            name="steel_finetune_3050",
            exist_ok=True,
            save=True,
        )

        print("训练结束。正在进行 TTA (Test Time Augmentation) 验证...")
        # 验证时开启 TTA，通过推理时多次变换来弥补分辨率的不足
        metrics = model.val(augment=True)
        print(f"TTA 验证完成。mAP50: {metrics.box.map50:.3f}")

    except Exception as e:
        print("\n发生错误:")
        print(e)
        if "out of memory" in str(e):
            print("\n建议: 显存溢出。请打开 run_finetune.py 将 BATCH_SIZE 改为 4 再试。")


if __name__ == "__main__":
    run_finetuning()
