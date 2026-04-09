import multiprocessing

from ultralytics import YOLO


def main():
    # 1. 初始化模型（加载你的 GhostConv 配置文件）
    model = YOLO(r"ultralytics/cfg/models/v8/yolov8_ghostconv.yaml", task="detect")

    # 2. 开始训练
    # 训练完成后，results 变量会记录训练过程
    model.train(
        data=r"L:\dasktop\A_Graduation_Project\ultralytics-main\ultralytics\datasets\data.yaml",
        epochs=120,
        imgsz=640,
        batch=16,
        workers=4,
        project="runs/detect",
        name="ghostconv_exp",
    )

    # 3. 核心步骤：训练完成后立即进行“期末考试”（测试集评估）
    print("\n" + "=" * 30)
    print("训练已结束，正在启动测试集(Test Set)最终评估...")
    print("=" * 30 + "\n")

    # 注意：这里不需要重新加载模型，train() 结束后 model 对象会自动驻留最优权重
    # 如果你想确保万无一失，也可以显式加载：model = YOLO(r"runs/detect/ghostconv_exp/weights/best.pt")

    test_results = model.val(
        split="test",  # 强制使用 test 路径
        project="runs/detect",
        name="ghostconv_exp_TEST_RESULTS",  # 结果会单独存放在这个文件夹
        save_json=True,  # 建议开启，方便后续做数据分析
    )

    print("\n测试集评估完成！结果保存在: runs/detect/ghostconv_exp_TEST_RESULTS")
    print(f"测试集 mAP50: {test_results.results_dict['metrics/m_ap50']:.4f}")


if __name__ == "__main__":
    # Windows 环境下多进程训练必须加这一行
    multiprocessing.freeze_support()
    main()
