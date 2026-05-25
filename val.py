from ultralytics import YOLO


def evaluate_and_predict():
    # 1. 加载你训练好的最佳权重 (请替换为你实际的 best.pt 路径)
    # 通常在 runs/detect/train/weights/best.pt 或者你最新训练的文件夹下
    model_path = "runs\detect\BiFPN_exp\weights/best.pt"
    model = YOLO(model_path)

    # 2. 定量评估：获取各个类别的详细指标 (mAP, Precision, Recall)
    print("=" * 50)
    print("开始在测试集上进行定量评估...")
    print("=" * 50)

    # 假设你的配置文件是 GC10-DET.yaml
    # split='test' 表示强制模型使用 yaml 文件中配置的 test 路径。
    # 如果你的 yaml 里只有 train 和 val，请把 split 删掉或者改成 split='val'
    metrics = model.val(
        data="L:\dasktop\A_Graduation_Project/ultralytics-main/ultralytics\datasets\data.yaml",
        split="test",  # 指定测试集
        imgsz=640,  # 保持与训练时一致的图像尺寸
        batch=16,  # 这里的 batch 只是为了加快推理速度，不影响精度
        conf=0.25,  # 置信度阈值
        iou=0.6,  # NMS 的 IoU 阈值
    )

    # 打印每个类别的结果日志提示
    print(f"\n定量评估完成！各类别详细指标已保存至: {metrics.save_dir}")
    print("你可以去该目录下查看 PR_curve.png 和 confusion_matrix.png 等图表。")

    # 3. 定性可视化：对测试集图片进行预测并画框
    print("\n" + "=" * 50)
    print("开始对测试集图片进行预测并生成可视化结果...")
    print("=" * 50)

    # 请将这里的 source 替换为你测试集图片所在的真实文件夹路径
    test_images_path = (
        "L:\dasktop\A_Graduation_Project/ultralytics-main/ultralytics\datasets\yiwuyahenzenqiang\images/test"
    )

    results = model.predict(
        source=test_images_path,
        save=True,  # 自动保存画好预测框的图片
        save_txt=True,  # (可选) 保存预测的坐标 txt 文件，方便后续做统计
        save_conf=True,  # (可选) 在 txt 中保存置信度
        imgsz=640,
        conf=0.25,  # 低于 0.25 的预测框将不会被画出来
        line_width=2,  # 画框的线条粗细，对于小目标建议设置细一点(如 1 或 2)
    )

    # 获取可视化结果的保存路径
    save_dir = results[0].save_dir
    print(f"\n定性预测完成！所有画好框的测试集图片已保存至: {save_dir}")
    print("请打开该文件夹肉眼检查各类别的实际预测效果（尤其是漏检和误检）。")


if __name__ == "__main__":
    evaluate_and_predict()
