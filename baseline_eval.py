import multiprocessing
from pathlib import Path

from ultralytics import YOLO


def main():

    # =========================
    # 1 模型路径
    # =========================
    model_path = r"runs/detect/train7/weights/best.pt"

    # =========================
    # 2 数据集配置
    # =========================
    data_yaml = r"L:\dasktop\A_Graduation_Project\ultralytics-main\ultralytics\datasets\data.yaml"

    # =========================
    # 3 加载模型
    # =========================
    model = YOLO(model_path)

    # =========================
    # 4 模型信息（参数量）
    # =========================
    print("\n===== MODEL INFO =====")
    model.info()

    # =========================
    # 5 运行验证
    # =========================
    metrics = model.val(
        data=data_yaml, imgsz=640, batch=16, workers=4, save_json=True, project="runs/detect", name="baseline"
    )

    # =========================
    # 6 打印关键指标
    # =========================
    print("\n===== BASELINE METRICS =====")

    print("Precision:", metrics.box.p)
    print("Recall:", metrics.box.r)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    # =========================
    # 7 保存指标到txt
    # =========================
    save_path = Path("runs/detect/baseline")
    result_file = save_path / "baseline_metrics.txt"

    with open(result_file, "w") as f:
        f.write("Baseline Results\n")
        f.write(f"Precision: {metrics.box.p}\n")
        f.write(f"Recall: {metrics.box.r}\n")
        f.write(f"mAP50: {metrics.box.map50}\n")
        f.write(f"mAP50-95: {metrics.box.map}\n")

    print("\nResults saved to:", result_file)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
