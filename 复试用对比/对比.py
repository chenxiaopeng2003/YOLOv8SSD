

from ultralytics import YOLO
def main():
    # 1️⃣ 加载训练好的模型
    model = YOLO("L:\dasktop\A_Graduation_Project/ultralytics-main/runs\detect/train7\weights/best.pt")

    # 2️⃣ 在验证集上评估
    metrics = model.val(
        data="L:\dasktop\A_Graduation_Project/ultralytics-main/ultralytics\datasets\data.yaml",   # 你的数据集配置文件
        split="val",        # 使用验证集
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        save_json=True,
        plots=True
    )

    # 3️⃣ 打印关键指标
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)

if __name__ == '__main__':
    main()