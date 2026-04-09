
from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"L:\dasktop\A_Graduation_Project/ultralytics-main/ultralytics\datasets/data.yaml",
        epochs=120,
        imgsz=640,
        batch=16,
        workers=4   # 推荐显式设置
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
