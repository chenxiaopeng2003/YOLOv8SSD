r"""
import cv2
import numpy as np
from ultralytics import YOLO.

# ======================
# 1. 参数配置
# ======================
model_path = "runs/detect/cbam_exp/weights/best.pt"
img_path = "L:\dasktop\A_Graduation_Project\数据及预训练模型\GC10-DET\images/train\img_01_425391600_00018.jpg"

tile_size = 640      # 每个切块大小（建议=训练尺寸）
overlap = 0.2        # 重叠比例（防止边缘漏检）
conf_thres = 0.25
iou_thres = 0.5

# ======================
# 2. 加载模型 & 图片
# ======================
model = YOLO(model_path)
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 步长
stride = int(tile_size * (1 - overlap))

# 存储所有检测框
all_boxes = []
all_scores = []
all_classes = []

# ======================
# 3. 滑窗检测
# ======================
for y in range(0, h, stride):
    for x in range(0, w, stride):

        tile = img[y:y+tile_size, x:x+tile_size]

        # 边界补齐（防止尺寸不够）
        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded

        # 推理
        results = model(tile, conf=conf_thres, iou=iou_thres, verbose=False)

        r = results[0]

        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        # ======================
        # 4. 坐标映射回原图
        # ======================
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box

            x1 += x
            y1 += y
            x2 += x
            y2 += y

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_classes.append(int(cls))

# ======================
# 5. NMS去重（关键！）
# ======================
indices = cv2.dnn.NMSBoxes(
    bboxes=all_boxes,
    scores=all_scores,
    score_threshold=conf_thres,
    nms_threshold=iou_thres
)

# ======================
# 6. 可视化
# ======================
for i in indices:
    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
    x1, y1, x2, y2 = map(int, all_boxes[i])
    score = all_scores[i]
    cls = all_classes[i]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{cls}:{score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

# ======================
# 7. 显示结果
# ======================
img_show = cv2.resize(img, (1280, 720))
cv2.imshow("Sliding Window Detection", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO("runs/detect/ghostconv_exp/weights/ghostconv.pt")

# 导出为 ONNX
# simplify=True 会自动调用 onnx-simplifer，这对 ncnn 转换非常重要
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
