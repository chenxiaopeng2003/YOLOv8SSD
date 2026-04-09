import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import re
import time
import datetime
from PIL import Image, ImageDraw, ImageFont

# ============================ 核心配置 ============================
MODEL_PATH = r'L:\dasktop\A_Graduation_Project\ultralytics-main\runs\detect\cbam_exp\weights\best.pt'
IMAGE_PATH = r"L:\dasktop\A_Graduation_Project\数据及预训练模型\每个类别图片-用于测试app\压痕\images\img_02_4402329100_00006.jpg"
GT_LABEL_PATH = r"L:\dasktop\A_Graduation_Project\数据及预训练模型\每个类别图片-用于测试app\压痕\labels\img_02_4402329100_00006.txt"

# 完整中英对照表
CLASS_MAP = {
    "chongkong": "冲孔", "hanfeng": "焊缝", "yueyawan": "月牙弯",
    "shuiban": "水斑", "youban": "油污", "siban": "丝斑",
    "yiwu": "异物", "yahen": "压痕", "zhehen": "折痕", "yaozhe": "腰折"
}

# 微软雅黑路径 (若报错请确认此文件存在)
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"


# ==================================================================

def draw_text_with_bg(img, text, x, y, color, font_size=30):
    """在图片上渲染带背景的中文"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()

    y = max(y, 10)
    text_bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


def load_gt_boxes_exact(gt_path, img_w, img_h, model_names):
    """精准解析包含干扰字符的 YOLO 标签文件"""
    gt_boxes = []
    if not os.path.exists(gt_path):
        return gt_boxes

    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            clean_line = re.sub(r'\[.*?\]', '', line).strip()
            parts = clean_line.split()

            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)

                label_en = model_names[cls_id] if cls_id in model_names else f"ID_{cls_id}"
                gt_boxes.append({
                    'label_en': label_en,
                    'label_cn': CLASS_MAP.get(label_en, label_en),
                    'box': [x1, y1, x2, y2]
                })
    return gt_boxes


def resize_image_for_display(img, target_h=700):
    """物理缩小图片像素，保证100%完整显示且不裁切"""
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def print_detection_report(results, img_w, img_h, inf_time_ms):
    """生成并打印详细的缺陷检测数据对比报告"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    boxes = results.boxes
    num_defects = len(boxes)
    names = results.names

    total_conf = 0
    max_conf = 0
    min_conf = 1.0 if num_defects > 0 else 0
    defect_counts = {}
    severity_counts = {"严重": 0, "中等": 0, "轻微": 0}

    report_str = f"""============================================================
钢材缺陷检测数据对比报告
============================================================

📋 检测基本信息
----------------------------------------
检测时间: {now_str}
使用模型: cbam
图片尺寸: {img_w} × {img_h} 像素
检测数量: {num_defects} 个缺陷
推理时间: {inf_time_ms:.2f} 毫秒

🔍 缺陷检测详情
----------------------------------------
"""

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        label_en = names[cls_id]
        label_cn = CLASS_MAP.get(label_en, label_en)
        conf = float(box.conf[0])

        # 边界框浮点坐标与整数坐标
        xyxy_f = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy_f)
        w, h = x2 - x1, y2 - y1
        area = w * h
        cx, cy = x1 + w // 2, y1 + h // 2

        # 统计数据累加
        total_conf += conf
        max_conf = max(max_conf, conf)
        min_conf = min(min_conf, conf)

        type_key = f"{label_cn} ({label_en})"
        defect_counts[type_key] = defect_counts.get(type_key, 0) + 1

        # 严重程度判定逻辑
        if conf >= 0.80:
            severity = "严重"
        elif conf >= 0.65:
            severity = "中等"
        else:
            severity = "轻微"

        severity_counts[severity] += 1
        advice = f"{label_cn}缺陷，通用模型检测"

        report_str += f"""缺陷 #{i + 1}:
  • 类型: {label_en} ({label_cn})
  • 置信度: {conf:.2f} ({int(conf * 100)}%)
  • 边界框: [{xyxy_f[0]:.4f}, {xyxy_f[1]:.4f}, {xyxy_f[2]:.4f}, {xyxy_f[3]:.4f}]
  • 像素坐标: ({x1}, {y1}) - ({x2}, {y2})
  • 尺寸: {w} × {h} 像素
  • 面积: {area} 像素
  • 中心点: ({cx}, {cy})
  • 严重程度: {severity}
  • 处理建议: {advice}

"""

    # 统计信息计算
    avg_conf = total_conf / num_defects if num_defects > 0 else 0

    report_str += f"""📊 检测统计信息
----------------------------------------
平均置信度: {avg_conf:.2f}
最高置信度: {max_conf:.2f}
最低置信度: {min_conf:.2f}

缺陷类型分布:
"""
    for k, v in defect_counts.items():
        report_str += f"  • {k}: {v} 个\n"

    report_str += "\n严重程度分布:\n"
    for k, v in severity_counts.items():
        if v > 0:
            report_str += f"  • {k}: {v} 个\n"

    report_str += """
💡 数据格式说明
----------------------------------------
• 边界框格式: [x1, y1, x2, y2] (绝对浮点坐标)
• 像素坐标: 基于原始图片尺寸
• 置信度: 0.0-1.0，越高表示检测越可靠
• 可与电脑端YOLO/PyTorch输出直接对比

============================================================
报告结束
============================================================"""

    print(report_str)


def run_comparison():
    # 1. 加载模型与推理计时
    model = YOLO(MODEL_PATH)
    t_start = time.perf_counter()
    results = model.predict(source=IMAGE_PATH, conf=0.2, verbose=False)[0]
    inf_time_ms = (time.perf_counter() - t_start) * 1000

    # 2. 读取原始图片
    img_orig = cv2.imread(IMAGE_PATH)
    if img_orig is None:
        print(f"❌ 读取图片失败: {IMAGE_PATH}")
        return
    h, w = img_orig.shape[:2]
    img_pred = img_orig.copy()
    img_gt = img_orig.copy()

    # 3. 绘制 AI 预测 (绿色)
    for box in results.boxes:
        xy = box.xyxy[0].cpu().numpy().astype(int)
        name = CLASS_MAP.get(model.names[int(box.cls[0])], model.names[int(box.cls[0])])
        cv2.rectangle(img_pred, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 0), 4)
        img_pred = draw_text_with_bg(img_pred, f"AI预测: {name}", xy[0], xy[1] - 45, (0, 180, 0))

    # 4. 绘制人工标注 (红色)
    gt_list = load_gt_boxes_exact(GT_LABEL_PATH, w, h, model.names)
    if not gt_list:
        img_gt = draw_text_with_bg(img_gt, "未解析到标签数据", 50, 50, (0, 0, 255), 40)
    else:
        for obj in gt_list:
            b = obj['box']
            cv2.rectangle(img_gt, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
            img_gt = draw_text_with_bg(img_gt, f"人工标签: {obj['label_cn']}", b[0], b[1] - 45, (0, 0, 255))

    # 5. 生成并打印终端报告
    print_detection_report(results, w, h, inf_time_ms)

    # 6. 物理缩小图像 (高度固定为 700 像素，等比例缩放) 解决裁切问题
    img_pred_display = resize_image_for_display(img_pred, target_h=700)
    img_gt_display = resize_image_for_display(img_gt, target_h=700)

    # 7. 显示独立窗口
    cv2.imshow("1. AI Prediction (Fully Visible)", img_pred_display)
    cv2.imshow("2. Ground Truth (Fully Visible)", img_gt_display)
    cv2.moveWindow("2. Ground Truth (Fully Visible)", img_pred_display.shape[1] + 20, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_comparison()