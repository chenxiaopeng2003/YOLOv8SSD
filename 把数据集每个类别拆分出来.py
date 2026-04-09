import os
import shutil
from tqdm import tqdm

# =========================
# 路径配置
# =========================
root = r"L:\dasktop\A_Graduation_Project\数据及预训练模型\GC10-DET"
images_root = os.path.join(root, "images")
labels_root = os.path.join(root, "labels")

output_root = os.path.join(root, "每个类别图片-用于测试app")

# 创建输出目录
os.makedirs(output_root, exist_ok=True)

# =========================
# 收集所有文件（不区分train/val/test）
# =========================
image_paths = []
label_paths = []

for split in ["train", "val", "test"]:
    img_dir = os.path.join(images_root, split)
    lab_dir = os.path.join(labels_root, split)

    for file in os.listdir(img_dir):
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(img_dir, file))

    for file in os.listdir(lab_dir):
        if file.endswith(".txt"):
            label_paths.append(os.path.join(lab_dir, file))

# 建立 label_name -> path 映射
label_map = {os.path.basename(p).replace(".txt", ""): p for p in label_paths}

# =========================
# 开始分类
# =========================
for img_path in tqdm(image_paths):
    img_name = os.path.basename(img_path)
    name = os.path.splitext(img_name)[0]

    # 找对应label
    if name not in label_map:
        continue

    label_path = label_map[name]

    # 读取类别（默认取第一个目标）
    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        continue

    first_class = int(lines[0].split()[0])

    # 创建类别文件夹
    class_dir = os.path.join(output_root, f"class_{first_class}")
    img_out = os.path.join(class_dir, "images")
    lab_out = os.path.join(class_dir, "labels")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)

    # 复制文件
    shutil.copy(img_path, os.path.join(img_out, img_name))
    shutil.copy(label_path, os.path.join(lab_out, name + ".txt"))

print("✅ 数据集拆分完成！")