import os
import shutil
import glob

# 配置
image_dir = r"\data\valid\images"
label_dir = r"data\valid\labels"
output_base = r"data\valid\sorted"

os.makedirs(output_base, exist_ok=True)

# 遍历所有标签文件
label_files = glob.glob(os.path.join(label_dir, "*.txt"))

for label_file in label_files:
    # 取主类别（第一行第一个数字）
    with open(label_file, "r") as f:
        line = f.readline()
        if not line.strip():
            continue  # 跳过空文件
        class_id = line.strip().split()[0]
    # 源图片路径
    img_name = os.path.splitext(os.path.basename(label_file))[0] + ".jpg"
    img_path = os.path.join(image_dir, img_name)
    # 目标类别文件夹
    class_folder = os.path.join(output_base, class_id)
    os.makedirs(class_folder, exist_ok=True)
    # 拷贝图片
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(class_folder, img_name))

print("分类完成，图片已按类别放入不同文件夹。")