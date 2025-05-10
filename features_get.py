import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def extract_features(video_list_file, features_file, labels_file, model, num_classes=9, seq_len=30):
    features_list = []
    labels_list = []
    with open(video_list_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"Extracting {video_list_file}"):
            video_path, label = line.strip().rsplit(' ', 1)
            label = int(label)
            results = model.predict(source=video_path, stream=True, verbose=False)
            frame_features = []
            for result in results:
                onehot = np.zeros(num_classes, dtype=np.float32)
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if 0 <= cls < num_classes:
                        onehot[cls] = 1.0
                frame_features.append(onehot)
            if len(frame_features) < seq_len:
                pad = [np.zeros(num_classes, dtype=np.float32)] * (seq_len - len(frame_features))
                frame_features.extend(pad)
            else:
                frame_features = frame_features[:seq_len]
            features_list.append(frame_features)
            labels_list.append(label)
    features = np.array(features_list)
    labels = np.array(labels_list)
    np.save(features_file, features)
    np.save(labels_file, labels)
    print(f"{video_list_file} 提取完成，样本数：{len(features_list)}，已保存为 {features_file}, {labels_file}")

if __name__ == "__main__":
    model = YOLO("runs/detect/train16/weights/best.pt")
    num_classes = 9
    seq_len = 30

    # 训练集
    extract_features(
        video_list_file="video_list_train.txt",
        features_file="features_train.npy",
        labels_file="labels_train.npy",
        model=model,
        num_classes=num_classes,
        seq_len=seq_len
    )

    # 验证集
    extract_features(
        video_list_file="video_list_valid.txt",
        features_file="features_valid.npy",
        labels_file="labels_valid.npy",
        model=model,
        num_classes=num_classes,
        seq_len=seq_len
    )