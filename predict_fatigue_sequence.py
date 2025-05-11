import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from ultralytics import YOLO

# 配置
class_names = [
    'eyes_closed', 'eyes_closed_head_left', 'eyes_closed_head_right',
    'focused', 'head_down', 'head_up', 'seeing_left', 'seeing_right', 'yarning'
]
num_classes = len(class_names)
seq_len = 30
input_size = num_classes
hidden_size = 64
num_layers = 1
use_model = "lstm"  # "lstm" or "transformer"

# LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Transformer模型
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, nhead=3, hidden_dim=64, seq_len=30):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

# 加载序列模型
if use_model == "transformer":
    seq_model = SimpleTransformerClassifier(
        input_dim=input_size,
        num_classes=num_classes,
        num_layers=2,
        nhead=3,
        hidden_dim=64,
        seq_len=seq_len
    )
    seq_model.load_state_dict(torch.load("transformer_model.pth"))
else:
    seq_model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
    seq_model.load_state_dict(torch.load("lstm_model.pth"))
seq_model.eval()

# 加载YOLO模型
yolo_model = YOLO("runs/detect/train16/weights/best.pt")

# 视频源
source = r"data/test/sorted/0"
results = yolo_model.predict(source=source, stream=True, verbose=False)

frame_features = []
out_video = None
frames = []  # 在循环前加上

for result in results:
    frame = result.orig_img.copy()
    onehot = np.zeros(num_classes, dtype=np.float32)
    for box in result.boxes:
        cls = int(box.cls[0])
        if 0 <= cls < num_classes:
            onehot[cls] = 1.0
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{yolo_model.names[cls]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    frame_features.append(onehot)
    frames.append(frame)

    # 构造序列特征
    if len(frame_features) < seq_len:
        features = frame_features + [np.zeros(num_classes, dtype=np.float32)] * (seq_len - len(frame_features))
    else:
        features = frame_features[-seq_len:]
    features_tensor = torch.tensor([features], dtype=torch.float32)

    # LSTM/Transformer预测
    with torch.no_grad():
        output = seq_model(features_tensor)
        prob = F.softmax(output, dim=1)[0]
        pred = prob.argmax().item()
        fatigue_score = (
            prob[0] + prob[1] + prob[2] + prob[8]
            + 0.5 * (prob[4] + prob[5] + prob[6] + prob[7])
        )

    # 叠加预测结果
    text1 = f"Prediction: {class_names[pred]}"
    text2 = f"Fatigue Level: {int(round(fatigue_score.item() * 10))}"
    cv2.putText(frame, text1, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, text2, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 展示和保存
    cv2.imshow("YOLO Realtime", frame)
    if out_video is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('yolo_detect_result_with_pred.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    out_video.write(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
if out_video is not None:
    out_video.release()

# === 用整个序列做一次总体预测（和predicate.py一致） ===
if len(frame_features) < seq_len:
    pad = [np.zeros(num_classes, dtype=np.float32)] * (seq_len - len(frame_features))
    features = frame_features + pad
else:
    features = frame_features[:seq_len]
features_np = np.array([features], dtype=np.float32)  # [1, seq_len, num_classes]
features_tensor = torch.tensor(features_np, dtype=torch.float32)

with torch.no_grad():
    output = seq_model(features_tensor)
    prob = F.softmax(output, dim=1)[0]
    pred = prob.argmax().item()
    print("Class probabilities:")
    for i, p in enumerate(prob):
        print(f"{class_names[i]}: {p.item():.4f}")
    print("Prediction:", pred, class_names[pred])
    fatigue_score = (
        prob[0] + prob[1] + prob[2] + prob[8]
        + 0.5 * (prob[4] + prob[5] + prob[6] + prob[7])
    )
    print("Fatigue Level:", int(round(fatigue_score.item() * 10)))

# === 在最后一帧上叠加“全局”预测结果并保存 ===
if len(frames) > 0:
    last_frame = frames[-1].copy()
    text1 = f"Prediction: {class_names[pred]}"
    text2 = f"Fatigue Level: {int(round(fatigue_score.item() * 10))}"
    cv2.putText(last_frame, text1, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(last_frame, text2, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 生成新视频，最后一帧带预测结果
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('yolo_detect_result_with_pred.mp4', fourcc, 20.0, (last_frame.shape[1], last_frame.shape[0]))
    for i in range(len(frames)-1):
        video.write(frames[i])
    video.write(last_frame)
    video.release()