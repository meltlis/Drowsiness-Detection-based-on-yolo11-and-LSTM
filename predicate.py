from ultralytics import YOLO
import numpy as np
import torch
import time
import cv2
import torch.nn.functional as F

class_names = [
    'eyes_closed', 'eyes_closed_head_left', 'eyes_closed_head_right',
    'focused', 'head_down', 'head_up', 'seeing_left', 'seeing_right', 'yarning'
]

# 0: eyes_closed, 1: eyes_closed_head_left, 2: eyes_closed_head_right, ...
fatigue_level_map = {
    0: "重度疲劳",  # eyes_closed
    1: "重度疲劳",  # eyes_closed_head_left
    2: "重度疲劳",  # eyes_closed_head_right
    3: "清醒",      # focused
    4: "轻度疲劳",  # head_down
    5: "轻度疲劳",  # head_up
    6: "轻度疲劳",  # seeing_left
    7: "轻度疲劳",  # seeing_right
    8: "重度疲劳"   # yarning
}

# 加载YOLO模型
model = YOLO("runs/detect/train16/weights/best.pt")
num_classes = 9
seq_len = 30

# 预测目标视频/图片序列
results = model.predict(source="data/test/sorted/1", stream=True, verbose=False)
frame_features = []
frames = []
for result in results:
    # 读取原始帧
    frame = result.orig_img.copy()
    frames.append(frame)
    onehot = np.zeros(num_classes, dtype=np.float32)
    for box in result.boxes:
        cls = int(box.cls[0])
        if 0 <= cls < num_classes:
            onehot[cls] = 1.0
        # 绘制检测框
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[cls]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    frame_features.append(onehot)
    # 实时展示
    cv2.imshow("YOLO Realtime", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

# 补齐/截断
if len(frame_features) < seq_len:
    pad = [np.zeros(num_classes, dtype=np.float32)] * (seq_len - len(frame_features))
    frame_features.extend(pad)
else:
    frame_features = frame_features[:seq_len]

features = np.array([frame_features])  # [1, seq_len, num_classes]
features = torch.tensor(features, dtype=torch.float32)

import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = num_classes
hidden_size = 64
num_layers = 1
num_classes = 9  # 你的LSTM输出类别数

lstm_model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
lstm_model.load_state_dict(torch.load("lstm_model.pth"))
lstm_model.eval()

with torch.no_grad():
    start_time = time.time()
    output = lstm_model(features)
    prob = F.softmax(output, dim=1)[0]
    pred = prob.argmax().item()
    end_time = time.time()
    print("各类别概率：")
    for i, p in enumerate(prob):
        print(f"{class_names[i]}: {p.item():.4f}")
    print("预测类别：", pred, class_names[pred])
    fatigue_score = (
        prob[0] + prob[1] + prob[2] + prob[8]  # 重度疲劳相关类别概率
        + 0.5 * (prob[4] + prob[5] + prob[6] + prob[7])  # 轻度疲劳相关类别概率
    )
    print("疲劳等级：", int(round(fatigue_score.item() * 10)))