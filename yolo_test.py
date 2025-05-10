#测试yolo模型效果
from ultralytics import YOLO

model = YOLO("runs/detect/train16/weights/best.pt")  # 训练好的权重路径
results = model.predict(source="data/test/sorted/0", show=True) # 设置需要预测的路径
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"类别: {cls}, 置信度: {conf:.2f}")
model.save("yolo11n_saved.pt")