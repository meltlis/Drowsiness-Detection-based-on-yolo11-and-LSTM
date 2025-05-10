from ultralytics import YOLO
if __name__ == "__main__":
    # 加载模型
    model = YOLO("yolo11n.pt")

    # 训练模型
    results = model.train(data="data/data.yaml", epochs=20, imgsz=640, batch=24, workers=0)  # train the model on custom data for 3 epochs

    # 评估模型
    results = model.val()  

    # 预测
    results = model.predict(source="data/test/images", show=True)  
    
    # 保存模型
    model.save("yolo11n_saved.pt")