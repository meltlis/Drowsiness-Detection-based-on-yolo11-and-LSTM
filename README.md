# Drowsiness-Detection-based-on-yolo11-and-LSTM


本项目基于 YOLO11 和 LSTM 实现驾驶员疲劳检测，支持视频/图片序列的目标检测、时序特征提取、疲劳等级分类与可视化展示。通过YOLO（You Only Look Once）模型对驾驶员的视频或图片序列进行目标检测，输出每一帧检测类别，并转化成one-hot特征，形成时许特征序列。利用LSTM对上述时序特征进行建模，学习驾驶员状态的时序变化，实现整段视频的疲劳预测。模型支持多类别识别，并可根据类别映射为疲劳等级（如清醒、轻度疲劳、重度疲劳），详见 `predicate.py` 中的 `fatigue_level_map`。

---

## 数据集说明

本项目测试时使用的数据集来自 [Roboflow - drowsiness detection](https://universe.roboflow.com/karthik-madhvan/drowsiness-detection-xsriz)。

- 提供者：Roboflow 用户
- License：Public Domain

本项目不包含数据集，数据集应该放置于data文件夹中，详情见 `data/README.dataset.txt`。

---

## 环境依赖
python == 3.13.2
对于带有CUDA的torch,建议按照torch官网推荐命令安装
```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```
剩余部分包
```bash
pip install -r requirments.txt
```

## 使用方法

### 1. YOLO训练
```bash
python yolo_train.py
```
可以使用yolo_test.py测试效果。
```bash
python yolo_test.py
```
### 2. 特征提取
```bash
python features_get.py
```
通过YOLO11训练的结果来提取特征。
### 3. LSTM训练
```bash
python data_sorted.py
python lstm_train.py
```
data_sorted.py根据图片标号来进行时序分类，依赖于训练数据集格式。
利用提取的特征训练LSTM模型。
### 4. 推理与可视化
```bash
python predicate.py
```
```bash
python predict_fatigue_sequence.py
```
利用模型进行推理。predicate.py用yolo提取了每一帧的特征, 然后用lstm或者transformer对整个序列进行了预测。
predict_fatigue_sequence.py则更进一步，不仅对整个序列进行了预测，还在每一帧用时序模型进行了预测。
最终预测结果在命令行中输出，并会输出对应的MP4视频。

## License

本项目代码遵循 MIT License。  
数据集 License 见 `data/README.dataset.txt`，为 Public Domain。
