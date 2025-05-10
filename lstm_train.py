import torch
import torch.nn as nn
import numpy as np

# 加载训练集
features = np.load('features_train.npy')
labels = np.load('labels_train.npy')

# 加载验证集
features_val = np.load('features_valid.npy')
labels_val = np.load('labels_valid.npy')

features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)
features_val = torch.tensor(features_val, dtype=torch.float32)
labels_val = torch.tensor(labels_val, dtype=torch.long)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

print("train features shape:", features.shape)
print("train labels shape:", labels.shape)
print("valid features shape:", features_val.shape)
print("valid labels shape:", labels_val.shape)
print("start training")

input_size = features.shape[2]
hidden_size = 64
num_layers = 1
num_classes = len(torch.unique(labels))

model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 200

for epoch in range(epochs):
    model.train()
    outputs = model(features)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = outputs.argmax(dim=1)
    acc = (pred == labels).float().mean()

    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(features_val)
        val_loss = criterion(val_outputs, labels_val)
        val_pred = val_outputs.argmax(dim=1)
        val_acc = (val_pred == labels_val).float().mean()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}, Val_Loss: {val_loss.item():.4f}, Val_Acc: {val_acc.item():.4f}")

torch.save(model.state_dict(), "lstm_model.pth")
print("LSTM模型已保存为 lstm_model.pth")