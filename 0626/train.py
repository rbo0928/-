from torch.utils.data import DataLoader
from dataset_cnn_lstm import LaneSeqDataset
from model_cnn_lstm import CNNLSTMModel
import torch
import torch.nn as nn

csv_path = "2025_06_25/1/log.csv"
img_dir = "2025_06_25/1/recorded_images"

dataset = LaneSeqDataset(csv_path, img_dir, seq_len=5, target="lane_offset")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = CNNLSTMModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
torch.save(model.state_dict(), "cnn_lstm_lane_offset.pth")
print("模型已儲存為 cnn_lstm_lane_offset.pth")