import torch
from model_cnn_lstm import CNNLSTMModel
from dataset_cnn_lstm import LaneSeqDataset
from torchvision import transforms
import cv2
import os

# 建立模型並載入權重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTMModel().to(device)
model.load_state_dict(torch.load("cnn_lstm_lane_offset.pth", map_location=device))
model.eval()

# 準備推論資料（示範讀入連續5張圖）
img_dir = "2025_06_25/1/recorded_images"
img_names = sorted(os.listdir(img_dir))[:5]  # 取前5張

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])
imgs = []
for img_name in img_names:
    img = cv2.imread(os.path.join(img_dir, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img)
    imgs.append(img_tensor)

x = torch.stack(imgs).unsqueeze(0).to(device)  # → shape: [1, 5, 3, 64, 64]

with torch.no_grad():
    pred = model(x)
    print(f"預測車道偏移量：{pred.item():.4f} 像素")
