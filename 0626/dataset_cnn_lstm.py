from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms

class LaneSeqDataset(Dataset):
    def __init__(self, csv_path, img_dir, seq_len=5, target="lane_offset", transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])
        self.target = target

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        imgs = []
        for i in range(self.seq_len):
            img_name = self.data.iloc[idx + i]['img_path']
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            imgs.append(img)
        x_seq = torch.stack(imgs)  # shape: [seq_len, C, H, W]

        label = torch.tensor(self.data.iloc[idx + self.seq_len][self.target], dtype=torch.float32)
        return x_seq, label