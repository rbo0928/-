"""
train_lane_center_transformer.py

Non-CLI version: configuration is provided by editing the variables in the CONFIG
block below. This version removes argparse and reads config directly from the
file so you don't need to pass CLI arguments.

You can edit the CONFIG block to set training data, validation data, and
training hyperparameters. Then simply run:

python train_lane_center_transformer.py

Directory layout (multi-folder mode):
  <data_dir>/log.csv
  <data_dir>/recorded_images/00000.png ...

Single CSV mode:
  -- set MODE = 'single' and provide CSV_PATH and IMAGES_DIR

CSV expected columns (header):
img_path,steering,throttle,lwheel,rwheel,speed_signed,timestamp
"""
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from tqdm import tqdm

import math
import logging
# logging.basicConfig(level=logging.INFO)

# ----------------------------- CONFIG ------------------------------------
# Edit these variables directly.
# MODE: either 'multi' to use TRAIN_DIRS / VAL_DIRS, or 'single' to use CSV_PATH
MODE = 'multi'  # 'multi' or 'single'

# For 'multi' mode: list dataset DIRS, each must contain log.csv and recorded_images/
TRAIN_DIRS = [
    '2025_08_07/2',  
    '2025_08_14/2',  
    '2025_08_14/3', 
    '2025_08_20/1',
    'new_data/混雜資料/2025_07_10/1',
    'new_data/混雜資料/2025_07_14/2',
    'new_data/混雜資料/2025_07_14/3',
    'new_data/混雜資料/2025_07_14/7',
    'new_data/混雜資料/2025_07_17/1',
    'new_data/混雜資料/2025_07_17/2',
    'new_data/RBO/2025_07_30/1',
    'new_data/RBO/2025_07_30/2',
    'new_data/RBO/2025_07_30/3',
    'new_data/RBO/2025_08_01/1',
    'new_data/RBO/2025_08_01/2',
    'new_data/混雜資料/2025_07_24/2',
    'new_data/混雜資料/2025_07_24/3',
    'new_data/混雜資料/2025_07_30/1',
    'new_data/混雜資料/2025_07_30/2',
]

# 驗證集資料夾（用於訓練過程中驗證模型性能）
VAL_DIRS = [
    '2025_08_07/3',  
    '2025_08_07/4',
    'new_data/RBO/2025_08_01/3',
    'new_data/RBO/2025_08_21/1',
    'new_data/RBO/2025_08_21/2',
    'new_data/混雜資料/2025_07_24/1',
]
# For 'single' mode: supply CSV_PATH and IMAGES_DIR
CSV_PATH = '/path/to/log.csv'
IMAGES_DIR = '/path/to/recorded_images'

# Output & training settings
OUTPUT_DIR = 'runs/exp'
SEQ_LEN = 8
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
VAL_SPLIT = 0.0  # fraction of train to hold as val if VAL_DIRS empty
SEED = 42
DEVICE = 'cuda'  # or 'cpu'
NO_AMP = True

# Dataset columns
TARGET_COLS = ("lwheel", "rwheel")
SENSOR_COLS = ("lwheel", "rwheel", "speed_signed")


# ----------------------------- Dataset -----------------------------------
class SequencedDrivingDataset(Dataset):
    """Dataset that returns sliding windows of frames and sensor values.

    Each item:
      imgs_seq: Tensor[T,3,H,W]
      sensors_seq: Tensor[T, sensor_dim]  # e.g. [lwheel, rwheel, speed_signed]
      target: Tensor[target_dim]  # e.g. [lwheel, rwheel] for last frame
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 8,
                 target_cols: List[str] = TARGET_COLS,
                 sensor_cols: List[str] = SENSOR_COLS,
                 transform=None, skip_incomplete: bool = True):
        # df must contain column 'img_abs' which is absolute path to each image file
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.target_cols = list(target_cols)
        self.sensor_cols = list(sensor_cols)
        self.transform = transform
        self.skip_incomplete = skip_incomplete

        # Precompute valid indices (we will produce windows ending at i)
        self.valid_indices = []
        N = len(self.df)
        for i in range(N):
            start = i - (seq_len - 1)
            if start < 0:
                if skip_incomplete:
                    continue
                else:
                    start = 0
            ok = True
            for j in range(start, i + 1):
                if not os.path.isfile(self.df.at[j, 'img_abs']):
                    ok = False
                    break
            if ok:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def _load_image(self, path: str):
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        start = i - (self.seq_len - 1)
        if start < 0:
            start = 0
        imgs = []
        sensors = []
        for j in range(start, i + 1):
            path = self.df.at[j, 'img_abs']
            img = self._load_image(path)
            imgs.append(img.unsqueeze(0))  # 1 x C x H x W
            sensor_row = self.df.loc[j, self.sensor_cols].values.astype(np.float32)
            sensors.append(sensor_row)

        # if we padded at start (when skip_incomplete=False), duplicate first frame to fill seq_len
        while len(imgs) < self.seq_len:
            imgs.insert(0, imgs[0].clone())
            sensors.insert(0, sensors[0])

        imgs_seq = torch.cat(imgs, dim=0)  # T x C x H x W
        sensors_seq = torch.from_numpy(np.array(sensors, dtype=np.float32))  # T x sensor_dim
        # 如果你想把 sensors_seq 變成 float32 tensor 明確一點：
        sensors_seq = sensors_seq.float()

        # target is for the last frame by default
        target_row = self.df.loc[i, self.target_cols].values.astype(np.float32)
        target = torch.tensor(target_row, dtype=torch.float32)

        return imgs_seq, sensors_seq, target


# ----------------------------- Utilities ---------------------------------
def load_dataset_dirs(dir_list: List[str]) -> pd.DataFrame:
    """Read multiple dataset DIRS and concat them into a single DataFrame.

    Each folder must contain:
      <folder>/log.csv
      <folder>/recorded_images/<img_path entries from csv>

    The returned DataFrame contains all columns from each CSV plus 'img_abs'
    which points to the absolute path of each image file.
    """
    frames = []
    for d in dir_list:
        dpath = Path(d)
        csv_path = dpath / 'log.csv'
        imgs_dir = dpath / 'recorded_images'
        if not csv_path.is_file():
            raise FileNotFoundError(f"Expected log.csv in dataset folder: {csv_path}")
        if not imgs_dir.is_dir():
            raise FileNotFoundError(f"Expected recorded_images folder: {imgs_dir}")
        df = pd.read_csv(csv_path)
        df = df.copy()
        df['img_abs'] = df['img_path'].apply(lambda p: str(imgs_dir / p))
        df['source_folder'] = str(dpath)
        frames.append(df)
    if len(frames) == 0:
        return pd.DataFrame()
    big = pd.concat(frames, axis=0, ignore_index=True)
    return big


def load_single_csv(csv_file: str, images_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = df.copy()
    imgs_dir = Path(images_dir)
    df['img_abs'] = df['img_path'].apply(lambda p: str(imgs_dir / p))
    df['source_folder'] = str(imgs_dir)
    return df


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------- Model -------------------------------------
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 512):
        super().__init__()
        if pretrained:
            res = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            res = models.resnet18(weights=None)
        modules = list(res.children())[:-1]  # remove fc
        self.encoder = nn.Sequential(*modules)
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B*T, 3, H, W)
        f = self.encoder(x)  # B*T x 512 x 1 x 1
        f = f.view(f.size(0), -1)
        return f


class LaneCenterTransformer(nn.Module):
    def __init__(self, feat_dim: int = 512, sensor_dim: int = 3, proj_dim: int = 512,
                 n_layers: int = 4, n_heads: int = 8, seq_len: int = 8, output_dim: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=True, out_dim=feat_dim)
        self.img_proj = nn.Linear(feat_dim, proj_dim)
        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_dim, proj_dim // 8),
            nn.ReLU(),
            nn.Linear(proj_dim // 8, proj_dim),
        )
        self.pos_emb = nn.Parameter(torch.randn(seq_len, proj_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=n_heads,
                                           dim_feedforward=proj_dim * 4, dropout=dropout,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, output_dim)
        )

        self.seq_len = seq_len
        self.proj_dim = proj_dim

    def forward(self, imgs_seq, sensors_seq):
        # imgs_seq: B x T x C x H x W
        # sensors_seq: B x T x sensor_dim
        B, T, C, H, W = imgs_seq.shape
        imgs = imgs_seq.view(B * T, C, H, W)
        feats = self.backbone(imgs)  # (B*T, feat_dim)
        feats = self.img_proj(feats)  # (B*T, proj_dim)
        feats = feats.view(B, T, -1)  # (B, T, proj_dim)

        sensor_emb = self.sensor_proj(sensors_seq)  # (B, T, proj_dim)

        tokens = feats + sensor_emb
        # add pos emb
        if self.pos_emb.shape[0] != T:
            pos = torch.nn.functional.interpolate(self.pos_emb.unsqueeze(0).permute(0,2,1), size=T, mode='linear', align_corners=False)
            pos = pos.permute(0,2,1).squeeze(0)
        else:
            pos = self.pos_emb
        tokens = tokens + pos.unsqueeze(0)
        out = self.transformer(tokens)  # (B, T, D)
        last = out[:, -1, :]  # B, D   -> 取最後 timestep 的向量
        y = self.head(last)
        return y


# ----------------------------- Training ----------------------------------
def train_one_epoch(model, optim, criterion, loader, device, scaler=None,
                    clip_max_norm=5.0, debug_nan=True):
    model.train()
    running_loss = 0.0
    n = 0
    for batch_idx, (imgs_seq, sensors_seq, target) in enumerate(tqdm(loader, desc='train', leave=False)):
        imgs_seq = imgs_seq.to(device)  # B x T x C x H x W
        sensors_seq = sensors_seq.to(device)
        target = target.to(device)

        optim.zero_grad()

        # forward + loss (AMP aware)
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(imgs_seq, sensors_seq)
                loss = criterion(pred, target)
        else:
            pred = model(imgs_seq, sensors_seq)
            loss = criterion(pred, target)

        # detect nan/inf in loss
        if not torch.isfinite(loss):
            logging.warning(f"Non-finite loss at batch {batch_idx}: {loss.item()}. Skipping batch.")
            # optionally continue without stepping optimizer
            continue

        # backward + gradient clipping + step
        if scaler is not None:
            scaler.scale(loss).backward()
            # unscale to inspect grads and clip
            scaler.unscale_(optim)
            # clip grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            # optional: compute grad norm for logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)
            # step
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            # grad norm logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)
            optim.step()

        running_loss += float(loss.item()) * imgs_seq.size(0)
        n += imgs_seq.size(0)

        # debug prints occasionally
        if batch_idx % 200 == 0:
            logging.info(f"[train] batch {batch_idx} loss={loss.item():.6f} grad_norm={total_norm:.4f}")

    return running_loss / max(1, n)


def evaluate(model, criterion, loader, device):
    model.eval()
    running_loss = 0.0
    n = 0
    with torch.no_grad():
        for imgs_seq, sensors_seq, target in tqdm(loader, desc='val', leave=False):
            imgs_seq = imgs_seq.to(device)
            sensors_seq = sensors_seq.to(device)
            target = target.to(device)
            pred = model(imgs_seq, sensors_seq)
            loss = criterion(pred, target)
            running_loss += float(loss.item()) * imgs_seq.size(0)
            n += imgs_seq.size(0)
    return running_loss / max(1, n)


# ----------------------------- Run (no CLI) -------------------------------
def main():
    seed_everything(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    if MODE == 'single':
        if not Path(CSV_PATH).is_file():
            raise FileNotFoundError(f"CSV path not found: {CSV_PATH}")
        if not Path(IMAGES_DIR).is_dir():
            raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
        print(f'Loading single CSV: {CSV_PATH} with images in {IMAGES_DIR}')
        train_df = load_single_csv(CSV_PATH, IMAGES_DIR)
    elif MODE == 'multi':
        if len(TRAIN_DIRS) == 0:
            raise ValueError('TRAIN_DIRS is empty while MODE=="multi". Edit the CONFIG block at top.')
        print('Loading training DIRS:', TRAIN_DIRS)
        train_df = load_dataset_dirs(TRAIN_DIRS)
        if len(VAL_DIRS) > 0:
            print('Loading validation DIRS:', VAL_DIRS)
            val_df = load_dataset_dirs(VAL_DIRS)
        elif VAL_SPLIT > 0.0:
            N = len(train_df)
            val_N = int(N * VAL_SPLIT)
            if val_N > 0:
                val_df = train_df.sample(n=val_N, random_state=SEED).reset_index(drop=True)
                train_df = train_df.drop(val_df.index).reset_index(drop=True)
    else:
        raise ValueError('Unknown MODE: set MODE to "single" or "multi" in the CONFIG block')

    dataset = SequencedDrivingDataset(train_df, seq_len=SEQ_LEN, transform=transform)
    val_dataset = None
    if len(val_df) > 0:
        val_dataset = SequencedDrivingDataset(val_df, seq_len=SEQ_LEN, transform=transform)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

    sensor_dim = len(SENSOR_COLS)
    output_dim = len(TARGET_COLS)
    model = LaneCenterTransformer(feat_dim=512, sensor_dim=sensor_dim, proj_dim=512,
                                  n_layers=4, n_heads=8, seq_len=SEQ_LEN, output_dim=output_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()

    scaler = None
    if (not NO_AMP) and device.type == 'cuda':
        scaler = torch.amp.GradScaler(device='cuda')

    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device, scaler)
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, criterion, val_loader, device)

        print(f"Epoch {epoch:03d}  Train Loss: {train_loss:.6f}  ", end='')
        if val_loss is not None:
            print(f"Val Loss: {val_loss:.6f}")
        else:
            print('Val Loss: N/A')

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, os.path.join(OUTPUT_DIR, f'ckpt_epoch_{epoch:03d}.pth'))

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'best.pth'))

    print('Training finished. Best val loss:', best_val)


if __name__ == '__main__':
    main()
