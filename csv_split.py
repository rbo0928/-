import pandas as pd
import os
import numpy as np

# --- 參數設定 ---
# 原始資料路徑
DATA_DIR = '2025_06_27/5/'
ORIGINAL_CSV_PATH = os.path.join(DATA_DIR, 'log.csv')

# 分割比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# 測試集比例會自動設為剩餘部分 (0.15)

# 輸出的檔案名稱
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_data.csv')
VAL_CSV_PATH = os.path.join(DATA_DIR, 'val_data.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test_data.csv')

def split_data_sequentially(csv_path):
    """
    讀取原始的CSV檔案，將其依時間順序分割為訓練、驗證與測試集，並儲存為新檔案。
    """
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到原始CSV檔案 '{csv_path}'。")
        return

    print(f"正在讀取資料來源: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 確保資料量足夠分割
    if len(df) < 20: # 至少要有足夠資料分給三個集合
        print("錯誤：資料量太少，無法進行有意義的分割。")
        return

    print(f"總資料筆數: {len(df)}")

    # --- 【修改點】改為依時間順序分割 ---
    # 計算分割點的索引
    train_end_index = int(len(df) * TRAIN_RATIO)
    val_end_index = train_end_index + int(len(df) * VAL_RATIO)

    # 使用 .iloc 進行切片，確保資料的連續性
    train_df = df.iloc[:train_end_index]
    val_df = df.iloc[train_end_index:val_end_index]
    test_df = df.iloc[val_end_index:]
    
    # 儲存檔案
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    val_df.to_csv(VAL_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    print("\n資料依時間順序分割完成！")
    print(f"訓練集大小: {len(train_df)} (資料前段) -> 已儲存至 {TRAIN_CSV_PATH}")
    print(f"驗證集大小: {len(val_df)} (資料中段) -> 已儲存至 {VAL_CSV_PATH}")
    print(f"測試集大小: {len(test_df)} (資料後段) -> 已儲存至 {TEST_CSV_PATH}")


if __name__ == '__main__':
    split_data_sequentially(ORIGINAL_CSV_PATH)
