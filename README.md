# 基於 Vision Transformer 的 PyBullet 自動駕駛專案
本專案利用 PyBullet 物理模擬器打造一個自動駕駛環境，旨在透過模仿學習 (Imitation Learning) 的方式，訓練一個基於 Vision Transformer 的深度學習模型，使其能夠自主駕駛車輛。  

### 專案核心 
資料收集: 在 PyBullet 環境中，透過手動駕駛 (手動新賽道1.py) 收集包含連續影像幀與對應車輪速度的資料集。  
模型架構: 採用混合式神經網路架構：  
CNN 特徵提取器 (EfficientNet-B0): 從每張連續的影像中提取關鍵的空間特徵。  
Transformer Encoder: 分析影像特徵的時間序列，捕捉動態變化，最終預測出左、右輪的目標速度。  
模型訓練: 使用收集到的資料對模型進行端到端的訓練 (transformer_model_beta.py)。  
自動駕駛: 將訓練完成的模型部署回 PyBullet 環境中，實現車輛的自動駕駛 (auto_drive_for_beta6.py)。  
### 檔案結構  
`手動新賽道1.py`:  
用途: 資料收集腳本。  
功能: 啟動 PyBullet 環境，讓使用者手動控制車輛。按下 'r' 鍵可開始/停止記錄影像與駕駛數據。
`transformer_model_beta.py`:
用途: 模型訓練腳本。
功能: 讀取收集到的資料，設定資料增強、模型參數，並進行 Vision Transformer 模型的訓練、驗證與測試。訓練完成後會儲存最佳模型 (.pth 檔案)。
`auto_drive_for_beta.py`:
用途: 自動駕駛模擬腳本。
功能: 載入訓練好的模型權重，啟動 PyBullet 環境。按下 'a' 鍵可切換至 AI 自動駕駛模式，模型會根據即時影像預測並控制車輛。  
### 環境設置與安裝指南  
#### 步驟 1: 安裝 CUDA Toolkit  
為了利用 GPU 進行高效的模型訓練與推論，請先根據您的 NVIDIA 顯示卡型號，安裝對應版本的 CUDA Toolkit。  
官方網站: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  
#### 步驟 2: 安裝 PyTorch  
PyTorch 的版本必須與您安裝的 CUDA Toolkit 版本相匹配。請前往 PyTorch 官網，選擇適合您的作業系統、套件管理工具及 CUDA 版本的安裝指令。

官方網站: [PyTorch Get Started](https://pytorch.org/get-started/locally/)

#### 步驟 3: 安裝其他 Python 模組  
本專案還依賴於其他幾個 Python 函式庫。您可以使用 pip 一次性安裝所有必要的模組：  
`pip install pybullet pandas numpy Pillow tqdm tensorboard opencv-python scikit-learn`  
### 使用流程  
#### 階段一：資料收集  
執行資料收集腳本：  
`python 手動新賽道1.py`  
在 PyBullet 視窗中，使用方向鍵手動駕駛車輛。  
按下鍵盤上的 'r' 鍵開始記錄。此時，您的駕駛影像和車輪速度數據會被儲存。再次按下 'r' 鍵則暫停記錄。  
結束程式後，數據會被儲存在以日期和序號命名的資料夾中 (例如 2025_08_27/1/)，包含 recorded_images 資料夾和 log.csv 檔案。  
重複此步驟，收集足夠多樣化的駕駛數據。  
#### 階段二：模型訓練  
配置資料路徑: 打開 `transformer_model_beta7.py` 腳本。  
找到 TRAIN_FOLDERS, VAL_FOLDERS, TEST_FOLDERS 這三個列表。  
將您在階段一收集到的資料夾路徑，依照需求分別填入這三個列表中。  

### Python

範例:  
TRAIN_FOLDERS = [

    "2025_08_27/1",
	
    "2025_08_27/2",
	
    # ... 其他訓練資料夾
	
]

VAL_FOLDERS = [

    "2025_08_27/3",
	
    # ... 其他驗證資料夾
	
]

TEST_FOLDERS = [

    "2025_08_27/4",
	
    # ... 其他測試資料夾
	
]
開始訓練: 執行訓練腳本。  
`python transformer_model_beta.py`  
腳本會開始訓練模型，並在主控台輸出每個週期的損失 (Loss) 和平均絕對誤差 (MAE)。訓練完成後，表現最佳的模型將被儲存為 .pth 檔案 (例如 beta7_at_0827.pth)。

#### 階段三：自動駕駛  
配置模型路徑: 打開 `auto_drive_for_beta.py` 腳本。  

找到 MODEL_PATH 變數，並將其值修改為您在階段二訓練好的模型檔案名稱。  

Python

範例:  
MODEL_PATH = 'beta7_at_0827.pth'  
啟動自動駕駛: 執行模擬腳本。  
python auto_drive_for_beta6.py  
在 PyBullet 視窗中，按下 'a' 鍵來啟動或關閉 AI 自動駕駛模式。

觀察 AI 的駕駛表現。您隨時可以按下 'a' 鍵切回手動模式，並使用方向鍵接管車輛。
