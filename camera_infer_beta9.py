import os
import sys
from collections import deque

import numpy as np
import torch
import torchvision.transforms as transforms


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# 讓 `python trainingmodel/camera_infer_beta9.py` 也能 import `trainingmodel.*`
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ====== 一般使用只要改這裡（不用參數） ======
WEIGHTS = "beta9_at_1015.pth"
CAMERA_INDEX = 0
DEVICE = "auto"  # auto/cpu/cuda
WHEEL_SOURCE = "pred"  # pred 或 zeros
MAX_CAMERA_INDEX_TO_TRY = 3  # CAMERA_INDEX 開不起來時，往後自動嘗試到這個 index
# ==========================================


def _resolve_weights_path(weights: str) -> str:
    if os.path.isabs(weights) and os.path.exists(weights):
        return weights

    candidate_paths = [
        os.path.join(REPO_ROOT, weights),
        os.path.join(THIS_DIR, weights),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path

    return weights  # 交給 torch.load 報錯


class DynamicTopCrop:
    """把訓練時的 top crop（以 480 高度為基準）套到任意解析度的鏡頭影像。"""

    def __init__(self, top_pixels: int, original_height: int = 480):
        self.top_pixels = int(top_pixels)
        self.original_height = int(original_height)

    def __call__(self, img):
        # img: PIL.Image
        w, h = img.size
        if h <= 1:
            return img

        scaled_top = int(round(self.top_pixels * (h / float(self.original_height))))
        scaled_top = max(0, min(scaled_top, h - 1))
        return img.crop((0, scaled_top, w, h))


def build_infer_transform(img_height: int, img_width: int, crop_top_pixels: int):
    return transforms.Compose(
        [
            DynamicTopCrop(crop_top_pixels),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _open_camera(preferred_index: int):
    import cv2

    def try_open(idx: int):
        # Windows 上 CAP_DSHOW 常比較穩（但不是必要）
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        return None

    cap = try_open(preferred_index)
    if cap is not None:
        return cap, preferred_index

    for idx in range(preferred_index + 1, MAX_CAMERA_INDEX_TO_TRY + 1):
        cap = try_open(idx)
        if cap is not None:
            return cap, idx

    return None, preferred_index


def main():

    # 延後 import：避免沒裝 opencv 時一開始就炸
    try:
        import cv2
        from PIL import Image
    except Exception as e:
        raise SystemExit(
            "缺少推論所需套件。請先安裝 opencv-python 與 pillow。\n"
            "例如：pip install opencv-python pillow\n"
            f"原始錯誤：{e}"
        )

    from trainingmodel.transformer_model_beta9 import (
        CROP_TOP_PIXELS,
        D_MODEL,
        DROPOUT,
        IMG_HEIGHT,
        IMG_WIDTH,
        N_HEAD,
        N_LAYERS,
        SEQUENCE_LENGTH,
        LaneCenteringController,
    )

    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    weights_path = _resolve_weights_path(WEIGHTS)

    model = LaneCenteringController(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=N_LAYERS,
        dropout=DROPOUT,
        seq_len=SEQUENCE_LENGTH,
        backbone_weights=None,  # 避免離線環境下載 torchvision 預訓練權重
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = build_infer_transform(IMG_HEIGHT, IMG_WIDTH, CROP_TOP_PIXELS)

    cap, opened_index = _open_camera(CAMERA_INDEX)
    if cap is None:
        raise SystemExit(
            f"無法開啟攝影機 index={CAMERA_INDEX}"
            + ("" if CAMERA_INDEX == MAX_CAMERA_INDEX_TO_TRY else f" (也已嘗試到 {MAX_CAMERA_INDEX_TO_TRY})")
            + "\n請修改本檔案上方的 CAMERA_INDEX，或確認相機權限/是否被其他程式佔用。"
        )

    frames = deque(maxlen=SEQUENCE_LENGTH)
    wheels = deque(maxlen=SEQUENCE_LENGTH)
    last_pred = np.array([0.0, 0.0], dtype=np.float32)

    print(
        f"開始推論：device={device} | weights={weights_path} | camera={opened_index} | seq_len={SEQUENCE_LENGTH}"
    )

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("讀取影像失敗，停止。")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            frames.append(pil_img)

            if WHEEL_SOURCE == "zeros":
                wheels.append([0.0, 0.0])
            else:
                # 以「上一幀預測」作為當前 wheel_speeds 的近似輸入
                wheels.append([float(last_pred[0]), float(last_pred[1])])

            if len(frames) < SEQUENCE_LENGTH or len(wheels) < SEQUENCE_LENGTH:
                continue

            seq_tensors = [transform(img.convert("RGB")) for img in list(frames)]
            input_tensor = torch.stack(seq_tensors).unsqueeze(0).to(device)
            wheel_tensor = torch.tensor(list(wheels), dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(input_tensor, wheel_tensor).detach().cpu().numpy().flatten()

            last_pred = pred.astype(np.float32)
            print(f"pred speed -> lwheel={pred[0]:.3f}, rwheel={pred[1]:.3f}")

    except KeyboardInterrupt:
        print("\n已中止。")
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
