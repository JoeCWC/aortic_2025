# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
from datetime import datetime
import time
import torch
import numpy as np

os.makedirs("./datasets/test", exist_ok=True)

# 測試資料根目錄
base_root = "./datasets/testing_images"

# 自動找到第一個「直屬子資料夾含 patient*」的目錄
patient_root = base_root
for dirpath, dirnames, _ in os.walk(base_root):
    if any(d.lower().startswith("patient") for d in dirnames):
        patient_root = dirpath
        break

# 收集所有圖片路徑（只看直屬的 patient 資料夾）
all_files = []
for patient_folder in os.listdir(patient_root):
    patient_path = os.path.join(patient_root, patient_folder)
    if os.path.isdir(patient_path) and patient_folder.lower().startswith("patient"):
        for fname in os.listdir(patient_path):
            if fname.lower().endswith(".png"):
                all_files.append(os.path.join(patient_path, fname))

# 按名稱排序
all_files.sort()

print(f"來源根目錄：{patient_root}")
print(f"共收集到 {len(all_files)} 張圖片")

# 載入模型（請確認 best.pt 檔案放在當前工作目錄，或改成絕對路徑）
model = YOLO("./runs/detect/train14/weights/best.pt")

# 開始計時
start_time = time.time()

# 執行推論
results = model.predict(
    source="./datasets/testing_images/*/*", # 測試圖片資料夾
    imgsz=640,                              # 輸入圖片大小
    batch=16,                               # 批次大小，視 GPU VRAM 調整
    device=0,                               # 使用 GPU:0；若要用 CPU，改成 device="cpu"
    stream=True,                            # 逐張圖片推論
    save=True,                              # 是否輸出預測結果圖片
    save_txt=True,                         # 是否輸出預測結果文字檔
    save_conf=True,                        # 在文字檔中包含信心分數
    conf=0.45,                             # 信心分數閾值（預設值0.25）- 提高 conf（例如 0.45 → 0.55）減少誤報（Precision ↑，Recall ↓），降低 conf（例如 0.35 → 0.25）增加檢出率（Recall ↑，Precision ↓）
    iou=0.45,                              # NMS 的 IoU 閾值（預設值0.45）,調整此值會影響 mAP,提高 iou → 保留更多重疊框（Recall ↑，Precision ↓，可能重複偵測）,降低 iou → 更嚴格地去除重疊框（Precision ↑，Recall ↓,可能漏檢）
    max_det=300,                           # 每張圖片的最大檢測數量

)

# ✅ 把 generator 轉成 list
results = list(results)

# 結束計時
end_time = time.time()
elapsed = end_time - start_time
minutes = elapsed / 60
total_time = end_time - start_time
num_images = len(results)
fps = num_images / total_time if total_time > 0 else 0

print(f"✅ 預測完成！總推論時間: 約 {minutes:.2f} 分鐘")
print(f"圖片數量: {num_images}")
print(f"平均推論時間: {total_time/num_images*1000:.2f} ms/張")
print(f"FPS: {fps:.2f}")

# 收集所有信心分數
all_confs = []
for r in results:
    if len(r.boxes) > 0:
        all_confs.extend(r.boxes.conf.cpu().numpy())

if len(all_confs) > 0:
    all_confs = np.array(all_confs)
    print(f"信心分數平均值: {all_confs.mean():.4f}")
    print(f"信心分數最大值: {all_confs.max():.4f}")
    print(f"信心分數最小值: {all_confs.min():.4f}")
else:
    print("⚠️ 沒有任何預測框，無法計算信心分數統計")

# print("✅ 預測完成！\n預測數量:", len(results))
# print('預測類別 : ',results[260].boxes.cls[0].item())
print('預測信心分數 : ',results[260].boxes.conf[0].item())
print('預測框座標 : ',results[260].boxes.xyxy[0].tolist())

# GPU 記憶體用量
if torch.cuda.is_available():
    mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"GPU 記憶體用量: {mem:.2f} MB")

# 建立輸出資料夾
os.makedirs("./predict_txt", exist_ok=True)

# 以日期時間命名檔案，例如 predict_20251008_000845.txt
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"./predict_txt/predict_{timestamp}.txt"

with open(output_path, "w") as output_file:
    for i in range(len(results)):
        # 取得圖片檔名（不含副檔名）
        filename = os.path.splitext(os.path.basename(results[i].path))[0]

        # 取得預測框數量
        boxes = results[i].boxes
        box_num = len(boxes.cls.tolist())

        if box_num > 0:
            for j in range(box_num):
                label = int(boxes.cls[j].item())   # 類別
                conf = boxes.conf[j].item()        # 信心度
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()

                line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                output_file.write(line)

print(f"✅ 已將預測結果輸出到 {output_path}")
