# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
from datetime import datetime

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

# 載入模型
model = YOLO("yolo12n.pt")  # 換成你的模型檔案

# 一次性預測所有圖片
results = model.predict(
    source=all_files,   # 直接丟入圖片清單
    imgsz=640,
    batch=16,
    device=0,            # GPU:0，如果要用 CPU 改成 device="cpu"
    stream=True
)
print("✅ 預測完成！")
print("測試集圖片數量 :", len(all_files))


# # 分兩批
# half = len(all_files) // 2
# batch1 = all_files[:half]
# batch2 = all_files[half:]

# print(f"➡️ 第一批 {len(batch1)} 張")
# results1 = model.predict(
#     source=batch1,
#     imgsz=640,
#     batch=16,
#     device=0,
#     stream=True
# )

# print(f"➡️ 第二批 {len(batch2)} 張")
# results2 = model.predict(
#     source=batch2,
#     imgsz=640,
#     batch=16,
#     device=0,
#     stream=True
# )
# print("✅ 兩批預測完成！")
# print("測試集圖片數量 :", len(all_files))

# 載入模型（請確認 best.pt 檔案放在當前工作目錄，或改成絕對路徑）
model = YOLO("./runs/detect/train/weights/best.pt")

# 執行推論
results = model.predict(
    source="./datasets/testing_images/*/*",  # 測試圖片資料夾
    save=True,                          # 是否輸出預測結果圖片
    imgsz=640,                          # 輸入圖片大小
    device=0                            # 使用 GPU:0；若要用 CPU，改成 device="cpu"
)

print("✅ 預測完成！\n預測數量:", len(results))
print('預測類別 : ',results[260].boxes.cls[0].item())
print('預測信心分數 : ',results[260].boxes.conf[0].item())
print('預測框座標 : ',results[260].boxes.xyxy[0].tolist())

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
