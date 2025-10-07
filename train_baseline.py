# -*- coding: utf-8 -*-
import locale
from ultralytics import checks, YOLO
import os
import shutil


def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
# 強制 Python 使用 UTF-8 編碼，無論系統預設的 locale 設定為何。
locale.getpreferredencoding = getpreferredencoding


checks()  # 檢查安裝是否正確


# 🔍 找出含有 patientXXXX 的資料夾
def find_patient_root(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(d.startswith("patient") for d in dirnames):
            return dirpath
    return root  # fallback

# 📁 找出根目錄
IMG_ROOT = find_patient_root("./datasets/raw/training_image")
LBL_ROOT = find_patient_root("./datasets/raw/training_label")

print("IMG_ROOT =", IMG_ROOT)
print("LBL_ROOT =", LBL_ROOT)

# 📁 建立輸出資料夾
for split in ["train", "val"]:
    os.makedirs(f"./datasets/{split}/images", exist_ok=True)
    os.makedirs(f"./datasets/{split}/labels", exist_ok=True)

# 🚚 複製訓練及驗證資料
def copy_patients(start, end, split):
    for i in range(start, end + 1):
        patient = f"patient{i:04d}"
        img_dir = os.path.join(IMG_ROOT, patient)
        lbl_dir = os.path.join(LBL_ROOT, patient)
        if not os.path.isdir(lbl_dir):
            print(f"❌ 缺少標註資料夾：{lbl_dir}")
            continue

        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue

            label_path = os.path.join(lbl_dir, fname)
            base, _ = os.path.splitext(fname)
            img_path = os.path.join(img_dir, base + ".png")

            if not os.path.exists(img_path):
                print(f"⚠️ 找不到圖片：{img_path}")
                continue
            if not os.path.exists(label_path):
                print(f"⚠️ 找不到標註：{label_path}")
                continue

            # 目標路徑
            dst_img = os.path.join(f"./datasets/{split}/images", os.path.basename(img_path))
            dst_lbl = os.path.join(f"./datasets/{split}/labels", os.path.basename(label_path))

            try:
                # 如果檔案已存在就略過
                if not os.path.exists(dst_img):
                    shutil.copy(img_path, dst_img)
                    print(f"✅ 已複製圖片 {patient}/{base}.png")
                else:
                    print(f"⏩ 略過已存在圖片 {patient}/{base}.png")

                if not os.path.exists(dst_lbl):
                    shutil.copy(label_path, dst_lbl)
                    print(f"✅ 已複製標註 {patient}/{base}.txt")
                else:
                    print(f"⏩ 略過已存在標註 {patient}/{base}.txt")

            except Exception as e:
                print(f"❌ 複製失敗：{e}")


# 執行複製
copy_patients(1, 30, "train")
copy_patients(31, 50, "val")

print('訓練集圖片數量 : ',len(os.listdir("./datasets/train/images")))
print('訓練集標記數量 : ',len(os.listdir("./datasets/train/labels")))
print('驗證集圖片數量 : ',len(os.listdir("./datasets/val/images")))
print('驗證集標記數量 : ',len(os.listdir("./datasets/val/labels")))

# 載入模型（可換成自己的 .pt 檔）
model = YOLO("yolo12n.pt")

# 開始訓練
results = model.train(
    data="./aortic_valve_colab.yaml",  # 資料集設定檔
    epochs=20,                         # 訓練回合數
    batch=16,                          # batch size
    imgsz=640,                         # 輸入圖片大小
    device=0                           # 指定 GPU (0 表示第一張 GPU；若要用 CPU，改成 'cpu')
)

