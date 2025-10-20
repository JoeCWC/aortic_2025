# -*- coding: utf-8 -*-
import locale
from ultralytics import checks, YOLO
import os
import shutil
import time
import torch

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
copy_patients(1, 50, "train")
copy_patients(1, 50, "val")

print('訓練集圖片數量 : ',len(os.listdir("./datasets/train/images")))
print('訓練集標記數量 : ',len(os.listdir("./datasets/train/labels")))
print('驗證集圖片數量 : ',len(os.listdir("./datasets/val/images")))
print('驗證集標記數量 : ',len(os.listdir("./datasets/val/labels")))

# 載入模型（可換成自己的 .pt 檔）
print("載入模型...")
# model = YOLO("yolov13n.pt")
model = YOLO("yolov13n_cbam_p3.yaml").load("yolov13n.pt")

# 記錄開始時間
print("紀錄開始時間...")
start_time = time.time()

# 開始訓練
print("開始訓練...")
results = model.train(
    data="./aortic_valve_colab.yaml",  
    epochs=120,                        
    batch=16,                          
    imgsz=640,                         
    device=0,                          
    optimizer="AdamW",                 
    lr0=0.001,                         
    lrf=0.01,                          
    cos_lr=True,                       
    weight_decay=0.0005,               
    label_smoothing=0.05,              
    dropout=0.2,                       
    patience=20,                       
    rect=True,                        
    val=True,                         
    save_period=50,                   
    plots=True,                       
    workers=16,                        
    seed=42,                          
)

'''
data="./aortic_valve_colab.yaml",  # 資料集設定檔
    # 訓練流程設定
    epochs=120,                        # 訓練回合數,醫學影像數據少，建議訓練更久，讓模型充分收斂,100-300
    batch=16,                          # 8-16,視 GPU VRAM 調整
    imgsz=640,                         # 輸入圖片大小
    device=0,                          # 指定 GPU (0 表示第一張 GPU；若要用 CPU，改成 'cpu')
    # 最佳化器
    optimizer="AdamW",                 # 選擇最佳化器（SGD、Adam、AdamW）
    lr0=0.001,                         # 初始學習率,避免過大導致震盪
    lrf=0.01,                          # 最終學習率（相對於初始學習率的比例）,搭配 cosine scheduler 效果佳
    cos_lr=True,                       # 使用使用 cosine learning rate scheduler學習率調整
    weight_decay=0.0005,               # 權重衰減,適度正則化，避免過擬合
    # 正則化
    label_smoothing=0.05,              # 0.05-0.1標籤平滑,減少過擬合，對小數據集有幫助
    dropout=0.2,                       # 0.2-0.3 在 head 層隨機捨棄神經元,避免過擬合,提升泛化能力
    # 早停策略
    patience=0,                       # 30-50若驗證集指標長時間無提升，自動停止訓練
    # 資料處理
    rect=True,                         # 使用長方形訓練,加速訓練過程,保持醫學影像比例，避免拉伸失真,- 設 rect=True 時，YOLO 會自動把 shuffle=False,- rect=True → 分組依長寬比，不能隨機打亂。- shuffle=True → 打亂樣本順序，和 rect 相衝突。
    # 驗證與輸出
    val=True,                          # 每個回合結束後進行驗證
    save_period=50,                    # 每n個回合保存一次模型
    plots=True,                         # 繪製訓練過程中的損失曲線等圖表
    # 其他
    workers=8,                         # 使用 8 個工作緒來加速資料載入
    seed=42,                           # 隨機種子，確保結果可重現
'''

# 記錄結束時間
end_time = time.time()
elapsed = end_time - start_time
minutes = elapsed / 60

# 在驗證集上計算 mAP
# metrics = model.val(data="./aortic_valve_colab.yaml", imgsz=640)

print(f"✅ 訓練完成！總訓練時間：約 {minutes:.2f} 分鐘")
print('訓練集圖片數量 : ',len(os.listdir("./datasets/train/images")))
print('訓練集標記數量 : ',len(os.listdir("./datasets/train/labels")))
print('驗證集圖片數量 : ',len(os.listdir("./datasets/val/images")))
print('驗證集標記數量 : ',len(os.listdir("./datasets/val/labels")))
# print(f"mAP@0.5:0.95 = {metrics.box.map:.4f}")
# print(f"mAP@0.5     = {metrics.box.map50:.4f}")
# print(f"mAP@0.75    = {metrics.box.map75:.4f}")

# GPU 記憶體用量
if torch.cuda.is_available():
    mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"GPU 記憶體用量: {mem:.2f} MB")