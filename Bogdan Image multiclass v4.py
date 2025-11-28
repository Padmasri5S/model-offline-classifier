import os
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_recall_curve
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
# =========================================================
# CONFIGURATION
# =========================================================
IMG_DIR = "images"
TRAIN_DIR = "images/train"
VAL_DIR   = "images/val"
TEST_DIR  = "images/test"
CSV_FILE  = "image_data.csv"
JSON_FILE = "image_data.json"   # not used by default, but left for compatibility
MODEL_PATH       = "model.pth"
CLASSES_JSON     = "classes.json"
THRESHOLDS_JSON  = "thresholds.json"
PREDICTIONS_FILE = "predictions.csv"
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 0.001
PATIENCE   = 3
USE_SQRT_SCALING = True
CLIP_MIN, CLIP_MAX = 1.0, 20.0
LOAD_PARTIAL_CHECKPOINT = True
THRESHOLDING_METHOD = "pr_curve"
VALID_COMPONENTS = {
    "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
    "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support",
    "Centre Support-Bolt", "Channel Bracket-Bolt", "Conveyor Support-Bolt", "Walkway Support-Bolt"
}
VALID_DEFECTS = {
    "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface", "Loose", "Missing", "Drummy", "Leaks"
}
# =========================================================
# DEVICE SETUP
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# =========================================================
# STEP 1: LOAD, CLEAN, AND MERGE METADATA
# =========================================================
csv_data = pd.read_csv(CSV_FILE)
metadata = csv_data.copy()
def clean_labels(label_list):
    if isinstance(label_list, str):
        labels = [l.strip() for l in str(label_list).split(";") if l.strip()]
    elif isinstance(label_list, list):
        labels = [l.strip() for l in label_list]
    else:
        labels = []
    return [l for l in labels if l not in {"component", "defect", "", "nan"}]
metadata["defect"] = metadata["defect"].apply(clean_labels)
def split_labels(label_list):
    comps = [l for l in label_list if l in VALID_COMPONENTS]
    defs  = [l for l in label_list if l in VALID_DEFECTS]
    return comps, defs
metadata["component"], metadata["defect"] = zip(*metadata["defect"].apply(split_labels))
metadata = metadata[(metadata["component"].apply(len) > 0) | (metadata["defect"].apply(len) > 0)]
print(f"Total rows after cleaning: {len(metadata)}")
print(f"Unique components: {len(set(metadata['component'].explode()))}")
print(f"Unique defects: {len(set(metadata['defect'].explode()))}")
unique_ids = metadata["ID"].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids   = train_test_split(temp_ids, test_size=0.5, random_state=42)
train_df = metadata[metadata["ID"].isin(train_ids)]
val_df   = metadata[metadata["ID"].isin(val_ids)]
test_df  = metadata[metadata["ID"].isin(test_ids)]
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)
def move_images(df, folder):
    for img_id in df["ID"]:
        src = os.path.join(IMG_DIR, f"{img_id}.jpg")
        dst = os.path.join(folder, f"{img_id}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)
move_images(train_df, TRAIN_DIR)
move_images(val_df,   VAL_DIR)
move_images(test_df,  TEST_DIR)
def filter_existing_images(df, folder):
    existing_ids = {os.path.splitext(f)[0] for f in os.listdir(folder)}
    return df[df["ID"].isin(existing_ids)]
train_df = filter_existing_images(train_df, TRAIN_DIR)
val_df   = filter_existing_images(val_df,   VAL_DIR)
test_df  = filter_existing_images(test_df,  TEST_DIR)
train_df.to_csv("train_metadata.csv", index=False)
val_df.to_csv("val_metadata.csv",   index=False)
test_df.to_csv("test_metadata.csv", index=False)
print(f"Cleaned train CSV: {len(train_df)} rows")
print(f"Cleaned val   CSV: {len(val_df)} rows")
print(f"Cleaned test  CSV: {len(test_df)} rows")
# =========================================================
# STEP 2: PREPARE MULTI-LABEL TARGETS
# =========================================================
def group_labels(df):
    grouped = df.groupby("ID").agg({
        "component": lambda col: list(set([item for sublist in col for item in sublist])),
        "defect":    lambda col: list(set([item for sublist in col for item in sublist]))
    }).reset_index()
    grouped["component"] = grouped["component"].apply(lambda xs: [c for c in xs if c in VALID_COMPONENTS])
    grouped["defect"]    = grouped["defect"].apply(lambda xs: [d for d in xs if d in VALID_DEFECTS])
    grouped["labels"]    = grouped.apply(lambda row: row["component"] + row["defect"], axis=1)
    return grouped
train_grouped = group_labels(train_df)
val_grouped   = group_labels(val_df)
test_grouped  = group_labels(test_df)
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_grouped["labels"])
y_val   = mlb.transform(val_grouped["labels"])
y_test  = mlb.transform(test_grouped["labels"])
label_classes = list(mlb.classes_)
num_classes   = len(label_classes)
print(f"Classes ({num_classes}): {label_classes}")
if os.path.exists(CLASSES_JSON):
    with open(CLASSES_JSON, "r") as f:
        saved_classes = json.load(f)
    if list(label_classes) != saved_classes:
        print("Warning: current label_classes differ from saved classes.json. "
              "Ensure consistent class order to avoid mislabeling.")
with open(CLASSES_JSON, "w") as f:
    json.dump(list(label_classes), f)
# =========================================================
# STEP 2.5: CLASS WEIGHTS (safer pos_weight)
# =========================================================
class_counts = y_train.sum(axis=0)
neg_counts   = len(y_train) - class_counts
pos_weight_raw = neg_counts / np.maximum(class_counts, 1)
if USE_SQRT_SCALING:
    pos_weight_vals = np.sqrt(pos_weight_raw)
else:
    pos_weight_vals = pos_weight_raw
pos_weight_vals = np.clip(pos_weight_vals, CLIP_MIN, CLIP_MAX)
pos_weight = torch.tensor(pos_weight_vals, dtype=torch.float32).to(device)
print("pos_weight used:", np.round(pos_weight_vals, 3).tolist())
# =========================================================
# STEP 3: DATASET
# =========================================================
class ImageDataset(Dataset):
    def __init__(self, img_dir, df_grouped, labels, transform=None):
        self.img_dir   = img_dir
        self.df        = df_grouped
        self.labels    = labels
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id  = self.df.iloc[idx]["ID"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, img_id
# =========================================================
# STEP 4: TRANSFORMS & DATALOADERS (WITH NORMALIZATION)
# =========================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
train_dataset = ImageDataset(TRAIN_DIR, train_grouped, y_train, transform=train_transform)
val_dataset   = ImageDataset(VAL_DIR,   val_grouped,   y_val,   transform=eval_transform)
test_dataset  = ImageDataset(TEST_DIR,  test_grouped,  y_test,  transform=eval_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
# =========================================================
# STEP 5: MODEL
# =========================================================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
# =========================================================
# STEP 6: OPTIONAL PARTIAL CHECKPOINT LOADING
# =========================================================
def load_partial_state_dict(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return False, []
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.size() == model_state[k].size():
            filtered_state[k] = v
        else:
            skipped.append(k)
    if filtered_state:
        model.load_state_dict(filtered_state, strict=False)
        return True, skipped
    return False, skipped
if LOAD_PARTIAL_CHECKPOINT and os.path.exists(MODEL_PATH):
    print("Attempting partial checkpoint load...")
    loaded_any, skipped = load_partial_state_dict(model, MODEL_PATH, device)
    print(f"Partial load: {'success' if loaded_any else 'no compatible keys'}")
    if skipped:
        print(f"Skipped keys ({len(skipped)}): {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")
# =========================================================
# STEP 7: TRAINING LOOP (BCEWithLogitsLoss + EarlyStopping)
# =========================================================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)
best_val_loss   = float("inf")
patience_counter = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / max(len(train_loader), 1)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / max(len(val_loader), 1)
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {avg_train_loss:.4f}   Val Loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
print("Training complete. Best model saved.")
# =========================================================
# STEP 8: THRESHOLD TUNING ON VALIDATION (Robust)
# =========================================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
all_probs_val = []
with torch.no_grad():
    for images, _, _ in val_loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs_val.append(probs)
all_probs_val = np.vstack(all_probs_val) if len(all_probs_val) > 0 else np.zeros((len(val_dataset), num_classes), dtype=np.float32)
optimal_thresholds = np.zeros((num_classes,), dtype=np.float32)
for class_idx in range(num_classes):
    y_true = y_val[:, class_idx]
    if len(np.unique(y_true)) <= 1:
        optimal_thresholds[class_idx] = 0.5
        continue
    if THRESHOLDING_METHOD == "pr_curve":
        y_scores = all_probs_val[:, class_idx]
        precision, recall, thresh = precision_recall_curve(y_true, y_scores)
        f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        if len(thresh) > 0 and np.any(np.isfinite(f1s[:-1])):
            best_idx = np.nanargmax(f1s[:-1])
            optimal_thresholds[class_idx] = float(thresh[best_idx])
        else:
            optimal_thresholds[class_idx] = 0.5
    else:
        thresholds = np.linspace(0.0, 1.0, 41)
        best_f1, best_t = -1.0, 0.5
        y_scores = all_probs_val[:, class_idx]
        for t in thresholds:
            preds_bin = (y_scores > t).astype(int)
            f1 = f1_score(y_true, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        optimal_thresholds[class_idx] = best_t
print("Optimal thresholds per class:", np.round(optimal_thresholds, 3).tolist())
max_threshold = 0.8
optimal_thresholds = np.where(optimal_thresholds > max_threshold, 0.5, optimal_thresholds)
print("Final thresholds after clipping:", np.round(optimal_thresholds, 3).tolist())
with open(THRESHOLDS_JSON, "w") as f:
    json.dump(optimal_thresholds.tolist(), f)
# =========================================================
# STEP 9: PREDICTION ON TEST SET + SAVE
# =========================================================
predictions = []
preds_all   = []
with torch.no_grad():
    for images, _, img_ids in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > optimal_thresholds).astype(int)
        preds_all.extend(preds)
        for i, img_id in enumerate(img_ids):
            prob_row = probs[i]
            pred_indices  = [j for j in range(num_classes) if preds[i][j] == 1]
            pred_labels   = [label_classes[j] for j in pred_indices]
            pred_components = [lbl for lbl in pred_labels if lbl in VALID_COMPONENTS]
            pred_defects    = [lbl for lbl in pred_labels if lbl in VALID_DEFECTS]
            comp_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_components]
            def_conf_pairs  = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_defects]
            avg_comp_conf = float(np.mean([prob_row[label_classes.index(lbl)] for lbl in pred_components])) if pred_components else 0.0
            avg_def_conf  = float(np.mean([prob_row[label_classes.index(lbl)] for lbl in pred_defects])) if pred_defects else 0.0
            overall_avg_conf = float((avg_comp_conf + avg_def_conf) / 2.0) if (pred_components or pred_defects) else 0.0
            predictions.append({
                "ID": img_id,
                "components": ";".join(pred_components),
                "defects":    ";".join(pred_defects),
                "components_confidence": ";".join(comp_conf_pairs),
                "defects_confidence":    ";".join(def_conf_pairs),
                "avg_component_confidence": round(avg_comp_conf, 4),
                "avg_defect_confidence":    round(avg_def_conf, 4),
                "overall_avg_confidence":   round(overall_avg_conf, 4)
            })
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(PREDICTIONS_FILE, index=False)
print("\nSample predictions with confidence:")
print(pred_df.head())
# =========================================================
# STEP 10: EVALUATION
# =========================================================
preds_all = np.array(preds_all) if len(preds_all) > 0 else np.zeros_like(y_test)
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)
hl = hamming_loss(y_test, preds_all)
f1_micro    = f1_score(y_test, preds_all, average='micro',    zero_division=0)
f1_macro    = f1_score(y_test, preds_all, average='macro',    zero_division=0)
f1_weighted = f1_score(y_test, preds_all, average='weighted', zero_division=0)
print(f"Hamming Loss: {hl:.4f}")
print(f"F1 Micro:     {f1_micro:.4f}")
print(f"F1 Macro:     {f1_macro:.4f}")
print(f"F1 Weighted:  {f1_weighted:.4f}")
f1_per_class = f1_score(y_test, preds_all, average=None, zero_division=0)
print("\nPer-Class F1 Scores:")
for i, label in enumerate(label_classes):
    print(f"  {label}: {f1_per_class[i]:.4f}")

'''
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model> python .\predict_all_images.py
Using device: cpu
Total rows after cleaning: 3352
Unique components: 1582
Unique defects: 1790
Cleaned train CSV: 2023 rows
Cleaned val   CSV: 441 rows
Cleaned test  CSV: 428 rows
Classes (20): ['Centre Support', 'Centre Support-Bolt', 'Channel Bracket', 'Channel Bracket-Bolt', 'Circular Connection', 'Circular Joint', 'Coating Failure', 'Conveyor Support', 'Conveyor Support-Bolt', 'Corrosion-Heavy', 'Corrosion-Surface', 'Drummy', 'Grout Hole', 'Leaks', 'Loose', 'Missing', 'Radial Connection', 'Radial Joint', 'Walkway Support', 'Walkway Support-Bolt']
pos_weight used: [4.194, 9.045, 1.767, 1.613, 1.805, 6.852, 12.281, 7.252, 9.492, 1.821, 2.15, 1.34, 2.357, 2.243, 6.214, 3.578, 2.832, 12.281, 6.51, 7.252]
Attempting partial checkpoint load...
Partial load: success
Epoch [1/10]  Train Loss: 0.3919   Val Loss: 0.6589
Epoch [2/10]  Train Loss: 0.3531   Val Loss: 0.5757
Epoch [3/10]  Train Loss: 0.3110   Val Loss: 0.3982
Epoch [4/10]  Train Loss: 0.2996   Val Loss: 0.4374
Epoch [5/10]  Train Loss: 0.2735   Val Loss: 0.4236
Epoch [6/10]  Train Loss: 0.2493   Val Loss: 0.5397
Early stopping triggered.
Training complete. Best model saved.
Optimal thresholds per class: [0.1289999932050705, 0.5, 0.28999999165534973, 0.578000009059906, 0.39100000262260437, 0.0989999994635582, 0.9100000262260437, 0.2240000069141388, 0.9779999852180481, 0.3919999897480011, 0.3160000145435333, 0.7149999737739563, 0.6880000233650208, 0.5630000233650208, 0.4090000092983246, 0.2720000147819519, 0.7289999723434448, 0.5270000100135803, 0.4050000011920929, 0.2460000067949295]
Final thresholds after clipping: [0.1289999932050705, 0.5, 0.28999999165534973, 0.578000009059906, 0.39100000262260437, 0.0989999994635582, 0.5, 0.2240000069141388, 0.5, 0.3919999897480011, 0.3160000145435333, 0.7149999737739563, 0.6880000233650208, 0.5630000233650208, 0.4090000092983246, 0.2720000147819519, 0.7289999723434448, 0.5270000100135803, 0.4050000011920929, 0.2460000067949295]

Sample predictions with confidence:
                                     ID                            components  ... avg_defect_confidence overall_avg_confidence
0  064b9581-87be-4a29-90a4-502fee1961c5  Channel Bracket;Channel Bracket-Bolt  ...                0.4396                 0.5061
1  07636456-22de-4097-9332-157374290e78  Channel Bracket;Channel Bracket-Bolt  ...                0.6454                 0.6611
2  0814d7ca-1173-43b5-838e-9e49ff0fe9f9  Channel Bracket;Channel Bracket-Bolt  ...                0.4784                 0.6462
3  085bd0aa-9105-4d77-b451-b073ab7ccfb5                                        ...                0.0000                 0.0000
4  0991c975-a5f4-4f7d-b4b3-a3bae0fcbbb1                            Grout Hole  ...                0.9051                 0.8913

[5 rows x 8 columns]

======================================================================
EVALUATION METRICS
======================================================================
Hamming Loss: 0.1211
F1 Micro:     0.5549
F1 Macro:     0.4250
F1 Weighted:  0.5942

Per-Class F1 Scores:
  Centre Support: 0.5000
  Centre Support-Bolt: 0.0000
  Channel Bracket: 0.5891
  Channel Bracket-Bolt: 0.6780
  Circular Connection: 0.6842
  Circular Joint: 0.1500
  Coating Failure: 0.2727
  Conveyor Support: 0.0000
  Conveyor Support-Bolt: 0.4000
  Corrosion-Heavy: 0.4412
  Corrosion-Surface: 0.4762
  Drummy: 0.8348
  Grout Hole: 0.6133
  Leaks: 0.7059
  Loose: 0.2222
  Missing: 0.3529
  Radial Connection: 0.4800
  Radial Joint: 0.0000
  Walkway Support: 0.5000
  Walkway Support-Bolt: 0.6000
'''
