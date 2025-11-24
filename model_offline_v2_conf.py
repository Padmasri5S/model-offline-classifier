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

# Class-weighting controls (safer than raw ratios)
USE_SQRT_SCALING = True
CLIP_MIN, CLIP_MAX = 1.0, 20.0

# Toggle partial checkpoint loading if a previous model.pth exists
LOAD_PARTIAL_CHECKPOINT = True

# Thresholding method: "pr_curve" (recommended) or "grid"
THRESHOLDING_METHOD = "pr_curve"

# Valid labels (kept from your original list)
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
# If you ever want to use JSON instead, uncomment:
# with open(JSON_FILE, "r") as f:
#     json_data = json.load(f)
metadata = csv_data.copy()

def clean_labels(label_list):
    """
    Accepts a string of ';'-separated labels or a list of labels; returns a list of stripped labels.
    Filters out placeholders ('component', 'defect', '', 'nan').
    """
    if isinstance(label_list, str):
        labels = [l.strip() for l in str(label_list).split(";") if l.strip()]
    elif isinstance(label_list, list):
        labels = [l.strip() for l in label_list]
    else:
        labels = []
    return [l for l in labels if l not in {"component", "defect", "", "nan"}]

# Original data had all labels in "defect" column; normalize first
metadata["defect"] = metadata["defect"].apply(clean_labels)

def split_labels(label_list):
    comps = [l for l in label_list if l in VALID_COMPONENTS]
    defs  = [l for l in label_list if l in VALID_DEFECTS]
    return comps, defs

metadata["component"], metadata["defect"] = zip(*metadata["defect"].apply(split_labels))

# Keep rows with either components OR defects (not strictly both)
metadata = metadata[(metadata["component"].apply(len) > 0) | (metadata["defect"].apply(len) > 0)]

print(f"Total rows after cleaning: {len(metadata)}")
print(f"Unique components: {len(set(metadata['component'].explode()))}")
print(f"Unique defects: {len(set(metadata['defect'].explode()))}")

# Split IDs into train/val/test
unique_ids = metadata["ID"].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids   = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_df = metadata[metadata["ID"].isin(train_ids)]
val_df   = metadata[metadata["ID"].isin(val_ids)]
test_df  = metadata[metadata["ID"].isin(test_ids)]

# Ensure folders exist
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

# Save cleaned CSVs with correct names
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
    """
    Groups rows by ID, unions component/defect lists across rows for that ID,
    filters to valid sets, and returns a frame with ['ID','component','defect','labels'].
    """
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

# Persist class order for downstream tools
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
class_counts = y_train.sum(axis=0)                  # positives per class
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
# STEP 4: TRANSFORMS & DATALOADERS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
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
    """
    Loads only compatible keys (matching names and shapes). Returns (loaded_any, skipped_keys).
    """
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
    # Train
    model.train()
    total_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)                 # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / max(len(train_loader), 1)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)             # logits
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
# Load best model for threshold tuning
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Collect validation probabilities
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
    # Degenerate class (all 0s or all 1s) -> fallback to 0.5
    if len(np.unique(y_true)) <= 1:
        optimal_thresholds[class_idx] = 0.5
        continue

    if THRESHOLDING_METHOD == "pr_curve":
        # Precise: pick threshold that maximizes F1 on PR curve
        y_scores = all_probs_val[:, class_idx]
        precision, recall, thresh = precision_recall_curve(y_true, y_scores)
        f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        # thresholds length is len(precision)-1; align safely
        if len(thresh) > 0 and np.any(np.isfinite(f1s[:-1])):
            best_idx = np.nanargmax(f1s[:-1])  # use indices aligned to thresholds
            optimal_thresholds[class_idx] = float(thresh[best_idx])
        else:
            optimal_thresholds[class_idx] = 0.5
    else:
        # Grid search across [0,1]
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

# --- CLIP HIGH THRESHOLDS ---
max_threshold = 0.8     # adjust as needed
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

        # Apply class-specific thresholds (shape matches)
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

# Save predictions
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
