import os
import json
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
# CONFIGURATION
IMG_DIR = "images"
TRAIN_DIR = "images/train"
VAL_DIR = "images/val"
TEST_DIR = "images/test"
CSV_FILE = "image_data.csv"
JSON_FILE = "image_data.json"
MODEL_PATH = "model.pth"
PREDICTIONS_FILE = "predictions.csv"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
PATIENCE = 3
VALID_COMPONENTS = {
    "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
    "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support",
    "Centre Support-Bolt", "Channel Bracket-Bolt", "Conveyor Support-Bolt", "Walkway Support-Bolt"
}
VALID_DEFECTS = {
    "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface", "Loose", "Missing", "Drummy", "Leaks"
}
# DEVICE SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# STEP 1: LOAD AND CLEAN METADATA
csv_data = pd.read_csv(CSV_FILE)
with open(JSON_FILE, "r") as f:
    json_data = json.load(f)
metadata = csv_data
def clean_labels(label_list):
    if isinstance(label_list, str):
        labels = [l.strip() for l in str(label_list).split(";") if l.strip()]
    elif isinstance(label_list, list):
        labels = label_list
    else:
        labels = []
    return [l for l in labels if l not in {"component", "defect", "", "nan"}]
metadata["component"] = metadata["component"].apply(clean_labels)
metadata["defect"] = metadata["defect"].apply(clean_labels)
metadata = metadata[(metadata["component"].apply(len) > 0) | (metadata["defect"].apply(len) > 0)]
# Split IDs into train/val/test
unique_ids = metadata["ID"].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
train_df = metadata[metadata["ID"].isin(train_ids)]
val_df = metadata[metadata["ID"].isin(val_ids)]
test_df = metadata[metadata["ID"].isin(test_ids)]
# Move images
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)
def move_images(df, folder):
    for img_id in df["ID"]:
        src = os.path.join(IMG_DIR, f"{img_id}.jpg")
        dst = os.path.join(folder, f"{img_id}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)
move_images(train_df, TRAIN_DIR)
move_images(val_df, VAL_DIR)
move_images(test_df, TEST_DIR)
def filter_existing_images(df, folder):
    existing_ids = {os.path.splitext(f)[0] for f in os.listdir(folder)}
    return df[df["ID"].isin(existing_ids)]
train_df = filter_existing_images(train_df, TRAIN_DIR)
val_df = filter_existing_images(val_df, VAL_DIR)
test_df = filter_existing_images(test_df, TEST_DIR)
# STEP 2: GROUP AND ENCODE LABELS
def group_labels(df):
    grouped = df.groupby("ID").agg({
        "component": lambda x: list(set([item for sublist in x for item in sublist])),
        "defect": lambda x: list(set([item for sublist in x for item in sublist]))
    }).reset_index()
    grouped["component"] = grouped["component"].apply(lambda x: [c for c in x if c in VALID_COMPONENTS])
    grouped["defect"] = grouped["defect"].apply(lambda x: [d for d in x if d in VALID_DEFECTS])
    grouped["labels"] = grouped.apply(lambda row: row["component"] + row["defect"], axis=1)
    return grouped
train_grouped = group_labels(train_df)
val_grouped = group_labels(val_df)
test_grouped = group_labels(test_df)
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_grouped["labels"])
y_val = mlb.transform(val_grouped["labels"])
y_test = mlb.transform(test_grouped["labels"])
label_classes = list(mlb.classes_)
print(f"Classes: {label_classes}")
# STEP 2.5: CLASS WEIGHTS
class_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - class_counts
pos_weight = torch.tensor(neg_counts / np.maximum(class_counts, 1), dtype=torch.float32).to(device)
# STEP 3: DATASET & DATALOADERS
class ImageDataset(Dataset):
    def __init__(self, img_dir, df, labels, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["ID"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, img_id
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = ImageDataset(TRAIN_DIR, train_grouped, y_train, transform=train_transform)
val_dataset = ImageDataset(VAL_DIR, val_grouped, y_val, transform=val_transform)
test_dataset = ImageDataset(TEST_DIR, test_grouped, y_test, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# STEP 4: MODEL
num_classes = len(label_classes)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
# Load partial weights if available
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
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    load_partial_state_dict(model, MODEL_PATH, device)
# STEP 5: TRAINING LOOP WITH EARLY STOPPING
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)
best_val_loss = float("inf")
patience_counter = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
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
# STEP 6: THRESHOLD TUNING ON VALIDATION
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
all_probs_val = []
with torch.no_grad():
    for images, _, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs_val.append(probs)
all_probs_val = np.vstack(all_probs_val)
optimal_thresholds = []
for class_idx in range(num_classes):
    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.1, 0.95, 0.05):
        preds_binary = (all_probs_val[:, class_idx] > threshold).astype(int)
        true_binary = y_val[:, class_idx]
        if len(np.unique(true_binary)) > 1:
            f1 = f1_score(true_binary, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
    optimal_thresholds.append(best_threshold)
optimal_thresholds = np.array(optimal_thresholds)
print("Optimal thresholds:", optimal_thresholds)
# STEP 7: PREDICTION ON TEST SET
predictions = []
preds_all = []
all_probs_test = []
with torch.no_grad():
    for images, _, img_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs_test.append(probs)
        preds = (probs > optimal_thresholds).astype(int)
        preds_all.extend(preds)
        for i, img_id in enumerate(img_ids):
            pred_indices = [j for j in range(len(label_classes)) if preds[i][j] == 1]
            pred_labels = [label_classes[j] for j in pred_indices]
            pred_components = [lbl for lbl in pred_labels if lbl in VALID_COMPONENTS]
            pred_defects = [lbl for lbl in pred_labels if lbl in VALID_DEFECTS]
            predictions.append({
                "ID": img_id,
                "components": ";".join(pred_components),
                "defects": ";".join(pred_defects)
            })
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(PREDICTIONS_FILE, index=False)
print("\nSample predictions:")
print(pred_df.head())
# STEP 8: EVALUATION METRICS
preds_all = np.array(preds_all)
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)
hl = hamming_loss(y_test, preds_all)
f1_micro = f1_score(y_test, preds_all, average='micro', zero_division=0)
f1_macro = f1_score(y_test, preds_all, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, preds_all, average='weighted', zero_division=0)
print(f"Hamming Loss: {hl:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")
# Jaccard Index
jaccard_scores = []
for i in range(y_test.shape[0]):
    gt = y_test[i].astype(int)
    pr = preds_all[i].astype(int)
    intersection = float(np.logical_and(gt, pr).sum())
    union = float(np.logical_or(gt, pr).sum())
    jaccard_scores.append(1.0 if union == 0 else intersection / union)
avg_jaccard = np.mean(jaccard_scores)
print(f"Average Jaccard Index: {avg_jaccard:.4f}")
# Per-class F1
f1_scores_per_class = f1_score(y_test, preds_all, average=None, zero_division=0)
print("\nPer-Class F1 Scores:")
for i, label in enumerate(label_classes):
    print(f"  {label}: {f1_scores_per_class[i]:.4f}")

'''
✅ Why are components missing in predictions?
Your output shows only defects because:

label_classes contains only defect labels:
['Coating Failure', 'Corrosion-Heavy', 'Corrosion-Surface', 'Drummy', 'Leaks', 'Loose', 'Missing']


This means component labels were never encoded. Likely cause:

Either the CSV has empty component fields or
The filtering step removed all components because they didn’t match VALID_COMPONENTS.



So the model is only predicting defects, not components.

✅ Has performance improved?
Yes, compared to your previous run:

Hamming Loss dropped from ~0.53 → 0.095 (much better).
F1 Micro jumped from ~0.72 (previously low) → 0.7226 (good).
Average Jaccard improved to 0.7027 (previously ~0.20).
Per-class F1 shows strong performance for Drummy (0.90) and decent for Corrosion-Heavy (0.75) and Leaks (0.72).

So the model is much better now, but only for defects.

✅ Next Steps


Fix component issue:

Check if your CSV actually has component labels.
Print a quick diagnostic:
Pythonprint(metadata["component"].head())print("Any components after filtering?", any(len(c) > 0 for c in metadata["component"]))Show more lines

If components exist, verify VALID_COMPONENTS matches actual names in the CSV.



If components are missing because of filtering, update VALID_COMPONENTS or remove that filter temporarily.



✅ Adding Confidence Scores & Averages
Since performance improved and defect names are printing correctly, adding confidence metrics will make results much more informative.
Here’s what I’ll add:

components_confidence and defects_confidence columns in CSV.
avg_component_confidence, avg_defect_confidence, and overall_avg_confidence per image.
Print mean confidence metrics across the test set.


🔥 Updated Prediction Block with Confidence Metrics
Replace your STEP 7 with this:
Python# STEP 7: PREDICTION ON TEST SET WITH CONFIDENCE METRICSpredictions = []preds_all = []with torch.no_grad():    for images, _, img_ids in test_loader:        images = images.to(device)        outputs = model(images)        probs = torch.sigmoid(outputs).cpu().numpy()        preds = (probs > optimal_thresholds).astype(int)        preds_all.extend(preds)        for i, img_id in enumerate(img_ids):            prob_row = probs[i]            pred_indices = [j for j in range(len(label_classes)) if preds[i][j] == 1]            pred_labels = [label_classes[j] for j in pred_indices]            pred_components = [lbl for lbl in pred_labels if lbl in VALID_COMPONENTS]            pred_defects = [lbl for lbl in pred_labels if lbl in VALID_DEFECTS]            comp_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_components]            defect_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_defects]            avg_comp_conf = float(sum([prob_row[label_classes.index(lbl)] for lbl in pred_components]) / len(pred_components)) if pred_components else 0.0            avg_def_conf = float(sum([prob_row[label_classes.index(lbl)] for lbl in pred_defects]) / len(pred_defects)) if pred_defects else 0.0            overall_avg_conf = float((avg_comp_conf + avg_def_conf) / 2.0) if (pred_components or pred_defects) else 0.0            predictions.append({                "ID": img_id,                "components": ";".join(pred_components),                "defects": ";".join(pred_defects),                "components_confidence": ";".join(comp_conf_pairs),                "defects_confidence": ";".join(defect_conf_pairs),                "avg_component_confidence": round(avg_comp_conf, 4),                "avg_defect_confidence": round(avg_def_conf, 4),                "overall_avg_confidence": round(overall_avg_conf, 4)            })# Save predictionspred_df = pd.DataFrame(predictions)pred_df.to_csv(PREDICTIONS_FILE, index=False)print("\nSample predictions with confidence:")print(pred_df.head())# Confidence summaryall_avg_comp_conf = [p["avg_component_confidence"] for p in predictions]all_avg_def_conf = [p["avg_defect_confidence"] for p in predictions]all_overall_conf = [p["overall_avg_confidence"] for p in predictions]print("\nConfidence Metrics:")print(f"Mean Component Confidence: {np.mean(all_avg_comp_conf):.4f}")print(f"Mean Defect Confidence: {np.mean(all_avg_def_conf):.4f}")print(f"Overall Average Confidence: {np.mean(all_overall_conf):.4f}")Show lessCode block expanded

✅ This will:

Print actual names (components + defects).
Add confidence scores per label.
Compute average confidence per image and global mean confidence.
'''
