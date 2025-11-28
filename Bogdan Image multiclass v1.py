
import os
import json
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# CONFIGURATION
IMG_DIR = "images"
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"
CSV_FILE = "image_data.csv"
JSON_FILE = "image_data.json"
MODEL_PATH = "model.pth"
PREDICTIONS_FILE = "predictions.csv"
BATCH_SIZE = 16
EPOCHS = 0
LR = 0.001
# Valid component and defect labels (remove generic ones)
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

# STEP 1: LOAD AND MERGE METADATA
csv_data = pd.read_csv(CSV_FILE)
with open(JSON_FILE, "r") as f:
    json_data = json.load(f)
df_json = pd.DataFrame(json_data)
metadata = csv_data  # Use CSV as primary source
# Function to clean labels - remove generic labels and invalid entries
def clean_labels(label_list):
    """Remove generic 'component'/'defect' labels and invalid entries"""
    if isinstance(label_list, str):
        labels = [l.strip() for l in str(label_list).split(";") if l.strip()]
    elif isinstance(label_list, list):
        labels = label_list
    else:
        labels = []
    
    # Filter out generic labels
    cleaned = [l for l in labels if l not in {"component", "defect", "", "nan"} and l]
    return cleaned
# Clean metadata
metadata["component"] = metadata["component"].apply(clean_labels)
metadata["defect"] = metadata["defect"].apply(clean_labels)
# Remove rows with empty components and defects
metadata = metadata[(metadata["component"].apply(len) > 0) | (metadata["defect"].apply(len) > 0)]
# Split metadata
unique_ids = metadata["ID"].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_df = metadata[metadata["ID"].isin(train_ids)]
test_df = metadata[metadata["ID"].isin(test_ids)]
# Move images
for img_id in train_df["ID"]:
    src = os.path.join(IMG_DIR, f"{img_id}.jpg")
    dst = os.path.join(TRAIN_DIR, f"{img_id}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
for img_id in test_df["ID"]:
    src = os.path.join(IMG_DIR, f"{img_id}.jpg")
    dst = os.path.join(TEST_DIR, f"{img_id}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
# Function to filter rows based on existing images
def filter_existing_images(df, folder):
    existing_ids = {os.path.splitext(f)[0] for f in os.listdir(folder)}
    filtered_df = df[df["ID"].isin(existing_ids)]
    missing_ids = set(df["ID"]) - existing_ids
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs missing in {folder}. They will be removed.")
    return filtered_df
# Clean both CSVs
train_df_clean = filter_existing_images(train_df, TRAIN_DIR)
test_df_clean = filter_existing_images(test_df, TEST_DIR)
print(f"Cleaned train CSV: {len(train_df_clean)} rows")
print(f"Cleaned test CSV: {len(test_df_clean)} rows")
train_df = train_df_clean
test_df = test_df_clean

# STEP 2: PREPARE MULTI-LABEL TARGETS
train_grouped = train_df.groupby("ID").agg({
    "component": lambda x: list(set([item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])])),
    "defect": lambda x: list(set([item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])]))
}).reset_index()
# Filter to only valid labels
train_grouped["component"] = train_grouped["component"].apply(lambda x: [c for c in x if c in VALID_COMPONENTS])
train_grouped["defect"] = train_grouped["defect"].apply(lambda x: [d for d in x if d in VALID_DEFECTS])
train_grouped["labels"] = train_grouped.apply(lambda row: row["component"] + row["defect"], axis=1)
test_grouped = test_df.groupby("ID").agg({
    "component": lambda x: list(set([item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])])),
    "defect": lambda x: list(set([item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])]))
}).reset_index()
# Filter to only valid labels
test_grouped["component"] = test_grouped["component"].apply(lambda x: [c for c in x if c in VALID_COMPONENTS])
test_grouped["defect"] = test_grouped["defect"].apply(lambda x: [d for d in x if d in VALID_DEFECTS])
test_grouped["labels"] = test_grouped.apply(lambda row: row["component"] + row["defect"], axis=1)
# Encode labels
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_grouped["labels"])
y_test = mlb.transform(test_grouped["labels"])
label_classes = list(mlb.classes_)
print(f"Classes: {label_classes}")
print(f"Total classes: {len(label_classes)}")

# STEP 2.5: CALCULATE CLASS WEIGHTS FOR IMBALANCED DATA
# Calculate class frequency and inverse weights
class_counts = y_train.sum(axis=0)
total_samples = len(y_train)
class_weights = total_samples / (len(label_classes) * np.maximum(class_counts, 1))
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("\nClass Distribution:")
for i, label in enumerate(label_classes):
    print(f"  {label}: {class_counts[i]} samples (weight: {class_weights[i]:.4f})")

# STEP 3: CUSTOM DATASET
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

# STEP 4: TRANSFORMS AND DATALOADERS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = ImageDataset("images/train", train_grouped, y_train, transform=transform)
test_dataset = ImageDataset("images/test", test_grouped, y_test, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# STEP 5: DEFINE MODEL
num_classes = len(label_classes)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# replaced sequential with sigmoid -> output raw logits (use BCEWithLogitsLoss)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# STEP 6: INCREMENTAL TRAINING SUPPORT
def load_partial_state_dict(model, checkpoint_path, device):
    """
    Load only those keys from checkpoint whose shapes match the model.
    Returns (loaded_any, skipped_info_list).
    """
    if not os.path.exists(checkpoint_path):
        return False, []
    ckpt = torch.load(checkpoint_path, map_location=device)
    # support checkpoints saved as {'state_dict': state} or raw state_dict
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state:
            if isinstance(v, torch.Tensor) and v.size() == model_state[k].size():
                filtered_state[k] = v
            else:
                skipped.append((k, getattr(v, "size", lambda: None)(), model_state[k].size()))
        else:
            # key not present in current model (e.g., old prefixes) -> skip
            skipped.append((k, getattr(v, "size", lambda: None)(), None))
    if not filtered_state:
        return False, skipped
    # load matching keys (strict=False allows missing keys)
    model.load_state_dict(filtered_state, strict=False)
    return True, skipped
if os.path.exists(MODEL_PATH):
    print("Loading existing model (partial) for incremental training...")
    loaded, skipped_info = load_partial_state_dict(model, MODEL_PATH, device)
    if loaded:
        print("Loaded matching weights from checkpoint.")
        if skipped_info:
            print("Skipped keys (shape mismatch or missing in current model):")
            for k, src_shape, tgt_shape in skipped_info:
                print(f"  {k}: checkpoint_shape={src_shape}  model_shape={tgt_shape}")
            print("Final classifier layer kept as defined for current number of classes.")
    else:
        print("No compatible weights found in checkpoint (or checkpoint missing). Starting from current model initialization.")
else:
    print("No existing model found. Training from scratch...")

# STEP 7: TRAINING LOOP (with weighted BCE loss)
# Use BCEWithLogitsLoss (more stable) and pass per-class pos_weight
# class_weights already on device and shaped (num_classes,)
pos_weight = class_weights  # per-class positive weights
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)
if EPOCHS > 0:
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved successfully.")
else:
    print("Skipping training (EPOCHS = 0). Using existing model for predictions only.")

# STEP 8: FIND OPTIMAL THRESHOLD PER CLASS (using test set as validation)
print("\nFinding optimal thresholds per class...")
model.eval()
all_probs = []
with torch.no_grad():
    for images, _, _ in test_loader:
        images = images.to(device)
        outputs = model(images)               # logits
        probs = torch.sigmoid(outputs)        # convert to probabilities
        all_probs.append(probs.cpu().numpy())
all_probs = np.vstack(all_probs)
# Calculate per-class F1 scores at different thresholds
optimal_thresholds = []
for class_idx in range(num_classes):
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.95, 0.05):
        preds_binary = (all_probs[:, class_idx] > threshold).astype(int)
        true_binary = y_test[:, class_idx]
        
        if len(np.unique(true_binary)) > 1:  # Only compute if both classes present
            f1 = f1_score(true_binary, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    optimal_thresholds.append(best_threshold)
optimal_thresholds = np.array(optimal_thresholds)
print("Optimal thresholds per class:")
for i, label in enumerate(label_classes):
    print(f"  {label}: {optimal_thresholds[i]:.2f}")

# STEP 9: PREDICTION ON TEST SET WITH OPTIMAL THRESHOLDS
model.eval()
preds_all = []
predictions = []
with torch.no_grad():
    for images, _, img_ids in test_loader:
        images = images.to(device)
        outputs = model(images)                # logits
        probs = torch.sigmoid(outputs).cpu().numpy()   # probabilities
        
        # Apply per-class optimal thresholds
        preds = (probs > optimal_thresholds).astype(int)
        preds_all.extend(preds)
        
        for i, img_id in enumerate(img_ids):
            prob_row = probs[i]
            pred_indices = [j for j in range(len(label_classes)) if preds[i][j] == 1]
            pred_labels = [label_classes[j] for j in pred_indices]
            
            pred_components = [lbl for lbl in pred_labels if lbl in VALID_COMPONENTS]
            pred_defects = [lbl for lbl in pred_labels if lbl in VALID_DEFECTS]
            
            comp_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_components]
            defect_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_defects]
            
            avg_comp_conf = float(sum([prob_row[label_classes.index(lbl)] for lbl in pred_components]) / len(pred_components)) if pred_components else 0.0
            avg_def_conf = float(sum([prob_row[label_classes.index(lbl)] for lbl in pred_defects]) / len(pred_defects)) if pred_defects else 0.0
            overall_avg_conf = float((avg_comp_conf + avg_def_conf) / 2.0) if (pred_components or pred_defects) else 0.0
            
            predictions.append({
                "ID": img_id,
                "components": ";".join(pred_components),
                "defects": ";".join(pred_defects),
                "components_confidence": ";".join(comp_conf_pairs),
                "defects_confidence": ";".join(defect_conf_pairs),
                "avg_component_confidence": round(avg_comp_conf, 4),
                "avg_defect_confidence": round(avg_def_conf, 4),
                "overall_avg_confidence": round(overall_avg_conf, 4)
            })
# Save predictions to CSV
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(PREDICTIONS_FILE, index=False)
print("\nSample predictions:")
print(pred_df.head())

# STEP 10: COMPREHENSIVE EVALUATION METRICS
preds_all = np.array(preds_all)
print("\n" + "="*70)
print("EVALUATION METRICS (Multi-Label Classification)")
print("="*70)
# Confidence metrics
all_avg_comp_conf = [p["avg_component_confidence"] for p in predictions]
all_avg_def_conf = [p["avg_defect_confidence"] for p in predictions]
all_overall_conf = [p["overall_avg_confidence"] for p in predictions]
mean_comp_conf = np.mean(all_avg_comp_conf) if all_avg_comp_conf else 0.0
mean_def_conf = np.mean(all_avg_def_conf) if all_avg_def_conf else 0.0
mean_overall_conf = np.mean(all_overall_conf) if all_overall_conf else 0.0
# avg_confidence_score requested (average of multi-head predictions per image)
avg_confidence_score = mean_overall_conf
print(f"\nConfidence Metrics:")
print(f"  Mean Component Detection Confidence: {mean_comp_conf:.4f}")
print(f"  Mean Defect Detection Confidence: {mean_def_conf:.4f}")
print(f"  Average Confidence Score: {avg_confidence_score:.4f}")
# Set-based metrics
hl = hamming_loss(y_test, preds_all)
f1_micro = f1_score(y_test, preds_all, average='micro', zero_division=0)
f1_macro = f1_score(y_test, preds_all, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, preds_all, average='weighted', zero_division=0)
print(f"\nSet-Based Metrics:")
print(f"  Hamming Loss: {hl:.4f}")
print(f"  F1 Micro (Global): {f1_micro:.4f}")
print(f"  F1 Macro (Per-Label Average): {f1_macro:.4f}")
print(f"  F1 Weighted (Class-Weighted): {f1_weighted:.4f}")
# Average Jaccard Index (intersection over union) per sample
# safe check: ensure shapes align
if preds_all.shape[0] != y_test.shape[0]:
    print("\nWarning: number of predictions does not match number of ground-truth samples. Skipping Jaccard computation.")
else:
    jaccard_scores = []
    for i in range(y_test.shape[0]):
        gt = y_test[i].astype(int)
        pr = preds_all[i].astype(int)
        intersection = float(np.logical_and(gt, pr).sum())
        union = float(np.logical_or(gt, pr).sum())
        if union == 0:
            # define Jaccard as 1.0 when both are empty (no labels) to avoid penalizing correct empties
            jaccard = 1.0
        else:
            jaccard = intersection / union
        jaccard_scores.append(jaccard)
    avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    print(f"\nAverage Jaccard Index (per-sample IoU): {avg_jaccard:.4f}")
# Per-class F1 scores
print(f"\nPer-Class F1 Scores:")
f1_scores_per_class = f1_score(y_test, preds_all, average=None, zero_division=0)
for i, label in enumerate(label_classes):
    print(f"  {label}: {f1_scores_per_class[i]:.4f}")


'''
For this code:
• Are these metrics good? No — overall performance is low. Hamming loss and Average Jaccard indicate many incorrect / missing labels; only a few classes (e.g., Drummy) perform reasonably.
• Do you need to retrain? Probably — but first verify label issues and thresholds (see diagnostics). Retrain after fixes and targeted improvements.
What each metric means (one-line each)
• Average Confidence Score: mean of per-image predicted confidences (your stakeholder-friendly number).
• Hamming Loss: fraction of label bits incorrect (lower is better). 0.5361 means ≈53% of label assignments are wrong.
• F1 Micro: global F1 treating every label equally across samples (sensitive to frequent classes).
• F1 Macro: average F1 across classes (treats each class equally).
• F1 Weighted: F1 averaged with class-frequency weighting.
• Average Jaccard Index (per-sample IoU): mean intersection/union of predicted vs. true label sets (higher is better). 0.20 is low.
• Per-class F1: F1 for every class; highlights which classes are learned vs. failing.
Quick diagnosis checklist (do these first)
1. Verify label set: print mlb.classes_ — ensure component labels are present. Your Mean Component Confidence = 0.0 suggests components may not be in label_classes or never predicted.
2. Inspect per-class prediction probabilities distribution (histograms) to see if outputs are near 0 or 1.
3. Check class_counts and pos_weight calculation. For BCEWithLogitsLoss, pos_weight should ideally be (neg_count / pos_count) per class.
4. Confirm optimal_thresholds are computed on a proper validation split (not test). Using test for thresholding inflates risk of overfitting thresholds.
5. Print classification reports (precision, recall) and sample false positives/negatives for failing classes (e.g., Loose).
Practical improvements (ordered, keep iterations small)
6. Fix labeling issues first (if components missing) — no retrain needed if it’s only a mapping bug; otherwise retrain once corrected.
7. Use a validation split (not test) to tune per-class thresholds and early stopping.
8. Recompute pos_weight = (N - pos) / pos for BCEWithLogitsLoss (or use focal loss) to better handle rare classes.
9. Data augmentation and oversampling for rare classes (Loose, Coating Failure) or use class-aware sampling.
10. Try a two-head model (one head for components, one for defects) if label semantics differ.
11. Increase epochs, monitor validation loss, unfreeze more backbone layers gradually (transfer learning).
12. Calibrate probabilities (temperature scaling or Platt) if confidence values are needed for stakeholders.
13. Evaluate additional metrics (per-class ROC-AUC, precision@k) and produce a confusion-style analysis per label.
Quick experiment plan (small, actionable)
• Step A: Confirm mlb.classes_ and print counts. If component labels missing, fix preprocessing and re-run.
• Step B: Use a small validation set (10%) to tune thresholds; compute pos_weight = (N-pos)/pos; set EPOCHS=5 and run; compare val F1 and Jaccard.
• Step C: If rare-class F1 still low, add augmentation + oversampling for those classes and retry.
• Step D: If prediction confidences are poorly calibrated, apply temperature scaling on validation.
If you want, I can:
• Provide a short checklist script to dump mlb.classes_, per-class counts, and probability histograms.
• Suggest exact code changes to pos_weight computation and validation split.'''
