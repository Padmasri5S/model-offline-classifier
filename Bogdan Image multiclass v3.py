"""
Multi-Label Image Classification Model - Offline Version 2

This script implements a multi-label image classification pipeline using ResNet18 for detecting
components and defects in industrial/structural images. It includes:
  - Data loading, cleaning, and splitting
  - Model training with early stopping and class weighting
  - Threshold tuning for optimal F1 scores
  - Prediction generation with confidence metrics
  - Comprehensive evaluation metrics (F1, Hamming Loss, Jaccard Index)

Author: [Your Name]
Date: November 17, 2025
"""

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

# ============================================================================
# CONFIGURATION
# ============================================================================
"""
Model and data pipeline configuration parameters.
"""
IMG_DIR = "images"  # Root directory containing all images
TRAIN_DIR = "images/train"  # Training set directory
VAL_DIR = "images/val"  # Validation set directory
TEST_DIR = "images/test"  # Test set directory
CSV_FILE = "image_data.csv"  # CSV metadata file
JSON_FILE = "image_data.json"  # JSON metadata file
MODEL_PATH = "model.pth"  # Path to save/load trained model
PREDICTIONS_FILE = "predictions.csv"  # Output predictions CSV
BATCH_SIZE = 16  # Batch size for dataloaders
EPOCHS = 10  # Maximum training epochs
LR = 0.001  # Learning rate for Adam optimizer
PATIENCE = 3  # Early stopping patience (epochs without improvement)

# ============================================================================
# DEVICE SETUP
# ============================================================================
"""
Initialize device for GPU/CPU training.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# STEP 1: LOAD AND CLEAN METADATA
# ============================================================================
"""
Load metadata from CSV and JSON files, then clean and validate labels.
"""
csv_data = pd.read_csv(CSV_FILE)
with open(JSON_FILE, "r") as f:
    json_data = json.load(f)
metadata = csv_data


def clean_labels(label_list):
    """
    Clean and normalize label strings.
    
    Args:
        label_list: String (semicolon-separated), list, or other format containing labels
        
    Returns:
        List of cleaned, lowercase labels with invalid entries removed
    """
    if isinstance(label_list, str):
        labels = [l.strip().lower() for l in str(label_list).split(";") if l.strip()]
    elif isinstance(label_list, list):
        labels = [l.lower().strip() for l in label_list]
    else:
        labels = []
    return [l for l in labels if l not in {"component", "defect", "", "nan"}]


# Clean defect column (contains all labels in dataset)
metadata["defect"] = metadata["defect"].apply(clean_labels)

# Define valid component and defect categories (normalized to lowercase)
VALID_COMPONENTS = {c.lower() for c in [
    "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
    "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support",
    "Centre Support-Bolt", "Channel Bracket-Bolt", "Conveyor Support-Bolt", "Walkway Support-Bolt"
]}
VALID_DEFECTS = {d.lower() for d in [
    "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface", "Loose", "Missing", "Drummy", "Leaks"
]}


def split_labels(label_list):
    """
    Categorize labels into components and defects based on valid sets.
    
    Args:
        label_list: List of labels to categorize
        
    Returns:
        Tuple of (components_list, defects_list)
    """
    comps = [l for l in label_list if l in VALID_COMPONENTS]
    defs = [l for l in label_list if l in VALID_DEFECTS]
    return comps, defs


# Split labels into components and defects
metadata["component"], metadata["defect"] = zip(*metadata["defect"].apply(split_labels))

# Remove rows with no valid labels
metadata = metadata[(metadata["component"].apply(len) > 0) | (metadata["defect"].apply(len) > 0)]

# Print sanity check statistics
print(f"Total rows after cleaning: {len(metadata)}")
print(f"Unique components: {len(set(metadata['component'].explode()))}")
print(f"Unique defects: {len(set(metadata['defect'].explode()))}")

# ============================================================================
# DATA SPLITTING AND IMAGE ORGANIZATION
# ============================================================================
"""
Split data into train (70%), validation (15%), and test (15%) sets,
then organize images into respective directories.
"""
unique_ids = metadata["ID"].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

train_df = metadata[metadata["ID"].isin(train_ids)]
val_df = metadata[metadata["ID"].isin(val_ids)]
test_df = metadata[metadata["ID"].isin(test_ids)]

# Create directories if they don't exist
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)


def move_images(df, folder):
    """
    Copy images from source directory to target dataset folder.
    
    Args:
        df: DataFrame containing image IDs
        folder: Destination folder path
    """
    for img_id in df["ID"]:
        src = os.path.join(IMG_DIR, f"{img_id}.jpg")
        dst = os.path.join(folder, f"{img_id}.jpg")
        if os.path.exists(src):
            shutil.copy(src, dst)


move_images(train_df, TRAIN_DIR)
move_images(val_df, VAL_DIR)
move_images(test_df, TEST_DIR)


def filter_existing_images(df, folder):
    """
    Filter dataset to only include images that exist in the folder.
    
    Args:
        df: DataFrame with image metadata
        folder: Folder path to check for existing images
        
    Returns:
        Filtered DataFrame containing only IDs with existing images
    """
    existing_ids = {os.path.splitext(f)[0] for f in os.listdir(folder)}
    return df[df["ID"].isin(existing_ids)]


# Filter to ensure all images exist
train_df = filter_existing_images(train_df, TRAIN_DIR)
val_df = filter_existing_images(val_df, VAL_DIR)
test_df = filter_existing_images(test_df, TEST_DIR)

# ============================================================================
# STEP 2: GROUP AND ENCODE LABELS
# ============================================================================
"""
Group labels by image ID and convert to multi-label binary format.
"""


def group_labels(df):
    """
    Aggregate labels by image ID and combine components and defects.
    
    Args:
        df: DataFrame with labels per image
        
    Returns:
        Grouped DataFrame with consolidated labels per image
    """
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

# Multi-label binarization: convert label lists to binary matrix
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_grouped["labels"])
y_val = mlb.transform(val_grouped["labels"])
y_test = mlb.transform(test_grouped["labels"])
label_classes = list(mlb.classes_)
print(f"Classes: {label_classes}")

# Consistency check with saved classes
if os.path.exists("classes.json"):
    with open("classes.json", "r") as f:
        saved_classes = json.load(f)
    if list(label_classes) != saved_classes:
        print("Warning: current label_classes differ from saved classes.json. "
              "Ensure consistent class order to avoid mislabeling.")

# ============================================================================
# STEP 2.5: CLASS WEIGHTS
# ============================================================================
"""
Calculate class weights to handle imbalanced labels.
Uses inverse frequency weighting: pos_weight = (neg_count / pos_count)
"""
class_counts = y_train.sum(axis=0)
neg_counts = len(y_train) - class_counts
pos_weight = torch.tensor(neg_counts / np.maximum(class_counts, 1), dtype=torch.float32).to(device)

# ============================================================================
# STEP 3: DATASET & DATALOADERS
# ============================================================================
"""
Define custom PyTorch Dataset and create DataLoaders with augmentation.
"""


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-label image classification.
    
    Attributes:
        img_dir: Directory containing images
        df: DataFrame with image metadata (including IDs)
        labels: Binary label matrix (samples x classes)
        transform: torchvision transforms to apply to images
    """
    def __init__(self, img_dir, df, labels, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return total number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load and return image, labels, and image ID.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Tuple of (image_tensor, label_vector, image_id)
        """
        img_id = self.df.iloc[idx]["ID"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label, img_id


# Define image transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet18 input size
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(10),  # Data augmentation
    transforms.ToTensor()
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create datasets
train_dataset = ImageDataset(TRAIN_DIR, train_grouped, y_train, transform=train_transform)
val_dataset = ImageDataset(VAL_DIR, val_grouped, y_val, transform=val_transform)
test_dataset = ImageDataset(TEST_DIR, test_grouped, y_test, transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# STEP 4: MODEL INITIALIZATION
# ============================================================================
"""
Initialize ResNet18 model with pretrained ImageNet weights and adapt for multi-label classification.
"""
num_classes = len(label_classes)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# Replace final fully connected layer for multi-label output
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


def load_partial_state_dict(model, checkpoint_path, device):
    """
    Load model weights selectively, handling size mismatches.
    Useful for resuming training with different label sets.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to saved checkpoint
        device: Device (CPU/GPU) to load onto
        
    Returns:
        Tuple of (success: bool, skipped_keys: list)
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


# Load existing model if available
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    load_partial_state_dict(model, MODEL_PATH, device)

# ============================================================================
# STEP 5: TRAINING LOOP WITH EARLY STOPPING
# ============================================================================
"""
Train model with BCEWithLogitsLoss (multi-label), Adam optimizer, and early stopping.
Saves best model based on validation loss.
"""
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
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

    # Validation phase
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

    # Early stopping logic
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

# ============================================================================
# STEP 6: THRESHOLD TUNING ON VALIDATION SET
# ============================================================================
"""
Optimize prediction thresholds per class on validation set to maximize F1 scores.
Default threshold of 0.5 may not be optimal for imbalanced labels.
"""
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

# Find optimal threshold for each class
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

# ============================================================================
# STEP 7: PREDICTION ON TEST SET WITH CONFIDENCE METRICS
# ============================================================================
"""
Generate predictions on test set with per-label and overall confidence scores.
Separates predictions into components and defects for better interpretability.
"""
predictions = []
preds_all = []
with torch.no_grad():
    for images, _, img_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > optimal_thresholds).astype(int)
        preds_all.extend(preds)

        for i, img_id in enumerate(img_ids):
            prob_row = probs[i]
            pred_indices = [j for j in range(len(label_classes)) if preds[i][j] == 1]
            pred_labels = [label_classes[j] for j in pred_indices]

            # Separate components and defects
            pred_components = [lbl for lbl in pred_labels if lbl in VALID_COMPONENTS]
            pred_defects = [lbl for lbl in pred_labels if lbl in VALID_DEFECTS]

            # Format confidence pairs (label:confidence)
            comp_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_components]
            defect_conf_pairs = [f"{lbl}:{prob_row[label_classes.index(lbl)]:.4f}" for lbl in pred_defects]

            # Calculate average confidence per category
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
print("\nSample predictions with confidence:")
print(pred_df.head())

# Print confidence summary statistics
all_avg_comp_conf = [p["avg_component_confidence"] for p in predictions]
all_avg_def_conf = [p["avg_defect_confidence"] for p in predictions]
all_overall_conf = [p["overall_avg_confidence"] for p in predictions]
print("\nConfidence Metrics:")
print(f"Mean Component Confidence: {np.mean(all_avg_comp_conf):.4f}")
print(f"Mean Defect Confidence: {np.mean(all_avg_def_conf):.4f}")
print(f"Overall Average Confidence: {np.mean(all_overall_conf):.4f}")

# ============================================================================
# STEP 8: EVALUATION METRICS
# ============================================================================
"""
Compute comprehensive multi-label classification metrics on test set.
Includes: Hamming Loss, F1 (micro/macro/weighted), and Jaccard Index.
"""
preds_all = np.array(preds_all)
print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

# Hamming Loss: fraction of incorrectly predicted labels
hl = hamming_loss(y_test, preds_all)

# F1 Scores at different averages
f1_micro = f1_score(y_test, preds_all, average='micro', zero_division=0)  # Global TP, FP, FN
f1_macro = f1_score(y_test, preds_all, average='macro', zero_division=0)  # Average of per-class F1
f1_weighted = f1_score(y_test, preds_all, average='weighted', zero_division=0)  # Weighted by support

print(f"Hamming Loss: {hl:.4f}")
print(f"F1 Micro: {f1_micro:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")

# Jaccard Index (Intersection over Union): measures label prediction overlap
jaccard_scores = []
for i in range(y_test.shape[0]):
    gt = y_test[i].astype(int)
    pr = preds_all[i].astype(int)
    intersection = float(np.logical_and(gt, pr).sum())
    union = float(np.logical_or(gt, pr).sum())
    jaccard_scores.append(1.0 if union == 0 else intersection / union)
avg_jaccard = np.mean(jaccard_scores)
print(f"Average Jaccard Index: {avg_jaccard:.4f}")

# Per-class F1 scores for detailed analysis
f1_scores_per_class = f1_score(y_test, preds_all, average=None, zero_division=0)
print("\nPer-Class F1 Scores:")
for i, label in enumerate(label_classes):
    print(f"  {label}: {f1_scores_per_class[i]:.4f}")

'''
It’s done — and the results look great! Here’s a summary of everything we did:

1. Initial Problem

Your original pipeline only predicted defects, not components.
mlb.classes_ contained only defect labels because the component column was empty or invalid.
Training was slow and metrics were poor (Hamming Loss ~0.53, Jaccard ~0.20).


2. Root Cause

The dataset had all labels (components + defects) in one column.
The component column was empty, so filtering removed all component labels.
VALID sets didn’t match the actual data format (case sensitivity, mixed columns).


3. Fixes Implemented
✅ Normalization & Splitting Logic

Converted all labels to lowercase and stripped spaces.
Used VALID_COMPONENTS and VALID_DEFECTS to split labels correctly from the single column.
Removed rows with no valid labels.

✅ Data Cleaning

Ensured only valid components and defects remain.
Printed summary counts instead of nan spam:
Total rows after cleaning: 3352
Unique components: 1582
Unique defects: 1790



✅ Model Adjustments

Confirmed mlb.classes_ now includes 20 classes (13 components + 7 defects).
Added confidence metrics and optimal threshold tuning.
Implemented early stopping for efficient training.

✅ Prediction Output

Predictions now include:

Components and defects names
Confidence scores per label
Average confidence per image


Saved to CSV with detailed columns.

✅ Evaluation Metrics

Hamming Loss: 0.0835 (↓ from 0.53)
F1 Micro: 0.6891
F1 Macro: 0.5674
Average Jaccard: 0.5957
Confidence Metrics:

Mean Component Confidence: 0.8374
Mean Defect Confidence: 0.8737
Overall Average Confidence: 0.8556



✅ Per-Class F1 Highlights

Components like channel bracket-bolt (0.75), grout hole (0.76), radial connection (0.73) are strong.
Defects like corrosion-heavy (0.84), drummy (0.87), leaks (0.81) are excellent.
Some rare classes (e.g., centre support-bolt, radial joint) need improvement.


4. Current Status
✔ Components and defects are correctly predicted.
✔ Confidence metrics and evaluation are meaningful.
✔ Model performance is much better than before.

Next Recommendations

Improve rare classes:

Oversample or augment images for low-frequency labels.


Train longer or unfreeze more layers for better feature learning.
Consider focal loss for imbalance.
Deploy GPU for faster training if available.


✅ You now have a working multi-label image classifier that outputs:

Components + defects
Confidence scores
Evaluation metrics

PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\imgannotp\label-studio-ml-backend> cd .\my_ml_backend\      
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\imgannotp\label-studio-ml-backend\my_ml_backend> python .\setup_backend.py
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\imgannotp\label-studio-ml-backend\my_ml_backend> cd ..
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\imgannotp\label-studio-ml-backend> label-studio-ml start my_ml_backend --port 9090
--------------------------------------------------------------------
Here's How to Prove the Model is Learning:
1. Check Training History File
powershellcd my_ml_backend
cat training_history.json
# Or open in notepad
notepad training_history.json
You should see records like:
json[
  {
    "timestamp": "2025-11-25T15:55:10",
    "event": "ANNOTATION_CREATED",
    "num_samples": 1,
    "num_epochs": 3,
    "learning_rate": 0.0001,
    "final_loss": 0.2490,
    "f1_micro": 0.5000,
    "f1_macro": 0.0500,
    "task_ids": [123],
    "thresholds_recomputed": true
  }
]
2. Test Predictions Before & After Training
Step A: Get a baseline prediction

Open an unlabeled image in Label Studio
Click "Retrieve Predictions"
Screenshot or note which labels it predicts

Step B: Correct the prediction
4. Fix the labels (add/remove as needed)
5. Submit
Step C: Test on the SAME image again
6. Go back to that same image (click back button)
7. Click "Retrieve Predictions" again
8. Compare - the predictions should now match your correction!
3. Test on Similar Images
The best test:

Label 5-10 images that all have "Centre Support"
Then open a NEW unlabeled image with "Centre Support"
Click "Retrieve Predictions"
The model should now predict "Centre Support" with higher confidence

4. Watch Loss Decrease Over Time
As you label more images, watch the terminal logs. You should see:

First few images: Loss ~0.4-0.6
After 10 images: Loss ~0.3-0.4
After 20 images: Loss ~0.2-0.3
After 50+ images: Loss ~0.1-0.2

5. Monitor Model File Timestamp
powershellcd my_ml_backend
dir model.pth
The timestamp updates every time you submit an annotation (showing it's being retrained).

Quick Experiment to Prove Learning
Try this 3-step test:
Test 1: Baseline (Random Model)

Open image 4aa0-40db-b3db-d0eb05ba8de1.jpg (the one you just labeled)
Note it predicted: ['Circular Connection', 'Drummy']

Test 2: Train on 3 More Similar Images
3. Label 3 more images with "Circular Connection"
4. Submit all 3
Test 3: Retest Original Image
5. Go back to 4aa0-40db-b3db-d0eb05ba8de1.jpg
6. Delete your annotation (or reload predictions)
7. Click "Retrieve Predictions"
8. You should see "Circular Connection" predicted with HIGHER confidence now

Visual Proof: Plot Training Progress
Create this script in my_ml_backend/:
python# plot_training.py
import json
import matplotlib.pyplot as plt

with open('training_history.json', 'r') as f:
    history = json.load(f)

losses = [h['final_loss'] for h in history]
f1_scores = [h['f1_micro'] for h in history]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, marker='o')
plt.xlabel('Training Session')
plt.ylabel('Loss')
plt.title('Training Loss (Lower is Better)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f1_scores, marker='o', color='green')
plt.xlabel('Training Session')
plt.ylabel('F1 Micro Score')
plt.title('Model Performance (Higher is Better)')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_progress.png')
print("✓ Saved training_progress.png")
Run it:
powershellpip install matplotlib
python plot_training.py
This creates a graph showing your model improving over time!

Expected Behavior
After labeling 20-30 images, you should see:

Correct predictions appearing automatically on similar images
Loss values dropping from ~0.5 to ~0.1
F1 scores increasing from ~0.2 to ~0.7+
Confidence scores getting higher on correct predictions


Your Current Stats
From your logs:

✅ 3 training sessions completed
✅ Loss improving: 0.2685 → 0.2490 (7% reduction in 3 epochs)
⚠️ F1 Macro is low (0.05) - Normal! You've only trained on 1-3 images. Needs 20+ samples

Keep labeling! After 20 images, you'll see dramatic improvements.
Want me to help you create an automated test that labels 10 images, then checks if predictions improve?
'''
