
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_all_images.py
Runs inference across ALL images in IMG_DIR and writes one CSV row per image:
- ID (filename stem)
- components_dict (JSON {component: confidence})
- defects_dict    (JSON {defect: confidence})
- components_list (semicolon list)
- defects_list    (semicolon list)
- all_scores_json (JSON {label: confidence}) for full auditability
Notes:
- Uses classes.json to preserve label ordering from training.
- Uses thresholds.json (from your PR-curve tuning). Defaults to 0.5 if missing.
- Handles multiple image extensions (.jpg/.jpeg/.png/.bmp/.tiff).
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
# ----------------------------
# CONFIG (adjust if needed)
# ----------------------------
IMG_DIR = "images"
MODEL_PATH = "model.pth"
CLASSES_JSON = "classes.json"
THRESHOLDS_JSON = "thresholds.json"
OUT_CSV = "predictions_all_images.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0  # set >0 if you want multi-process loading locally
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
# Keep consistent with your training script
VALID_COMPONENTS = {
    "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
    "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support",
    "Centre Support-Bolt", "Channel Bracket-Bolt", "Conveyor Support-Bolt", "Walkway Support-Bolt"
}
VALID_DEFECTS = {
    "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface", "Loose", "Missing", "Drummy", "Leaks"
}
# Same eval transform as training script
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# ----------------------------
# DATASET: all images in IMG_DIR
# ----------------------------
class AllImagesDataset(Dataset):
    def __init__(self, img_dir: str, valid_ext: set, transform=None):
        self.transform = transform
        self.items: List[Dict] = []
        root = Path(img_dir)
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in valid_ext:
                self.items.append({"path": str(p), "id": p.stem})
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {img_dir} with extensions {valid_ext}")
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec["id"], rec["path"]
# ----------------------------
# MODEL LOADING: ResNet18 head
# ----------------------------
def load_model(num_classes: int, model_path: str, device: torch.device):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    # allow either pure state_dict or checkpoint dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
# ----------------------------
# UTILS
# ----------------------------
def load_classes(classes_json: str) -> List[str]:
    if not os.path.exists(classes_json):
        raise FileNotFoundError(f"{classes_json} not found; you must save class order during training.")
    with open(classes_json, "r") as f:
        classes = json.load(f)
    if not isinstance(classes, list):
        raise ValueError("classes.json should contain a JSON list of class names.")
    return classes
def load_thresholds(thresholds_json: str, num_classes: int) -> torch.Tensor:
    if os.path.exists(thresholds_json):
        with open(thresholds_json, "r") as f:
            arr = json.load(f)
        if not isinstance(arr, list) or len(arr) != num_classes:
            print("WARNING: thresholds.json shape mismatch; defaulting all thresholds to 0.5")
            return torch.full((num_classes,), 0.5, dtype=torch.float32)
        return torch.tensor(arr, dtype=torch.float32)
    else:
        print("WARNING: thresholds.json not found; defaulting all thresholds to 0.5")
        return torch.full((num_classes,), 0.5, dtype=torch.float32)
def split_types(labels: List[str]):
    comps = [l for l in labels if l in VALID_COMPONENTS]
    defs  = [l for l in labels if l in VALID_DEFECTS]
    return comps, defs
# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=IMG_DIR)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--classes_json", default=CLASSES_JSON)
    parser.add_argument("--thresholds_json", default=THRESHOLDS_JSON)
    parser.add_argument("--out_csv", default=OUT_CSV)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load class order + thresholds
    label_classes = load_classes(args.classes_json)
    num_classes = len(label_classes)
    thresholds = load_thresholds(args.thresholds_json, num_classes).to(device)
    # Dataset / Loader
    ds = AllImagesDataset(args.images_dir, VALID_EXT, transform=EVAL_TRANSFORM)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    # Model
    model = load_model(num_classes, args.model_path, device)
    # Inference loop
    out_rows = []
    with torch.no_grad():
        for images, ids, paths in dl:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)                 # [B, C]
            preds = (probs > thresholds).cpu().numpy()    # [B, C]
            probs_np = probs.cpu().numpy()
            for i in range(len(ids)):
                img_id = ids[i]
                prob_row = probs_np[i]
                pred_mask = preds[i].astype(bool)
                # Build dict of predicted labels with confidence
                pred_labels = [label_classes[j] for j, m in enumerate(pred_mask) if m]
                pred_components, pred_defects = split_types(pred_labels)
                components_dict = {lbl: float(prob_row[label_classes.index(lbl)]) for lbl in pred_components}
                defects_dict    = {lbl: float(prob_row[label_classes.index(lbl)]) for lbl in pred_defects}
                # Optional: full label→score map for auditability
                all_scores_dict = {lbl: float(prob_row[j]) for j, lbl in enumerate(label_classes)}
                out_rows.append({
                    "ID": img_id,
                    "components_dict": json.dumps(components_dict),
                    "defects_dict": json.dumps(defects_dict),
                    "components_list": ";".join(pred_components),
                    "defects_list": ";".join(pred_defects),
                    "all_scores_json": json.dumps(all_scores_dict),
                    "image_path": paths[i],  # handy for tracebacks
                })
    # Save CSV with 3321 rows (one per image if present)
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions for {len(df)} images to {args.out_csv}")
if __name__ == "__main__":
    main()

'''
# Confidence guidance (recommended starting points, tune per dataset)
# ---------------------------------------------------------------------
# Components (larger, more visually persistent):
# - Walkway Support / Conveyor Support / Centre Support .......... 0.50–0.60
# - Channel Bracket .............................................. 0.55–0.65
# - Grout Hole (may be small aperture) ........................... 0.55–0.65
# - Radial / Circular Connection / Joint (structural interfaces) . 0.60–0.70
# - *-Bolt classes (small objects) ................................ 0.65–0.75
#
# Defects (risk-sensitive, aim to reduce false positives):
# - Corrosion-Heavy .............................................. 0.65–0.75
# - Coating Failure .............................................. 0.55–0.65
# - Corrosion-Surface ............................................ 0.45–0.55
# - Leaks (texture/gloss cues can be subtle) ..................... 0.55–0.70
# - Missing / Loose (discrete object absence/presence) ........... 0.60–0.75
# - Drummy (visual proxy; if only image cues) .................... 0.50–0.60
#
# Why: components are often spatially larger and more consistent,
# while defect classes vary in granularity and appearance, so thresholds
# err higher for “safety-critical” types (Missing, Loose, Heavy Corrosion).
#
# Final tip: keep using your PR-curve-derived thresholds.json as the source
# of truth. Use these ranges for manual overrides only if validation curves
# are unstable. Consider calibrating logits via temperature scaling if
# probabilities look over/under-confident.

PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model> python predict_all_images.py
Using device: cpu
Saved predictions for 3321 images to predictions_all_images.csv
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model>

Outputs
ID	components_dict	defects_dict	components_list	defects_list	image_path
Sample
008f1080-a442-4fee-b66a-327d55730927	{"Conveyor Support": 0.35695481300354004, "Conveyor Support-Bolt": 0.9627693891525269, "Walkway Support-Bolt": 0.9323155283927917}	{"Corrosion-Heavy": 0.6942402124404907}	Conveyor Support;Conveyor Support-Bolt;Walkway Support-Bolt	Corrosion-Heavy	images\008f1080-a442-4fee-b66a-327d55730927.jpg

Requires further refinement of output format and streamlined confidence scores.
'''
