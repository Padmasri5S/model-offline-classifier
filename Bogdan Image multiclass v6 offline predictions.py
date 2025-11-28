#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_all_images.py (slim output)
- Emits labels only (no per-label scores) + average confidences by type.
- Works across ALL images in IMG_DIR (annotations not required).
"""
import os
import json
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd
# ----------------------------
# CONFIG
# ----------------------------
IMG_DIR = "images"
MODEL_PATH = "model.pth"
CLASSES_JSON = "classes.json"
THRESHOLDS_JSON = "thresholds.json"
OUT_CSV = "predictions_all_images.csv"
BATCH_SIZE = 32
NUM_WORKERS = 0
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VALID_COMPONENTS = {
    "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
    "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support",
    "Centre Support-Bolt", "Channel Bracket-Bolt", "Conveyor Support-Bolt", "Walkway Support-Bolt"
}
VALID_DEFECTS = {
    "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface", "Loose", "Missing", "Drummy", "Leaks"
}
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# ----------------------------
# DATASET
# ----------------------------
class AllImagesDataset(Dataset):
    def __init__(self, img_dir: str, valid_ext: set, transform=None):
        self.transform = transform
        self.items = []
        root = Path(img_dir)
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in valid_ext:
                self.items.append({"path": str(p.resolve()), "id": p.stem})
        if not self.items:
            raise RuntimeError(f"No images found in {img_dir} with {valid_ext}")
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec["id"], rec["path"]
# ----------------------------
# MODEL / UTILS
# ----------------------------
def load_classes(classes_json: str) -> List[str]:
    with open(classes_json, "r") as f:
        classes = json.load(f)
    if not isinstance(classes, list):
        raise ValueError("classes.json must be a list of class names.")
    return classes
def load_thresholds(thresholds_json: str, num_classes: int) -> torch.Tensor:
    if os.path.exists(thresholds_json):
        with open(thresholds_json, "r") as f:
            arr = json.load(f)
        if isinstance(arr, list) and len(arr) == num_classes:
            return torch.tensor(arr, dtype=torch.float32)
        print("WARNING: thresholds.json length mismatch; defaulting to 0.5")
    else:
        print("WARNING: thresholds.json not found; defaulting to 0.5")
    return torch.full((num_classes,), 0.5, dtype=torch.float32)
def load_model(num_classes: int, model_path: str, device: torch.device):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
def split_types(labels: List[str]):
    components = [l for l in labels if l in VALID_COMPONENTS]
    defects    = [l for l in labels if l in VALID_DEFECTS]
    return components, defects
# ----------------------------
# MAIN
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    label_classes = load_classes(CLASSES_JSON)
    num_classes = len(label_classes)
    thresholds = load_thresholds(THRESHOLDS_JSON, num_classes).to(device)
    ds = AllImagesDataset(IMG_DIR, VALID_EXT, transform=EVAL_TRANSFORM)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    model = load_model(num_classes, MODEL_PATH, device)
    rows = []
    with torch.no_grad():
        for images, ids, paths in dl:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)  # [B, C]
            preds = (probs > thresholds).cpu().numpy()
            probs_np = probs.cpu().numpy()
            for i in range(len(ids)):
                img_id = ids[i]
                prob_row = probs_np[i]
                mask = preds[i].astype(bool)
                pred_labels = [label_classes[j] for j, m in enumerate(mask) if m]
                pred_components, pred_defects = split_types(pred_labels)
                # Build minimal dicts (presence only)
                components_dict = {lbl: True for lbl in pred_components}
                defects_dict    = {lbl: True for lbl in pred_defects}
                # Averages by type (0.0 if none)
                comp_scores = [float(prob_row[label_classes.index(lbl)]) for lbl in pred_components]
                def_scores  = [float(prob_row[label_classes.index(lbl)]) for lbl in pred_defects]
                avg_comp_conf = round(sum(comp_scores)/len(comp_scores), 4) if comp_scores else 0.0
                avg_def_conf  = round(sum(def_scores)/len(def_scores), 4) if def_scores else 0.0
                rows.append({
                    "ID": img_id,
                    "components_dict": json.dumps(components_dict, ensure_ascii=False),
                    "defects_dict": json.dumps(defects_dict, ensure_ascii=False),
                    "components_list": ";".join(pred_components),
                    "defects_list": ";".join(pred_defects),
                    "avg_component_confidence": avg_comp_conf,
                    "avg_defect_confidence": avg_def_conf,
                    "image_path": paths[i],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
if __name__ == "__main__":
    main()

'''
# Confidence guidance (comment-only; tune with validation)

# Components (larger, more consistent):
#   Walkway/Centre/Conveyor Support ............. threshold ~0.50–0.60
#   Channel Bracket ............................. threshold ~0.55–0.65
#   Grout Hole .................................. threshold ~0.55–0.65
#   Radial/Circular Connection/Joint ............ threshold ~0.60–0.70
#   *-Bolt (small objects) ...................... threshold ~0.65–0.75
#
# Defects (higher stakes, reduce false positives):
#   Corrosion-Heavy ............................. threshold ~0.65–0.75
#   Coating Failure ............................. threshold ~0.55–0.65
#   Corrosion-Surface ........................... threshold ~0.45–0.55
#   Leaks ....................................... threshold ~0.55–0.70
#   Missing / Loose ............................. threshold ~0.60–0.75
#   Drummy ...................................... threshold ~0.50–0.60
#
# Keep your PR-curve thresholds.json as the source of truth; only override
# if validation curves are unstable. Consider temperature scaling if scores
# are systematically over/under-confident.

PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model> python predict_all_images.py
Using device: cpu
Saved 3321 rows to predictions_all_images.csv
PS C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model>

Output generated (first 9 rows)
ID	components_dict	defects_dict	components_list	defects_list	avg_component_confidence	avg_defect_confidence	image_path
00114446-13df-40e8-89ea-3034c75e2da9	{"Radial Connection": true}	{"Drummy": true}	Radial Connection	Drummy	0.9292	0.9495	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\00114446-13df-40e8-89ea-3034c75e2da9.jpg
0013464e-df19-4dfd-8ed6-6957ca31277d	{"Circular Joint": true}	{"Leaks": true}	Circular Joint	Leaks	0.1038	0.5852	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\0013464e-df19-4dfd-8ed6-6957ca31277d.jpg
00551810-d75e-466b-a77d-c2c28abbd958	{}	{}			0	0	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\00551810-d75e-466b-a77d-c2c28abbd958.jpg
008f1080-a442-4fee-b66a-327d55730927	{"Conveyor Support": true, "Conveyor Support-Bolt": true, "Walkway Support-Bolt": true}	{"Corrosion-Heavy": true}	Conveyor Support;Conveyor Support-Bolt;Walkway Support-Bolt	Corrosion-Heavy	0.7507	0.6942	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\008f1080-a442-4fee-b66a-327d55730927.jpg
009befbe-a8bc-4911-92fd-f9a7b327d26b	{"Circular Joint": true, "Grout Hole": true}	{"Leaks": true}	Circular Joint;Grout Hole	Leaks	0.4362	0.7263	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\009befbe-a8bc-4911-92fd-f9a7b327d26b.jpg
009e8891-71a2-4c2d-9057-cece436eab37	{"Circular Joint": true, "Grout Hole": true}	{"Leaks": true}	Circular Joint;Grout Hole	Leaks	0.4839	0.8744	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\009e8891-71a2-4c2d-9057-cece436eab37.jpg
0133f008-c71f-4717-8c05-9b534e3230ff	{"Channel Bracket": true, "Channel Bracket-Bolt": true}	{"Coating Failure": true, "Corrosion-Surface": true, "Missing": true}	Channel Bracket;Channel Bracket-Bolt	Coating Failure;Corrosion-Surface;Missing	0.9141	0.728	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\0133f008-c71f-4717-8c05-9b534e3230ff.jpg
0135ab13-042d-4be8-987c-3f14ca845e4f	{"Circular Connection": true}	{"Drummy": true}	Circular Connection	Drummy	0.532	0.8374	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\0135ab13-042d-4be8-987c-3f14ca845e4f.jpg
013aa420-2af4-4123-93ea-9c929e742e48	{"Grout Hole": true}	{"Drummy": true, "Leaks": true}	Grout Hole	Drummy;Leaks	0.9373	0.8054	C:\Users\ps18286\OneDrive - Surbana Jurong Private Limited\Desktop\ImageAnnot\best_img_model\images\013aa420-2af4-4123-93ea-9c929e742e48.jpg
What this writes (one row per image)
	• ID — filename stem
	• components_dict — JSON dict of component names only (no scores), e.g. {"Channel Bracket": true, "Channel Bracket-Bolt": true}
	• defects_dict — JSON dict of defect names only (no scores)
	• components_list — ;-separated component names
	• defects_list — ;-separated defect names
	• avg_component_confidence — mean of confidences for detected component labels (0.0 if none)
	• avg_defect_confidence — mean of confidences for detected defect labels (0.0 if none)
	• image_path — absolute path
	Note: We still compute probabilities per class to (a) pick labels above threshold and (b) compute the averages, but we don’t emit per‑label scores.
• No per‑label confidence strings: we only emit presence (True) in the dicts and the lists.
• Averages only: avg_component_confidence, avg_defect_confidence summarise the confidence for the detected labels, giving you short numbers you can sort/filter by.
• All images: iterates the images/ directory, so your 3321 files will be included regardless of annotation status.
If you also want the origin snippet like your UI example ({value: {choices: [...]}, id: ..., origin: prediction-changed}), we can add another column called ui_payload_json that packs components_list + defects_list into that shape.

Future direction?
I assume it's better to stick to image classification for now. Object detection would draw bounding boxes around each component/defect, and give spatial localization but requires 5-10x more annotation effort (to re-annotate all the images).
 
The current performance issues (mainly insufficient training data for some classes, class imbalance, possible data quality issues (mislabelled images)). These can be fixed with better architecture, data augmentation, larger models (ResNet50), other pre-trained models, or hyperparameters (focal loss), and more diverse training examples (200-500 per class).
An approach would be to use Grad-CAM for approximate spatial highlighting (generates heatmaps showing "where the model is looking" without the need for extra annotations) with the existing trained model.
 
Only switch to object detection if: (When it's worth it)
You need precise locations (down to the pixel) for automated repair/inspections
Multiple instances of same object in one image (e.g., "Find all 15 bolts in this image")
Components overlap or touch each other (partial in this case)
Generating heatmaps/visual inspection reports is a must
It would take ~2 months + for re-annotation of at least 3k (machine learning - How many images(minimum) should be there in each classes for training YOLO? - St…) images - considering 100 images per class label (20*100) as the baseline.
Classification hits a performance ceiling after trying improvements above.
 
Bottom line: 
Multi-label classification i.e. "This image has corrosion" → Good enough for triage/labelling. The improvements should get us to better scores with way less effort/time than Object detection, which would be useful in: "Corrosion at geolocation (234, 567)" detection cases.
'''
