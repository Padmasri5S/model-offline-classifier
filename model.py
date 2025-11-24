from typing import List, Dict, Optional
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_local_path

def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y")

def _load_json_or_whitespace_list(path: str, expect_float: bool = False):
    """
    Tries json.load; if it fails, falls back to whitespace-separated parsing.
    Returns a list[str] or list[float] depending on expect_float.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[LS-ML] File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure correct type
        if expect_float:
            return [float(x) for x in data]
        else:
            return [str(x) for x in data]
    except Exception as e:
        print(f"[LS-ML] json.load failed for {path} ({e}); falling back to whitespace parsing.")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        items = [tok for tok in raw.split() if tok]
        if expect_float:
            return [float(x) for x in items]
        else:
            return items

class NewModel(LabelStudioMLBase):
    """
    Label Studio ML backend that loads a multi-label ResNet18 checkpoint
    and returns predictions to two multi-select controls: 'component' and 'defect',
    attached to <Image name="image" />.
    """

    def setup(self):
        """Load model/checkpoint once when the server starts."""
        self.set("model_version", os.getenv("MODEL_VERSION", "0.0.4"))

        # --- File paths ---
        model_dir     = os.getenv("MODEL_DIR", ".")
        ckpt_path     = os.path.join(model_dir, os.getenv("CHECKPOINT", "model.pth"))
        classes_path  = os.path.join(model_dir, os.getenv("CLASSES_PATH", "classes.json"))
        thresholds_path = os.path.join(model_dir, os.getenv("THRESHOLDS_PATH", "thresholds.json"))

        # --- Label Studio control names (must match your XML exactly) ---
        self.image_name = os.getenv("LS_IMAGE_NAME", "image")       # <Image name="image" ...>
        self.comp_name  = os.getenv("LS_COMPONENT_NAME", "component")  # <Choices name="component" ...>
        self.def_name   = os.getenv("LS_DEFECT_NAME", "defect")        # <Choices name="defect" ...>

        # --- Device ---
        prefer_device = os.getenv("DEVICE", "").lower()
        if prefer_device in ("cuda", "gpu") and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif prefer_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # --- Load ordered classes (must match training order) ---
        self.label_classes: List[str] = _load_json_or_whitespace_list(classes_path, expect_float=False)
        self.num_classes = len(self.label_classes)
        if self.num_classes == 0:
            raise ValueError("[LS-ML] classes.json contains no classes.")
        # Component/Defect membership
        self.components_set = {
            "Centre Support", "Channel Bracket", "Circular Connection", "Circular Joint",
            "Conveyor Support", "Grout Hole", "Radial Connection", "Radial Joint", "Walkway Support",
            "Centre Support-Bolt", "Channel Bracket-Bolt", "Conveyor Support-Bolt", "Walkway Support-Bolt"
        }
        self.defects_set = {
            "Corrosion-Heavy", "Coating Failure", "Corrosion-Surface",
            "Loose", "Missing", "Drummy", "Leaks"
        }

        # --- Load per-class thresholds ---
        thresholds = _load_json_or_whitespace_list(thresholds_path, expect_float=True)

        # Align thresholds length with classes length
        if len(thresholds) < self.num_classes:
            # Pad missing with 0.5
            thresholds = thresholds + [0.5] * (self.num_classes - len(thresholds))
            print(f"[LS-ML] thresholds length < classes; padded to {self.num_classes}.")
        elif len(thresholds) > self.num_classes:
            thresholds = thresholds[:self.num_classes]
            print(f"[LS-ML] thresholds length > classes; truncated to {self.num_classes}.")
        self.class_thresholds = np.array(thresholds, dtype=np.float32)

        # Optional clipping to avoid unreachable thresholds (e.g., >0.9)
        clip_max = float(os.getenv("THRESH_CLIP_MAX", "0.85"))
        replace_val = float(os.getenv("THRESH_REPLACE_VAL", "0.5"))
        self.class_thresholds = np.where(self.class_thresholds > clip_max, replace_val, self.class_thresholds)

        # --- Build the SAME architecture as training: Linear head only ---
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feats, self.num_classes)  # <-- matches 'fc.weight'/'fc.bias' in checkpoint

        # --- Load state dict (strict first; fallback to backbone-only) ---
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[LS-ML] Checkpoint not found: {ckpt_path}")
        state_obj = torch.load(ckpt_path, map_location="cpu")
        state_dict = self._extract_state_dict(state_obj)
        try:
            self.net.load_state_dict(state_dict, strict=True)
            print("[LS-ML] Checkpoint loaded with strict=True.")
        except RuntimeError as e:
            print(f"[LS-ML] Strict load failed; loading backbone only. Details: {e}")
            current_sd = self.net.state_dict()
            filtered = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
            filtered = {k: v for k, v in filtered.items()
                        if k in current_sd and current_sd[k].shape == v.shape}
            current_sd.update(filtered)
            self.net.load_state_dict(current_sd, strict=False)
            print("[LS-ML] Backbone loaded; classifier head left as current init.")

        self.net.to(self.device)
        self.net.eval()

        # --- Inference transforms (match your training eval pipeline) ---
        self.use_pretrained_norm = _env_flag("USE_PRETRAINED_NORM", "0")
        if self.use_pretrained_norm:
            weights = ResNet18_Weights.DEFAULT
            self.tf = T.Compose([T.Resize((224, 224))] + list(weights.transforms().transforms))
            tf_desc = "Resize(224)+ToTensor+Normalize(weights.DEFAULT)"
        else:
            self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            tf_desc = "Resize(224)+ToTensor (no normalize)"

        print(f"[LS-ML] Loaded checkpoint: {ckpt_path}")
        print(f"[LS-ML] Loaded classes ({self.num_classes}): {self.label_classes}")
        print(f"[LS-ML] Thresholds (clipped @ {clip_max}→{replace_val}): {self.class_thresholds.tolist()}")
        print(f"[LS-ML] Device: {self.device}")
        print(f"[LS-ML] Transforms: {tf_desc}")
        print(f"[LS-ML] LS names: image='{self.image_name}', component='{self.comp_name}', defect='{self.def_name}'")

    # ----------------------- PREDICT -----------------------
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Run inference on incoming tasks and return two multi-choice results."""
        predictions = []
        for task in tasks:
            data = task.get("data", {})
            image_url = data.get(self.image_name) or data.get("image")
            if not image_url:
                predictions.append({
                    "result": [],
                    "score": 0.0,
                    "model_version": self.get("model_version")
                })
                continue

            local_path = get_local_path(image_url, task_id=task.get("id"))
            with Image.open(local_path).convert("RGB") as im:
                x = self.tf(im).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.net(x)                # shape [1, num_classes]
                probs  = torch.sigmoid(logits)[0]   # shape [num_classes], in [0,1]

            chosen = []
            for i in range(self.num_classes):
                if float(probs[i].item()) >= float(self.class_thresholds[i]):
                    chosen.append(self.label_classes[i])

            chosen_components = [c for c in chosen if c in self.components_set]
            chosen_defects    = [d for d in chosen if d in self.defects_set]

            result = []
            if chosen_components:
                result.append({
                    "from_name": self.comp_name,
                    "to_name": self.image_name,
                    "type": "choices",
                    "value": {"choices": chosen_components}
                })
            if chosen_defects:
                result.append({
                    "from_name": self.def_name,
                    "to_name": self.image_name,
                    "type": "choices",
                    "value": {"choices": chosen_defects}
                })

            predictions.append({
                "model_version": self.get("model_version"),
                "score": float(torch.max(probs).item()) if probs.numel() > 0 else 0.0,
                "result": result
            })

        return ModelResponse(predictions=predictions)

    # ----------------------- FIT --------------------------
    def fit(self, event, data, **kwargs):
        """No online training in this backend."""
        return {"status": "ok", "event": event}

    # ----------------------- HELPERS ----------------------
    @staticmethod
    def _extract_state_dict(state_obj) -> dict:
        """Support plain state_dict or {'state_dict': ...} formats."""
        if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
            return state_obj["state_dict"]
        if isinstance(state_obj, dict):
            return state_obj
        raise RuntimeError("[LS-ML] Unsupported checkpoint format: expected dict or {'state_dict': dict}")