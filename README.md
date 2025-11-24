Multi-Label Image Classification Model for Component & Defect Detection
This script trains a ResNet18-based classifier to simultaneously detect multiple components and defects in industrial inspection images. Each image can have any combination of component types (e.g., "Centre Support", "Channel Bracket") and defect types (e.g., "Corrosion-Heavy", "Missing").

Pipeline Overview
The training pipeline follows these stages:
1.	Data Loading & Cleaning – Parse labels, split dataset, organize images
2.	Multi-Label Encoding – Convert string labels to binary vectors
3.	Class Balancing – Calculate weights for imbalanced classes
4.	Model Training – Fine-tune ResNet18 with early stopping
5.	Threshold Optimization – Tune per-class decision thresholds on validation set
6.	Test Evaluation – Generate predictions and compute metrics

Configuration & Setup
Key Parameters:
•	BATCH_SIZE = 16 – Images per training batch
•	EPOCHS = 10 – Maximum training iterations
•	LR = 0.001 – Adam optimizer learning rate
•	PATIENCE = 3 – Early stopping patience (epochs without improvement)
•	THRESHOLDING_METHOD = "pr_curve" – Method for finding optimal thresholds
Class Weighting:
•	USE_SQRT_SCALING = True – Apply square-root scaling to class weights
•	CLIP_MIN, CLIP_MAX = 1.0, 20.0 – Prevent extreme weight values
Valid Labels:
•	Components (13 types): "Centre Support", "Channel Bracket", "Circular Connection", etc.
•	Defects (7 types): "Corrosion-Heavy", "Coating Failure", "Loose", "Missing", etc.
Step 1: Data Loading & Cleaning
Reads: image_data.csv
Outputs: train_metadata.csv, val_metadata.csv, test_metadata.csv
1.	Label Parsing – clean_labels() converts semicolon-separated strings into lists, filtering out placeholders ("component", "defect", "nan", empty strings)
2.	Label Separation – split_labels() categorizes each label as either a component or defect based on predefined valid sets
3.	Dataset Splitting:
o	70% training / 15% validation / 15% test (by unique image ID)
o	Ensures same ID doesn't appear in multiple splits
4.	Image Organization:
o	Copies images to images/train/, images/val/, images/test/
o	Filters metadata to only include images that physically exist
o	Removes rows for missing image files

Step 2: Multi-Label Target Preparation
Function: group_labels()
•	Groups metadata by image ID
•	Unions all component/defect labels across multiple annotations for the same image
•	Filters labels against valid sets
•	Combines components + defects into single label list per image
Encoding:
•	MultiLabelBinarizer converts label lists into binary vectors
•	Example: Image with ["Centre Support", "Corrosion-Heavy"] → [0, 1, 0, ..., 0, 1, 0]
•	Label order saved to classes.json for reproducibility

Step 2.5: Class Weight Calculation
Addresses severe class imbalance (some defects/components are rare):
1.	Calculates pos_weight = (# negatives) / (# positives) per class
2.	Optionally applies square-root scaling to moderate extreme weights
3.	Clips weights between 1.0–20.0 to prevent training instability
4.	Used in BCEWithLogitsLoss to upweight rare classes

Step 3: Custom Dataset Class
ImageDataset (PyTorch Dataset):
•	Loads image from disk by ID
•	Applies transformations (resize, augmentation)
•	Returns: (image_tensor, label_vector, image_id)

Step 4: Data Augmentation & Loaders
Training Transforms:
•	Resize to 224×224 (ResNet18 input size)
•	Random horizontal flip (50% probability)
•	Random rotation (±10°)
•	Convert to tensor
Validation/Test Transforms:
•	Resize only (no augmentation)
DataLoaders:
•	Shuffle training data each epoch
•	Batch size = 16

Step 5: Model Architecture
•	Base: ResNet18 pre-trained on ImageNet (ResNet18_Weights.DEFAULT)
•	Modification: Replace final fully connected layer 
o	Original: 512 → 1000 (ImageNet classes)
o	Modified: 512 → num_classes (our components + defects)
•	Output: Raw logits (no activation), suitable for BCEWithLogitsLoss

Step 6: Partial Checkpoint Loading (Optional)
load_partial_state_dict():
•	Attempts to load weights from previous training run
•	Only loads layers with matching name and shape
•	Skips incompatible layers (e.g., if class count changed)
•	Useful for continuing training or transfer learning

Step 7: Training Loop
Loss Function: BCEWithLogitsLoss with class weights
Optimizer: Adam (lr=0.001)
Each Epoch:
1.	Training Phase:
o	Forward pass through model
o	Compute binary cross-entropy loss (multi-label)
o	Backpropagation
o	Optimizer step
2.	Validation Phase:
o	Evaluate on validation set (no gradients)
o	Track validation loss
3.	Early Stopping:
o	Save model checkpoint when validation loss improves
o	Stop training if no improvement for PATIENCE epochs

Step 8: Threshold Optimization
Default threshold (0.5) is suboptimal for imbalanced classes. This step finds better per-class thresholds:
1.	Generate probability predictions on validation set
2.	For each class independently: 
o	PR Curve Method (default): Find threshold that maximizes F1 on precision-recall curve
o	Grid Search Method: Test 41 thresholds [0.0–1.0], pick best F1
3.	Clamp thresholds above 0.8 back to 0.5 (avoids overfitting)
4.	Save to thresholds.json
Why per-class thresholds?
A rare defect might need threshold=0.3 (high recall), while a common component might need 0.7 (high precision).

Step 9: Test Set Predictions
Outputs: predictions.csv
For each test image:
1.	Run inference with best model
2.	Apply per-class thresholds to convert probabilities → binary predictions
3.	Separate predictions into components vs. defects
4.	Calculate confidence scores: 
o	Per-label confidence (raw probability)
o	Average component confidence
o	Average defect confidence
o	Overall average confidence
CSV Columns:
•	ID – Image identifier
•	components – Predicted components (semicolon-separated)
•	defects – Predicted defects (semicolon-separated)
•	components_confidence – Label:confidence pairs
•	defects_confidence – Label:confidence pairs
•	avg_component_confidence, avg_defect_confidence, overall_avg_confidence

Step 10: Evaluation Metrics
Computed on test set:
•	Hamming Loss – Fraction of incorrect label predictions (lower is better)
•	F1-Micro – Overall F1 treating all labels equally
•	F1-Macro – Unweighted average F1 across classes (emphasizes rare classes)
•	F1-Weighted – Weighted average F1 by class frequency
•	Per-Class F1 – Individual F1 score for each component/defect
All metrics use zero_division=0 to handle classes with no predictions.

Key Design Decisions
1.	Multi-Label Learning: Images can have multiple components AND defects simultaneously (not mutually exclusive)
2.	Class Imbalance Handling: Three-pronged approach:
o	Weighted loss function
o	Per-class threshold tuning
o	Square-root scaling of weights
3.	Per-Class Thresholds: Global threshold (0.5) performs poorly when classes have vastly different frequencies
4.	Partial Checkpoint Loading: Enables iterative development (e.g., adding new defect types without retraining from scratch)
5.	ID-Level Splitting: Ensures same physical object doesn't appear in both train and test sets

File Outputs Summary
File	Description
model.pth	Best model weights (lowest validation loss)
classes.json	Ordered list of all label classes
thresholds.json	Optimized decision threshold per class
predictions.csv	Per-image predictions with confidences
train_metadata.csv	Cleaned training metadata
val_metadata.csv	Cleaned validation metadata
test_metadata.csv	Cleaned test metadata

Usage Notes
•	GPU Device Recommended: Training will be slow on CPU
•	Data Requirements: Images must exist in images/ directory with IDs matching CSV
•	Resuming Training: Set LOAD_PARTIAL_CHECKPOINT = True to continue from saved model
•	Threshold Method: Switch to "grid_search" if PR curve method fails on small datasets.
