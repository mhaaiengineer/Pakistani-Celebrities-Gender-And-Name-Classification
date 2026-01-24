
# Install ultralytics if not already installed
!pip install -q ultralytics

# ✅ Imports
from ultralytics import YOLO
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import seaborn as sns
from PIL import Image
import torch
import numpy as np
import pandas as pd

#IF YOU ARE TRAINING LOCALLY THAN NO NEED TO RUN THIS CELL
from google.colab import drive
drive.mount('/content/drive')

#IF YOU ARE TRANING ON LOCAL DEIVCE RATHER THAN COLAB SO YOU THERES NO NEED TO RUN THIS CELL
import zipfile

with zipfile.ZipFile("PATH OF DATASET IN YOUR DRIVE OR ON DEVICE", 'r') as zip_ref:
    zip_ref.extractall("/content/")

# ✅ Define dataset.yaml
dataset_yaml = """
path: DATASET_PATH
train: TRAIN_IMAGES_PATH
val: VALIDATION_IMAGES_PATH
test: TEST_IMAGES_PATH

names:
  0: female
  1: male
"""

with open("dataset.yaml", "w") as f:
    f.write(dataset_yaml)

# ✅ Load and train YOLOv8 model
model = YOLO("yolov8m.pt")  # or use yolov8s.pt etc.

results = model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=224,
    batch=16,
    name="yolov8_gender_classification",
    save=True,
    project="runs/detect"
)

# ✅ Save best.pt model to working directory
best_path = "runs/detect/yolov8_gender_classification/weights/best.pt"
if os.path.exists(best_path):
    shutil.copy(best_path, "/content/best.pt")
    print("✅ best.pt copied to content directory")
else:
    print("❌ best.pt not found, training might have failed")

# ✅ Show training results.png from YOLO
def plot_metrics(results):
    metrics_path = os.path.join(results.save_dir, "results.png")
    if os.path.exists(metrics_path):
        img = Image.open(metrics_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title("YOLOv8 Training Metrics")
        plt.show()
    else:
        print("❌ Training metrics image not found.")

plot_metrics(results)

# ✅ Evaluate on test set
metrics = model.val(data="dataset.yaml", split="test")

import os
import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Setup ---
MODEL_PATH = "/content/best.pt"
TEST_IMG_DIR = "/content/test/images"
TEST_LABEL_DIR = "/content/test/labels"
CLASS_NAMES = ["female", "male"]

# Load model
model = YOLO(MODEL_PATH)

# Init
y_true = []
y_pred = []

# Process images
image_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
for img_path in image_paths:
    # Get matching label
    label_path = os.path.join(TEST_LABEL_DIR, os.path.basename(img_path).replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    # Ground truth labels
    with open(label_path, "r") as f:
        labels = [int(line.split()[0]) for line in f.readlines()]

    # YOLOv8 prediction
    results = model(img_path)[0]
    preds = [int(cls.item()) for cls in results.boxes.cls]

    # Match labels and predictions by count (simple assumption)
    for gt, pr in zip(labels, preds):
        y_true.append(gt)
        y_pred.append(pr)

# --- Metrics ---
print("✅ Accuracy:", accuracy_score(y_true, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import os
import glob
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup ---
MODEL_PATH = "/content/best.pt"  # adjust if needed
VAL_IMG_DIR = "/content/valid/images"
VAL_LABEL_DIR = "/content/valid/labels"
CLASS_NAMES = ["female", "male"]

# Load model
model = YOLO(MODEL_PATH)

# Store true and predicted labels
y_true, y_pred = [], []

# Loop through validation images
image_paths = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))
for img_path in image_paths:
    label_path = os.path.join(VAL_LABEL_DIR, os.path.basename(img_path).replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    # Ground truth
    with open(label_path, "r") as f:
        labels = [int(line.split()[0]) for line in f.readlines()]

    # Predictions
    results = model(img_path)[0]
    preds = [int(cls.item()) for cls in results.boxes.cls]

    # Match by count (simple assumption)
    for gt, pr in zip(labels, preds):
        y_true.append(gt)
        y_pred.append(pr)

# --- Metrics ---
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
acc = accuracy_score(y_true, y_pred)

print(f"✅ validation Accuracy: {acc:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🌀 Recall: {recall:.4f}")
print(f"📊 F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Validation Set)")
plt.show()

# --- Optional: Bar Chart of Metrics ---
plt.figure(figsize=(6, 4))
plt.bar(["Precision", "Recall", "F1 Score"], [precision, recall, f1], color=["green", "orange", "blue"])
plt.ylim(0, 1)
plt.title("YOLOv8 Validation Metrics")
plt.grid(axis='y')
plt.show()

import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths (update if needed)
MODEL_PATH = "/content/best.pt"
TEST_IMG_DIR = "/content/test/images"
CLASS_NAMES = ["female", "male"]
CONF_THRESHOLD = 0.3
NUM_IMAGES = 10  # number of images to display (in grid)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Get image paths
image_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))[:NUM_IMAGES]

# Prepare grid size (e.g., 2x3 for 6 images)
cols = 5
rows = (NUM_IMAGES + cols - 1) // cols

# Setup plot
plt.figure(figsize=(8, 3* rows))

for idx, img_path in enumerate(image_paths):
    # Load and preprocess image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Predict
    results = model(img_path, conf=CONF_THRESHOLD)[0]

    # Draw boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"
        color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show in grid
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(img_rgb)
    plt.title(os.path.basename(img_path), fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()
