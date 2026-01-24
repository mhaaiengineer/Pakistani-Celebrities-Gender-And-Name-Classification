

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from google.colab.patches import cv2_imshow

# --- Load models once globally ---
YOLO_MODEL_PATH = 'YOUR TRAINED YOLO PATH'
CLASSIFIER_MODEL_PATH = "PATH OF TRAINED KERAS MODEL FOR CLASSIFICATION"
yolo_model = YOLO(YOLO_MODEL_PATH)
classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)

# --- Label Maps ---
FACE_CLASSES = {
    0: 'Unknown', 1: 'Iqra Aziz', 2: 'Hamuyun Saeed', 3: 'Atif Aslam', 4: 'Fahad Mustafa', 5: 'Fawad Khan',
    6: 'Hamza Ali Abbasi', 7: 'Hania Amir', 8: 'Qubra Khan', 9: 'Maira Khan', 10: 'Naseem Shah', 11: 'Noman Ijaz',
    12: 'Neelam Muneer', 13: 'Ramsha Khan', 14: 'Sajal Ali', 15: 'Shaheen Shah Afridi'
}
gender_map = {0: "Female", 1: "Male"}
GENDER_COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}  # Green: Female, Blue: Male

# --- Function ---
def detect_and_classify_faces(image_path, output_path='annotated_output.jpg'):
    IMAGE_SIZE = (224, 224)

    # Read image
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(f"❌ Could not read image: {image_path}")

    img_height, img_width = img_cv.shape[:2]
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # YOLO face + gender detection
    results = yolo_model.predict(img_cv, conf=0.3, iou=0.4, imgsz=640, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("❌ No faces detected.")
        return

    print(f"\n✅ Detected {len(boxes)} face(s):\n")

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

        gender = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        color = GENDER_COLORS.get(gender, (255, 255, 255))

        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, IMAGE_SIZE)
        face_array = np.expand_dims(face_resized, axis=0)
        face_processed = resnet_preprocess_input(face_array)

        # Classify identity
        output = classifier_model.predict(face_processed, verbose=0)
        pred_class = int(np.argmax(output, axis=1)[0])
        label = FACE_CLASSES.get(pred_class, 'Unknown')
        gender_text = gender_map.get(gender, 'Unknown')

        # Final label text
        text = f"{gender_text}, {label}, {confidence:.2f}"
        print(f"Face {i+1}: {text}")

        # Draw text and box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        margin = 4
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = max(min(x1, img_width - text_w - margin), margin)
        text_y = y1 - margin if y1 - text_h - margin > margin else y2 + text_h + margin
        text_y = min(text_y, img_height - margin)

        # Draw background for text
        cv2.rectangle(img_cv,
                      (text_x - margin, text_y - text_h - margin),
                      (text_x + text_w + margin, text_y + margin),
                      color, -1)
        cv2.putText(img_cv, text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)

        # Draw bounding box (thin)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 1)

    # Save and show image
    cv2.imwrite(output_path, img_cv)
    print(f"\n📸 Saved annotated image to: {output_path}")
    cv2_imshow(img_cv)

import cv2
import numpy as np
import os

def upscale_image(image_path, scale_factor=2, output_path='upscaled_temp.jpg'):
    """
    Upscales an image using OpenCV and saves it to a temporary file.

    Args:
        image_path (str): Path to the original image.
        scale_factor (int): Factor by which to upscale the image.
        output_path (str): Path to save the upscaled image.
Q
    Returns:
        str: Path to the upscaled image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ Could not read image: {image_path}")

    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)

    # Use INTER_CUBIC for better quality when upscaling
    upscaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(output_path, upscaled_img)
    return output_path

original_image_path = 'IMAGE PATH FOR CLASSIFICATION'
#YOU CAN SKIP UPSCALIN BY CHANGING SCALE_FACTOR=1
upscaled_image_path = upscale_image(original_image_path, scale_factor=2)

# Now pass the upscaled image path to the detection function
detect_and_classify_faces(upscaled_image_path)

# Optional: Remove the temporary upscaled image file
# os.remove(upscaled_image_path)
