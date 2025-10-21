from google.colab import drive
drive.mount('/content/drive')


pip install numpy==1.24.3 



# Install if needed: pip install ultralytics tensorboard psutil matplotlib opencv-python
from ultralytics import YOLO
import os
import psutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

print(f"✅ Starting from: {os.getcwd()}")
print(f"💾 Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# Load model
model = YOLO("yolov8n-seg.pt")

# LOW-MEMORY Training parameters
train_params = {
    "data": r"/content/drive/MyDrive/Dataset/data.yaml",
    "epochs": 25,
    "imgsz": 212,
    "project": r"runs\segment",
    "name": "train_pizza_lowmem",
    "batch": 1,
    "device": 'cpu',
    "workers": 1,
    "cache": False,
    "patience": 50,
    "augment": True,
    "exist_ok": True
}

# === TRAINING ===
print("🚀 Starting training...")
results = model.train(**train_params)

# === VALIDATION ===
print("✅ Validating...")
results_val = model.val()

# === DATASET CHECK ===
data_yaml = train_params["data"]
if os.path.exists(data_yaml):
    print(f"✅ data.yaml found!")
    with open(data_yaml, 'r') as f:
        content = f.read()
        print("📄 data.yaml preview:\n", content[:300])
else:
    print(f"❌ {data_yaml} not found!")
    exit()

# === EXPORT ===
print("💾 Exporting ONNX...")
exported_model = model.export(format="onnx")
print(f"✅ Exported: {exported_model}")

# === SAMPLE INFERENCE ===
sample_image = r"/content/drive/MyDrive/Dataset/test/images/pizza_121-jpg_jpg.rf.23aed9ba57126f39d49f0938104b054f.jpg"
if os.path.exists(sample_image):
    print("🎯 Running sample inference...")
    results_predict = model(sample_image)
    results_predict[0].show()
else:
    print(f"❌ Sample not found. Available: {os.listdir(r'/content/drive/MyDrive/Dataset/test/images')[:3]}")










import cv2
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")  # or yolov8n.pt

image_path = "/content/drive/MyDrive/Dataset/download (1).png"  # your provided frame
results = model.predict(image_path, conf=0.25)

results[0].show()  # display image with boxes
results[0].save("pizza_detected.png")  # save output
print("✅ Detection complete – saved as pizza_detected.png")














# 🚀 Select YOLOv8 Logger
logger = 'TensorBoard'  # options: 'TensorBoard' or 'Weights & Biases'

if logger == 'TensorBoard':
    !yolo settings tensorboard=True
    %load_ext tensorboard
    %tensorboard --logdir runs
elif logger == 'Weights & Biases':
    !yolo settings wandb=True








import os
import cv2
from ultralytics import YOLO

# ✅ 1. Load Model (change to "best.pt" if you trained your own)
model = YOLO("yolov8n.pt")

# ✅ 2. Folder containing your images
input_folder = "/content/drive/MyDrive/videos-frames"  # change if needed
output_folder = "/content/drive/MyDrive/output"
os.makedirs(output_folder, exist_ok=True)

# ✅ 3. Loop through all image files
valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]

print(f"✅ Found {len(image_files)} images. Starting detection...")

for idx, filename in enumerate(image_files, start=1):
    image_path = os.path.join(input_folder, filename)

    # Run YOLO detection
    results = model.predict(image_path, conf=0.25, stream=False)

    # Annotate and save the output
    annotated = results[0].plot()  # draw bounding boxes
    out_name = f"detected_{filename}"
    out_path = os.path.join(output_folder, out_name)

    cv2.imwrite(out_path, annotated)

    print(f"[{idx}/{len(image_files)}] ✅ Saved: {out_path}")

print("\n🎉 All images processed!")
print(f"✅ Annotated results saved in: {output_folder}")



















import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

# Load model again (for extraction)
model = YOLO("yolov8n.pt")

# Input/Output folders (same as before)
input_folder = "/content/drive/MyDrive/videos-frames"
output_folder = "/content/drive/MyDrive/output"
valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

# Get sorted image files (ensure order matches saved outputs)
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)])
output_files = sorted([f for f in os.listdir(output_folder) if f.startswith('detected_')])

print(f"✅ Analyzing {len(image_files)} processed images...")

# Collect data
data = []
class_counts = defaultdict(int)
confidences = []

for idx, filename in enumerate(image_files):
    image_path = os.path.join(input_folder, filename)
    results = model.predict(image_path, conf=0.25, stream=False)[0]  # Re-infer for metrics
    
    boxes = results.boxes
    num_dets = len(boxes) if boxes is not None else 0
    avg_conf = boxes.conf.mean().item() if num_dets > 0 else 0
    confidences.extend(boxes.conf.tolist() if num_dets > 0 else [])
    
    for cls_id in boxes.cls.tolist():
        class_name = results.names[int(cls_id)]
        class_counts[class_name] += 1
    
    data.append({
        'image': filename,
        'detections': num_dets,
        'avg_confidence': avg_conf
    })
    
    print(f"[{idx+1}/{len(image_files)}] {filename}: {num_dets} detections, avg conf {avg_conf:.2f}")

# Save summary CSV
df = pd.DataFrame(data)
summary_path = os.path.join(output_folder, 'detection_summary.csv')
df.to_csv(summary_path, index=False)
print(f"\n✅ Summary saved to: {summary_path}")

# Class distribution
class_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['count']).sort_values('count', ascending=False)

# Plot 1: Detections per Image (Bar Chart)
plt.figure(figsize=(10, 6))
plt.bar(df['image'], df['detections'], color='#36A2EB')
plt.title('Number of Detections per Image')
plt.xlabel('Image File')
plt.ylabel('Detection Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'detections_per_image.png'))
plt.show()

# Plot 2: Average Confidence per Image (Line Chart)
plt.figure(figsize=(10, 6))
plt.plot(df['image'], df['avg_confidence'], marker='o', color='#FF6384')
plt.title('Average Confidence per Image')
plt.xlabel('Image File')
plt.ylabel('Avg Confidence')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'avg_conf_per_image.png'))
plt.show()

# Plot 3: Class Distribution (Pie Chart)
if not class_df.empty:
    plt.figure(figsize=(8, 8))
    plt.pie(class_df['count'], labels=class_df.index, autopct='%1.1f%%', colors=plt.cm.tab20.colors)
    plt.title('Overall Class Distribution')
    plt.savefig(os.path.join(output_folder, 'class_distribution.png'))
    plt.show()

# Plot 4: Confidence Histogram
plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=20, color='#4BC0C0', edgecolor='black')
plt.title('Distribution of Detection Confidences')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_folder, 'confidence_hist.png'))
plt.show()

print("\n🎉 Plots saved in output folder!")
print(" - detections_per_image.png")
print(" - avg_conf_per_image.png")
print(" - class_distribution.png")
print(" - confidence_hist.png")