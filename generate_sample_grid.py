import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np

classes = ['bottle', 'grass', 'branch', 'milk-box', 'plastic-bag', 'plastic-garbage', 'ball', 'leaf', 'pile']

# Distinct colors for each class
colors = [
    (255, 56, 56),    # bottle
    (255, 157, 151),  # grass
    (255, 112, 31),   # branch
    (255, 178, 29),   # milk-box
    (207, 210, 49),   # plastic-bag
    (72, 249, 10),    # plastic-garbage
    (146, 204, 23),   # ball
    (61, 219, 134),   # leaf
    (26, 147, 52)     # pile
]

base_dir = r"C:\Users\induk\Documents\trash-detection-edge\data\datasets\RFT"
img_dir = os.path.join(base_dir, "images", "val")
lbl_dir = os.path.join(base_dir, "labels", "val")

# Pick some interesting images that we know have good classes
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
random.seed(42)
selected_imgs = random.sample(img_files, 20) # get 20, pick 4 best

best_4 = []
for img_name in selected_imgs:
    lbl_name = img_name.replace('.jpg', '.txt')
    lbl_path = os.path.join(lbl_dir, lbl_name)
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 2: # Prefer images with at least a few objects
                best_4.append(img_name)
    if len(best_4) == 4:
        break

if len(best_4) < 4:
    best_4 = img_files[:4]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, img_name in enumerate(best_4):
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))
    
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Read labels and draw boxes
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])
                x_c, y_c, bw, bh = map(float, parts[1:])
                
                # Convert YOLO format to pixel coordinates
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                
                color = colors[cls_id]
                cls_name = classes[cls_id]
                
                # Draw Box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(h/200)))
                
                # Draw Label Background
                label_size, _ = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                
                # Draw Label Text
                cv2.putText(img, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"Contoh Citra {i+1}", fontsize=14)

plt.tight_layout()
output_path = r"C:\Users\induk\.gemini\antigravity-ide\brain\655eeafe-bbbf-4634-a6de-129c0aaa65ea\contoh_anotasi_skripsi.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")
