import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------------
# Parameters
# -----------------------------
num_per_bg = 5
target_size = 256
csv_file = "/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/syntheticdata/metadata.csv"

save_dir = "/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/syntheticdata"
background_dir = "/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/croppeddata"
ball_dir = "/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/ball"

os.makedirs(save_dir, exist_ok=True)
metadata = []

# -----------------------------
# Load images
# -----------------------------
backgrounds = [cv2.imread(os.path.join(background_dir, f)) for f in os.listdir(background_dir)]
balls = [cv2.imread(os.path.join(ball_dir, f), cv2.IMREAD_UNCHANGED) for f in os.listdir(ball_dir)]

# -----------------------------
# Generate synthetic images
# -----------------------------
for i, bg in enumerate(backgrounds):
    if bg is None:
        continue

    h_bg, w_bg, _ = bg.shape

    for j in range(num_per_bg):
        img = bg.copy()

        # Pick a random ball
        ball = random.choice(balls)
        if ball is None:
            continue

        # Resize ball randomly
        scale = random.uniform(0.1, 0.5)
        new_w = int(target_size * scale)
        new_h = int(target_size * scale)
        ball_resized = cv2.resize(ball, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Extract alpha channel
        alpha = ball_resized[:, :, 3]
        coords = np.column_stack(np.where(alpha > 0))
        if coords.size == 0:
            continue  # skip fully transparent ball

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        ball_w = x_max - x_min + 1
        ball_h = y_max - y_min + 1

        # Random position on the background, fully visible
        x_bg = random.randint(0, w_bg - ball_w)
        y_bg = random.randint(0, h_bg - ball_h)

        # Overlay the visible part of the ball
        for c in range(3):
            alpha_channel = alpha[y_min:y_max+1, x_min:x_max+1] / 255.0
            img[y_bg:y_bg+ball_h, x_bg:x_bg+ball_w, c] = (
                alpha_channel * ball_resized[y_min:y_max+1, x_min:x_max+1, c] +
                (1 - alpha_channel) * img[y_bg:y_bg+ball_h, x_bg:x_bg+ball_w, c]
            )

        # Save the image
        filename = f"synthetic_{i}_{j}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), img)

        # Compute bounding box centered on visible ball
        x_center = x_bg + ball_w / 2
        y_center = y_bg + ball_h / 2
        box_width = max(ball_w, ball_h)  # square box

        metadata.append({
            "filename": filename,
            "bbox": [1.0, x_center, y_center, box_width]
        })

    # Save background image as negative example
    filename = f"background_{i}_{j}.jpg"
    cv2.imwrite(os.path.join(save_dir, filename), bg)
    metadata.append({
        "filename": filename,
        "bbox": [0, 0, 0, 0]
    })

# -----------------------------
# Save CSV
# -----------------------------
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "p_ball", "x_center", "y_center", "width"])
    for item in metadata:
        writer.writerow([item["filename"], *item["bbox"]])

# -----------------------------
# Visualization (sanity check)
# -----------------------------
num_samples = 5
samples = random.sample(metadata, num_samples)

plt.figure(figsize=(15,5))
for i, item in enumerate(samples):
    img_path = os.path.join(save_dir, item["filename"])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = plt.subplot(1, num_samples, i+1)
    ax.imshow(img)
    ax.axis("off")

    p_ball, x_center, y_center, width = item["bbox"]
    if p_ball > 0:
        x1 = int(x_center - width/2)
        y1 = int(y_center - width/2)
        rect = plt.Rectangle((x1, y1), width, width,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        ax.text(5, 15, f"x:{x_center:.0f}, y:{y_center:.0f}, w:{width:.0f}",
                color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
