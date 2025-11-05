# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# 1. Parameters
# -----------------------------
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
P_BALL_WEIGHT = 5.0  # weight for p_ball loss

# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv(f"/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/syntheticdata/labels.csv")  # must contain: filename, p_ball, x_center, y_center, width

IMAGE_FOLDER = "/Users/research/PycharmProjects/PhDProjects/RadarProcessingProject/dataset/syntheticdata/images"

def load_and_preprocess(filename):
    full_path = os.path.join(IMAGE_FOLDER, filename)
    img = load_img(full_path)
    img = img_to_array(img) / 255.0
    return img

images = np.array([load_and_preprocess(f) for f in df["filename"]])
labels = df[["p_ball", "x_center", "y_center", "width"]].values.astype(np.float32)

train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

train_coords = train_labels[:, 1:]
train_p_ball = train_labels[:, 0:1]

val_coords = val_labels[:, 1:]
val_p_ball = val_labels[:, 0:1]

# -----------------------------
# 3. Build two-head CNN
# -----------------------------
inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(16, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(32, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

# Regression head for coordinates
coords_output = layers.Dense(3, name="coords")(x)  # x_center, y_center, width

# Classification head for p_ball
p_ball_output = layers.Dense(1, activation='sigmoid', name="p_ball")(x)

model = models.Model(inputs=inputs, outputs=[coords_output, p_ball_output])

# -----------------------------
# 4. Define custom loss for masked coords
# -----------------------------
def masked_coords_loss(y_true, y_pred):
    # just MSE on coordinates
    return tf.reduce_mean(tf.square(y_true - y_pred))

# -----------------------------
# 5. Compile model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "coords": masked_coords_loss,
        "p_ball": tf.keras.losses.BinaryCrossentropy(from_logits=False)
    },
    loss_weights={
        "coords": 1.0,
        "p_ball": P_BALL_WEIGHT
    },
    metrics={"p_ball": "accuracy"}
)

model.summary()

# %%
# -----------------------------
# 7. Train model
# -----------------------------
history = model.fit(
    train_images,
    {"coords": train_coords, "p_ball": train_p_ball},
    validation_data=(val_images, {"coords": val_coords, "p_ball": val_p_ball}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# %%
# -----------------------------
# 8. Visualization
# -----------------------------
preds_coords, preds_p_ball = model.predict(val_images)

# Rescale coordinates back to pixels
preds_coords_px = preds_coords * IMG_SIZE[0]
val_coords_px = val_coords * IMG_SIZE[0]

num_show = 5
plt.figure(figsize=(15,5))
for i in range(num_show):
    ax = plt.subplot(1, num_show, i+1)
    i = i + 60
    ax.imshow(val_images[i])
    ax.axis("off")

    # Ground truth box
    true_x, true_y, true_w = val_coords[i]
    rect_gt = plt.Rectangle(
        (true_x - true_w / 2, true_y - true_w / 2),
        true_w, true_w, linewidth=2, edgecolor='green', facecolor='none'
    )
    ax.add_patch(rect_gt)

    # Predicted box
    pred_x, pred_y, pred_w = preds_coords[i]
    rect_pred = plt.Rectangle(
        (pred_x - pred_w / 2, pred_y - pred_w / 2),
        pred_w, pred_w, linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect_pred)

    # Print predicted values on image
    ax.text(5, 20, f"p_ball: {preds_p_ball[i,0]:.2f}", color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5))
    ax.text(5, 35, f"x: {pred_x:.1f}, y: {pred_y:.1f}, w: {pred_w:.1f}", color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Get predictions
pred_coords, pred_p_ball = model.predict(val_images)

# Round classification predictions for metrics
pred_p_ball_bin = (pred_p_ball > 0.5).astype(int)
true_p_ball = val_labels[:, 0].astype(int)

# --- Classification stats for p_ball ---
acc = accuracy_score(true_p_ball, pred_p_ball_bin)
prec = precision_score(true_p_ball, pred_p_ball_bin, zero_division=0)
rec = recall_score(true_p_ball, pred_p_ball_bin, zero_division=0)
f1 = f1_score(true_p_ball, pred_p_ball_bin, zero_division=0)

print("p_ball metrics:")
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# --- Regression stats for coordinates ---
true_coords = val_labels[:, 1:]  # x, y, width
mse = mean_squared_error(true_coords, pred_coords)
mae = mean_absolute_error(true_coords, pred_coords)

print("\nCoordinates metrics:")
print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")

# Optional: per-coordinate MAE
mae_per_coord = np.mean(np.abs(true_coords - pred_coords), axis=0)
print(f"MAE per coordinate (x, y, width): {mae_per_coord}")

