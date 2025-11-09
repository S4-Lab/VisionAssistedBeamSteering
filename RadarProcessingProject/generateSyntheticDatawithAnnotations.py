#!/usr/bin/env python3
from pathlib import Path
import csv, random, shutil
import numpy as np
import cv2

# ==============================
# Paths (relative to this file)
# ==============================
BASE        = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "dataset"
BG_DIR      = DATASET_DIR / "croppeddata"
BALL_DIR    = DATASET_DIR / "ball"
OUT_DIR     = DATASET_DIR / "syntheticdata"
IMAGES_DIR  = OUT_DIR / "images"
CSV_PATH    = OUT_DIR / "labels.csv"   # matches the viewer

# ==============================
# Controls
# ==============================
TOTAL_IMAGES   = 100       # total images to generate (mixed positives + negatives)
NEGATIVE_PROB  = 0.10      # per-image probability of generating a background-only (no ball)
TARGET_SIZE    = 256       # base size used to draw random ball scales
SCALE_RANGE    = (0.1, 0.5)# ball size as a fraction of TARGET_SIZE
RANDOM_SEED    = 42        # set to None for non-deterministic runs
MAX_POS_TRIES  = 20        # per-image retry budget when a positive attempt fails (e.g., ball too big)
SHOW_PREVIEW   = False     # set True for a small sanity-check figure at the end

# ==============================
# Prep & clean
# ==============================
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Purge old images
for p in IMAGES_DIR.iterdir():
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)

# ==============================
# Helpers
# ==============================
def list_images_with_paths(folder: Path):
    """Return list of (path, image) with IMREAD_UNCHANGED (keep alpha)."""
    items = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            items.append((p, img))
    return items

def ensure_bgra(img):
    """Return image as BGRA (adds alpha=255 if absent)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge((b, g, r, a))
    elif img.shape[2] == 4:
        pass
    else:
        raise ValueError("Unsupported channel count.")
    return img

def to_bgr(img):
    """Ensure 3-channel BGR background."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def try_make_positive(backgrounds, balls):
    """
    Attempt to compose a single positive (ball present) image.
    Returns (img, x_center, y_center, width, bg_name, ball_name) or None if fail.
    """
    bg_path, bg_raw = random.choice(backgrounds)
    ball_path, ball0 = random.choice(balls)

    bg = to_bgr(bg_raw)
    h_bg, w_bg, _ = bg.shape

    ball = ensure_bgra(ball0)

    # Resize ball randomly
    scale = random.uniform(*SCALE_RANGE)
    new_w = max(1, int(TARGET_SIZE * scale))
    new_h = max(1, int(TARGET_SIZE * scale))
    ball_resized = cv2.resize(ball, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Visible region from alpha
    alpha = ball_resized[:, :, 3]
    coords = np.column_stack(np.where(alpha > 0))
    if coords.size == 0:
        return None  # fully transparent after resize

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    ball_w = int(x_max - x_min + 1)
    ball_h = int(y_max - y_min + 1)

    # Skip if visible ball patch is larger than the background
    if ball_w > w_bg or ball_h > h_bg:
        return None

    # Random top-left on the background, fully visible
    x_bg = random.randint(0, w_bg - ball_w)
    y_bg = random.randint(0, h_bg - ball_h)

    # Compose
    img = bg.copy()
    alpha_vis = alpha[y_min:y_max+1, x_min:x_max+1].astype(np.float32) / 255.0
    for c in range(3):
        src = ball_resized[y_min:y_max+1, x_min:x_max+1, c].astype(np.float32)
        dst = img[y_bg:y_bg+ball_h, x_bg:x_bg+ball_w, c].astype(np.float32)
        img[y_bg:y_bg+ball_h, x_bg:x_bg+ball_w, c] = alpha_vis * src + (1.0 - alpha_vis) * dst

    # Square bounding box centered on visible region
    x_center = x_bg + ball_w / 2.0
    y_center = y_bg + ball_h / 2.0
    box_width = float(max(ball_w, ball_h))

    return img, x_center, y_center, box_width, bg_path.name, ball_path.name

# ==============================
# Load sources
# ==============================
backgrounds = list_images_with_paths(BG_DIR)  # [(path, img), ...]
balls       = list_images_with_paths(BALL_DIR)

if not backgrounds:
    raise RuntimeError(f"No readable background images in: {BG_DIR}")
if not balls:
    raise RuntimeError(f"No readable ball images in: {BALL_DIR}")

# ==============================
# Generate mixed set
# ==============================
metadata = []  # [filename, p_ball, x_center, y_center, width, background file, ball file]
seq = 0
num_pos = 0
num_neg = 0

for _ in range(TOTAL_IMAGES):
    make_negative = (random.random() < NEGATIVE_PROB)

    if make_negative:
        # negative: pick a background only
        bg_path, bg_raw = random.choice(backgrounds)
        bg = to_bgr(bg_raw)
        filename = f"synthetic_{seq:06d}.jpg"
        cv2.imwrite(str(IMAGES_DIR / filename), bg)
        metadata.append([filename, 0, 0, 0, 0, bg_path.name, ""])
        seq += 1
        num_neg += 1
        continue

    # positive: try a few times, else fallback to a negative to keep total count constant
    success = None
    for _try in range(MAX_POS_TRIES):
        success = try_make_positive(backgrounds, balls)
        if success is not None:
            break

    if success is None:
        # fallback negative
        bg_path, bg_raw = random.choice(backgrounds)
        bg = to_bgr(bg_raw)
        filename = f"synthetic_{seq:06d}.jpg"
        cv2.imwrite(str(IMAGES_DIR / filename), bg)
        metadata.append([filename, 0, 0, 0, 0, bg_path.name, ""])
        seq += 1
        num_neg += 1
    else:
        img, x_center, y_center, box_width, bg_name, ball_name = success
        filename = f"synthetic_{seq:06d}.jpg"
        cv2.imwrite(str(IMAGES_DIR / filename), img)
        metadata.append([filename, 1, x_center, y_center, box_width, bg_name, ball_name])
        seq += 1
        num_pos += 1

# ==============================
# Write CSV fresh
# ==============================
with open(CSV_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "p_ball", "x_center", "y_center", "width", "background file", "ball file"])
    writer.writerows(metadata)

print(f"Generated {seq} images into {IMAGES_DIR}")
print(f"Counts -> positives(with balls): {num_pos}, negatives(background-only): {num_neg}")
print(f"Wrote CSV: {CSV_PATH}")

# ==============================
# Optional preview
# ==============================
if SHOW_PREVIEW and seq > 0:
    import matplotlib.pyplot as plt
    n = min(5, seq)
    picks = random.sample(metadata, n)
    plt.figure(figsize=(15, 5))
    for i, row in enumerate(picks):
        fn, p_ball, xc, yc, w, bg_name, ball_name = row
        img = cv2.imread(str(IMAGES_DIR / fn))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(1, n, i+1)
        ax.imshow(img)
        ax.axis("off")
        if p_ball:
            x1 = float(xc) - float(w)/2.0
            y1 = float(yc) - float(w)/2.0
            rect = plt.Rectangle((x1, y1), float(w), float(w),
                                 edgecolor='red', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, 15, f"x:{xc:.0f}, y:{yc:.0f}, w:{w:.0f}",
                    color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.show()
