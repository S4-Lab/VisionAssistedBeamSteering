from ultralytics import YOLO
import torch, random, shutil, csv, os
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from glob import glob

# =========================
# Paths (relative to script)
# =========================
BASE = Path(__file__).parent.resolve()

# Example locations (adjust if needed)
IMAGES_DIR      = (BASE / "dataset" / "syntheticdata" / "images").resolve()
LABELS_CSV_PATH = (BASE / "dataset" / "syntheticdata" / "labels.csv").resolve()

# Single source of truth for work dir (subset + plots)
WORK_DIR        = (BASE / "work_subset").resolve()

# =========================================
# Knobs — set these for your setup
# =========================================
PRETRAINED_WEIGHTS = "yolov8s.pt"  # or yolo11s.pt / yolov8n.pt / etc.

# Start from a previous run's checkpoint instead of COCO weights
USE_FINETUNE       = True
FINETUNE_WEIGHTS   = BASE / "runs" / "detect" / "train6" / "weights" / "best.pt"  # <-- change if needed

# Your CSV coords are in pixels; boxes are square:
COORDS_NORMALIZED  = False
ASSUME_SQUARE_BOX  = True

# Data usage & split
PCT_DATA_USED      = 0.10    # e.g., 0.10 to use 10% of the images
TRAIN_PCT          = 0.90    # train vs val split

# Training knobs
BATCH_SIZE         = 32
EPOCHS             = 10
IMG_SIZE           = 256
FREEZE             = 10
PATIENCE           = 20
RANDOM_SEED        = 42

# Outputs (subset + plot)
SUBSET_NAME        = "dataset_sub"
PLOT_PATH          = WORK_DIR / "train_vs_val.png"
CLASS_NAMES        = ["target"]   # single class

# =========================================
# Utilities (leave at top-level)
# =========================================
def ensure_empty_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def canon(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")

def parse_bool_or_float(v: str) -> float:
    if v is None:
        return 0.0
    s = str(v).strip().lower()
    if s in {"1","true","t","yes","y"}:
        return 1.0
    if s in {"0","false","f","no","n",""}:
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def list_images(root: Path):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    files = []
    for ext in exts:
        files.extend(glob(str(root / ext)))
    return [Path(p) for p in sorted(files)]

def load_csv_labels(csv_path: Path):
    """
    Returns dict: fname -> dict(fields) with keys:
      presence (0/1 float), x_center, y_center, width, height (optional),
      plus raw meta if present.
    """
    label_map = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_c = {canon(k): v for k, v in row.items()}
            fname = row_c.get("filename", "")
            if not fname:
                continue
            p_ball = parse_bool_or_float(row_c.get("p_ball"))
            xc = row_c.get("x_center")
            yc = row_c.get("y_center")
            w  = row_c.get("width")
            h  = row_c.get("height", None)

            def to_float(x):
                try:
                    return float(x)
                except:
                    return None

            xc_f = to_float(xc)
            yc_f = to_float(yc)
            w_f  = to_float(w)
            h_f  = to_float(h) if h is not None else None

            label_map[fname] = {
                "presence": 1.0 if p_ball >= 0.5 else 0.0,
                "x_center": xc_f,
                "y_center": yc_f,
                "width": w_f,
                "height": h_f,   # may be None
                "raw": row_c
            }
    return label_map

def yolo_label_line(xc, yc, w, h):
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"

def write_label_from_info(dst_lbl_path: Path, img_path: Path, info: dict):
    presence = 1.0 if info.get("presence", 0.0) >= 0.5 else 0.0
    if presence < 0.5:
        dst_lbl_path.write_text("")   # negative sample
        return

    xc = info.get("x_center", None)
    yc = info.get("y_center", None)
    w  = info.get("width", None)
    h  = info.get("height", None)

    # If height missing and square boxes assumed, set h = w
    if h is None and ASSUME_SQUARE_BOX:
        h = w

    if None in (xc, yc, w, h):
        dst_lbl_path.write_text("")
        return

    if COORDS_NORMALIZED:
        xc_n, yc_n, w_n, h_n = xc, yc, w, h
    else:
        # Normalize by image size (Pillow required)
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit("[ERR] COORDS_NORMALIZED=False requires Pillow: pip install pillow")
        with Image.open(img_path) as im:
            W, H = im.size
        if W <= 0 or H <= 0:
            dst_lbl_path.write_text(""); return
        xc_n = xc / W
        yc_n = yc / H
        w_n  = w  / W
        h_n  = h  / H

    def clamp01(x): return max(0.0, min(1.0, float(x)))
    line = yolo_label_line(clamp01(xc_n), clamp01(yc_n), clamp01(w_n), clamp01(h_n))
    dst_lbl_path.write_text(line)

def copy_and_label(pairs_list, dst_img_dir: Path, dst_lbl_dir: Path):
    for img_path, info in pairs_list:
        dst_img = dst_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)
        dst_lbl = dst_lbl_dir / (img_path.stem + ".txt")
        write_label_from_info(dst_lbl, img_path, info)

# =========================================
# Main (everything that executes goes here)
# =========================================
def main():
    random.seed(RANDOM_SEED)

    images_root = Path(IMAGES_DIR).resolve()
    csv_path    = Path(LABELS_CSV_PATH).resolve()

    print("[INFO] WORK_DIR:", WORK_DIR)
    if not images_root.exists():
        raise SystemExit(f"[ERR] Images dir not found: {images_root}")
    if not csv_path.exists():
        raise SystemExit(f"[ERR] Labels CSV not found: {csv_path}")

    label_map = load_csv_labels(csv_path)
    all_images = list_images(images_root)
    if not all_images:
        raise SystemExit(f"[ERR] No images found under {images_root}")

    # Join images with CSV rows; keep only images that appear in the CSV
    pairs = []
    miss_csv = 0
    for img in all_images:
        key = img.name
        if key not in label_map:
            miss_csv += 1
            continue
        info = label_map[key]
        pairs.append((img, info))

    if not pairs:
        raise SystemExit("[ERR] None of the images matched filenames in the CSV. Check names/paths.")
    if miss_csv:
        print(f"[WARN] {miss_csv} images skipped (no matching CSV row).")

    # Take a fraction
    random.shuffle(pairs)
    k = max(1, int(len(pairs) * PCT_DATA_USED))
    subset = pairs[:k]

    # Stratified split on presence
    pos = [p for p in subset if p[1].get("presence", 0.0) >= 0.5]
    neg = [p for p in subset if p[1].get("presence", 0.0) < 0.5]
    random.shuffle(pos); random.shuffle(neg)

    def split_list(lst, frac):
        n = int(len(lst) * frac)
        return lst[:n], lst[n:]

    pos_train, pos_val = split_list(pos, TRAIN_PCT)
    neg_train, neg_val = split_list(neg, TRAIN_PCT)

    train_pairs = pos_train + neg_train
    val_pairs   = pos_val   + neg_val
    random.shuffle(train_pairs); random.shuffle(val_pairs)

    print(f"[INFO] Using {len(subset)}/{len(pairs)} samples ({PCT_DATA_USED*100:.1f}%).")
    print(f"[INFO] Train: {len(train_pairs)} | Val: {len(val_pairs)} "
          f"(pos_train={len(pos_train)}, neg_train={len(neg_train)}, pos_val={len(pos_val)}, neg_val={len(neg_val)})")

    # Prepare subset dirs
    subset_root = WORK_DIR / SUBSET_NAME
    imgs_train = subset_root / "images" / "train"
    imgs_val   = subset_root / "images" / "val"
    lbls_train = subset_root / "labels" / "train"
    lbls_val   = subset_root / "labels" / "val"

    ensure_empty_dir(subset_root)
    imgs_train.mkdir(parents=True, exist_ok=True)
    imgs_val.mkdir(parents=True, exist_ok=True)
    lbls_train.mkdir(parents=True, exist_ok=True)
    lbls_val.mkdir(parents=True, exist_ok=True)

    # Copy images + write labels
    copy_and_label(train_pairs, imgs_train, lbls_train)
    copy_and_label(val_pairs,   imgs_val,   lbls_val)

    # Write subset dataset.yaml for Ultralytics (we'll reference this path)
    subset_yaml = WORK_DIR / "dataset_sub.yaml"
    subset_yaml.write_text(
        yaml.safe_dump({
            "path": str(subset_root.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "names": CLASS_NAMES
        }, sort_keys=False)
    )
    print(f"[INFO] Subset dataset created at: {subset_root}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    # Choose weights: previous checkpoint if enabled and exists; else default COCO weights
    if USE_FINETUNE and Path(FINETUNE_WEIGHTS).exists():
        start_weights = str(FINETUNE_WEIGHTS)
        print(f"[INFO] Starting from previous checkpoint: {start_weights}")
    else:
        start_weights = PRETRAINED_WEIGHTS
        if USE_FINETUNE:
            print(f"[WARN] FINETUNE_WEIGHTS not found at {FINETUNE_WEIGHTS}. Falling back to {start_weights}")

    # Build model from chosen weights
    model  = YOLO(start_weights).to(device)

    # Optional: print device from Ultralytics trainer
    def on_fit_start(trainer):
        print(f"[ultralytics] training on device: {trainer.device} | start_weights: {start_weights}")
    model.add_callback("on_fit_start", on_fit_start)

    # Train
    results = model.train(
        data=str(subset_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        freeze=FREEZE,
        patience=PATIENCE,
        device=device,
        workers=4,   # multiprocessing data loader (Windows-safe under __main__)
        max_det=1,   #make sure we are only detecting one object per image
        classes=[0],
        plots=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0
    )

    # Best checkpoint path
    best_path = Path(getattr(model, "ckpt_path", "")) if hasattr(model, "ckpt_path") else Path("runs/detect/train/weights/best.pt")
    if not best_path.exists():
        run_dirs = sorted(Path("runs/detect").glob("train*"), key=os.path.getmtime)
        if run_dirs:
            best_path = run_dirs[-1] / "weights" / "best.pt"
    print("Best weights:", best_path)

    # Plot train vs val error per epoch
    csv_path = best_path.parent.parent / "results.csv"   # runs/detect/train*/results.csv
    if not csv_path.exists():
        run_dirs = sorted(Path("runs/detect").glob("train*"), key=os.path.getmtime)
        for rd in reversed(run_dirs):
            alt = rd / "results.csv"
            if alt.exists():
                csv_path = alt
                break

    epochs, train_err, test_err = [], [], []
    if csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    e = int(float(row.get("epoch", len(epochs))))
                except:
                    e = len(epochs)
                def getf(key, default=0.0):
                    s = row.get(key)
                    try:
                        return float(s)
                    except:
                        return default
                train_box = getf("train/box_loss", getf("train/box_loss(B)", 0.0))
                train_cls = getf("train/cls_loss", getf("train/cls_loss(B)", 0.0))
                train_dfl = getf("train/dfl_loss", getf("train/dfl_loss(B)", 0.0))
                total_train = train_box + train_cls + train_dfl

                mAP50 = row.get("metrics/mAP50(B)", row.get("metrics/mAP50", None))
                te = (1.0 - float(mAP50)) if (mAP50 not in (None, "", "nan")) else float("nan")

                epochs.append(e)
                train_err.append(total_train)
                test_err.append(te)

        plt.figure()
        plt.plot(epochs, train_err, label="Train Loss (box+cls+dfl)")
        plt.plot(epochs, test_err, label="Val Error (1 − mAP50)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Error")
        plt.title("Train vs Validation Error per Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_PATH, dpi=150)
        print(f"[INFO] Saved plot: {PLOT_PATH}")
    else:
        print("[WARN] results.csv not found; skipping plot.")

    # Optional: quick presence helper (max_det=1)
    infer = YOLO(str(best_path)).to(device)
    def present(img, conf=0.25, iou=0.7):
        r = infer.predict(img, conf=conf, iou=iou, max_det=1, verbose=False)[0]
        return len(r.boxes) > 0
    # Example:
    # print("Present?", present(str((subset_root/'images'/'val').glob('*.jpg').__iter__().__next__())))

# =========================
# Windows-safe entry point
# =========================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
