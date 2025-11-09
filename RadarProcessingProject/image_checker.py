#!/usr/bin/env python3
# File: view_boxes_random.py
# pip install pillow
from pathlib import Path
import csv, random
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

# ========= EDIT THESE FOR YOUR PROJECT =========
BASE        = Path(__file__).parent.resolve()
CSV_PATH    = BASE / "dataset" / "syntheticdata" / "labels.csv"
IMAGES_DIR  = BASE / "dataset" / "syntheticdata" / "images"
WINDOW_W    = 1200
WINDOW_H    = 800
LINE_THICK  = 3
SHOW_ONLY_EXISTING = True
# ==============================================

FOOTER_BG = "#1e1e1e"
FOOTER_FG = "#e6e6e6"
CANVAS_BG = "#111111"
OVERLAY_FG = "white"
OVERLAY_SHADOW = "#000000"

def to_int(x): return int(round(float(x)))
def clamp(v, lo, hi): return max(lo, min(hi, v))
def norm_key(k: str) -> str: return "".join(k.strip().lower().split())

def load_manifest(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV appears empty or malformed.")
        fmap = {norm_key(k): k for k in reader.fieldnames}

        # Required columns (first five)
        req = ["filename", "p_ball", "x_center", "y_center", "width"]
        miss = [k for k in req if norm_key(k) not in fmap]
        if miss:
            raise ValueError(f"CSV missing required columns: {miss}\nFound: {reader.fieldnames}")

        # Optional new columns
        bg_key   = fmap.get(norm_key("background file"))
        ball_key = fmap.get(norm_key("ball file"))

        rows = []
        for row in reader:
            try:
                fn   = row[fmap[norm_key("filename")]].strip()
                flag = to_int(row[fmap[norm_key("p_ball")]])
                cx   = to_int(row[fmap[norm_key("x_center")]])
                cy   = to_int(row[fmap[norm_key("y_center")]])
                size = to_int(row[fmap[norm_key("width")]])
                bg_name   = row[bg_key].strip()   if bg_key   else ""
                ball_name = row[ball_key].strip() if ball_key else ""
            except Exception as e:
                print(f"[skip row] parse error: {e}")
                continue
            rows.append(dict(
                filename=fn, p_ball=flag, cx=cx, cy=cy, size=size,
                bg_name=bg_name, ball_name=ball_name
            ))
        return rows

def fit_to_window(img, max_w, max_h):
    w, h = img.size
    if w <= 0 or h <= 0: return img, 1.0, 1.0
    scale = min(max_w / w, max_h / h, 1.0)  # avoid upscaling
    new_w, new_h = int(w*scale), int(h*scale)
    return (img.resize((new_w, new_h), Image.BICUBIC) if scale < 1.0 else img,
            new_w / w, new_h / h)

def draw_square(img, cx, cy, size, thickness):
    if size <= 0: return img
    w, h = img.size
    half = size // 2
    x0 = clamp(cx - half, 0, w - 1)
    y0 = clamp(cy - half, 0, h - 1)
    x1 = clamp(cx + half, 0, w - 1)
    y1 = clamp(cy + half, 0, h - 1)
    d = ImageDraw.Draw(img)
    for t in range(thickness):
        d.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline="green")
    return img

class Viewer(tk.Tk):
    def __init__(self, items, images_dir: Path):
        super().__init__()
        self.title("Random Box Viewer")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.items = items[:]
        random.shuffle(self.items)
        self.images_dir = images_dir
        self.idx = 0

        self.canvas = tk.Canvas(self, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        footer = tk.Frame(self, bg=FOOTER_BG)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        self.info_var = tk.StringVar(value="")
        self.info_label = tk.Label(footer, textvariable=self.info_var, bg=FOOTER_BG, fg=FOOTER_FG, anchor="w", justify="left")
        self.info_label.pack(side=tk.LEFT, padx=8, pady=6, fill=tk.X, expand=True)

        def mk_btn(txt, cmd):
            return tk.Button(footer, text=txt, command=cmd, bg="#2a2a2a", fg=FOOTER_FG,
                             activebackground="#333", activeforeground=FOOTER_FG, bd=0, padx=10, pady=6)
        mk_btn("⟵ Prev [Left]", self.prev_image).pack(side=tk.RIGHT, padx=4, pady=4)
        mk_btn("Next [Right] ⟶", self.next_image).pack(side=tk.RIGHT, padx=4, pady=4)
        mk_btn("Shuffle [S]", self.shuffle_all).pack(side=tk.RIGHT, padx=4, pady=4)

        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Left>",  lambda e: self.prev_image())
        self.bind("<s>",     lambda e: self.shuffle_all())
        self.bind("<S>",     lambda e: self.shuffle_all())
        self.bind("<Escape>",lambda e: self.destroy())
        self.bind("<Configure>", self._on_resize)

        self.tk_img = None
        self.display_current()

    def shuffle_all(self):
        random.shuffle(self.items)
        self.idx = 0
        self.display_current()

    def next_image(self):
        if not self.items: return
        self.idx = (self.idx + 1) % len(self.items)
        self.display_current()

    def prev_image(self):
        if not self.items: return
        self.idx = (self.idx - 1) % len(self.items)
        self.display_current()

    def _on_resize(self, event):
        if event.widget is self:
            self.display_current()

    def _draw_overlay_text(self, it):
        """Top-left overlay with filename / coords / size / bg / ball (with shadow)."""
        pad = 12
        # Build multi-line overlay to avoid super-long single line
        lines = [
            f"{self.idx+1}/{len(self.items)} • {it['filename']} • "
            f"{'BOX' if it['p_ball']==1 else 'NO BOX'}",
            f"X={it['cx']}  Y={it['cy']}  Size={it['size']} px",
        ]
        # Only add bg/ball if present (keeps compatibility with older CSVs)
        if it.get("bg_name") or it.get("ball_name"):
            lines.append(f"BG: {it.get('bg_name','')}")
            lines.append(f"BALL: {it.get('ball_name','')}")

        y = pad
        for line in lines:
            # Shadow
            self.canvas.create_text(pad+1, y+1, text=line, anchor="nw",
                                    fill=OVERLAY_SHADOW, font=("Segoe UI", 12, "bold"))
            # Foreground
            self.canvas.create_text(pad, y, text=line, anchor="nw",
                                    fill=OVERLAY_FG, font=("Segoe UI", 12, "bold"))
            y += 18  # line spacing

    def display_current(self):
        self.canvas.delete("all")
        if not self.items:
            self.info_var.set("No images to display.")
            return

        it = self.items[self.idx]
        img_path = self.images_dir / it["filename"]
        if not img_path.exists():
            self.info_var.set(f"[missing] {img_path}  ({self.idx+1}/{len(self.items)})")
            return

        try:
            original = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.info_var.set(f"[open error] {img_path}: {e}")
            return

        cw = max(self.canvas.winfo_width() - 10, 100)
        ch = max(self.canvas.winfo_height() - 10, 100)
        disp_img, sx, sy = fit_to_window(original, cw, ch)

        if it["p_ball"] == 1:
            cx = int(round(it["cx"] * sx))
            cy = int(round(it["cy"] * sy))
            size = max(1, int(round(it["size"] * (sx + sy) / 2.0)))
            draw_square(disp_img, cx, cy, size, LINE_THICK)

        self.tk_img = ImageTk.PhotoImage(disp_img)
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img, anchor=tk.CENTER)

        flag = "BOX" if it["p_ball"] == 1 else "NO BOX"
        # Footer text (single line, includes bg/ball names if present)
        extra = ""
        if it.get("bg_name") or it.get("ball_name"):
            extra = f" • BG: {it.get('bg_name','')} • BALL: {it.get('ball_name','')}"
        self.info_var.set(
            f"{self.idx+1}/{len(self.items)} • {it['filename']} • {flag} • "
            f"X={it['cx']}  Y={it['cy']}  Size={it['size']} px{extra}"
        )

        # On-image overlay (multi-line)
        self._draw_overlay_text(it)

def main():
    csv_path = Path(CSV_PATH)
    images_dir = Path(IMAGES_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    rows = load_manifest(csv_path)
    if SHOW_ONLY_EXISTING:
        rows = [r for r in rows if (images_dir / r["filename"]).exists()]
        if not rows:
            print("No images found that exist on disk with the given CSV/paths.")
            return

    app = Viewer(rows, images_dir)
    app.mainloop()

if __name__ == "__main__":
    main()
