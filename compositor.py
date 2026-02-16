"""
Composite multiple images into one 16:9 frame for I2V first-frame input.
Used when a shot requires multiple ingredients (or last frame + new ingredients).
"""
from pathlib import Path
from typing import List

# Optional: Pillow for compositing; fallback to single image if not available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Target 16:9 for video (e.g. 1280x720)
COMPOSITE_WIDTH = 1280
COMPOSITE_HEIGHT = 720


def composite_images(image_paths: List[Path], out_path: Path) -> Path:
    """
    Place N images in a grid on a 16:9 canvas (1280x720).
    If only one image, resize to canvas and save.
    If PIL is not available and N > 1, copy the first image and save (fallback).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        raise ValueError("composite_images requires at least one image path")

    if not HAS_PIL:
        if len(image_paths) == 1:
            import shutil
            shutil.copy2(image_paths[0], out_path)
            return out_path
        # Fallback: use first image only
        import shutil
        shutil.copy2(image_paths[0], out_path)
        return out_path

    images = []
    for p in image_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        images.append(img)

    if len(images) == 1:
        out = images[0].copy()
        out = out.resize((COMPOSITE_WIDTH, COMPOSITE_HEIGHT), Image.Resampling.LANCZOS)
        out.save(out_path)
        return out_path

    # Grid: fit N images (e.g. 2 -> 1x2, 4 -> 2x2, 3 -> 1x3 or 2x2 with one empty)
    n = len(images)
    if n <= 2:
        cols, rows = 2, 1
    elif n <= 4:
        cols, rows = 2, 2
    else:
        cols = 3
        rows = (n + cols - 1) // cols

    cell_w = COMPOSITE_WIDTH // cols
    cell_h = COMPOSITE_HEIGHT // rows

    canvas = Image.new("RGB", (COMPOSITE_WIDTH, COMPOSITE_HEIGHT), (30, 30, 30))

    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        x = col * cell_w
        y = row * cell_h
        # Scale to fit cell (keep aspect ratio, then crop or pad)
        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        # Center in cell
        px = x + (cell_w - img.width) // 2
        py = y + (cell_h - img.height) // 2
        canvas.paste(img, (px, py))

    canvas.save(out_path)
    return out_path
