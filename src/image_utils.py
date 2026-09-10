"""Image utility: enrich any image to 16:9 1920×1080 with blurred-fill background."""
from __future__ import annotations
from pathlib import Path
from typing import Union


def enrich_to_16x9(
    src: Union[str, Path],
    dst: Union[str, Path],
    out_w: int = 1920,
    out_h: int = 1080,
    blur_radius: int = 60,
) -> Path:
    """Fit image into 16:9 canvas; fill bars with blurred+darkened version of same image."""
    from PIL import Image, ImageFilter
    import numpy as np

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    iw, ih = img.size
    target_ratio = out_w / out_h

    # Background: scale to fill entire canvas, then heavy blur + darken
    bg_scale = max(out_w / iw, out_h / ih)
    bg = img.resize((int(iw * bg_scale), int(ih * bg_scale)), Image.LANCZOS)
    # crop to exact canvas
    bg_x = (bg.width  - out_w) // 2
    bg_y = (bg.height - out_h) // 2
    bg = bg.crop((bg_x, bg_y, bg_x + out_w, bg_y + out_h))
    bg = bg.filter(ImageFilter.GaussianBlur(blur_radius))
    # darken so subject stands out
    bg_arr = (np.array(bg) * 0.45).astype("uint8")
    canvas = Image.fromarray(bg_arr)

    # Foreground: fit inside canvas keeping aspect ratio
    fg_scale = min(out_w / iw, out_h / ih)
    fg = img.resize((int(iw * fg_scale), int(ih * fg_scale)), Image.LANCZOS)
    x = (out_w - fg.width)  // 2
    y = (out_h - fg.height) // 2
    canvas.paste(fg, (x, y))

    canvas.save(str(dst), quality=95)
    return dst
