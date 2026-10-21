"""
Visual Contact Sheet Generator.
Assembles 4x4 or NxN thumbnail grids into a single contact sheet image
with overlay labels for rapid human visual auditing.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont


def create_contact_sheet(
    image_paths: List[str],
    labels: List[str],
    output_path: str,
    cols: int = 4,
    thumb_size: int = 240,
    title: str = "Visual Audit Sheet"
) -> bool:
    """
    Generate an NxN contact sheet grid from individual image files.
    """
    if not image_paths:
        return False

    n = len(image_paths)
    rows = (n + cols - 1) // cols

    pad = 10
    header_h = 40
    footer_h = 24

    cell_w = thumb_size
    cell_h = thumb_size + footer_h

    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad + header_h

    sheet = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    # Draw Header
    draw.text((pad, 10), title, fill=(240, 240, 240))

    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        r = idx // cols
        c = idx % cols

        x = pad + c * (cell_w + pad)
        y = header_h + pad + r * (cell_h + pad)

        if os.path.exists(path):
            try:
                with Image.open(path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((cell_w, thumb_size), Image.Resampling.LANCZOS)
                    tx = x + (cell_w - im_rgb.width) // 2
                    ty = y + (thumb_size - im_rgb.height) // 2
                    sheet.paste(im_rgb, (tx, ty))
            except Exception:
                draw.rectangle([x, y, x + cell_w, y + thumb_size], fill=(60, 20, 20))
                draw.text((x + 10, y + 10), "ERR", fill=(255, 100, 100))
        else:
            draw.rectangle([x, y, x + cell_w, y + thumb_size], fill=(50, 50, 50))
            draw.text((x + 10, y + 10), "MISSING", fill=(180, 180, 180))

        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(80, 80, 80), width=1)
        short_lbl = label[:28] if len(label) > 28 else label
        draw.text((x + 5, y + thumb_size + 4), f"{idx+1}. {short_lbl}", fill=(200, 200, 200))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sheet.save(output_path, "JPEG", quality=85)
    return True
