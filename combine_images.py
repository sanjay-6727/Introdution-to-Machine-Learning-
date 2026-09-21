"""
combine_images.py - Combine each experiment's output images into grid images,
split into an EDA group and a Results group, so reports can use one figure
per section instead of dozens of separate files.

Usage:
    python combine_images.py

For each ExperimentN, writes (inside ExperimentN/outputs):
    ExperimentN_eda_combined.png / .eps      -> exploratory data analysis plots
    ExperimentN_results_combined.png / .eps  -> model results / evaluation plots
"""

import os
import glob

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)

THUMB_W, THUMB_H = 640, 480
CAPTION_H = 28
PADDING = 12
MARGIN = 30
NCOLS = 4

try:
    FONT = ImageFont.load_default(size=16)
    TITLE_FONT = ImageFont.load_default(size=28)
except TypeError:
    FONT = ImageFont.load_default()
    TITLE_FONT = ImageFont.load_default()

# how each experiment's images are split into "eda" vs "results"
EDA_MARKERS = {
    "Experiment_1": ["_eda", "00_library_exploration_demo", "digits_sample_images"],
    "Experiment_2": ["01_class_distribution", "eda"],
    "Experiment_3": ["01_target_distribution", "02_feature_vs_target_scatter", "eda"],
    "Experiment_4": ["01_class_distribution", "02_correlation_heatmap_full", "eda"],
}


def is_eda_image(exp_name, path, outputs_dir):
    rel = os.path.relpath(path, outputs_dir).replace("\\", "/")
    return any(marker in rel for marker in EDA_MARKERS[exp_name])


def build_grid(image_paths, outputs_dir, title):
    nrows = (len(image_paths) + NCOLS - 1) // NCOLS
    cell_w = THUMB_W + PADDING
    cell_h = THUMB_H + CAPTION_H + PADDING
    title_h = 60

    canvas_w = MARGIN * 2 + NCOLS * cell_w
    canvas_h = MARGIN * 2 + title_h + nrows * cell_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, MARGIN // 2), f"{title} ({len(image_paths)} images)",
              fill="black", font=TITLE_FONT)

    for i, path in enumerate(image_paths):
        row, col = divmod(i, NCOLS)
        x = MARGIN + col * cell_w
        y = MARGIN + title_h + row * cell_h

        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H))

        paste_x = x + (THUMB_W - img.width) // 2
        paste_y = y + (THUMB_H - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))

        label = os.path.relpath(path, outputs_dir).replace("\\", "/").replace(".png", "")
        draw.text((x, y + THUMB_H + 4), label[:60], fill="black", font=FONT)

    return canvas


def combine_experiment(exp_name):
    outputs_dir = os.path.join(HERE, exp_name, "outputs")
    if not os.path.isdir(outputs_dir):
        print(f"skip {exp_name}: no outputs folder")
        return

    all_paths = sorted(glob.glob(os.path.join(outputs_dir, "**", "*.png"), recursive=True))
    all_paths = [p for p in all_paths if "_combined" not in os.path.basename(p)]
    all_paths = [p for p in all_paths if "feature_selection" not in os.path.basename(p)]
    if not all_paths:
        print(f"skip {exp_name}: no png images found")
        return

    eda_paths = [p for p in all_paths if is_eda_image(exp_name, p, outputs_dir)]
    result_paths = [p for p in all_paths if p not in eda_paths]

    for group_name, paths in [("eda", eda_paths), ("results", result_paths)]:
        if not paths:
            continue
        print(f"{exp_name} [{group_name}]: combining {len(paths)} images")
        canvas = build_grid(paths, outputs_dir, f"{exp_name} - {group_name}")

        png_path = os.path.join(outputs_dir, f"{exp_name}_{group_name}_combined.png")
        canvas.save(png_path)
        print("saved:", png_path, f"  ({canvas.width}x{canvas.height}px)")

        eps_path = os.path.join(outputs_dir, f"{exp_name}_{group_name}_combined.eps")
        canvas.save(eps_path, "EPS", dpi=(600, 600))
        print("saved:", eps_path)


if __name__ == "__main__":
    for exp in ["Experiment_1", "Experiment_2", "Experiment_3", "Experiment_4"]:
        combine_experiment(exp)
