from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# 1. Row / column configuration (match current filename rules)
# ============================================================
row_names = ['original', 'Tiulpin', 'ours', 'multi']
kl_levels = ['KL0', 'KL1', 'KL2', 'KL3', 'KL4']
sides = ['medial', 'lateral']

img_dir = "./demo"

# ============================================================
# 2. Scan all images to determine the maximum image size
#    (so all subplots are aligned consistently)
# ============================================================
max_w, max_h = 0, 0

for row in row_names:
    # KL 0 (single image)
    fpath = os.path.join(img_dir, f"{row}_KL0.png")
    if os.path.exists(fpath):
        with Image.open(fpath) as im:
            max_w = max(max_w, im.size[0])
            max_h = max(max_h, im.size[1])

    # KL 1–4 (medial / lateral)
    for kl in kl_levels[1:]:
        for side in sides:
            fpath = os.path.join(img_dir, f"{row}_{kl}_{side}.png")
            if os.path.exists(fpath):
                with Image.open(fpath) as im:
                    max_w = max(max_w, im.size[0])
                    max_h = max(max_h, im.size[1])

# ============================================================
# 3. Create figure and GridSpec
#    KL0 = 1 column
#    KL1–KL4 = 2 columns each → total 9 columns
# ============================================================
fig = plt.figure(figsize=(26, 10))  # <-- key change: wider figure
gs = GridSpec(
    nrows=len(row_names),
    ncols=9,
    figure=fig,
    wspace=0.08,   # reduce horizontal spacing
    hspace=0.025     # slightly increase vertical spacing
)

fig.subplots_adjust(
    left=0.24,      # more space for row labels
    right=0.99,
    bottom=0.03,
    top=0.92
)

axes = {}

# ============================================================
# 4. Create all axes
# ============================================================
col_idx = 0
for kl in kl_levels:
    if kl == "KL0":
        for r in range(len(row_names)):
            axes[(r, kl)] = fig.add_subplot(gs[r, col_idx])
        col_idx += 1
    else:
        for side in sides:
            for r in range(len(row_names)):
                axes[(r, kl, side)] = fig.add_subplot(gs[r, col_idx])
            col_idx += 1

# ============================================================
# 5. Load and render images
# ============================================================
for i, row in enumerate(row_names):

    # ---- KL 0 ----
    ax = axes[(i, "KL0")]
    fpath = os.path.join(img_dir, f"{row}_KL0.png")
    if os.path.exists(fpath):
        img = Image.open(fpath).resize(
            (max_w, max_h), Image.Resampling.LANCZOS
        )
        ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- KL 1–4 ----
    for kl in kl_levels[1:]:
        for side in sides:
            ax = axes[(i, kl, side)]
            fpath = os.path.join(img_dir, f"{row}_{kl}_{side}.png")
            if os.path.exists(fpath):
                img = Image.open(fpath).resize(
                    (max_w, max_h), Image.Resampling.LANCZOS
                )
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

    # ---- Row labels (left side) ----
    if row == "ours":
        label = "ours (KL)"
    elif row == "multi":
        label = "ours (KL + OARSI)"
    else:
        label = row

    axes[(i, "KL0")].set_ylabel(
        label,
        fontsize=22,
        rotation=0,
        va="center",
        ha="right",
        labelpad=22
    )

# ============================================================
# 6. Column titles
# ============================================================

# KL 0 title
axes[(0, "KL0")].set_title("KL 0", fontsize=22, pad=14)

# KL 1–4 titles + medial / lateral
# for kl in kl_levels[1:]:
#     left = axes[(0, kl, "medial")]
#     right = axes[(0, kl, "lateral")]

#     # Center KL title across medial + lateral
#     x = (left.get_position().x0 + right.get_position().x1) / 2
#     y = left.get_position().y1 + 0.025
#     fig.text(
#         x, y,
#         kl.replace("KL", "KL "),
#         ha="center",
#         fontsize=22
#     )

#     # Side titles (larger and clearer)
#     left.set_title("medial", fontsize=16, pad=6)
#     right.set_title("lateral", fontsize=16, pad=6)
for kl in kl_levels[1:]:
    left = axes[(0, kl, "medial")]
    right = axes[(0, kl, "lateral")]

    # ---- KL title (centered over medial + lateral) ----
    x_center = (left.get_position().x0 + right.get_position().x1) / 2
    y_top = left.get_position().y1 + 0.030

    fig.text(
        x_center, y_top,
        kl.replace("KL", "KL "),
        ha="center",
        va="bottom",
        fontsize=22
    )

    # ---- medial / lateral (figure-level text) ----
    y_side = left.get_position().y1 + 0.006

    fig.text(
        (left.get_position().x0 + left.get_position().x1) / 2,
        y_side,
        "medial",
        ha="center",
        va="bottom",
        fontsize=20
    )

    fig.text(
        (right.get_position().x0 + right.get_position().x1) / 2,
        y_side,
        "lateral",
        ha="center",
        va="bottom",
        fontsize=18
    )

# ============================================================
# 7. Save figure
# ============================================================
fig.savefig(
    os.path.join(img_dir, "cam_comparison.png"),
    dpi=150,
    bbox_inches="tight"
)
fig.savefig(
    os.path.join(img_dir, "cam_comparison.pdf"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)
