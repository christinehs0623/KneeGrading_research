import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------------------------------
# 1. Task order and display names
# --------------------------------------------------
tasks = ["JSNM", "JSNL", "OSFM", "OSTM", "OSTL", "OSFL"]

# Example:
# labels_dict = {
#     "JSNM": y_true_jsnm,
#     "JSNL": y_true_jsnl,
#     ...
# }
# preds_dict = {
#     "JSNM": y_pred_jsnm,
#     "JSNL": y_pred_jsnl,
#     ...
# }

# --------------------------------------------------
# 2. Create figure (one row, six columns)
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 6,
    figsize=(24, 5),   # wide figure is KEY
    sharey=True
)

# --------------------------------------------------
# 3. Plot each confusion matrix
# --------------------------------------------------
for ax, task in zip(axes, tasks):

    disp = ConfusionMatrixDisplay.from_predictions(
        labels_dict[task],
        preds_dict[task],
        normalize="true",
        values_format=".2f",
        cmap=plt.cm.Greens,
        ax=ax
    )

    # ---- Cell text (numbers inside matrix)
    for text in ax.texts:
        text.set_fontsize(18)
        text.set_fontweight("bold")

    # ---- Tick labels
    ax.tick_params(axis="both", which="major", labelsize=14)

    # ---- Axis labels (only leftmost keeps Y label)
    ax.set_xlabel("Predicted", fontsize=16)
    if ax is axes[0]:
        ax.set_ylabel("True", fontsize=16)
    else:
        ax.set_ylabel("")

    # ---- Title
    ax.set_title(task, fontsize=18, pad=8)

    # ---- Colorbar font (if exists)
    if disp.im_.colorbar is not None:
        disp.im_.colorbar.ax.tick_params(labelsize=12)

# --------------------------------------------------
# 4. Layout & save
# --------------------------------------------------
plt.tight_layout()
plt.savefig("confusion_matrices_oarsi.png", dpi=150, bbox_inches="tight")
plt.savefig("confusion_matrices_oarsi.pdf", dpi=150, bbox_inches="tight")
plt.show()
