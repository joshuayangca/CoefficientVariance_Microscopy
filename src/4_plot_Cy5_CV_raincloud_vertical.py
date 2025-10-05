import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# -------- CONFIG --------
script_path = os.path.abspath(__file__)
base_folder = os.path.dirname(script_path)
input_csv = os.path.join(base_folder, "Cy5_CV_results.csv")
output_plot_folder = os.path.join(base_folder, "Cy5_CV_halfviolin_boxpoints_clean")

treatments = ["A", "B", "C", "D"] #INSERT TREATMENT NAMES
cell_lines = ["Alpha", "Beta", "Gamma", "Delta"]
treat_colors = {
    "A": "#808080",
    "B": "#0039A6",
    "C": "#E66100",
    "D": "#750096",
}

remove_quantiles = (0.01, 0.99)   # trim global 1–99%
rng = np.random.default_rng(0)

# layout tuning
group_width   = 1     # span for treatments inside one cell line
group_spacing = 0.5     # extra space between cell line groups
violin_width  = 0.3     # full violin width before clipping
box_width     = 0.08    # thinner box
point_size    = 4
alpha_violin  = 0.65
alpha_points  = 0.8
alpha_box     = 0.8

# ------------------------

os.makedirs(output_plot_folder, exist_ok=True)

# --- Load & clean ---
df = pd.read_csv(input_csv)
df["cell_line"] = pd.Categorical(df["cell_line"], categories=cell_lines, ordered=True)
df["treatment"] = pd.Categorical(df["treatment"], categories=treatments,  ordered=True)

# --- Global outlier trim ---
lo = df["cv_intensity"].quantile(remove_quantiles[0])
hi = df["cv_intensity"].quantile(remove_quantiles[1])
df = df[(df["cv_intensity"] >= lo) & (df["cv_intensity"] <= hi)]
print(f"⚖️ Filtering outside {remove_quantiles[0]*100:.0f}–{remove_quantiles[1]*100:.0f}: {lo:.3f} → {hi:.3f}")

# --- FIGURE SETUP ---
sns.set_style("white")

n = len(cell_lines)
panel_size = 3.0  # inches per subplot (square)
fig_w = panel_size
fig_h = panel_size * n

# ✅ This defines `fig` and `axes` properly
fig, axes = plt.subplots(
    n, 1,
    figsize=(fig_w, fig_h),
    sharex=True, sharey=True,
    constrained_layout=False
)

# Ensure `axes` is always iterable (even if n=1)
if n == 1:
    axes = [axes]

xmin = df["cv_intensity"].min()
xmax = df["cv_intensity"].max()
xpad = 0.10 * (xmax - xmin if xmax > xmin else 1.0)
x_spacing = 0.6
x_centers = np.arange(len(treatments)) * x_spacing

# --- PLOT PER CELL LINE ---
for i, (ax, cl) in enumerate(zip(axes, cell_lines)):
    for j, tr in enumerate(treatments):
        vals = df.loc[
            (df["cell_line"] == cl) & (df["treatment"] == tr),
            "cv_intensity"
        ].dropna().values
        if vals.size == 0:
            continue

        color = treat_colors[tr]
        center = x_centers[j]

        # --- 1) Half-violin ---
        vp = ax.violinplot(
            [vals],
            positions=[center],
            widths=violin_width,
            showmeans=False, showmedians=False, showextrema=False
        )
        body = vp["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(alpha_violin)
        body.set_linewidth(1.0)
        clip_rect = mpatches.Rectangle(
            (center - 10.0, xmin - xpad * 5),
            width=10.0,
            height=(xmax - xmin) + xpad * 10,
            transform=ax.transData
        )
        body.set_clip_path(clip_rect)

        # --- 2) Box ---
        bp = ax.boxplot(
            [vals],
            positions=[center],
            widths=box_width,
            patch_artist=True, whis=(5, 95),
            showfliers=False, zorder=5
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=alpha_box, edgecolor="black", linewidth=1.0)
        for elem in ["whiskers", "caps", "medians"]:
            for line in bp[elem]:
                line.set(color="black", linewidth=1.0)

        # --- 3) Points ---
        jitter = rng.uniform(-0.05, 0.08, size=vals.size)
        ax.scatter(center + 0.05 + jitter, vals,
                   s=point_size, c=color, alpha=alpha_points,
                   edgecolors="none", zorder=2)

    # --- Axis formatting ---
    ax.set_xlim(-0.5, x_centers[-1] + 0.5)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(treatments, fontsize=9, fontweight='bold')

    # Hide bottom subplot x-labels
    if i == n - 1:
        ax.set_xticklabels([])

    #ax.set_ylabel(cl, fontsize=11, fontweight='bold', color='black', labelpad=8)

    # Y-axis ticks
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(2.5))

    # X/Y tick styling (ticks on all sides)
    ax.tick_params(which='minor', left=True, right=True)
    ax.tick_params(top=True, bottom=True, left=True, right=True, direction='in')
    ax.tick_params(axis='x', which='major', length=5, width=1.0, labelsize=9, color='black', top=True)
    ax.tick_params(axis='y', which='major', length=5, width=1.0, labelsize=9, color='black')

    # Make side minor ticks visible and styled 
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', length=3, width=0.8, color='black', direction='in')

    # --- Full rectangular spines ---
    for side in ['top', 'bottom', 'left']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color('black')

    ax.grid(False)

# --- Shared Y label ---
fig.supylabel("Coefficient of Variation (CV) of Nuclear Cy5-αSC35 Signal ", fontsize=15, fontweight='bold', x=0.05)

# --- Label treatments on top (colored) ---
top_ax = axes[0]  # top subplot

# Create top x-axis twin for colored labels
ax_top = top_ax.twiny()

# Match tick positions to treatments
x_spacing = 0.6
x_centers = np.arange(len(treatments)) * x_spacing
ax_top.set_xlim(top_ax.get_xlim())
ax_top.set_xticks(x_centers)

# Colored labels for each treatment
for tick_label, label in zip(ax_top.set_xticklabels(treatments), treatments):
    tick_label.set_color(treat_colors[label])
    tick_label.set_fontweight('bold')
    tick_label.set_fontsize(9)

# Style top axis only (no frame)
ax_top.tick_params(axis='x', which='major', length=0, width=0)
for side in ['top', 'bottom', 'left', 'right']:
    ax_top.spines[side].set_visible(False)

# --- Layout ---
plt.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08, hspace=0.1)

# --- Save ---
out = os.path.join(output_plot_folder, "Raincloud_Square_Ticks_TopLabel.png")
plt.savefig(out, dpi=600, bbox_inches=None, pad_inches=0, transparent=True)
plt.close()
print(f"✅ Saved → {out}")