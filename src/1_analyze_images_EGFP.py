#KEEP THIS AS THE SAME FOLDER AS IMAGES
#SCRIPT FOLDER WITH SUBFOLDER NAMED WITH TREATMENT NAME AS SUBFOLDERS
#TIFF OR JPEG IMAGES SORTED IN TREATMENT SUBFOLDERS SHOULD HAVE CELL LINE NAME INCLUDED #SEE LINE 24
#ALSO OUTPUT IMAGES WITH DAPI MASK OUTLINE


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import io, measure
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, disk
from scipy.stats import skew, kurtosis

from cellpose import models

# -------- CONFIG --------
script_path = os.path.abspath(__file__)
base_folder = os.path.dirname(script_path)
treatments = ["A", "B", "C", "D"] #INSERT TREATMENT NAMES
cell_lines = ["Alpha", "Beta", "Gamma", "Delta"]
output_csv = os.path.join(base_folder, "EGFP_CV_results.csv") #EGFP CHANNEL CV
output_plot_folder = os.path.join(base_folder, "EGFP_CV_boxplots_by_treatment")
qc_folder = os.path.join(base_folder, "EGFP_CV_QC_overlays")
valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Thresholding (backup if Otsu fails)
threshold_percentile = 10
MIN_GFP_MEAN = 10
bg_radius = 10  # pixels for local background ring
# ------------------------

os.makedirs(output_plot_folder, exist_ok=True)
os.makedirs(qc_folder, exist_ok=True)

# --- Init Cellpose model ---
use_gpu = True  # safer for M1/M2/M3 Macs
try:
    model = models.CellposeModel(gpu=True, pretrained_model="nuclei")
    print("✅ Loaded Cellpose nuclei model")
except Exception as e:
    print(f"⚠️ Failed to load Cellpose model: {e}")
    raise

results = []

# --- Loop through treatments and images ---
for treatment in treatments:
    # dynamic folder detection (case-insensitive)
    found_folder = None
    for f in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, f)) and treatment.lower() in f.lower():
            found_folder = os.path.join(base_folder, f)
            break
    if not found_folder:
        print(f"⚠️ Treatment folder for '{treatment}' not found, skipping.")
        continue
    treatment_folder = found_folder

    for root, _, files in os.walk(treatment_folder):
        folder_name = os.path.basename(root).upper()

        # detect cell line from folder name
        cell_line = "Unknown"
        for line in cell_lines:
            if line.upper() in folder_name:
                cell_line = line
                break

        # find DAPI and EGFP images
        dapi_files = [f for f in files if "DAPI" in f.upper() and f.lower().endswith(valid_ext)]
        gfp_files = [f for f in files if "EGFP" in f.upper() and f.lower().endswith(valid_ext)]

        if not dapi_files or not gfp_files:
            continue

        for dapi_fname in dapi_files:
            dapi_path = os.path.join(root, dapi_fname)
            gfp_path = os.path.join(root, gfp_files[0])  # take first EGFP

            # Load images
            dapi = io.imread(dapi_path)
            gfp = io.imread(gfp_path)

            # Convert to grayscale if RGB
            if dapi.ndim == 3:
                dapi = (rgb2gray(dapi) * 255).astype(np.uint8)
            if gfp.ndim == 3:
                gfp = (rgb2gray(gfp) * 255).astype(np.uint8)

            # --- Segment nuclei with Cellpose ---
            try:
                dapi_f32 = dapi.astype(np.float32)

                masks, flows, styles = model.eval(
                    [dapi_f32],
                    diameter=150,
                    flow_threshold=0.4,
                    cellprob_threshold=-6.0,
                    normalize=True
                )

                nuclei_mask = masks[0]

            except Exception as e:
                print(f"⚠️ Cellpose failed on {dapi_fname}: {e}")
                continue

            # --- Measure GFP per nucleus (with background subtraction) ---
            region_data = []
            for region in measure.regionprops(nuclei_mask, intensity_image=gfp):
                intensities = region.intensity_image[region.image]
                mean_intensity = intensities.mean()

                # local background ring
                dilated = dilation(region.image, disk(bg_radius))
                ring_mask = dilated ^ region.image
                if ring_mask.sum() > 0:
                    bg_vals = region.intensity_image[ring_mask]
                    bg_mean = bg_vals.mean() if bg_vals.size > 0 else 0
                else:
                    bg_mean = 0

                corrected_intensity = max(mean_intensity - bg_mean, 0)
                region_data.append((region, corrected_intensity, mean_intensity, bg_mean))

            if len(region_data) < 5:
                print(f"⚠️ Too few nuclei found in {dapi_fname}, skipping.")
                continue

            mean_intensities = np.array([corr for (_, corr, _, _) in region_data])

            # Adaptive threshold (Otsu first, fallback to percentile)
            if mean_intensities.std() < 1e-6:
                bg_threshold = max(np.percentile(mean_intensities, threshold_percentile), MIN_GFP_MEAN)
            else:
                try:
                    bg_threshold = threshold_otsu(mean_intensities)
                except Exception:
                    bg_threshold = max(np.percentile(mean_intensities, threshold_percentile), MIN_GFP_MEAN)

            # --- Apply filtering ---
            kept_labels = set()
            dim_labels = set()
            interphase_labels = set()

            for region, corr_intensity, raw_intensity, bg_mean in region_data:
                if corr_intensity < max(bg_threshold, MIN_GFP_MEAN):
                    dim_labels.add(region.label)
                    continue

                intensities = region.intensity_image[region.image]
                std_intensity = intensities.std()
                cv_intensity = std_intensity / corr_intensity if corr_intensity > 0 else np.nan

                skewness = skew(intensities, bias=False)
                kurt = kurtosis(intensities, bias=False)

                # Interphase exclusion
                if cv_intensity < 0.12 and abs(skewness) < 0.5 and kurt < 1.0:
                    interphase_labels.add(region.label)
                    continue

                # Keep valid nuclei
                kept_labels.add(region.label)
                results.append({
                    "treatment": treatment,
                    "cell_line": cell_line,
                    "folder": folder_name,
                    "image": dapi_fname,
                    "nucleus_label": region.label,
                    "mean_intensity_raw": raw_intensity,
                    "mean_intensity_corrected": corr_intensity,
                    "local_bg_mean": bg_mean,
                    "cv_intensity": cv_intensity,
                    "skewness": skewness,
                    "kurtosis": kurt,
                    "background_threshold": bg_threshold,
                    "filter_reason": "kept"
                })

            # --- QC overlays (green = kept, red = dim, red = interphase) ---
            gfp_rgb = np.zeros((*gfp.shape, 3), dtype=float)
            gfp_rgb[..., 1] = gfp / gfp.max()  # pure green background

            overlay = gfp_rgb.copy()

            for region, _, _, _ in region_data:
                mask_single = (nuclei_mask == region.label).astype(np.uint8)
                boundaries = find_boundaries(mask_single, mode="outer")

                if region.label in kept_labels:
                    overlay[boundaries] = (0, 1, 0)  # green
                elif region.label in dim_labels:
                    overlay[boundaries] = (1, 0, 0)  # red
                elif region.label in interphase_labels:
                    overlay[boundaries] = (1, 0, 0)  # red

            qc_name = f"{treatment}_{cell_line}_{os.path.splitext(dapi_fname)[0]}_EGFP_outline_overlay.png"
            qc_path = os.path.join(qc_folder, qc_name)
            plt.imsave(qc_path, overlay)
            print(f"📸 Saved outline overlay: {qc_path}")

# --- Save CSV ---
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"✅ Saved EGFP CV results to {output_csv}")