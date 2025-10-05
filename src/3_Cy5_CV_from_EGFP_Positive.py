import os
import numpy as np
import pandas as pd
from skimage import io, measure
from skimage.color import rgb2gray
from skimage.morphology import dilation, disk
from scipy.stats import skew, kurtosis

# -------- CONFIG --------
script_path = os.path.abspath(__file__)
base_folder = os.path.dirname(script_path)

# input EGFP results to identify GFP+ cells
egfp_csv = os.path.join(base_folder, "EGFP_CV_results.csv")
output_csv = os.path.join(base_folder, "Cy5_CV_results.csv")

# same folder layout
treatments = ["A", "B", "C", "D"] #INSERT TREATMENT NAMES
cell_lines = ["Alpha", "Beta", "Gamma", "Delta"]
valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# analysis parameters
bg_radius = 10
MIN_SIGNAL = 10
threshold_percentile = 10
# ------------------------

# --- load EGFP+ nuclei ---
egfp_df = pd.read_csv(egfp_csv)
egfp_positive = egfp_df[egfp_df["filter_reason"] == "kept"]
print(f"✅ Loaded {len(egfp_positive)} EGFP+ nuclei")

results = []

# --- iterate over all EGFP+ nuclei groups (by image) ---
for (treatment, cell_line, folder, image), subset in egfp_positive.groupby(["treatment", "cell_line", "folder", "image"]):
    print(f"\n📂 Processing {treatment} / {cell_line} / {image}")

    # detect folder containing image
    treatment_folder = None
    for f in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, f)) and treatment.lower() in f.lower():
            treatment_folder = os.path.join(base_folder, f)
            break
    if not treatment_folder:
        print(f"⚠️ Missing folder for {treatment}")
        continue

    # locate Cy5 image in the same folder as DAPI
    for root, _, files in os.walk(treatment_folder):
        if folder.lower() not in root.lower():
            continue
        cy5_files = [f for f in files if "CY5" in f.upper() and f.lower().endswith(valid_ext)]
        if not cy5_files:
            continue
        cy5_path = os.path.join(root, cy5_files[0])

        # load Cy5
        cy5 = io.imread(cy5_path)
        if cy5.ndim == 3:
            cy5 = (rgb2gray(cy5) * 255).astype(np.uint8)

        # use the same segmentation mask from EGFP stage
        # (requires same naming pattern for DAPI → you can reuse your DAPI segmentation)
        # Here, we resegment DAPI just like before:
        dapi_files = [f for f in files if "DAPI" in f.upper() and f.lower().endswith(valid_ext)]
        if not dapi_files:
            print(f"⚠️ No DAPI found for {image}")
            continue

        dapi_path = os.path.join(root, dapi_files[0])
        dapi = io.imread(dapi_path)
        if dapi.ndim == 3:
            dapi = (rgb2gray(dapi) * 255).astype(np.uint8)

        # --- segment nuclei quickly (same params as before) ---
        from cellpose import models
        model = models.CellposeModel(gpu=True, pretrained_model="nuclei")

        try:
            masks, _, _ = model.eval([dapi.astype(np.float32)],
                                     diameter=150,
                                     flow_threshold=0.4,
                                     cellprob_threshold=-6.0,
                                     normalize=True)
            nuclei_mask = masks[0]
        except Exception as e:
            print(f"⚠️ Cellpose failed on {image}: {e}")
            continue

        # --- quantify Cy5 in only EGFP+ nuclei ---
        for region in measure.regionprops(nuclei_mask, intensity_image=cy5):
            if region.label not in subset["nucleus_label"].values:
                continue  # skip EGFP− cells

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
            std_intensity = intensities.std()
            cv_intensity = std_intensity / corrected_intensity if corrected_intensity > 0 else np.nan
            skewness = skew(intensities, bias=False)
            kurt = kurtosis(intensities, bias=False)

            results.append({
                "treatment": treatment,
                "cell_line": cell_line,
                "folder": folder,
                "image": image,
                "nucleus_label": region.label,
                "mean_intensity_raw": mean_intensity,
                "mean_intensity_corrected": corrected_intensity,
                "local_bg_mean": bg_mean,
                "cv_intensity": cv_intensity,
                "skewness": skewness,
                "kurtosis": kurt,
                "filter_reason": "kept"  # inherited from GFP+
            })

# --- Save ---
df_out = pd.DataFrame(results)
df_out.to_csv(output_csv, index=False)
print(f"\n✅ Saved Cy5 CV results → {output_csv}")