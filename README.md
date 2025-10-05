Fluorescence heterogeneity across individual cells was quantified using a coefficient of variation (CV)–based metric computed from image-segmented single-cell intensity measurements.

Cell nuclei or GFP-positive regions were segmented using a hybrid approach combining pre-trained machine learning segmentation model [![Cellpose](https://img.shields.io/badge/GitHub-MouseLand%2Fcellpose-181717?logo=github)](https://github.com/MouseLand/cellpose) (for cell boundary initialization) and intensity thresholding in the GFP channel for refinement.

The variability in expression within a population was expressed as the coefficient of variation (CV) of pixel intensity for each image field or individual cell population.


⸻

**Before you start**

All conda .ymal were exported and tested on apple silicon OSX-ARM64 platform

⸻

**Installation**

You can resolve environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```
cd [the location of src folder]
```

activate this for segmentation and calculation #1 and #3 script#
```
conda env create --file=co_var_image_analysis.yaml 
```

activate this for plotting #2 and #4 script#
```
conda env create --file=co_var_image_analysis_plot.yaml 
```
⸻

**Usage**

```
conda activate co_var_image_analysis
python ~/src/1_analyze_images_EGFP.py
```

&

```
conda activate co_var_image_analysis_plot
python ~/src/2_plot_EGFP_CV_raincloud_vertical.py
```
⸻

**Definition**

For each segmented nucleus $$\sigma_i$$, let
$$\bar{I}i = \{ I_{i1}, I_{i2}, \dots, I_{in_i} \}$$
be the set of GFP pixel intensities within that nucleus.

The coefficient of variation (CV) is defined as:

$$
\text{CV}_i = \frac{\sigma_i}{\mu_i^{\text{corr}}}
$$

where:
$$\sigma_i$$: standard deviation of GFP intensities within nucleus $$\sigma_i$$


$$
\sigma_i = \sqrt{\frac{1}{n_i - 1} \sum_{j=1}^{n_i} (I_{ij} - \bar{I}_i)^2}
$$

$$\bar{I}i$$ : mean GFP intensity inside the nucleus

$$
\bar{I}i = \frac{1}{n_i} \sum{j=1}^{n_i} I{ij}
$$

$$\mu_i^{\text{corr}}$$ : background-corrected mean intensity


$$\mu_i^{\text{corr}} = \max(\bar{I}_i - \bar{B}_i,, 0)$$
where $$\bar{B}_i$$ is the mean intensity of the local background ring surrounding nucleus  $$\sigma_i$$.

⸻

**Interpretation**

	•	The numerator $$\sigma_i$$ represents intranuclear variability of GFP intensity — higher values indicate stronger spatial heterogeneity (e.g., speckles or puncta).
	•	The denominator $$\mu_i^{\text{corr}}$$ normalizes this variability to the corrected mean brightness, producing a dimensionless ratio independent of absolute intensity.
	•	Thus, $$\text{CV}_i$$ reflects the relative dispersion of signal within each nucleus and serves as a quantitative descriptor of texture or punctate patterning.

⸻

**Final Expression**

The full background-corrected coefficient of variation is:

$$
\boxed{
\text{CV}i =
\frac{
\sqrt{
\dfrac{1}{n_i - 1}
\sum{j=1}^{n_i} (I_{ij} - \bar{I}_i)^2
}
}{
\max(\bar{I}_i - \bar{B}_i,, 0)
}
}
$$

⸻

**Calculation for other channels**

eg. Cy5/Cy3
	•	EGFP were used as a metric to include the cell for channal CV calculation. See _3_Cy5_CV_from_EGFP_Positive.py_ and input your desired channel. 

⸻

**Output Example**
<img width="1930" height="528" alt="image" src="https://github.com/user-attachments/assets/46520588-0c1f-4f10-9f4c-5561183e4620" />
