# Volume 1: Satellite Data Engineering & Radiometry Guide

**Project:** Multi-Temporal Satellite Super-Resolution (Sentinel-2 to Airbus SPOT)  
**Dataset:** WorldStrat (JIF Configuration, ESA & Airbus Paired Dataset)  
**Resolution Mapping:** 10m Ground Sample Distance (GSD) $\rightarrow$ 1.5m GSD ($\approx 6.67\times$ scaling)

---

## 1. Multi-Spectral Satellite Data & Physical Radiometry

### 1.1 The Sentinel-2 Multi-Spectral Spectrum (12 Bands)
Unlike 3-channel RGB photography, Sentinel-2 Level-2A (L2A) imagery captures Earth surface reflectance across 12 distinct optical bands. Each wavelength band targets specific physical properties:

| Band Index | S2 Band | Spectral Region | Central Wavelength | Native GSD | Primary Remote Sensing Utility |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **B1** | Deep Coastal Blue | $443\text{ nm}$ | $60\text{m}$ | Coastal aerosols, shallow water, bathymetry |
| **1** | **B2** | **Blue** | $490\text{ nm}$ | **$10\text{m}$** | True-color display, soil vs vegetation |
| **2** | **B3** | **Green** | $560\text{ nm}$ | **$10\text{m}$** | True-color display, vegetation green peak |
| **3** | **B4** | **Red** | $665\text{ nm}$ | **$10\text{m}$** | True-color display, chlorophyll absorption |
| **4** | **B5** | Red Edge 1 | $705\text{ nm}$ | $20\text{m}$ | Vegetation stress, leaf area index (LAI) |
| **5** | **B6** | Red Edge 2 | $740\text{ nm}$ | $20\text{m}$ | Vegetation health and canopy chlorophyll |
| **6** | **B7** | Red Edge 3 | $783\text{ nm}$ | $20\text{m}$ | Biomass estimation |
| **7** | **B8** | **Broad NIR** | $842\text{ nm}$ | **$10\text{m}$** | High surface reflectance, land/water boundary |
| **8** | **B8A**| Narrow NIR | $865\text{ nm}$ | $20\text{m}$ | Atmospheric correction, water vapor reference |
| **9** | **B9** | Water Vapor | $945\text{ nm}$ | $60\text{m}$ | Atmospheric absorption correction |
| **10**| **B11**| SWIR 1 | $1610\text{ nm}$ | $20\text{m}$ | Moisture content, cloud/snow differentiation |
| **11**| **B12**| SWIR 2 | $2190\text{ nm}$ | $20\text{m}$ | Soil geology, burn scar detection |

> [!NOTE]
> GDAL and GeoTIFF band indices are 1-indexed (`1` to `12`). In PyTorch and NumPy, arrays are 0-indexed (`0` to `11`). True-color RGB corresponds to Bands $[4, 3, 2]$, which in zero-index notation is $[3, 2, 1]$.

---

### 1.2 Radiometric Precision: 12-Bit Depth vs. 8-Bit RGB
* **Standard Images (JPEG / PNG):** 8-bit unsigned integers $[0, 255]$ with non-linear sRGB gamma encoding.
* **Sentinel-2 L2A Imagery:** 12-bit physical surface reflectance (Bottom-Of-Atmosphere, BOA). Values represent physical dimensionless reflectance scaled by $10,000$ (e.g., $1,000 = 0.10$ reflectance).
* **Airbus SPOT 6/7 Imagery:** High-resolution optical sensors recording 12-bit digital numbers normalized by $4,095$ (8-bit equivalent $255$).

---

## 2. The Radiometry & Daylight Visualization Breakthrough

### 2.1 The Mystery of the Pitch-Black Inputs & Nighttime Streetlights
During initial exploratory analysis, standard `plt.imshow(img)` produced two major visual defects:
1. **Sentinel-2 Input appeared pitch black:** Earth surface reflectance is physically low ($0.02\text{--}0.11$). When divided by $1.0$, pixel values are $\approx 0.04$, which renders as solid black on computer monitors.
2. **Airbus SPOT Ground Truth looked like night streetlights:** Highly reflective surfaces (e.g., corrugated metal roofs, solar panels, specular water glints) recorded outlier values up to $1,088$ (median was $92$). Linear min-max scaling ($x / x_{\max}$) squashed the entire landscape to $0.08$, rendering the daylight image as nighttime with bright specular light bulbs.

### 2.2 The Solution: 2%–98% Percentile Daylight Stretching
In professional remote sensing, images are not displayed using linear min-max scaling. Instead, a **$2\%\text{--}98\%$ cumulative percentile contrast stretch** is applied per channel:

$$\hat{x} = \text{clip}\left(\frac{x - P_2}{P_{98} - P_2 + \epsilon}, \; 0.0, \; 1.0\right)$$

* **$P_2$:** Discards dark shadow outliers and sensor noise floor.
* **$P_{98}$:** Discards specular metal roof glints and cloud saturation.
* **The Result:** Satellite landscapes appear in natural daylight with full dynamic contrast while preserving physical raw tensors for PSNR/SSIM evaluation.

---

## 3. DataLoader Engineering & Multi-Temporal Contracts

### 3.1 The 5D Multi-Temporal Batch Contract
Unlike single-image super-resolution (SISR) which uses 4D tensors `(B, C, H, W)`, multi-frame super-resolution (MFSR) processes time-series satellite revisits in 5D:

| Key | Tensor Shape | Dtype | Range | Description |
| :--- | :--- | :---: | :---: | :--- |
| **`lr`** | `(B, 8, 12, 160, 160)` | `float32` | $\approx \mathcal{N}(0, 1)$ | 8 temporal S2 revisits, 12 bands, Z-score normalized |
| **`lrc`** | `(B, 8, 1, 160, 160)` | `float32` | $[0, 11]$ | Categorical Cloud Mask (CLM) per revisit |
| **`hr_pansharpened`**| `(B, 1, 3, 1054, 1054)` | `float32` | $[0.0, 1.0]$ | Target Airbus SPOT True-Color RGB ($1.5\text{m}$ GSD) |

### 3.2 Python 3.14 Multiprocessing Deadlock & The Fix
* **The Problem:** In Python 3.14, using `num_workers > 0` with inline `lambda` functions inside torchvision transforms crashes with `PicklingError` or deadlocks during inter-process serialization.
* **The Engineering Fix:**
  1. Set `num_workers=0` in `DataLoader` instances.
  2. Implement deterministic splitting with explicit `torch.Generator().manual_seed(42)` and a `seed_worker` callback.
  3. Replace inline lambda serialization with standalone top-level functions.

---

## 4. De-Normalization & Cross-Sensor Domain Matching

### 4.1 The WorldStrat Baseline Bug Audit
In the reference repository, the heuristic bicubic baseline contained two critical bugs:
1. **Inverted Order-of-Operations in De-Normalization:** The code subtracted mean before dividing by standard deviation, inverting standard Z-score math:
   $$\text{Correct: } x_{\text{denorm}} = x \cdot \sigma + \mu \quad \text{vs. Wrong: } (x - \mu) / \sigma$$
2. **Radiometric Destruction via `x /= torch.max(x)`:** Normalizing by per-image max destroyed calibrated surface reflectance, causing severe color drift.

### 4.2 The Golden Rule of Benchmark Evaluation
* **Neural Networks (HighResNet):** Operate in Z-score normalized space ($\mu=0, \sigma=1$) for smooth gradient propagation. The network learns the linear mapping from standardized Sentinel-2 to $[0.0, 1.0]$ SPOT.
* **Heuristics (Bicubic Baseline):** Have no learnable parameters. They MUST receive de-normalized physical reflectance values, or the baseline PSNR collapses from $30.03\text{ dB}$ to $<15\text{ dB}$.
