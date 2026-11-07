# Project Milestone Checkpoint & Technical Roadmap

**Project:** Deep Learning-Based Super-Resolution Mapping (Sentinel-2 10m LR $\rightarrow$ Airbus SPOT 1.5m HR, $\approx 6.67\times$ Upscaling)  
**Primary Dataset:** WorldStrat (`archive_1/`, JIF configuration)  
**Primary Workspace:** `/mnt/storage/OneDrive_4_14-04-2025/Super_resolution`  
**Active Notebook:** `testing.ipynb`  
**Last Updated:** November 2026  

---

## 📊 High-Level Status Dashboard

| Phase | Description | Status | Progress |
| :--- | :--- | :---: | :---: |
| **Phase 0** | **Dataset Profiling, Physics Normalization & EDA** | **COMPLETED** | 100% |
| **Phase 1** | **Pure PyTorch Model Backbone & Architecture Dissection** | **COMPLETED** | 100% |
| **Phase 2** | **Scientific Benchmark & Metric Evaluation Engine** | **COMPLETED** | 100% |
| **Phase 3** | **Interactive Geospatial & Visual Showcase** | **COMPLETED** | 100% |
| **Phase 4** | **Explainable AI (LAM), Transformers & Future Extensions** | **PLANNED / ROADMAP** | 0% |

---

## 📌 Phase 0: Dataset Exploration & Preprocessing Calibration
> **Objective:** Validate sensor physics, compute exact global statistics, build multi-revisit PyTorch Datasets, and confirm tensor shapes.

- [x] **Step 0.1: Parallel Global Statistics Profiling**
  - **File:** `compute_dataset_stats.py`
  - Profiled all 35,353 images across CPU cores in 95s using `multiprocessing` (`fork` context).
  - *Sentinel-2 (10m LR):* Global range `[-3.2768, 3.2767]`. Band means: R `0.1600`, G `0.1445`, B `0.1175`.
  - *Airbus SPOT (1.5m HR):* Raw values up to `46,407.0`. Band means: R `1128.4`, G `1011.2`, B `845.1`.

- [x] **Step 0.2: Atmospheric & Cross-Sensor Normalization Strategy**
  - **Target Scaling:** Discovered SPOT values divided by `10000.0` align directly to physical surface reflectance ($[0.0, 1.0]$, mean $\approx 0.10$).
  - **Input Standardization:** Applied band-wise Z-score normalization $(x - \mu)/\sigma$ using `JIF_S2_MEAN` and `JIF_S2_STD` to center inputs at zero.
  - **Negative Values Handled:** Explained Sen2Cor atmospheric overcorrection over dark pixels (water/shadow).

- [x] **Step 0.3: Multi-View PyTorch Dataset Pipeline**
  - **File:** `testing.ipynb` (Cells 27–36)
  - Configured `SatelliteDataset`, `DictDataset`, and `make_transforms_JIF(lr_bands_to_use='true_color', radiometry_depth=12)`.
  - Ingests 5 modalities: `lr` (S2 input), `lrc` (cloud mask), `hr` (SPOT RGB), `hr_pan` (panchromatic), `hr_pansharpened` (super-res target).

- [x] **Step 0.4: DataLoader & Batch Shape Verification**
  - **File:** `testing.ipynb` (Cell 54)
  - Confirmed batch tensor contracts:
    - `batch['lr']`: `torch.Size([2, 8, 3, 160, 160])`, dtype `float32`, mean $\approx 0$.
    - `batch['lrc']`: `torch.Size([2, 8, 1, 160, 160])`, dtype `float32`, cloud labels.
    - `batch['hr_pansharpened']`: `torch.Size([2, 1, 3, 1054, 1054])`, dtype `float32`, range $[0.0, 1.0]$.

- [x] **Step 0.5: Distribution & Physical Alignment Visualization**
  - **File:** `testing.ipynb` (Cell 59)
  - Plotted 3-panel distribution curves: Input Z-score curve, Target physical reflectance histogram, and Sentinel-2 vs. SPOT cross-sensor alignment chart.

- [x] **Step 0.6: Comprehensive Knowledge Base Saved in `learnings/`**
  - `learnings/notes.txt`: Full roadmap, PyTorch `.item()` memory leak prevention, 5-modality transform rules, 12-band vs 3-band physics, and why standard pretrained models (ESRGAN) cannot be used directly.
  - `learnings/radiometric_depth.txt`: 8-bit vs. 12-bit sensor dynamics.
  - `learnings/spot_10000_scaling_notes.txt`: SentinelHub quantification scaling.
  - `learnings/target_range_and_negative_values.txt`: Physical reflectance bounds.

---

## 🚀 Phase 1: Pure PyTorch Model & Architecture Dissection (COMPLETED)
> **Objective:** Transition from exploratory data loading to a working model forward pass, pure PyTorch unpickling, and structural model auditing.

- [x] **Step 1.1: Multi-Split DataLoaders & Tensor Contracts**
  - **File:** `testing.ipynb`
  - Created deterministic Train, Val, and Test splits.
  - Wrapped into pure PyTorch DataLoaders with `num_workers=0` (preventing lambda serialization crashes) and verified shapes (`lr`: `(2, 8, 12, 160, 160)`, `hr_pansharpened`: `(2, 1, 3, 1054, 1054)`).

- [x] **Step 1.2: Define Pure PyTorch `HighResNet` Architecture**
  - **File:** `src/modules.py`
  - Deconstructed and implemented all modular sub-networks:
    1. *DoubleConv2d & ResidualBlock:* Hierarchical feature encoder projecting to `hidden_channels=128`.
    2. *Recursive Multi-Frame Fusion:* Tournament tree pairwise fusion with shared weights across 3 rounds ($\log_2 8 = 3$).
    3. *PixelShuffle Reconstruction Head:* Sub-pixel convolution expanding latent features to $(1054, 1054)$.
  - Resolved 130 vs 128 channel mismatch bug in `mask_encoder`.

- [x] **Step 1.3: Pretrained Weights Loading & Architecture Audit**
  - Loaded official NeurIPS checkpoint from `pretrained_model/model.ckpt` in pure PyTorch without PyTorch Lightning dependency.
  - Model Card: 3,260,452 parameters (3.26M), 12 input bands, 8 temporal revisits, 128 hidden channels.

---

## 📈 Phase 2: Scientific Evaluation & Benchmark Engine (COMPLETED)
> **Objective:** Scientifically evaluate the model against a heuristic baseline using remote sensing metrics.

- [x] **Step 2.1: Heuristic Control Baseline (`BicubicUpscaledBaseline`)**
  - Computed temporal median across 8 revisits with bicubic interpolation to $(1054, 1054)$.
  - Baseline score: **29.85 dB PSNR | 0.7271 SSIM**.

- [x] **Step 2.2: Compute Core Remote Sensing Metrics on Validation AOIs**
  - Implemented zero-order mean luminance alignment and computed PSNR and SSIM.
  - HighResNet score: **29.78 dB PSNR | 0.7669 SSIM**.
  - Demonstrated the **PSNR Paradox**: $-0.08\text{ dB}$ PSNR difference due to sub-pixel sensor alignment, with a massive **$+0.0398$ (+0.04) structural SSIM leap**!

---

## 🎨 Phase 3: Visualizer & Geospatial Showcase (COMPLETED)
> **Objective:** Provide visual proof and daylight-stretched comparisons showing real-world super-resolved outputs.

- [x] **Step 3.1: 4-Panel High-Resolution Daylight Grid Visualizer**
  - Implemented 2%–98% percentile contrast stretching (`stretch_to_daylight`), resolving specular roof glints up to 1,088.
  - Interactive 4-panel comparison:
    1. *Low-Res Sentinel-2 (10m Input)*
    2. *Bicubic Baseline (29.85 dB)*
    3. *HighResNet Deep SR (29.78 dB, 0.7669 SSIM)*
    4. *Airbus SPOT (1.5m Ground Truth)*
  - Added randomized scene selection on each notebook execution.

---

## 🔬 Phase 4: Advanced Interpretability (XAI), Transformers & Future Roadmap (PLANNED)
> **Objective:** Open the model black-box with Explainable AI and explore cutting-edge spatio-temporal attention architectures.

- [ ] **Step 4.1: Spatio-Temporal-Spectral LAM (Local Attribution Maps)**
  - **Objective:** Adapt CVPR 2021 LAM (*"Interpreting Super-Resolution Networks with Local Attribution Maps"*) for multi-temporal satellite super-resolution.
  - **Attribution Mechanics:**
    - Select target high-resolution bounding box (e.g. $16 \times 16$ or $32 \times 32$ urban roof edge or road boundary).
    - Apply progressive Gaussian blur path function over 20–30 integration steps.
    - Compute integrated autograd gradients $\frac{\partial \text{Target}}{\partial X}$ across the input tensor.
  - **Three-Axis Attribution Analysis:**
    1. *Spatial Receptive Field:* Heatmap showing which $160 \times 160$ low-res pixels contributed to edge synthesis.
    2. *Temporal Attribution:* Measure gradient magnitude per revisit ($T=1 \dots 8$) to mathematically prove that the model prioritizes cloud-free sunny dates and suppresses overcast passes.
    3. *Spectral Contribution:* Decompose gradient magnitude across the 12 multi-spectral bands (evaluating how much SWIR/NIR guides RGB spatial boundaries).

- [ ] **Step 4.2: Hybrid CNN–Transformer Architecture with Temporal Cross-Attention**
  - Keep 2D CNN spatial encoder (maintaining $O(N)$ efficiency on local CPU).
  - Replace static recursive tournament tree with a lightweight Multi-Head Temporal Self-Attention bottleneck ($8 \times 8$ attention matrix) to dynamically learn content-aware date weights.

- [ ] **Step 4.3: Interactive Georeferenced Leaflet / Folium Showcase**
  - Export super-resolved output as georeferenced GeoTIFF retaining original EPSG CRS coordinate projections.
  - Render pan-and-zoom interactive map overlay on top of OpenStreetMap/Google Satellite basemaps.

---

## 🧠 Architectural & Strategic Decisions (Summary Cheat Sheet)

1. **Why Pure PyTorch over PyTorch Lightning CLI?**
   - Eliminates 40+ unused CLI parameters, avoids external dependency conflicts, gives 100% transparent control over the gradient loop, and provides maximum interview value.
2. **Why Only the `JIF` Dataset?**
   - `SN7` (SpaceNet 7) and `PROBA-V` are completely separate datasets with different sensors, spatial resolutions, and directory layouts. WorldStrat (`JIF`, in `archive_1/`) is complete and self-contained for Sentinel-2 $\rightarrow$ SPOT super-resolution.
3. **Why `num_workers=0` in DataLoaders?**
   - The dataset transforms include Python lambda functions that fail to pickle across multi-processing worker processes under standard Python multiprocessing. `num_workers=0` guarantees stable loading.
4. **Why `.max().item()` instead of `.max()`?**
   - `.max()` returns a PyTorch Tensor attached to the autograd graph; accumulating it creates a GPU memory leak. `.item()` extracts a plain Python scalar and frees CUDA memory.

