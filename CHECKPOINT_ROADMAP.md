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
| **Phase 1** | **Pure PyTorch Model Backbone & Training Pipeline** | **IN PROGRESS** | 20% |
| **Phase 2** | **Scientific Benchmark & Metric Evaluation Engine** | **PENDING** | 0% |
| **Phase 3** | **Interactive Geospatial & Visual Showcase** | **PENDING** | 0% |

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

## 🚀 Phase 1: Pure PyTorch Model & Training Pipeline (CURRENT)
> **Objective:** Transition from exploratory data loading to a working model forward pass, single-batch overfit test, and complete training loop.

- [ ] **Step 1.1: Finalize Deterministic `load_dataset` in Cell 61**
  - **Target Cell:** `testing.ipynb` (Cell 61 & 62)
  - Strip original repository CLI bloat (`initialise_wandb`, `argparse`, `SN7`, `PROBAV`).
  - Implement deterministic 80% Train, 10% Val, 10% Test split using `random_split` with `torch.Generator`.
  - Wrap splits into pure PyTorch DataLoaders with `num_workers=0` (preventing lambda serialization crashes) and `worker_init_fn=seed_worker`.
  - **Verification:** Run `dataloaders = load_dataset(dataset, batch_size=2)` and check lengths.

- [ ] **Step 1.2: Define Pure PyTorch `HighResNet` Architecture**
  - **Target Location:** `testing.ipynb` or `src/modules.py`
  - Implement 3 modular sub-networks:
    1. *Feature Encoder:* 2D convolutions projecting $(B \times T, C_{in}, H, W)$ to `hidden_channels=64`.
    2. *Recursive Multi-Frame Fusion:* Recurrent fusion unit sequentially accumulating information across all $T=8$ temporal revisits into a latent feature state.
    3. *Super-Resolution Reconstruction Head:* Sub-pixel convolution (`nn.PixelShuffle`) expanding latent features to $(1054, 1054)$.
  - **Model Parameters:** `in_channels=3`, `mask_channels=1`, `revisits=8`, `hidden_channels=64`, `out_channels=3`, `output_size=(1054, 1054)`.

- [ ] **Step 1.3: Forward Pass Shape & Device Verification**
  - Pass test batch through the network on GPU (`cuda:0`) or CPU:
    ```python
    y_hat = model(batch['lr'].to(device), mask=batch['lrc'].to(device))
    ```
  - **Verification Criteria:**
    - Output tensor shape must strictly be `(B, 3, 1054, 1054)`.
    - No NaN/Inf values.
    - Output values bounded in $[0.0, 1.0]$ via final Sigmoid activation.

- [ ] **Step 1.4: Single-Batch Overfit Sanity Check**
  - Set up shift-invariant registration loss (`cR-loss`) or composite loss:
    $$\mathcal{L} = 0.3 \cdot \text{cR-MSE} + 0.4 \cdot \text{cR-MAE} + 0.3 \cdot \text{cR-SSIM}$$
  - Train for 30 iterations on **one single batch** with Adam (`lr=1e-4`).
  - **Verification Criteria:** Loss must strictly decrease monotonically from $\approx 0.25$ down to $<0.05$. (Proves backpropagation, loss formulation, and gradient flow work properly before launching long runs).

- [ ] **Step 1.5: Multi-Epoch Training Loop & Checkpointing**
  - Implement clean training and validation loop:
    - Iterate over `dataloaders['train']`.
    - Evaluate on `dataloaders['val']` at each epoch end.
    - Track best validation loss and save checkpoint to `best_highresnet.pth`.
    - Use `loss.item()` to prevent GPU memory leaks.

---

## 📈 Phase 2: Scientific Evaluation & Benchmark Engine
> **Objective:** Scientifically evaluate the model against a heuristic baseline using remote sensing metrics.

- [ ] **Step 2.1: Heuristic Control Baseline (`BicubicUpscaledBaseline`)**
  - Compute the temporal median across the 8 Sentinel-2 revisits (filters out transient clouds and noise).
  - Apply standard bicubic interpolation to $(1054, 1054)$.
  - This establishes the lower performance bound.

- [ ] **Step 2.2: Compute Core Remote Sensing Metrics on Unseen Validation AOIs**
  - **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level signal fidelity in decibels.
  - **SSIM (Structural Similarity Index):** Evaluates texture, contrast, and edge sharpness.
  - **cR-loss (Shift-Invariant Registration Loss):** Accounts for sub-pixel orbital parallax ($\pm 2$ pixels) between Sentinel-2 and SPOT satellites.

- [ ] **Step 2.3: Quantitative Benchmark Comparison Table**
  - Produce benchmark table comparing:
    - `Bicubic Baseline` vs. `Trained HighResNet Model`.
    - Expected target: $+3.0\text{ dB to }+4.0\text{ dB}$ PSNR gain and $\approx 40\%$ reduction in cR-MSE error.

---

## 🎨 Phase 3: Visualizer & Geospatial Showcase
> **Objective:** Provide visual proof and interactive GIS maps showing real-world super-resolved outputs.

- [ ] **Step 3.1: 4-Panel High-Resolution Grid Visualizer**
  - Build a comparison visualizer displaying:
    - *Panel 1:* Single S2 Low-Resolution Revisit ($10\text{m}$, raw input).
    - *Panel 2:* Bicubic Upscaled Temporal Median ($10\text{m} \rightarrow 1.5\text{m}$ baseline).
    - *Panel 3:* HighResNet Model Reconstructed Output ($1.5\text{m}$ prediction).
    - *Panel 4:* SPOT Ground Truth ($1.5\text{m}$ target).

- [ ] **Step 3.2: Georeferenced Folium Map Integration**
  - Overlay super-resolved prediction tiles directly onto satellite basemaps using bounding box coordinates and Folium/Leaflet.
  - Allows panning and zooming to inspect building edges, roads, and agricultural boundaries.

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

