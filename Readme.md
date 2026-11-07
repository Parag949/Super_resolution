# Multi-Temporal Satellite Super-Resolution (Sentinel-2 $\rightarrow$ Airbus SPOT)

A PyTorch implementation and evaluation of deep multi-temporal super-resolution mapping **10m Sentinel-2 multi-spectral imagery** to **1.5m Airbus SPOT commercial imagery** ($\approx 6.67\times$ spatial scaling) using the [WorldStrat (NeurIPS 2022)](https://worldstrat.github.io/) benchmark.

![4-Panel Daylight Comparison](image.png)
*Figure: Real validation scene comparison. (1) Raw Sentinel-2 10m input; (2) Bicubic baseline (29.85 dB); (3) HighResNet reconstruction (29.78 dB, 0.7669 SSIM); (4) Airbus SPOT 1.5m ground truth target.*

---

## Benchmark Scorecard

Evaluated on the WorldStrat validation split using zero-order brightness alignment:

| Metric | Bicubic Baseline | HighResNet (Ours) | Delta ($\Delta$) | What It Means |
| :--- | :---: | :---: | :---: | :--- |
| **SSIM** | 0.7271 | **0.7669** | **+0.0398** | **+0.04 structural gain**: Sharp building edges, road corridors, and field boundaries clearly resolved. |
| **PSNR** | 29.85 dB | 29.78 dB | -0.07 dB | Parity ($\pm 0.08$ dB): Unaligned pixel MSE penalizes sharp edges under sub-pixel sensor registration shifts. |
| **Input Specs** | 12 Bands $\times$ 8 Revisits | 12 Bands $\times$ 8 Revisits | — | Ingests 8 temporal passes across the full multi-spectral Sentinel-2 range ($160 \times 160$). |
| **Parameters** | 0 (Heuristic) | 3,260,452 (3.26M) | — | Lightweight recursive tournament fusion with shared weights. |
| **Scaling** | $\approx 6.67\times$ | $\approx 6.67\times$ | — | $160 \times 160$ input ($10\text{m}$) $\rightarrow$ $1054 \times 1054$ target ($1.5\text{m}$), decomposing 1 pixel into 44 sub-pixels. |

---

## Key Engineering Insights & Problems Solved

### 1. The "PSNR Paradox" in Satellite Imagery
When evaluating super-resolution across different satellite constellations (Sentinel-2 and Airbus SPOT), raw pixel-wise Mean Squared Error (MSE/PSNR) has an inherent **blurry bias**:
* If a model predicts a razor-sharp building boundary that is shifted by even $0.5$ pixels due to orbital parallax, pixel-by-pixel squared differences penalize both edges severely.
* A blurry, washed-out prediction (like Bicubic) averages pixel values, producing a lower numerical MSE penalty.
* **SSIM (Structural Similarity Index)** evaluates local luminance, contrast, and structural covariance. HighResNet achieves a **$+0.0398$ leap in SSIM**, confirming genuine sub-pixel geometric recovery despite the identical PSNR numbers.

### 2. Solving Display Clipping & Specular Glints (2%–98% Percentile Stretch)
* **The Problem:** Sentinel-2 surface reflectance values are physically calibrated in $[0.02, 0.11]$, making raw linear RGB plots look pitch black. Conversely, SPOT imagery contains metal roof specular glints reaching values up to $1,088$ (well above physical 1.0 reflectance), causing standard min-max clipping to look like night streetlights.
* **The Fix:** Implemented a robust `stretch_to_daylight` function using $2\%\text{--}98\%$ cumulative percentile contrast stretching on visualized arrays while keeping tensor calculations in pure physical reflectance.

### 3. Why Regression Models over Generative Models (GANs / Diffusion)
* Generative models (ESRGAN, Stable Diffusion) invent plausible-looking synthetic high-frequency noise. While visually appealing for portraits or consumer photos, **generative hallucination is dangerous for Earth Observation**: it creates fake building footprints, distorts NDVI agricultural vegetation indices, and shifts riverbanks.
* HighResNet uses $L_1$ and SSIM regression, which mathematically computes the **conditional expectation $\mathbb{E}[y|x]$**. It outputs the physically honest consensus of what the sensor actually captured across time.

### 4. Pure PyTorch Architecture Deconstruction (`src/modules.py`)
* Extracted the entire architecture out of legacy, abandoned PyTorch Lightning CLI scripts into clean, dependency-free PyTorch classes:
  * `DoubleConv2d`: $5\times 5$ effective receptive field using stacked $3\times 3$ kernels ($28\%$ fewer weights).
  * `RecursiveFusion`: Pairwise tournament fusion tree ($8 \rightarrow 4 \rightarrow 2 \rightarrow 1$) with shared weights across rounds, reducing parameter count by $66\%$.
  * `PixelShuffleBlock`: Sub-pixel convolutional upsampling to avoid checkerboard artifacts.
* **Fixed bugs in the original codebase:**
  * Patched the 130 vs 128 channel mismatch in `mask_encoder`.
  * Resolved torchvision interpolation enum incompatibilities with Kornia.
  * Formulated a memory-efficient composite loss ($0.4 L_1 + 0.3 \text{MSE} + 0.3 \text{SSIM}$) preventing CPU Out-of-Memory crashes.

---

## Repository Structure

```text
├── src/
│   ├── modules.py            # Clean pure PyTorch architecture definitions (HighResNet, SRCNN, etc.)
│   └── __init__.py
├── learnings/                # 3-Volume Technical Handbook & Engineering Notes
│   ├── README.md             # Master Table of Contents
│   ├── 01_satellite_data_and_radiometry.md   # Sensor bands, 12-bit depth, display glint fixes
│   ├── 02_highresnet_architecture_and_pytorch.md # PyTorch idioms, convolution math, tournament tree
│   ├── 03_training_evaluation_and_benchmarks.md  # PSNR paradox, loss functions, model card
│   └── model_notes.txt       # Raw chronological debugging and engineering logs
├── compute_dataset_stats.py  # Parallel global statistics profiler across 35,353 images
├── testing.ipynb             # Interactive notebook: DataLoader tests, benchmark engine & visualizer
├── CHECKPOINT_ROADMAP.md     # Milestone tracking across all project phases
└── image.png                 # Saved 4-panel daylight benchmark comparison
```

---

## How to Run & Reproduce

### 1. Environment Setup
```bash
git clone https://github.com/Parag949/Super_resolution.git
cd Super_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy matplotlib pytorch-msssim kornia pandas
```

### 2. Run the Evaluation Notebook
Launch Jupyter and open `testing.ipynb`:
* **Cell 1–10:** Loads multi-spectral Sentinel-2 and SPOT paired tiles.
* **Cell 15–20:** Initializes `BicubicUpscaledBaseline` and loads the pure PyTorch `HighResNet` checkpoint.
* **Cell 25:** Executes the quantitative validation benchmark (PSNR, SSIM).
* **Cell 30:** Renders the randomized 4-Panel Daylight Comparison visualizer.

---

## Future Roadmap

- [ ] **Explainable AI (XAI) with Local Attribution Maps (LAM):** Adapt CVPR 2021 LAM for multi-temporal super-resolution to visualize which input pixels, spectral bands, and revisit dates contribute most to sharp high-res edges.
- [ ] **Hybrid CNN–Transformer Bottleneck:** Replace static recursive tournament fusion with a lightweight $8 \times 8$ Temporal Self-Attention matrix to dynamically weigh cloud-free dates.
- [ ] **Georeferenced Web Map (Folium/Leaflet):** Export super-resolved GeoTIFFs preserving original EPSG coordinate reference systems for interactive web inspection.

---

## Citations & Acknowledgements

This project builds upon the open-source **WorldStrat Dataset** and benchmark:

```bibtex
@misc{cornebise_open_2022,
  title = {Open High-Resolution Satellite Imagery: The WorldStrat Dataset -- With Application to Super-Resolution},
  author = {Cornebise, Julien and Or{\v s}oli{\'c}, Ivan and Kalaitzis, Freddie},
  year = {2022},
  eprint = {2207.06418},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
