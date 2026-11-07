# Multi-Temporal Satellite Super-Resolution: Master Technical Handbook

**Repository:** `/mnt/storage/OneDrive_4_14-04-2025/Super_resolution`  
**Dataset:** WorldStrat (JIF Benchmark: Sentinel-2 10m L2A $\rightarrow$ Airbus SPOT 1.5m HR)  
**Architecture:** HighResNet (Recursive Tournament Fusion Backbone)  

---

## Master Table of Contents

The complete documentation of this project has been restructured and optimized into three core volumes:

### 📖 [Volume 1: Satellite Data Engineering & Radiometry Guide](01_satellite_data_and_radiometry.md)
* **1. Multi-Spectral Satellite Data & Physical Radiometry:**
  * Full 12-band Sentinel-2 spectrum (Coastal, RGB, Red-Edge 1-3, NIR, Water Vapor, SWIR 1-2).
  * 12-bit physical surface reflectance vs. 8-bit standard sRGB photography.
* **2. The Radiometry & Daylight Visualization Breakthrough:**
  * Why Sentinel-2 appeared pitch black and SPOT looked like night streetlights.
  * The $2\%\text{--}98\%$ cumulative percentile contrast stretching solution (`stretch_to_daylight`).
* **3. DataLoader Engineering & Multi-Temporal Contracts:**
  * The 5D Multi-Temporal Batch Contract: `(Batch, Revisits=8, Channels=12, H=160, W=160)`.
  * Python 3.14 multiprocessing deadlock resolution (`num_workers=0` & deterministic seeds).
* **4. De-Normalization & Cross-Sensor Domain Matching:**
  * Auditing the reference repository bugs (order-of-operations and `x /= max(x)`).
  * Why neural networks map normalized features while heuristics require raw physical units.

---

### 🏛️ [Volume 2: PyTorch Architecture & HighResNet Engineering](02_highresnet_architecture_and_pytorch.md)
* **1. The Keras-to-PyTorch Mental Bridge:**
  * Why PyTorch uses classes for everything (`nn.Module`).
  * The Two-Part Anatomy (`__init__` vs `forward`).
  * The Blueprint vs. House object rule and the danger of instantiating layers inside `forward()`.
* **2. Layer & Convolutional Mechanics:**
  * 3D filter cubes across 12 input channels.
  * Why Double Convolutions (`DoubleConv2d`): $5\times 5$ receptive field with $28\%$ fewer weights.
  * Reflection padding vs. zero padding.
  * Why `nn.BatchNorm2d` is intentionally disabled in satellite super-resolution (`use_batchnorm=False`).
* **3. HighResNet Building Blocks:**
  * `ResidualBlock` (Addition: $C \rightarrow C$) vs. `DenseBlock` (Concatenation: $C \rightarrow 2C$).
  * ESPCN `nn.PixelShuffle` (Sub-pixel convolutional origami) vs. Deconv checkerboard artifacts.
  * Fractional scaling via anti-aliased bilinear `Resize` ($160 \rightarrow 320 \rightarrow 1054$).
* **4. Multi-Temporal Engineering & Recursive Fusion:**
  * The "Batch Folding" trick for 5D video convolutions (`x.view(B * T, C, H, W)`).
  * The "Piggy Bank" weight-sharing trick in `RecursiveFusion` (saving $66\%$ parameters).
  * PyTorch constructor inheritance without defining `forward()`.
* **5. Architectural Comparison:**
  * Rigid 512-channel early fusion (`SRCNN`) vs. Hierarchical pairwise tournament reduction (`HighResNet`).

---

### 📊 [Volume 3: Training, Benchmarks & The Science of Evaluation](03_training_evaluation_and_benchmarks.md)
* **1. Training Loss vs. Evaluation Metrics:**
  * Differentiable steering wheel (Optimizer) vs. Human scorecard (PSNR/SSIM).
  * Why we do not train directly on PSNR (MSE blur penalty and log saturation).
  * The crash-proof composite loss formulation ($0.4\text{ MAE} + 0.3\text{ MSE} + 0.3\text{ SSIM}$).
* **2. The "PSNR Paradox" in Remote Sensing:**
  * Why unaligned pixel-wise MSE rewards blurry smudges when sub-pixel shifts exist.
  * Why SSIM is the true measure of structural fidelity (**$+0.0398$ SSIM leap**).
  * The rationale for shift-invariant registration metrics (`cR-loss`).
* **3. HighResNet Official Model Card & Benchmark Scorecard:**
  * Complete model card: $3,260,452$ parameters, CC-BY and BSD licenses.
  * Quantitative benchmark table (Bicubic Baseline vs. Pretrained HighResNet).
* **4. Interview Defense Script & Resume Showcase Guide:**
  * Professional resume bullet points for remote sensing AI engineering.
  * How to confidently answer: *"Did you train from scratch or use pretrained weights?"*
* **5. Why Super-Resolved Imagery Looks Softer Than Native Commercial Ground Truth:**
  * The $44\times$ Area Expansion Problem ($6.67\times$ scaling: $100\text{ m}^2$ into 44 sub-pixels).
  * Regression vs. GANs: Why satellite science prioritizes physical honesty over fake hallucination.
  * Multi-week time-series consensus vs. instantaneous single-day optical shutter snapshots.
  * The proper benchmark comparison: Panel 1 (10m Input) vs. Panel 3 (HighResNet).
* **6. Future Technical Roadmap & Planned Extensions:**
  * Explainable AI (XAI) via Multi-Temporal Spatio-Spectral Local Attribution Maps (LAM).
  * Hybrid CNN-Transformer architecture with Temporal Cross-Attention.

---

### 📁 Legacy Chronological Notes (Preserved for History)
* `model_notes.txt`: Original 23 chronological Q&A sections from interactive debugging.
* `notes.txt`: Original exploratory and exploratory data analysis notes.
* `radiometric_depth.txt`, `spot_10000_scaling_notes.txt`, `target_range_and_negative_values.txt`: Early sensor calibration records.
