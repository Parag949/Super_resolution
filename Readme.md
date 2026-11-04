## Currently developing this Phase 1 feature locally. A commit will be pushed upon its complete implementation
# Project Requirements Specification

## Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries

---

### 1. Executive Summary
Development of an end-to-end deep learning framework designed to ingest medium-resolution satellite imagery (Sentinel-2 at 10m Ground Sample Distance) and reconstruct high-resolution outputs (<4m GSD). The solution must guarantee radiometric, spectral, and geospatial fidelity to serve downstream remote sensing analytics and decision-support systems.

---

### 2. Core Functional Requirements

#### 2.1 Data Ingestion & Pre-Processing
* **Input Specifications:** Multi-band Sentinel-2 imagery (native 10m spatial resolution across VNIR bands).
* **Ground-Truth Paired Datasets:** Construction of geographically aligned pairs of low-resolution (LR, 10m) inputs and high-resolution (HR, sub-4m) reference data (e.g., PlanetScope, SPOT, or aerial orthophotos).
* **Data Cleansing & Normalization:**
  * Cloud, cloud-shadow, and atmospheric artifact masking.
  * Orthorectification and sub-pixel spatial coregistration.
  * Band-wise normalization preserving physical surface reflectance units.

#### 2.2 Model Architecture & Training Framework
* **Deep Learning Paradigms:** Implementation and tuning of generative or deep feature-extraction models:
  * Generative Adversarial Networks (e.g., ESRGAN, Real-ESRGAN).
  * Denoising Diffusion Probabilistic Models (DDPMs / SR3).
  * Vision Transformers (e.g., SwinIR, HAT) or CNN-Transformer hybrid networks.
* **Target Scale Factor:** Minimum 2.5x to 4x super-resolution scaling (10m $\rightarrow$ <4m).
* **Multi-Objective Loss Formulation:**
  * Pixel-level reconstruction loss ($L_1$ or Charbonnier loss).
  * Edge/gradient consistency loss for structural sharpening.
  * Perceptual loss (VGG or remote-sensing foundation model embeddings).
  * Dedicated spectral angle and radiometric preservation penalties.

#### 2.3 Scientific Integrity & Uncertainty Quantification
* **Spectral Fidelity:** Prevention of hallucinated spectral shifts; output bands must yield mathematically consistent radiometric indices (e.g., NDVI, NDWI, EVI).
* **Geospatial Rigor:** Full retention of spatial reference systems (CRS/EPSG codes), bounding boxes, affine transformation matrices, and GeoTIFF metadata.
* **Uncertainty Mapping:** Generation of pixel-level confidence/variance masks to distinguish verified observed structures from generative inferences.

---

### 3. Evaluation & Validation Framework

#### 3.1 Quantitative Metrics

| Evaluation Dimension | Metric | Objective |
| :--- | :--- | :--- |
| **Pixel Reconstruction** | Peak Signal-to-Noise Ratio (PSNR) | Minimize mean squared error across bands |
| **Structural Fidelity** | Structural Similarity Index (SSIM) | Maximize luminance, contrast, and structural alignment |
| **Perceptual Realism** | Learned Perceptual Patch Similarity (LPIPS) | Preserve realistic ground textures without blurring |
| **Spectral Integrity** | Spectral Angle Mapper (SAM) | Minimize spectral vector distortion across all bands |
| **Synthesis Error** | ERGAS | Minimize overall synthesis error relative to mean radiance |

#### 3.2 Qualitative & Cross-Terrain Validation
* Validation across heterogeneous land-use/land-cover (LULC) typologies:
  * Dense urban fabrics (small building footprints, alleys, road networks).
  * Precision agriculture (cadastral field parcels, irrigation channels).
  * Natural resources (riparian boundaries, water edges, forest corridors).
  * Disaster zones (localized flood boundaries, infrastructure damage).

---

### 4. Downstream Analytical Utility
The enhanced imagery must demonstrate measurable performance gains over native 10m Sentinel-2 inputs in:
* **Automated Land Cover Classification:** Higher overall accuracy and Kappa coefficient.
* **Building & Road Extraction:** Improved IoU (Intersection over Union) on fine infrastructure.
* **Change Detection:** Reduction in false positives along high-contrast boundaries.

## Acknowledgements & Citations

This project utilizes the **WorldStrat Dataset** for training and validating cross-sensor super-resolution mappings between Sentinel-2 and high-resolution Airbus SPOT imagery.

If you use this work or benchmark, please cite the original WorldStrat creators:

```bibtex
@misc{cornebise_open_2022,
  title = {Open High-Resolution Satellite Imagery: The WorldStrat Dataset -- With Application to Super-Resolution},
  author = {Cornebise, Julien and Or{\v s}oli{\'c}, Ivan and Kalaitzis, Freddie},
  year = {2022},
  month = jul,
  number = {arXiv:2207.06418},
  eprint = {2207.06418},
  eprinttype = {arxiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2207.06418}
}

@article{cornebise_worldstrat_zenodo_2022,
  title = {The WorldStrat Dataset},
  author = {Cornebise, Julien and Orsolic, Ivan and Kalaitzis, Freddie},
  year = {2022},
  month = jul,
  journal = {Dataset on Zenodo},
  doi = {10.5281/zenodo.6810792}
}
