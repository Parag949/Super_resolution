# Volume 2: PyTorch Architecture & HighResNet Engineering

**Architecture:** HighResNet (Multi-Frame Recursive Fusion Super-Resolution Network)  
**Reference:** WorldStrat Project (NeurIPS 2022)  
**Total Parameters:** 3,260,452 (~3.26 Million)  
**Input Channels:** 12 Multi-Spectral Bands × 8 Temporal Revisits  
**Hidden Trunk Width:** 128 Channels (with PReLU activations & reflection padding)  
**Upsampling Strategy:** Sub-Pixel Convolution (`nn.PixelShuffle`) + Bilinear Fractional Resize

---

## 1. The Keras-to-PyTorch Mental Bridge

### 1.1 Why PyTorch Uses Classes for Everything
* **In Keras / TensorFlow:** Models and layers can be assembled with functional functions. Keras tracks weights globally under the hood.
* **In PyTorch:** Plain Python functions **cannot** retain learnable weights or save checkpoints to GPU/disk. Any block that holds learnable weights (`Conv2d`, `PReLU`, `Linear`) **must** be a class inheriting from `nn.Module`.

### 1.2 The Two-Part Anatomy of Every PyTorch Module
Every custom `nn.Module` has exactly two required methods:
1. `def __init__(self, ...)` — **"The Hardware Store"**:
   * Instantiates all layer weights, filters, and sub-blocks.
   * Runs **once** when the model object is built.
   * Must begin with `super().__init__()` (the ignition key that registers parameters in PyTorch's autograd engine).
2. `def forward(self, x)` — **"The Conveyor Belt"**:
   * Defines how input tensors flow through the layers.
   * Executes dynamically on every training step.

> [!IMPORTANT]
> **The Golden Rule:** Never instantiate layers inside `forward()`. Instantiating a layer inside `forward()` re-initializes its weights randomly on every single batch, completely wiping out trained knowledge!

### 1.3 Blueprint vs. House & The Power of Parentheses `()`
* `obj = ClassName` (No parentheses): Stores the **Blueprint** (class definition).
* `obj = ClassName()` (With parentheses): Executes the constructor, building a **House** (physical object in RAM).
* **The Loop Nuance:**
  * `[ClassName() for _ in range(3)]` $\rightarrow$ Parentheses inside loop: builds **3 distinct objects** with independent weights.
  * `obj = ClassName(); [obj for _ in range(3)]` $\rightarrow$ Parentheses outside loop: 1 object referenced 3 times (**Shared Weights**).

---

## 2. Layer & Convolutional Mechanics

### 2.1 3D Filter Cubes in `Conv2d`
* In a 2D convolution over images with $C_{\text{in}}$ channels (e.g., $12$ multi-spectral bands), each filter is not a flat $3 \times 3$ square; it is a **3D filter cube of size $(C_{\text{in}} \times 3 \times 3)$**.
* The filter slides across spatial height and width, summing across all channels simultaneously to produce 1 scalar output pixel per filter.
* An $N$-filter layer has weight tensor shape $(C_{\text{out}}, C_{\text{in}}, K_H, K_W)$.

### 2.2 Why Double Convolutions (`DoubleConv2d`)?
Instead of a single large $5 \times 5$ convolution, modern vision networks stack two consecutive $3 \times 3$ convolutions:
1. **Receptive Field Equivalence:** Two $3 \times 3$ convs cover the exact same $5 \times 5$ spatial receptive field.
2. **28% Fewer Parameters:** 
   $$\text{One } 5 \times 5 \text{ filter: } 25 \cdot C^2 \quad \text{vs.} \quad \text{Two } 3 \times 3 \text{ filters: } 2 \times (9 \cdot C^2) = 18 \cdot C^2$$
3. **Double Non-Linearity:** Two non-linear activation functions allow learning more expressive boundaries.

### 2.3 Reflection Padding vs. Zero Padding
* Standard zero-padding adds a halo of black zeros around satellite images, corrupting border features.
* **Reflection Padding (`padding_mode="reflect"`):** Mirrors actual ground reflectance across the image boundaries, eliminating edge distortion in super-resolution.

### 2.4 Why `nn.BatchNorm2d` is Disabled in Super-Resolution
In high-level tasks (e.g., classification), BatchNorm stabilizes training by forcing features to mean 0 and variance 1. In **satellite super-resolution**, BatchNorm is intentionally disabled (`use_batchnorm=False` $\rightarrow$ `nn.Identity()`):
1. **Destruction of Absolute Radiometric Scale:** Satellite super-resolution requires reconstructing physical ground reflectance and contrast. Normalizing feature maps erases physical dynamic range.
2. **Mini-Batch Artifacts:** Small batch sizes common in remote sensing ($B=2$ or $4$) create noisy batch statistics, causing color tinting and halo artifacts.

---

## 3. HighResNet Building Blocks

### 3.1 `ResidualBlock` vs. `DenseBlock`
* **`ResidualBlock` (Addition):** Computes $y = x + \text{DoubleConv}(x)$. Preserves identical channel width ($C \rightarrow C$) while creating skip connections that allow gradients to flow directly backward without vanishing.
* **`DenseBlock` (Concatenation):** Computes $y = [x, \; \text{DoubleConv}(x)]$. Doubles the channel count ($C \rightarrow 2C$), accumulating features along the depth dimension.

### 3.2 ESPCN `PixelShuffle` (Sub-Pixel Convolutional Origami)
Instead of older `ConvTranspose2d` (which causes severe grid-like "checkerboard artifacts" due to uneven kernel overlap), `HighResNet` uses **PixelShuffle** (Shi et al., 2016):
* **Zero Learnable Weights:** Pure memory tensor reshaping. It trades channel depth for spatial height and width.
* **The Mathematical Formula:**
  $$(B, \; C \cdot r^2, \; H, \; W) \quad \xrightarrow{\text{PixelShuffle}(r)} \quad (B, \; C, \; H \cdot r, \; W \cdot r)$$
* **Example ($r=2$):** Consumes $r^2 = 4$ depth channels per spatial pixel. A $(B, 64, 160, 160)$ tensor unfolds into $(B, 16, 320, 320)$.

### 3.3 Fractional Scaling via Bilinear `Resize`
* Sentinel-2 native resolution: $160 \times 160$.
* Airbus SPOT target resolution: $1054 \times 1054$.
* Scale Factor: $1054 / 160 = \mathbf{6.5875\times}$ (fractional!).
* **The Strategy:** `PixelShuffle` performs the integer sub-pixel feature generation ($160 \rightarrow 320$), hallucinating sharp details. Then an anti-aliased bilinear `Resize` snaps the features smoothly to the exact SPOT tile boundaries ($320 \rightarrow 1054$).

---

## 4. Multi-Temporal Engineering & Recursive Fusion

### 4.1 The "Batch Folding" Trick for 5D Video Tensors
PyTorch `Conv2d` only accepts 4D tensors `(B, C, H, W)`, but multi-temporal data is 5D `(B, T, C, H, W)`:
```python
batch_size, revisits, channels, height, width = x.shape
x = x.view(batch_size * revisits, channels, height, width)  # (2*8=16, 12, 160, 160)
x = self.encoder(x)                                         # (16, 128, 160, 160)
```
* Folds time into the batch dimension so all 16 revisits pass through GPU convolutions in a single parallel step.
* Guarantees every date is encoded by the exact same feature filters without slow Python loops.

### 4.2 The "Piggy Bank" Weight-Sharing Trick in `RecursiveFusion`
In `RecursiveFusion`, 8 temporal revisits are fused into 1 master feature map via a tournament tree:
* **Round 1:** $8 \text{ revisits} \rightarrow 4 \text{ fused revisits}$
* **Round 2:** $4 \text{ revisits} \rightarrow 2 \text{ fused revisits}$
* **Round 3:** $2 \text{ revisits} \rightarrow 1 \text{ master fused revisit}$

```python
pairwise_fusion = FusionBlock(in_channels, kernel_size)
self.fusion = nn.Sequential(*(pairwise_fusion for _ in range(num_fusion_layers)))
```
* **The Pointer Nuance:** `pairwise_fusion` was constructed **once**. The generator references the exact same object in RAM across all 3 rounds.
* **Benefits:** Saves $66\%$ of parameters and forces the network to learn a scale-invariant, fractal fusion algorithm that works regardless of whether 8, 4, or 2 dates are present.

### 4.3 PyTorch Inheritance Without `forward()`
`PixelShuffleBlock` inherits from `ConvTransposeBlock`:
```python
class PixelShuffleBlock(ConvTransposeBlock):
```
* `PixelShuffleBlock` defines `__init__()` (swapping `self.upsample` to use `PixelShuffle`), but does **not** define `forward()`.
* When called, Python automatically climbs up to `ConvTransposeBlock.forward(x)`, which executes `return self.upsample(x)`. Because `self.upsample` was overwritten in the child constructor, it runs the new sub-pixel pipeline seamlessly.

---

## 5. Architectural Comparison: SRCNN vs. HighResNet

| Architectural Property | `SRCNN` (Multi-Frame Baseline) | `HighResNet` (Flagship Model) |
| :--- | :--- | :--- |
| **Fusion Mechanism** | **Naive Early Concatenation** | **Hierarchical Tournament Tree** |
| **Fusion Tensor Shape** | Glues all revisits into channels ($8 \times 64 = 512$) | Pairs revisits hierarchically ($8 \rightarrow 4 \rightarrow 2 \rightarrow 1$) |
| **Weight Sharing** | None; massive flat conv matrix | Complete weight sharing across all rounds |
| **Temporal Flexibility** | Rigid; hardcoded to fixed revisit count | Flexible; handles arbitrary revisits |
| **Permutation Invariance**| Vulnerable to date ordering | Highly robust to revisit sequence |
| **Parameter Efficiency** | High parameter redundancy | Highly compressed ($3.26\text{M}$ parameters) |
