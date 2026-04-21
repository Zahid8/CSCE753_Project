# Methodology: Hybrid Learning-Based and Physics-Guided Underwater Image Enhancement

## 1. Problem Formulation and Design Objective

Underwater image enhancement is formulated as supervised restoration from a degraded observation to a visually and structurally improved target. Let an input underwater image be denoted by \(I \in [0,1]^{3 \times H \times W}\), and the desired clean image by \(J \in [0,1]^{3 \times H \times W}\). The model predicts \(\hat{J}\) by combining:

1. A compact learnable enhancement backbone that captures local detail and global context.
2. A physics-guided pathway that predicts a pseudo-transmission prior \(T\), representing spatially varying degradation severity.
3. A modulation mechanism that injects \(T\)-conditioned correction weights into intermediate backbone features.

This design explicitly balances representational flexibility (data-driven learning) with physically meaningful guidance (degradation prior), while remaining efficient enough for high-resolution training and inference on a single GPU.

## 2. Data Representation, Pair Construction, and Preprocessing

### 2.1 Input Assumptions

All images are treated as 3-channel RGB tensors normalized to \([0,1]\). Inputs are loaded from disk, converted to RGB, and transformed from height-width-channel memory layout to channel-height-width layout required by deep learning frameworks.

### 2.2 Paired Data Construction

Paired supervision is formed by matching degraded and clean samples through robust key-based correspondence. The matching key is obtained from a normalized filename stem (or configurable regex policy in generalized deployments). Every degraded sample must resolve to exactly one clean counterpart; unresolved entries are either:

- flagged as integrity errors during dataset indexing, or
- skipped with explicit audit logging, depending on strictness configuration.

This guarantees deterministic one-to-one supervision in paired datasets such as UIEB- or LSUI-style structures where directory patterns differ but sample identity is preserved.

### 2.3 Synchronized Augmentation for Paired Samples

For paired training, all geometric augmentation must be **synchronized** across degraded and clean images to preserve pixel correspondence. Random variables are sampled once per sample pair and applied identically to both images.

Applied augmentation primitives:

1. Random crop:
   - If source dimensions exceed crop size, a shared random top-left coordinate is sampled.
   - If source dimensions are smaller, controlled resize/pad fallback is used.
2. Random horizontal flip:
   - Shared Bernoulli decision.
3. Optional random vertical flip:
   - Shared Bernoulli decision.

Color-domain augmentations that alter supervision consistency are avoided for strict reconstruction training unless explicitly configured.

### 2.4 Size Handling and Tensor Conversion

Each sample is converted to floating-point tensor with contiguous memory layout. Any resizing uses interpolation appropriate for natural images and avoids alias amplification. During batching, tensor shapes are guaranteed consistent by preprocessing policy.

## 3. Hybrid Network Architecture

The enhancement network consists of three tightly coupled subsystems:

1. A lightweight encoder-decoder enhancement stream.
2. A parallel pseudo-transmission estimation stream.
3. A feature modulation operator applied at multiple scales.

### 3.1 Enhancement Stream (Lightweight U-Shaped Encoder-Decoder)

The main stream follows a multi-scale U-shaped design:

1. **Encoder** progressively downsamples spatial resolution while increasing channel capacity.
2. **Bottleneck** performs high-level feature transformation at reduced resolution.
3. **Decoder** progressively upsamples and fuses skip information from encoder stages.

Each stage uses computationally efficient building blocks (small-kernel convolutions and lightweight normalization/activation policies) to maintain favorable memory and throughput characteristics for high-resolution inputs.

Skip connections preserve high-frequency details and stabilize optimization by reducing information loss through deep downsampling paths.

### 3.2 Physics-Guided Prior Stream

A dedicated prior stream processes the same degraded image \(I\) and outputs pseudo-transmission:

\[
T = g_{\psi}(I), \quad T \in [0,1]^{1 \times H \times W}
\]

The stream is intentionally compact and fully convolutional, enabling dense per-pixel prior estimation with low overhead. A bounded output activation enforces physically plausible transmission range.

This predicted prior is not treated as final physical inversion output. Instead, it is used as a guidance signal to condition enhancement features in the main stream.

### 3.3 Feature Modulation Mechanism

At each selected backbone stage with intermediate feature map \(F \in \mathbb{R}^{C \times h \times w}\), transmission is spatially resized to stage resolution and transformed by a learnable mapping:

\[
W = \phi(T_{h,w}), \quad W \in \mathbb{R}^{C \times h \times w}
\]

Then modulation is applied channel-wise and spatially:

\[
F' = F \odot W
\]

where \(\odot\) is Hadamard product.

Interpretation:

- Regions with severe degradation can be upweighted or reconditioned.
- Regions with reliable content can be preserved by attenuation control.
- Stage-specific \(\phi(\cdot)\) allows distinct correction behavior at coarse and fine scales.

The mapping \(\phi\) is implemented as shallow learnable convolutional transforms producing bounded modulation weights, ensuring stability during mixed precision training.

### 3.4 Multi-Scale Injection Strategy

Prior-conditioned modulation is not restricted to one layer. It is applied at multiple encoder and decoder depths so that:

1. Coarse stages receive global correction cues for color cast and haze distribution.
2. Fine stages receive local cues for edge restoration and texture recovery.

This reduces the chance of prior underutilization and improves consistency across scales.

### 3.5 Output Head

The decoder output is mapped back to 3 channels and bounded to \([0,1]\). Final prediction is:

\[
\hat{J} = f_{\theta}(I, T)
\]

The model can optionally return \(T\) alongside \(\hat{J}\) for diagnostics, interpretability, or auxiliary visualization.

## 4. Objective Function Design

Training uses a weighted composite loss that combines pixel fidelity and perceptual structural consistency.

### 4.1 Pixel Reconstruction Term

Two robust options are supported:

1. L1 loss:
\[
\mathcal{L}_{pix} = \| \hat{J} - J \|_1
\]
2. Charbonnier loss:
\[
\mathcal{L}_{pix} = \sqrt{(\hat{J} - J)^2 + \epsilon^2}
\]

Charbonnier behaves as a differentiable robust penalty and is often preferred for outlier-resistant restoration.

### 4.2 Structural Similarity Term

A differentiable SSIM-based loss term enforces local luminance, contrast, and structure consistency:

\[
\mathcal{L}_{ssim} = 1 - \text{SSIM}(\hat{J}, J)
\]

Gaussian-window local statistics are used for stable structural comparison across channels.

### 4.3 Combined Objective

\[
\mathcal{L}_{total} = \lambda_{pix}\mathcal{L}_{pix} + \lambda_{ssim}\mathcal{L}_{ssim}
\]

Weights \(\lambda_{pix}\) and \(\lambda_{ssim}\) are exposed as tunable hyperparameters to trade off strict pixel alignment and perceptual structure quality.

## 5. Evaluation Metrics

### 5.1 Full-Reference Metrics

For paired validation:

1. PSNR:
   - Computed from per-image MSE.
   - Higher is better.
2. SSIM:
   - Structural fidelity with local statistics.
   - Higher is better.

Batch-level and epoch-level aggregation are performed through mean reduction.

### 5.2 Non-Reference Underwater Quality Metrics

For scenarios without reliable ground truth or for additional perceptual reporting, non-reference underwater quality scores are integrated as:

1. UIQM
2. UCIQE

If full external implementations are unavailable in a given environment, standardized integration stubs are provided so external evaluators can be plugged in with minimal interface changes. This keeps the training/evaluation pipeline stable while allowing method-compliant metric replacement.

## 6. Optimization and Mixed Precision Training

### 6.1 Optimizer and State

An adaptive optimizer is used with configurable learning rate and momentum terms. Parameters from both enhancement and prior streams are optimized jointly.

### 6.2 Automatic Mixed Precision

To support high-resolution processing efficiently, training uses automatic mixed precision:

1. Forward and loss computation are run under autocast.
2. Gradients are scaled before backpropagation.
3. Optimizer step is executed through gradient-scaling utilities.
4. Scale factor is dynamically updated to mitigate underflow.

This reduces memory footprint and improves throughput while preserving numerical stability.

### 6.3 Gradient and Stability Considerations

The implementation follows stable ordering:

1. Zero gradients.
2. Forward pass.
3. Loss computation.
4. Scaled backward pass.
5. Scaled optimizer step.
6. Scale update.

Optional gradient clipping can be enabled for highly unstable settings.

## 7. Training Loop Semantics

Each epoch executes:

1. Data iteration over training batches.
2. Forward prediction of enhanced output (and optional prior map).
3. Composite loss computation.
4. Mixed precision backward and update.
5. Running statistics accumulation.

At configured intervals:

1. Validation computes PSNR/SSIM means on held-out paired data.
2. Checkpoints are saved (latest and best).
3. Logs are written for loss and metrics.

Early stopping is supported using validation PSNR/SSIM plateau monitoring with patience control.

## 8. Logging, Monitoring, and Experiment Traceability

### 8.1 Scalar Logging

Per-step and per-epoch logs include:

1. Total loss.
2. Pixel loss.
3. SSIM loss component.
4. Validation PSNR.
5. Validation SSIM.

Backends:

1. Native tensorboard event logging.
2. Optional experiment tracker stub for online dashboards.

### 8.2 Artifact Logging

Key artifacts are persisted:

1. Model checkpoints.
2. Configuration snapshot.
3. Validation summaries.
4. Optional image grids (input, prediction, reference, prior visualization).

This supports full experiment reproducibility and retrospective error analysis.

## 9. Checkpointing and Resume Strategy

Saved state includes:

1. Network parameters.
2. Optimizer state.
3. Mixed precision scaler state.
4. Epoch/global-step counters.
5. Best-metric trackers.

Resume loads all states atomically so training continuity is preserved, including optimizer moments and loss scaling behavior.

## 10. Inference and Deployment Behavior

Inference runs in evaluation mode with gradient computation disabled. For each input:

1. Image is loaded and normalized.
2. Forward pass produces enhanced output (and optionally prior map).
3. Output is clamped to valid intensity range.
4. Result is converted back to image format and saved.

The pipeline supports single image and directory-level batch inference, with deterministic preprocessing to maintain deployment consistency.

## 11. Efficiency and Practical GPU Considerations

Efficiency constraints are addressed by:

1. Lightweight stage design in the enhancement stream.
2. Compact prior stream.
3. Multi-scale modulation with shallow transforms.
4. Mixed precision acceleration.
5. Moderate channel growth schedule.

This allows practical training and inference on commodity single-GPU setups while preserving restoration quality.

## 12. Numerical Conventions and Range Safety

Implementation conventions:

1. Internal tensors are float32/float16 depending on autocast context.
2. Image range remains \([0,1]\) throughout training and metric computation.
3. Final outputs are clamped before export.
4. Small epsilons are used in divisions and square roots to avoid NaN/Inf propagation.

These safeguards improve robustness in long training runs.

## 13. Reproducibility Controls

Reproducibility is enforced through:

1. Random seed initialization across Python, NumPy, and deep learning backend.
2. Deterministic or semi-deterministic data ordering controls.
3. Full hyperparameter serialization with each run.
4. Stable validation protocol and metric aggregation.

## 14. Failure Modes and Mitigations

Common failure modes and implementation responses:

1. Pair mismatch in dataset:
   - strict key checks and startup validation.
2. Over-smoothing:
   - structural term balancing and multi-scale skip fusion.
3. Color distortion:
   - prior-guided modulation conditioning across decoder stages.
4. Training instability under mixed precision:
   - dynamic gradient scaling and bounded modulation outputs.
5. Metric inconsistency across batches:
   - per-image computation followed by explicit mean aggregation.

## 15. End-to-End Operational Summary

The complete flow is:

1. Load paired degraded-clean samples.
2. Apply synchronized augmentations.
3. Predict pseudo-transmission prior.
4. Extract multi-scale enhancement features.
5. Modulate features with learned prior-conditioned weights.
6. Decode and reconstruct enhanced output.
7. Optimize with pixel + structural composite loss.
8. Validate with full-reference metrics and report underwater quality metrics.
9. Log all training/validation signals and persist checkpoints for reproducibility.

This implementation realizes a coherent hybrid methodology where physical prior estimation is not a post-processing add-on, but an integrated conditioning signal that continuously guides learned restoration at multiple representational levels.
