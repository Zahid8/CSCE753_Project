# Architecture Change Record (Updated)

This document is rewritten to explicitly capture **each architecture-level change** that has been applied, why it was made, and what effect it is expected to have on training behavior and final image quality.

## Current Architecture (after all changes)

The model is now a hybrid physics-guided residual U-Net:

1. A compact physics branch estimates a pseudo-transmission map `T`.
2. A multi-scale encoder-decoder restoration branch processes image features.
3. At every major scale, features are modulated with learned transmission-conditioned weights using:
   - `F' = F ⊙ phi(T_resized)`
4. The output head predicts a bounded residual RGB correction and adds it to the input:
   - `residual = s * tanh(head(feature))`
   - `output = clamp(input + residual, 0, 1)`

Where `s = 0.2` is a residual scaling factor.

## Change 1: Replaced attention-heavy core with residual U-Net blocks

### What changed

- Removed the earlier full-resolution transformer-style feature path.
- Replaced with a U-shaped encoder-decoder using:
  - stride-2 convolutional downsampling blocks
  - bilinear upsample + skip fusion decoder blocks
  - depthwise-separable residual refinement blocks

### Why

- Full-token attention becomes expensive and unstable for high-resolution underwater images.
- A residual U-Net is more compute-efficient and usually stronger for dense image restoration on limited hardware.

### Expected effect

- Better memory efficiency.
- Faster training iteration throughput.
- Improved local texture restoration and edge continuity.

## Change 2: Introduced depthwise-separable residual blocks with channel gating

### What changed

Each refinement block now uses:

1. depthwise `3x3` convolution
2. pointwise `1x1` convolution
3. squeeze-excitation style channel attention
4. residual skip

### Why

- Improves representation quality per FLOP.
- Channel gating helps suppress noisy responses and emphasize informative channels.

### Expected effect

- Better detail recovery and color correction with lower compute than dense conv stacks.

## Change 3: GroupNorm everywhere in feature blocks

### What changed

- Replaced normalization logic with GroupNorm in conv blocks.
- Group count is selected so channel divisibility is always valid.

### Why

- Underwater enhancement often uses small batch sizes due high resolution.
- BatchNorm can become unstable in that regime; GroupNorm is robust.

### Expected effect

- More stable optimization with small/variable batch sizes.

## Change 4: Expanded multi-scale physics-guided modulation

### What changed

- Physics prior branch still predicts `T`, but modulation is now applied at:
  - encoder stages
  - bottleneck
  - decoder stages
- Dedicated modulation heads (`phi`) are used per scale.

### Why

- Different scales need different prior conditioning behavior.
- Early scales need coarse correction; late scales need fine-detail correction.

### Expected effect

- Stronger coupling between degradation prior and restoration features.
- Better consistency of enhancement across image regions.

## Change 5: Switched to residual image prediction (instead of direct RGB only)

### What changed

- Output now predicts a residual correction map that is added to input.

### Why

- Restoration is typically a relatively small correction to input.
- Learning residuals is easier than learning full image reconstruction from scratch.

### Expected effect

- Faster convergence.
- Better structure preservation.
- Lower risk of washed-out outputs.

## Change 6: Stabilized residual head with zero-initialization and scaling

### What changed

- Final output conv is zero-initialized (`weight = 0`, `bias = 0`).
- Residual amplitude is scaled by `0.2`:
  - `residual = 0.2 * tanh(...)`

### Why

- Prevents catastrophic early outputs from random initialization.
- Makes initial model behavior close to identity mapping.

### Expected effect

- Initial validation starts near input baseline rather than collapsing.
- Dramatically better training stability in early epochs.

## Change 7: Kept transmission output for interpretability

### What changed

- Model still returns both enhanced output and transmission map `T`.

### Why

- Enables debugging and qualitative inspection of physics branch behavior.
- Helps verify whether conditioning is meaningful spatially.

### Expected effect

- Easier failure analysis and ablation-driven iteration.

## Stage layout summary

Let base width be `B`.

- Encoder channels: `B, 2B, 4B, 8B, 16B`
- Decoder channels: `8B, 4B, 2B, B`
- Output: 3-channel residual correction
- Prior: 1-channel transmission map

For input `N x 3 x H x W` (with `H, W` divisible by 16):

- `e1`: `N x B x H x W`
- `e2`: `N x 2B x H/2 x W/2`
- `e3`: `N x 4B x H/4 x W/4`
- `e4`: `N x 8B x H/8 x W/8`
- `b`: `N x 16B x H/16 x W/16`
- `d4`: `N x 8B x H/8 x W/8`
- `d3`: `N x 4B x H/4 x W/4`
- `d2`: `N x 2B x H/2 x W/2`
- `d1`: `N x B x H x W`
- `T`: `N x 1 x H x W`
- `output`: `N x 3 x H x W`

## Hyperparameters (Current)

### A) Architecture hyperparameters

| Hyperparameter | Meaning | Current default | Typical strong setting |
|---|---|---:|---:|
| `base` | Base channel width `B` for encoder/decoder (`B,2B,4B,8B,16B`) | `48` | `48` (or `32` if memory-limited) |
| `depth` | Backward-compatible constructor arg (not used in current internal topology) | `4` | `4` |
| `blocks-per-stage` | Backward-compatible constructor arg (not used in current internal topology) | `1` | `1` |
| `residual_scale` | Output residual amplitude multiplier | `0.2` (fixed in model) | `0.2` |
| output head init | Final residual conv initialization | all zeros | all zeros |
| normalization | Feature normalization type | GroupNorm | GroupNorm |
| activation | Nonlinearity | SiLU | SiLU |

### B) Loss hyperparameters

| Hyperparameter | Meaning | Current default | Typical strong setting |
|---|---|---:|---:|
| `w-l1` | Reconstruction term weight (Charbonnier by default) | `1.0` | `1.0` |
| `w-ssim` | Structural loss weight | `0.3` | `0.3` |
| `w-edge` | Edge-consistency loss weight | `0.1` | `0.1` |
| `w-nr` | Non-reference proxy quality loss weight | `0.0` | `0.0` for paired training |
| `use-l1-only` | If enabled, use L1 instead of Charbonnier for reconstruction | `False` | `False` |
| Charbonnier `eps` | Robustness constant in reconstruction term | `1e-3` (fixed) | `1e-3` |
| SSIM window | SSIM local window size | `11` (fixed) | `11` |
| SSIM sigma | SSIM Gaussian sigma | `1.5` (fixed) | `1.5` |

### C) Optimizer and schedule hyperparameters

| Hyperparameter | Meaning | Current default | Typical strong setting |
|---|---|---:|---:|
| optimizer | Parameter update rule | AdamW | AdamW |
| `lr` | Initial learning rate | `2e-4` | `2e-4` |
| `min-lr` | Final cosine LR floor | `1e-6` | `1e-6` |
| betas | AdamW momentum terms | `(0.9, 0.99)` (fixed) | `(0.9, 0.99)` |
| `weight-decay` | L2-style regularization | `1e-4` | `1e-4` |
| LR schedule | Epoch-wise cosine annealing | enabled (fixed) | enabled |
| `grad-clip` | Max gradient norm | `1.0` | `1.0` |
| `amp` | Mixed precision training | `True` | `True` |
| `ema-decay` | Exponential moving average for eval model | `0.0` | `0.0` (or `0.999` after stable convergence) |

### D) Data and batching hyperparameters

| Hyperparameter | Meaning | Current default | Typical strong setting |
|---|---|---:|---:|
| `batch` | Training batch size | `8` | `8` (reduce to `4` if OOM) |
| `val-batch` | Validation batch size | `4` | `4` |
| `crop` | Training crop size | `256` | `256` (increase if memory allows) |
| `val-crop` | Validation resize/crop size | `256` | `256` |
| `workers` | DataLoader worker processes | `4` | `4` (set `0` for debug) |
| augmentations | Training paired augmentations | synced crop + flip + optional rotate | same |
| paired dataset input | multiple domains via comma-separated paths | supported | scenes + dark + imagenet |

### E) Run-control hyperparameters

| Hyperparameter | Meaning | Current default | Typical strong setting |
|---|---|---:|---:|
| `epochs` | Total training epochs | `120` | `200` |
| `val-interval` | Validate every N epochs | `2` | `1` or `2` |
| `save-interval` | Save checkpoint every N epochs | `5` | `5` |
| `log-interval` | Training log frequency (steps) | `20` | `1` for detailed monitoring |
| `early-stop-patience` | Stop if no PSNR gain for N validation checks | `15` | `15` |
| `max-steps` | Cap steps per epoch (`0` means full epoch) | `0` | `0` |
| `seed` | Random seed | `42` | `42` |
| `tb` | TensorBoard logging | `True` | `False`/`True` as needed |

### F) Recommended configuration for current setup

This is the exact high-quality training profile currently recommended:

- multi-domain paired training: `underwater_scenes + underwater_dark + underwater_imagenet`
- `epochs=200`
- `base=48`
- `batch=8`, `val-batch=4`
- `crop=256`, `val-crop=256`
- `lr=2e-4`, `min-lr=1e-6`, `weight-decay=1e-4`
- `w-l1=1.0`, `w-ssim=0.3`, `w-edge=0.1`, `w-nr=0.0`
- `grad-clip=1.0`
- `amp=True`
- `ema-decay=0.0`

## Practical note for future updates

Whenever architecture changes are made again, this file should be updated in this exact format:

1. What changed.
2. Why it changed.
3. Expected effect.

This keeps architecture evolution and hyperparameter choices traceable for experiments and paper writing.
