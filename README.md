# Physics-Guided Residual U-Net for Underwater Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/License-Research%20Only-lightgrey.svg)]()

Official research codebase for a hybrid **learning-based + physics-guided** underwater image enhancement framework, developed for CVPR-style experimentation.  
The method combines a residual U-shaped restoration network with a physics-prior transmission branch and multi-scale feature modulation.

## Abstract

Underwater image formation is affected by wavelength-dependent attenuation and backscatter, which jointly degrade contrast, color fidelity, and visibility. This repository implements a physics-guided enhancement architecture where a pseudo-transmission prior is estimated and injected into restoration features at multiple scales. The restoration stream predicts bounded residual corrections rather than full-image synthesis, improving optimization stability and structure preservation. The training pipeline supports mixed precision, multi-domain paired training, and standard full-reference metrics (PSNR/SSIM), with optional underwater no-reference proxy metrics (UCIQE/UIQM proxies).

## Method Summary

Given degraded input \(I\), the model predicts:

1. A pseudo-transmission map \(T = g_{\psi}(I)\) from the physics branch.
2. Multi-scale restoration features \(F^{(s)}\) from the main U-shaped branch.
3. Modulated features:
   \[
   F^{(s)}_{\text{mod}} = F^{(s)} \odot \phi_s(\text{resize}(T))
   \]
4. A bounded residual correction:
   \[
   \Delta I = 0.2 \cdot \tanh(\text{Head}(F^{(1)}_{\text{mod}}))
   \]
5. Final enhanced output:
   \[
   \hat I = \text{clip}(I + \Delta I, 0, 1)
   \]

## Repository Structure

```text
.
├── main.py          # CLI entrypoint: train / eval / infer
├── models.py        # Physics-guided residual U-Net architecture
├── data.py          # Paired/unpaired datasets and synchronized augmentations
├── losses.py        # Composite loss (reconstruction + SSIM + edge + optional NR)
├── metrics.py       # PSNR / SSIM / UCIQE-proxy / UIQM-proxy
├── utils.py         # Seeding and checkpoint utilities
├── detail.md        # Architecture change log + hyperparameter documentation
├── info.md          # Methodology narrative
├── diag.tex         # LaTeX/TikZ architecture + backprop diagram
└── outputs*         # Experiment artifacts
```

## Environment

### Recommended

```bash
conda activate gappy
python -V
```

### Minimal dependencies

```bash
pip install torch torchvision numpy pillow tqdm matplotlib tensorboard
```

## Dataset Protocol

### Paired supervision

- Degraded images: `trainA`
- Clean targets: `trainB`
- Pairing rule: basename matching (e.g., `im_xxx_.jpg` ↔ `im_xxx_.jpg`)

### Multi-domain paired training

The training CLI accepts **comma-separated path lists** (quoted as one argument), enabling joint training across domains (e.g., `underwater_scenes`, `underwater_dark`, `underwater_imagenet`).

## Training

### Recommended CVPR-style recipe (multi-domain paired)

```bash
python -u /home/zahid/Projects/CVPR/main.py train \
  --paired-degraded "/home/zahid/Projects/CVPR/EUVP/Paired/underwater_scenes/trainA,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_dark/trainA,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_imagenet/trainA" \
  --paired-clean "/home/zahid/Projects/CVPR/EUVP/Paired/underwater_scenes/trainB,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_dark/trainB,/home/zahid/Projects/CVPR/EUVP/Paired/underwater_imagenet/trainB" \
  --val-degraded "/home/zahid/Projects/CVPR/EUVP/test_samples/Inp" \
  --val-clean "/home/zahid/Projects/CVPR/EUVP/test_samples/GTr" \
  --out-dir "/home/zahid/Projects/CVPR/outputs_v3" \
  --epochs 200 \
  --batch 8 \
  --val-batch 4 \
  --crop 256 \
  --val-crop 256 \
  --base 48 \
  --lr 2e-4 \
  --min-lr 1e-6 \
  --weight-decay 1e-4 \
  --w-l1 1.0 \
  --w-ssim 0.3 \
  --w-edge 0.1 \
  --grad-clip 1.0 \
  --workers 4 \
  --log-interval 1 \
  --val-interval 2 \
  --save-interval 5 \
  --amp \
  --no-tb
```

### Notes

- Use quotes around comma-separated directory lists.
- If memory is limited: reduce `--batch` (first) and then `--base`.
- Validation metrics are printed only at epochs matching `--val-interval`.

## Evaluation (paired)

```bash
python -u /home/zahid/Projects/CVPR/main.py eval \
  --val-degraded /home/zahid/Projects/CVPR/EUVP/test_samples/Inp \
  --val-clean /home/zahid/Projects/CVPR/EUVP/test_samples/GTr \
  --checkpoint /home/zahid/Projects/CVPR/outputs_v3/train/train_203448_19042026/ckpt_best.pt \
  --out-dir /home/zahid/Projects/CVPR/outputs_v3 \
  --crop 256 \
  --base 48 \
  --workers 4 \
  --save-images 32 \
  --save-transmission
```

### Eval outputs

```text
outputs_v3/eval/eval_<timestamp>/
├── metrics.json
├── metrics.csv
└── images/
    ├── 00000_inp_pred_gt.png
    ├── 00000_T.png
    └── ...
```

Panel order in `*_inp_pred_gt.png`:

1. input degraded image
2. predicted enhanced image
3. ground-truth clean image

## Inference (unpaired folder)

```bash
python -u /home/zahid/Projects/CVPR/main.py infer \
  --in-dir /home/zahid/Projects/CVPR/EUVP/test_samples/Inp \
  --out-dir /home/zahid/Projects/CVPR/outputs_v3/infer_samples \
  --checkpoint /home/zahid/Projects/CVPR/outputs_v3/train/train_203448_19042026/ckpt_best.pt \
  --base 48 \
  --save-transmission
```

Outputs include enhanced images and optional transmission maps (`*_T.png`).

## Loss Function

Training uses:

\[
\mathcal{L} = \lambda_{rec}\mathcal{L}_{rec} + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{edge}\mathcal{L}_{edge} + \lambda_{nr}\mathcal{L}_{nr}
\]

Current default supervised configuration:

- \(\lambda_{rec}=1.0\)
- \(\lambda_{ssim}=0.3\)
- \(\lambda_{edge}=0.1\)
- \(\lambda_{nr}=0.0\)

## Metrics

### Full-reference

- PSNR
- SSIM

### Underwater no-reference proxies

- UCIQE proxy
- UIQM proxy

## Reproducibility

- Set random seed via `--seed` (default: `42`).
- Training args are saved in each run directory (`args.json`).
- Checkpoints are periodically saved (`ckpt_epoch_*.pt`) and best model as `ckpt_best.pt`.

## Expected Artifacts

### Train

```text
outputs_v3/train/train_<timestamp>/
├── args.json
├── ckpt_best.pt
└── ckpt_epoch_*.pt
```

### Eval

```text
outputs_v3/eval/eval_<timestamp>/
├── metrics.json
├── metrics.csv
└── images/
```

### Infer

```text
outputs_v3/infer_samples/
├── <image>.jpg
├── <image>_T.png
└── ...
```

## Diagram

Architecture/backprop figure source:

- `diag.tex`

Compile:

```bash
pdflatex -interaction=nonstopmode -halt-on-error diag.tex
```

Output:

- `diag.pdf`

## Troubleshooting

### `ambiguous option: --w- ...`

This is a shell line-break issue. Re-run the command without broken flags (e.g., keep `--w-l1` intact).

### `No paired samples found`

Check basename matching between degraded/clean paths and ensure path list counts match for multi-domain mode.

### No logs visible

Use `python -u ...`, and for debug runs use:

- `--workers 0`
- `--log-interval 1`

## Citation

If you use this repository in academic work, please cite your paper/release corresponding to this implementation.

```bibtex
@misc{physics_guided_underwater_2026,
  title        = {Physics-Guided Residual U-Net for Underwater Image Enhancement},
  author       = {Your Name et al.},
  year         = {2026},
  note         = {Code repository}
}
```

## Contact

For technical issues or reproducibility questions, open an issue with:

- exact command
- error trace
- run directory (`outputs*/train/train_<timestamp>`)
