# GAM-SACB-Net for longitudinal head-and-neck registration

This branch extends the official [SACB-Net](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_SACB-Net_Spatial-awareness_Convolutions_for_Medical_Image_Registration_CVPR_2025_paper.html) backbone with one compact research path at the L4 scale:

1. **Gaussian Anatomy Correspondence Module (GACM)** extracts shared diagonal-Gaussian tokens and matches them with feature/position softmax correspondence.
2. A **lightweight geometry-conditioned residual corrector** rasterizes the matched token-feature residual and uses it to refine the original SACB dense flow.

The minimal-v2 model has no learned visibility, confidence, anatomical type
head, unbalanced transport, Bures covariance cost, standalone Gaussian flow,
Gaussian/dense gate, or cross-scale context propagation.

The original `SACB_Net` remains available. The new entry point is `GAM_SACB_Net` in `model_gam.py`.

## Environment

```bash
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

The pinned environment follows the original PyTorch 1.13/CUDA 11.7 setup and adds SimpleITK for metadata-safe medical-image preprocessing.

## HNTS-MRG24 protocol

The primary experiment is within-patient longitudinal T2 MRI registration: raw preRT is the moving image and midRT is fixed. The challenge-provided deformably registered preRT image is never used as a model input or training target. Tumor labels (`1=GTVp`, `2=GTVn`) are used only for validation/test metrics.

Preprocessing performs deterministic rigid then affine mutual-information prealignment, resamples into a centered physical frame at 1.5 mm isotropic resolution and `(D,H,W)=(128,160,160)`, applies robust 0.5–99.5 percentile MRI normalization, checks tumor ROI coverage, and creates patient-disjoint stratified 80/10/10 manifests.

```bash
python prepare_hntsmrg24.py \
  --source-root /path/to/HNTSMRG24_train \
  --output-root /path/to/HNTSMRG24_gam_preprocessed \
  --num-workers 2
```

If any case fails geometry, registration, label, or crop QA, preprocessing exits nonzero and records the reason in `dataset_summary.json`. Do not use `--allow-failures` for a final paper experiment without reporting exclusions.

## Training

Run on one selected 40 GB A100. `CUDA_VISIBLE_DEVICES` should be chosen after checking current GPU occupancy.

```bash
CUDA_VISIBLE_DEVICES=0 python train_gam.py \
  --config configs/gam_sacb_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gam_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gam_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gam_preprocessed/manifests/validation.csv \
  --output-dir runs/gam_sacb_minimal_v2_hntsmrg24_seed2026 \
  --device cuda:0
```

The same command is packaged as
`scripts/train_gam_v2_hntsmrg24.sh DATA_ROOT GPU_ID RUN_DIR [BASELINE_CKPT]`.
To initialize all shared backbone weights, add
`--baseline-checkpoint /path/to/baseline.pth`. Minimal-v2 is structurally
incompatible with checkpoints from the earlier dual-GCDR GAM model. To
continue a minimal-v2 run, use `--resume runs/.../latest.pt`; the script
rejects a resume if the config or manifest hashes differ.

Training is unsupervised with respect to anatomy labels. It uses LNCC,
displacement smoothness, multi-scale image similarity, and a small Jacobian
folding penalty. There are no token, confidence, transport, or anchor-flow
auxiliary losses. Logs are written to TensorBoard and `metrics.jsonl`. The best
checkpoint is selected by validation image NCC rather than validation Dice.

## Evaluation

Run the held-out test manifest once after model/hyperparameter selection:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_gam.py \
  --checkpoint runs/gam_sacb_minimal_v2_hntsmrg24_seed2026/best_validation_ncc.pt \
  --data-root /path/to/HNTSMRG24_gam_preprocessed \
  --manifest /path/to/HNTSMRG24_gam_preprocessed/manifests/test.csv \
  --output-dir results/gam_sacb_minimal_v2_hntsmrg24_seed2026 \
  --device cuda:0 \
  --save-predictions
```

The evaluator writes patient-level CSV/JSON and bootstrap 95% confidence intervals for pre/post NCC and Dice, HD95, ASSD, Jacobian folding, and deformation magnitude. A tumor class is eligible only when present in both original timepoints; if an eligible structure disappears after warping, it receives Dice 0 and an image-diagonal surface-distance penalty instead of being skipped. TRE is not reported because HNTS-MRG24 has no landmark annotations.

## Controlled original SACB-Net baseline

The repository retains the original SACB-Net architecture and provides a thin
training-interface adapter so it can use the same HNTS-MRG24 split, optimizer,
schedule, AMP safeguards, validation-NCC checkpoint selection, and held-out
metrics as GAM-SACB-Net. The baseline objective contains only the losses shared
with the original architecture: LNCC (weight 1.0) and displacement smoothness
(weight 0.3). No GACM/GCDR auxiliary loss is applied.

```bash
CUDA_VISIBLE_DEVICES=0 python train_sacb_baseline.py \
  --config configs/sacb_baseline_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gam_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gam_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gam_preprocessed/manifests/validation.csv \
  --output-dir runs/sacb_baseline_hntsmrg24_seed2026 \
  --device cuda:0
```

Evaluate the validation-NCC-selected baseline checkpoint exactly once on the
held-out test split:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_sacb_baseline.py \
  --checkpoint runs/sacb_baseline_hntsmrg24_seed2026/best_validation_ncc.pt \
  --data-root /path/to/HNTSMRG24_gam_preprocessed \
  --manifest /path/to/HNTSMRG24_gam_preprocessed/manifests/test.csv \
  --output-dir results/sacb_baseline_hntsmrg24_seed2026_test \
  --device cuda:0 \
  --save-predictions
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The focused suite covers compact Gaussian tokenization, softmax
correspondence, bounded residual correction, full-model backward propagation,
baseline checkpoint compatibility, parameter budget, DHW flow conventions,
objective composition, medical metrics, and manifest/preprocessing behavior.

## Original datasets and weights

The original project supports [Learn2Reg abdomen CT-CT](https://learn2reg.grand-challenge.org/Datasets/) and LPBA. The original SACB-Net weights are available from [Google Drive](https://drive.google.com/drive/folders/1XW19iuyCyg3YGmCpLFGGFjdPFi73xxwh?usp=share_link).

## Citation
```bibtex
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Xinxing and Zhang, Tianyang and Lu, Wenqi and Meng, Qingjie and Frangi, Alejandro F. and Duan, Jinming},
    title     = {SACB-Net: Spatial-awareness Convolutions for Medical Image Registration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5227-5237}
}
```

## Acknowledgments
We sincerely acknowledge the [ModeT](https://github.com/ZAX130/SmileCode), [CANNet](https://github.com/Duanyll/CANConv) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects.
