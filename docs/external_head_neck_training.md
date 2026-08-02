# Additional head-and-neck datasets

The external-data pipeline uses these fixed protocols:

- HaN-Seg: CT-to-CT inter-patient registration and a patient-disjoint 80/10/10
  split. Incomplete `case_19` annotations (`OAR_data.csv < 1`) are excluded
  class-by-class rather than rejecting the patient.
- SegRap2023: contrast-enhanced CT-to-CT inter-patient registration using the
  120 labelled training cases and a patient-disjoint 80/10/10 split. The two
  GTV masks are retained as channels but excluded from this OAR benchmark; all
  45 OARs are evaluated.
- Head-Neck-CBCT-CT: paired CBCT-to-CT registration and a patient-disjoint
  80/10/10 split. This copy has no segmentations, so checkpoint selection and
  evaluation use image similarity and deformation topology, not Dice.

HaN-Seg and SegRap masks are stored as independent binary channels. Do not
merge them into one integer map because several structures overlap or nest.

## 1. Preprocessing

Run from `/home/student3/data2t/TouJing/GaussianReg`. The cross-patient default
performs body centring and rigid/affine alignment to one training-only atlas.
Patient identities are split before atlas selection and pairing.
Each moving subject is paired with three deterministic, non-self fixed
subjects inside the same split; no pair crosses a patient split.

```bash
/home/student3/miniconda3/envs/SACB/bin/python prepare_head_neck_datasets.py \
  --dataset han-seg \
  --source-root /home/student3/data2t/TouJing/HaN-Seg \
  --output-root /home/student3/data2t/TouJing/HaN-Seg_gaussian_native_preprocessed \
  --num-workers 4
```

```bash
/home/student3/miniconda3/envs/SACB/bin/python prepare_head_neck_datasets.py \
  --dataset segrap2023 \
  --source-root /home/student3/data2t/TouJing/SegRap2023_Training_Set_120cases \
  --output-root /home/student3/data2t/TouJing/SegRap2023_gaussian_native_preprocessed \
  --segrap-modality contrast \
  --num-workers 4
```

```bash
/home/student3/miniconda3/envs/SACB/bin/python prepare_head_neck_datasets.py \
  --dataset head-neck-cbct-ct \
  --source-root /home/student3/data2t/TouJing/Head-Neck-CBCT-CT \
  --output-root /home/student3/data2t/TouJing/Head-Neck-CBCT-CT_gaussian_native_preprocessed \
  --num-workers 4
```

Each command writes `dataset_summary.json`, per-subject QA metadata, and
`manifests/{train,validation,test}.csv`. A preprocessing run with any failure
exits non-zero; inspect the summary rather than adding `--allow-failures` for a
paper experiment.

## 2. Training

Create the log directory once:

```bash
mkdir -p /home/student3/data2t/TouJing/GaussianReg/logs
```

Run the datasets sequentially on the single A100. HaN-Seg:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
  /home/student3/miniconda3/envs/SACB/bin/python train_gaussian_native.py \
  --config configs/gaussian_native_v12_han_seg.json \
  --data-root /home/student3/data2t/TouJing/HaN-Seg_gaussian_native_preprocessed \
  --train-manifest /home/student3/data2t/TouJing/HaN-Seg_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /home/student3/data2t/TouJing/HaN-Seg_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v12_han_seg_seed2026 \
  --device cuda:0 \
  > logs/gaussian_native_v12_han_seg_seed2026.log 2>&1 &
```

SegRap2023:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
  /home/student3/miniconda3/envs/SACB/bin/python train_gaussian_native.py \
  --config configs/gaussian_native_v12_segrap2023.json \
  --data-root /home/student3/data2t/TouJing/SegRap2023_gaussian_native_preprocessed \
  --train-manifest /home/student3/data2t/TouJing/SegRap2023_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /home/student3/data2t/TouJing/SegRap2023_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v12_segrap2023_seed2026 \
  --device cuda:0 \
  > logs/gaussian_native_v12_segrap2023_seed2026.log 2>&1 &
```

Head-Neck-CBCT-CT:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 \
  /home/student3/miniconda3/envs/SACB/bin/python train_gaussian_native.py \
  --config configs/gaussian_native_v12_head_neck_cbct_ct.json \
  --data-root /home/student3/data2t/TouJing/Head-Neck-CBCT-CT_gaussian_native_preprocessed \
  --train-manifest /home/student3/data2t/TouJing/Head-Neck-CBCT-CT_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /home/student3/data2t/TouJing/Head-Neck-CBCT-CT_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v12_head_neck_cbct_ct_seed2026 \
  --device cuda:0 \
  > logs/gaussian_native_v12_head_neck_cbct_ct_seed2026.log 2>&1 &
```

## 3. Test-set evaluation

HaN-Seg and SegRap use `best_validation_dice.pt`:

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  evaluate_gaussian_native.py \
  --checkpoint runs/gaussian_native_v12_han_seg_seed2026/best_validation_dice.pt \
  --data-root /home/student3/data2t/TouJing/HaN-Seg_gaussian_native_preprocessed \
  --manifest /home/student3/data2t/TouJing/HaN-Seg_gaussian_native_preprocessed/manifests/test.csv \
  --output-dir results/gaussian_native_v12_han_seg_test \
  --device cuda:0
```

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  evaluate_gaussian_native.py \
  --checkpoint runs/gaussian_native_v12_segrap2023_seed2026/best_validation_dice.pt \
  --data-root /home/student3/data2t/TouJing/SegRap2023_gaussian_native_preprocessed \
  --manifest /home/student3/data2t/TouJing/SegRap2023_gaussian_native_preprocessed/manifests/test.csv \
  --output-dir results/gaussian_native_v12_segrap2023_test \
  --device cuda:0
```

CBCT-CT uses `best_validation_ncc.pt` and intentionally emits no Dice fields:

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  evaluate_gaussian_native.py \
  --checkpoint runs/gaussian_native_v12_head_neck_cbct_ct_seed2026/best_validation_ncc.pt \
  --data-root /home/student3/data2t/TouJing/Head-Neck-CBCT-CT_gaussian_native_preprocessed \
  --manifest /home/student3/data2t/TouJing/Head-Neck-CBCT-CT_gaussian_native_preprocessed/manifests/test.csv \
  --output-dir results/gaussian_native_v12_head_neck_cbct_ct_test \
  --device cuda:0
```

For the final paper, repeat each training configuration with at least three
seeds and compare against baselines using exactly the generated subject splits.
