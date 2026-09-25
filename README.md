# Self-Supervised Learning for Induction Motor Fault Diagnosis

Companion code for the dissertation *"A Self-Supervised Deep Learning Framework for Enhancing the Reliability of Induction Motor Fault Diagnosis in Data-Scarce Environments"*.

This repository implements a self-supervised pretraining and fine-tuning pipeline for motor fault diagnosis, evaluated against supervised and meta-learning (MAML) baselines, with model compression tooling. It is structured to be adapted to a different dataset, motor type, or fault category with minimal changes to the core pipeline.

## Structure

```
pipeline/       Core preprocessing, model, and pretraining/fine-tuning
maml/           MAML meta-learning baseline
compression/    Channel pruning, validation, and edge-latency measurement
```

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

This repository does not include the datasets themselves. To reproduce the results, download:

- **CWRU**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Paderborn**: https://groups.uni-paderborn.de/kat/BearingDataCenter/

Organise CWRU files with descriptive names (e.g. `Normal_0.mat`, `IR007_1.mat`, `OR007@6_2.mat`) and place Paderborn bearing folders (e.g. `K001`, `KA01`, `KI01`) under a shared parent directory. See `pipeline/labels.py` for the exact naming convention the loaders expect.

---

## 1. Pipeline: Pretraining and Fine-Tuning

### Pretrain an encoder

```bash
python pipeline/pretrain.py \
  --dataset paderborn \
  --data_dir path/to/paderborn \
  --fs 64000 \
  --segment_len 4096 \
  --epochs 20 \
  --out pretrained_paderborn.pt
```

Use `--dataset cwru --fs 12000 --segment_len 2048` for CWRU (vibration).

### Fine-tune and compare conditions

```bash
# Self-supervised (pretrained)
python pipeline/finetune.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 --epochs 30 --lr 1e-4 \
  --pretrained pretrained_paderborn.pt \
  --seed 42 --out results_ssl.csv

# Supervised baseline (omit --pretrained)
python pipeline/finetune.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 --epochs 30 --lr 1e-4 \
  --seed 42 --out results_supervised.csv
```

Repeat with `--seed 123` and `--seed 7` for the multi-seed comparison used in the dissertation. Add `--save_model path.pt` to save the trained model at full label availability, for use in the compression phase below.

---

## 2. MAML Baseline

```bash
python maml/maml_pretrain.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 --meta_iterations 300 \
  --seed 42 --out maml_paderborn.pt

python pipeline/finetune.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 --epochs 30 --lr 1e-4 \
  --maml_checkpoint maml_paderborn.pt \
  --seed 42 --out results_maml.csv
```

---

## 3. Compression

### Prune and retrain

```bash
python compression/compress.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 \
  --checkpoint trained_model_100pct.pt \
  --pruning_ratios 0.1 0.2 0.3 0.4 0.5 \
  --lr 5e-5 --retrain_epochs 40 --seed 123 \
  --out compression_results.csv \
  --save_pruned_model pruned_30pct.pt --save_ratio 0.3
```

### Validate a pruned model's predictions

```bash
python compression/validate_pruned_model.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 \
  --pruned_model pruned_30pct.pt \
  --widths_file pruned_30pct_widths.txt
```

### Measure inference latency on edge hardware

Copy `compression/measure_on_pi.py` plus a saved checkpoint to the target device (no other project files needed; this script is self-contained, torch-only, no dataset required):

```bash
python3 measure_on_pi.py \
  --checkpoint trained_model_100pct.pt --widths 32 64 128

python3 measure_on_pi.py \
  --checkpoint pruned_30pct.pt --widths 22 45 90
```

---

## Citation

If you use this code, please cite the accompanying dissertation:

```
[Nilanjith, 2026] Nilanjith, T.M.L.K., A Self-Supervised Deep Learning Framework for Enhancing the Reliability of Induction Motor Fault Diagnosis in Data-Scarce Environments, vol. BSc (Honours). Kelaniya, Sri Lanka: University of Kelaniya, 2026.
```
