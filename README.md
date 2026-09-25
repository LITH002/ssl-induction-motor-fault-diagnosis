# Self-Supervised Learning for Induction Motor Fault Diagnosis

Companion code for the dissertation "A Self-Supervised Deep Learning Framework for Enhancing the Reliability of Induction Motor Fault Diagnosis in Data-Scarce Environments".

This repository implements a self-supervised pretraining and fine-tuning pipeline for motor fault diagnosis, evaluated against supervised and meta-learning (MAML) baselines, with model compression and real-world 
validation tooling. It is structured to be adapted to a different dataset, motor type, or fault category with minimal changes to the core pipeline.

## Requirements

```
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
numpy
scipy
scikit-learn
python-docx
pandas
```

## Datasets

This repository does not include the datasets themselves. To reproduce the results, download:

- CWRU: https://engineering.case.edu/bearingdatacenter/download-data-file
- Paderborn: https://groups.uni-paderborn.de/kat/BearingDataCenter/

Organise CWRU files with descriptive names (e.g. `Normal_0.mat`, `IR007_1.mat`, `OR007@6_2.mat`) and place Paderborn bearing folders (e.g. `K001`, `KA01`, `KI01`) under a shared parent directory. 
See `pipeline/labels.py` for the exact naming convention the loaders expect.

---

## 1. Pipeline: Pretraining and Fine-Tuning

### Pretrain an encoder

```
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

```
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

Repeat with `--seed 123` and `--seed 7` for the multi-seed comparison used in the dissertation. Add `--save_model path.pt` to save the trained model at full label availability, for use in the compression 
phase below.

---

## 2. MAML Baseline

```
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

```
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

```
python compression/validate_pruned_model.py \
  --dataset paderborn --data_dir path/to/paderborn \
  --fs 64000 --segment_len 4096 \
  --pruned_model pruned_30pct.pt \
  --widths_file pruned_30pct_widths.txt
```

### Measure inference latency on edge hardware

Copy `compression/measure_on_pi.py` plus a saved checkpoint to the target device (no other project files needed; this script is self-contained, torch-only, no dataset required):

```
python3 measure_on_pi.py \
  --checkpoint trained_model_100pct.pt --widths 32 64 128

python3 measure_on_pi.py \
  --checkpoint pruned_30pct.pt --widths 22 45 90
```

---

## 4. Real-World Validation Tooling

These tools were built for a specific plant's report format during this research; adapt the parsing logic in `parse_weekly_summaries.py` if your own reports use a different layout. 
Real plant data is not included in this repository for confidentiality reasons; only the code.

```
# Parse a folder of weekly MCSA reports into a structured CSV
python real_world_validation/parse_weekly_summaries.py \
  --input_dir weekly_reports/ --out weekly_data.csv

# Extract embedded graph images (rotor bar, torque oscillation, etc.)
python real_world_validation/extract_report_images.py \
  --docx report.docx --out_dir extracted_images

# Check one new report against a trained model, flagging disagreements
python real_world_validation/check_new_report.py \
  --docx new_report.docx --model tabular_model.pt --scaler tabular_scaler.json

# Build the supplementary tabular model and test it against real data
python real_world_validation/build_tabular_dataset.py \
  --data_dir path/to/paderborn --fs 64000 --out tabular_paderborn.csv

python real_world_validation/train_tabular_model.py \
  --csv tabular_paderborn.csv --out tabular_model.pt --scaler_out tabular_scaler.json

python real_world_validation/test_on_real_data.py \
  --weekly_csv weekly_data.csv --model tabular_model.pt --scaler tabular_scaler.json
```

---

## Citation

If you use this code, please cite the accompanying dissertation:

```
Nilanjith, T.M.L.K. (2026). A Self-Supervised Deep Learning Framework for Enhancing the Reliability of Induction Motor Fault Diagnosis in Data-Scarce
Environments. BSc dissertation, University of Kelaniya, Sri Lanka.
```

## License

[Choose one — MIT is a reasonable default for academic code]
