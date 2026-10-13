# PhyGRec

PhyGRec recovers spatial transcriptomic expression by combining physical
backprojection, adaptive modulation and graph-context correction.

This repository provides the implementation and configurations for the paper's
main experiments, receiver-LOCO evaluation, three component ablations and seven
comparison baselines, with shared expression, cell-type annotation and clustering
evaluation.

[Setup](#setup) · [Experiments](#experiments) · [Evaluation](#evaluation) ·
[Training protocol](#training-protocol) · [Code structure](#code-structure)

## Setup

### Install

Use Python 3.11 or newer; annotation and clustering require Python 3.12 or newer.
The PhyGRec training configurations use one GPU. Install a PyTorch build for your
accelerator, then run the following from the repository root:

```bash
python -m pip install -e .
```

Use the editable installation above with benchmark files in `data/`.
[environment.txt](environment.txt) records the tested runtime;
[pyproject.toml](pyproject.toml) declares package dependencies.

### Prepare the benchmark

Place the separately distributed benchmark files in this repository's `data/`
directory. See [data/readme.md](data/readme.md) for the file layout, split definitions
and evaluation resources. The main split should be at
`data/benchmark/main/{train.h5,val.h5,test.h5,genes.txt}`.

Verify manifest hashes, gene order and split isolation:

```bash
python scripts/verify_release.py
```

The prepared benchmark contains mixing scenes and Reference expression targets.
Model inputs use mixed expression, receiver/candidate indices, distances and masks.

## Experiments

### Start with one seed

Inspect the execution plan, then train and evaluate the main model:

```bash
python scripts/main.py --experiment main --seeds 20260816 --dry-run
python scripts/main.py --experiment main --seeds 20260816
```

Each run trains a model, selects its checkpoint using validation scores and
scores the test split. Results are saved under `runs/results/`.

### Main, receiver-LOCO and ablations

Omitting `--seeds` runs all three paper seeds: `20260816`, `20260817`, `20260818`.

| Experiment | Command | Training runs |
| --- | --- | ---: |
| Main | `python scripts/main.py --experiment main` | 3 |
| Receiver-LOCO: A9, L7, NA | `python scripts/main.py --experiment loco` | 9 |
| Full model + three ablations | `python scripts/main.py --experiment ablation` | 12 |
| All, sharing full-model runs | `python scripts/main.py --experiment all` | 21 |

Ablations retain the settings of the remaining components:

| Configuration | Removed component |
| --- | --- |
| [rb](configs/ablation/rb/) | Physical backprojection increment; circle operator and candidate graph retained |
| [aim](configs/ablation/aim/) | Adaptive gain and momentum |
| [gcc](configs/ablation/gcc/) | Graph-context correction |

Use a fresh output directory for each repeated training attempt to preserve its
validation records.

### Seven comparison baselines

| Method | CLI name |
| --- | --- |
| GATv2 | `gatv2` |
| GraphSAGE | `graphsage` |
| GAT | `gat` |
| GCN | `gcn` |
| MPNN | `mpnn` |
| VAE | `vae` |
| Physical PGD | `physical_pgd` |

Fixed [baseline configurations](configs/baselines/) are reused across main and
LOCO splits. Run all methods, or replace `all` with one CLI name:

```bash
python scripts/baselines.py reproduce --method all --experiment main
python scripts/baselines.py reproduce --method all --experiment loco
```

Use `--experiment all` for both main and LOCO. Both experiment entry points accept
`--seeds`, `--dry-run` and `--structure` (annotation and clustering; see below).

## Evaluation

### Evaluate a locally trained model

For an individual PhyGRec run:

```bash
phygrec-cli fit --config configs/main/seed20260816.yaml
python scripts/evaluate.py --split main --seed 20260816
```

Training automatically records the best validation checkpoint in `best.json`.
Evaluation reads this record, or accepts `--checkpoint PATH`. For an ablation,
train its configuration and specify the same variant when evaluating:

```bash
phygrec-cli fit --config configs/ablation/gcc/seed20260816.yaml
python scripts/evaluate.py --split main --seed 20260816 --variant gcc
```

For an individual baseline:

```bash
python scripts/baselines.py train --method gatv2 --split main --seed 20260816
python scripts/baselines.py evaluate --method gatv2 --split main --seed 20260816
```

Individual evaluation commands print JSON. `--device cpu` or `--device cuda`
controls PhyGRec evaluation and baseline training/evaluation.

### Include annotation and clustering

Install the optional dependencies and add `--structure` to an experiment or
individual evaluation command:

```bash
python -m pip install -e '.[structure]'
python scripts/main.py --experiment main --structure
python scripts/baselines.py reproduce --method all --experiment loco --structure
```

`--structure` evaluates cell-type annotation and clustering for Reference, Mixed
and recovered expression. Within each training seed, receiver-scene expression
is averaged by physical cell identity. Frozen nonzero-Reference cohorts contain
6,394 cells for main and 2,027/2,127/2,240 for A9/L7/NA. Zero-Reference receivers
remain in expression evaluation.

- **Annotation:** fit standardization on Reference log-normalized counts, clip
  to `[-10, 10]` and apply five fixed marker programs. Report Accuracy, Macro
  Precision, Balanced Accuracy, Macro F1, MCC, Cohen's kappa, Macro AP from
  softmax marker scores, and a five-class confusion matrix. Reference hard-label
  agreement is one; its Macro AP is omitted.
- **Clustering:** independently fit each expression state using standardized,
  clipped 20-PC expression, a 30-neighbor cosine graph and Leiden resolution
  `0.35` with three fixed initializations. Report ARI/NMI against frozen Reference
  partitions, pairwise stability ARI, cluster counts and frozen-label silhouette
  on at most 4,000 cells. Leiden variation and training-seed variation are
  recorded separately; Reference and Mixed scores are point estimates.

Evaluation uses the cohorts, gene order and Reference partitions in
`data/structure/`. Scoring runs in a separate process with one-thread settings
and verifies resource and aggregated Reference-expression hashes.

### Results and aggregation

| Output | Location |
| --- | --- |
| PhyGRec per-run scores | `runs/results/{main,a9,l7,na,rb,aim,gcc}/seedSEED.json` |
| PhyGRec summaries | `runs/results/{main,loco,ablation,all}_summary.json` |
| Baseline per-run scores | `runs/results/baselines/METHOD/SPLIT/seedSEED.json` |
| Baseline summaries | `runs/results/baselines/{main,loco,all}_summary.json` |

These files are written by the experiment entry points. With `--structure`,
they also contain annotation and clustering results.

Expression metrics are Count-MAE, Log-MAE, relative L1, Count-RMSE, Log-RMSE and
relative L2 over receiver-scene entries. Relative L2 averages each receiver's
error norm divided by its Reference norm, floored at one. Summaries report mean
and sample SD across training seeds; a single seed has no sample SD. LOCO first
averages the three folds equally within each seed.

### Public checkpoints

Public checkpoints are pending retraining with this implementation. The commands
above train models and evaluate their validation-selected checkpoints.
For `--evaluate-only`, the experiment entry points use these checkpoint locations:

| Experiment | Expected checkpoint source |
| --- | --- |
| PhyGRec main / LOCO | `checkpoints/{main,a9,l7,na}/seedSEED.ckpt` |
| PhyGRec ablations | Locally selected ablation checkpoints, plus the main checkpoints above |
| Baselines | `runs/baselines/METHOD/SPLIT/seedSEED/best.pt` |

### Paper result records

`results/ablation_expression.json` contains the reported ablation scores, selected
epochs and parameter counts for the original experiment checkpoints. Files in
`results/` accompany the paper; new evaluations are saved to `runs/results/`.

## Training protocol

### PhyGRec

The full model has four blocks and 2,832,681 parameters: 1,000 genes, width-16
AIM, and two width-128 GATv2 layers with four heads and zero dropout. Each block
combines physical backprojection, adaptive modulation, momentum and graph
correction before nonnegative projection.

| Setting | Value |
| --- | --- |
| Learning rates: solver / circle / AIM / GCC | `0.003125 / 0.06 / 0.003 / 0.01` |
| Schedule | 100 epochs, one GPU |
| Training batch / gradient accumulation | 3 scene graphs / 2 batches (6 effective graphs) |
| Validation and test batch | 6 scene graphs |
| Loss | Raw relative L1 + normalized-log MAE, unit weights |
| Optimizer / weight decay | AdamW / `1e-4` |
| Gradient clipping / EMA decay | `1 / 0.99` |
| Validation interval | Every 10 epochs |

Validation selection maximizes the mean of Log-MAE and relative-L1 recovery
relative to Mixed, breaking ties by earliest epoch. Validation and test use EMA
parameters, with test scoring performed after checkpoint selection.
Circle-total normalization uses the detached median of receiver and valid
donor-slot totals across the packed batch. The validation callback updates `best.json` beside its `val_epoch_*.json`
records whenever the score improves.

### Baselines

Neural baselines use 100 epochs, raw relative L1 plus normalized-log MAE, AdamW
and gradient clipping at 5. Validation runs at epoch 1 and every five epochs;
the earliest checkpoint with the highest recovery score is selected.

- VAE adds KL weight `0.02` and evaluates with its latent mean.
- MPNN uses fixed distance-kernel weights in residual messages.
- Physical PGD fits a four-parameter circle operator for 20 epochs on training
  data, then freezes it for 100 PGD iterations with step size `0.1`. Main fitting
  averages three independent scene losses per update; LOCO packs six scenes.

## Code structure

```text
data/                     # Prepared benchmark and evaluation resources
checkpoints/              # Public pretrained weights (pending release)
results/                  # Paper result records
runs/                     # Generated training logs, checkpoints and evaluation results
configs/                  # Fixed main, LOCO, ablation and baseline settings
scripts/
  main.py                 # PhyGRec experiment orchestration
  baselines.py            # Baseline training, evaluation and experiments
  evaluate.py             # Evaluate a PhyGRec checkpoint
  verify_release.py       # Verify the prepared benchmark
src/phygrec/
  models/                 # Recovery solver, operators, AIM and GCC
  training/               # Lightning training and validation checkpoint selection
  evaluation/
    expression.py         # Expression metrics and validation/test loops
    annotation.py         # Marker-based cell-type annotation
    clustering.py         # PCA, Leiden and clustering metrics
    scoring.py            # Cell aggregation, scoring dispatch and result summaries
  data/                   # Scene stores, paths and batching
  baselines/              # Seven methods and their training/evaluation logic
  __init__.py             # Shared split, seed, ablation and path definitions
  utils.py                # Hashing and numerical transforms
tests/                    # Model, protocol and evaluation checks
```

The main method and baselines share evaluation functions. Training selects
checkpoints; scripts coordinate experiments; `evaluation/scoring.py` aggregates
cells and summarizes folds and seeds. Optional annotation and clustering modules
load inside the isolated scoring process. Benchmark resources live in
`data/`; `src/phygrec/data/` contains the Python data-loading code. Data, published
weights and paper result records are distributed separately and excluded from Git.
Each has a `MANIFEST.json` with paths relative to its own directory.

### Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Install `.[structure]` as well to run tests that require annotation and clustering
dependencies.

## License

Code license: pending selection.
