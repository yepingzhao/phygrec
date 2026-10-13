# PhyGRec

Physics-guided graph recovery for spatial transcriptomic expression. This repository
contains the paper's final method, receiver-LOCO experiments, three component-removal
ablations, seven comparison baselines, and expression, annotation and clustering
evaluation.

## Installation

Use Python 3.11 or newer for expression experiments, or Python 3.12 or newer for
annotation and clustering. Run all commands from this repository's root:

```bash
python -m pip install -e .
```

PhyGRec training configurations use one GPU. Install a PyTorch build appropriate
for your accelerator; [environment.txt](environment.txt) records the tested runtime.
To run the test suite:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Benchmark data

Obtain the final benchmark files separately and place `phygrec-data/` beside the
repository. The expected layout is:

```text
phygrec/
phygrec-data/
  MANIFEST.json
  benchmark/
    main/        # train.h5, val.h5, test.h5 and genes.txt
    loco/
      fold_a9/   # train.h5, val.h5, test.h5 and genes.txt
      fold_l7/   # train.h5, val.h5, test.h5 and genes.txt
      fold_na/   # train.h5, val.h5, test.h5 and genes.txt
  results/ablation_expression.json
  structure/
    main.npz
    a9.npz
    l7.npz
    na.npz
    manifest.json
```

Verify the files listed in the manifest, gene order and split isolation before
running experiments:

```bash
python scripts/verify_release.py
```

The HDF5 files contain mixing scenes and Reference expression targets. Model
inputs are restricted to mixed expression, receiver/candidate indices, distances
and masks. The release uses prepared benchmark files; data construction is a
separate workflow.

## Reproduce the experiments

Each command trains the configured runs, selects checkpoints using validation
scores, evaluates on the test split and saves scores under `runs/results/`.
All experiments use seeds `20260816`, `20260817` and `20260818`.

| Experiment | Command | Training runs |
| --- | --- | ---: |
| Main | `python scripts/main.py --experiment main` | 3 |
| Receiver-LOCO (A9, L7, NA) | `python scripts/main.py --experiment loco` | 9 |
| Full model and three removals | `python scripts/main.py --experiment ablation` | 12 |
| All of the above, sharing the full-model runs | `python scripts/main.py --experiment all` | 21 |

Use `--seeds 20260816` for one seed or `--dry-run` to inspect a plan without
training or evaluating:

```bash
python scripts/main.py --experiment all --seeds 20260816 --dry-run
```

The ablation configurations retain all settings of the remaining components:

| Configuration directory | Removed component |
| --- | --- |
| `configs/ablation/rb/` | Physical backprojection increment; circle operator and candidate graph retained |
| `configs/ablation/aim/` | Adaptive gain and momentum |
| `configs/ablation/gcc/` | Graph-context correction |

### Seven comparison baselines

The final methods are GATv2, GraphSAGE, GAT, GCN, MPNN, VAE and Physical PGD.
Their CLI names are `gatv2`, `graphsage`, `gat`, `gcn`, `mpnn`, `vae` and
`physical_pgd`. Fixed configurations in [configs/baselines/](configs/baselines/)
are reused across the main split and LOCO folds.

```bash
python scripts/baselines.py reproduce --method all --experiment main
python scripts/baselines.py reproduce --method all --experiment loco
```

These commands also accept `--seeds`, `--dry-run` and `--evaluate-only`.
Use `--method gatv2` to run one baseline, or `--experiment all` for both main
and LOCO. Baseline scores use the same evaluation functions as PhyGRec.

Neural baselines train for 100 epochs with raw relative L1 plus normalized-log
MAE, AdamW and gradient clipping at 5. Validation runs at epoch 1 and every five
epochs; the earliest checkpoint with the highest recovery score is selected.
VAE adds KL weight `0.02` and evaluates with its latent mean. MPNN uses fixed
distance-kernel weights in residual messages. Physical PGD fits a four-parameter
circle operator for 20 epochs on training data, then freezes it for 100 PGD
iterations with step size `0.1`; main fitting averages three independent scene
losses per update, while LOCO fitting packs six scenes per update.

### Annotation and clustering

Install the optional dependencies, then add `--structure` to either reproduction
entry point:

```bash
python -m pip install -e '.[structure]'
python scripts/main.py --experiment main --structure
python scripts/baselines.py reproduce --method all --experiment loco --structure
```

Within each training seed, raw receiver-scene predictions are averaged by physical
cell identity. Scoring uses frozen nonzero-Reference cohorts: 6,394 cells for main
and 2,027/2,127/2,240 for A9/L7/NA. Zero-Reference receivers remain in expression
evaluation.

Annotation standardization is fitted once on Reference log-normalized counts,
clipped to `[-10, 10]` and applied to five fixed marker programs. Outputs include
Accuracy, Macro Precision, Balanced Accuracy, Macro F1, MCC, Cohen's kappa,
Macro AP from softmax marker scores, and a five-class confusion matrix. Reference
hard-label agreement is one; its Macro AP is omitted.

Clustering fits each expression state independently using standardized, clipped
20-PC expression, a 30-neighbor cosine graph and Leiden resolution `0.35` with
three fixed initializations. Outputs include ARI/NMI against frozen Reference
partitions, pairwise stability ARI, cluster counts and frozen-label silhouette
on at most 4,000 cells. Leiden variation is recorded separately from training-seed
variation; Reference and Mixed scores are point estimates.

Frozen cohorts, gene order and Reference partitions are supplied in
`../phygrec-data/structure/` with the benchmark data. Scoring checks
resource and aggregated Reference-expression hashes and runs in a separate
process with one-thread settings. Reference labels are used only for evaluation.

## Individual runs and saved checkpoints

To train and evaluate one PhyGRec configuration:

```bash
phygrec-cli fit --config configs/main/seed20260816.yaml
python scripts/evaluate.py --split main --seed 20260816
```

Training records its best validation checkpoint automatically. Evaluation reads
that record; use `--checkpoint PATH` to score an explicitly supplied checkpoint.
For an ablation, use its YAML and pass `--variant` to evaluation:

```bash
phygrec-cli fit --config configs/ablation/gcc/seed20260816.yaml
python scripts/evaluate.py --split main --seed 20260816 --variant gcc
```

Individual baseline commands are:

```bash
python scripts/baselines.py train --method gatv2 --split main --seed 20260816
python scripts/baselines.py evaluate --method gatv2 --split main --seed 20260816
```

Both evaluation scripts print JSON and accept `--structure`. `--device cpu` or
`--device cuda` controls PhyGRec evaluation; for baselines it controls both training
and evaluation.

**Public checkpoints will be released after retraining with the finalized code.**
PhyGRec `--evaluate-only` expects validation-selected checkpoints at
`../phygrec-data/checkpoints/{main,a9,l7,na}/seedSEED.ckpt`. Ablation evaluation
reads the best-checkpoint records produced during removal training and also
requires the main checkpoints. Baseline `--evaluate-only` uses existing
`runs/baselines/METHOD/SPLIT/seedSEED/best.pt` files.

## Training and scoring protocol

The full model has four Blocks and 2,832,681 parameters, with 1,000 genes,
width-16 AIM, and two width-128 GATv2 layers with four heads and zero dropout.
Each Block combines physical backprojection, adaptive modulation, momentum and
graph correction before a nonnegative projection.

| PhyGRec setting | Value |
| --- | --- |
| Learning rates: solver / circle / AIM / GCC | `0.003125 / 0.06 / 0.003 / 0.01` |
| Training schedule | 100 epochs, one GPU |
| Training batch / gradient accumulation | 3 scene graphs / 2 batches (6 effective graphs) |
| Validation and test batch | 6 scene graphs |
| Loss | Raw relative L1 + normalized-log MAE, unit weights |
| Optimizer / weight decay | AdamW / `1e-4` |
| Gradient clipping / EMA decay | `1 / 0.99` |
| Validation interval | Every 10 epochs |

Validation selection maximizes the mean of Log-MAE and relative-L1 recovery
relative to Mixed, breaking ties by earliest epoch. Validation and test use EMA
parameters. Test data do not enter selection. Circle-total normalization uses
the detached median of receiver and valid donor-slot totals across the packed
batch. Start repeated training attempts in a fresh output directory: existing
validation points are never overwritten.

The validation callback updates `best.json` beside its `val_epoch_*.json` records
whenever a checkpoint improves the score. This records the checkpoint and score
report paths without copying the weights or running a separate selection step.

Expression metrics are Count-MAE, Log-MAE, relative L1, Count-RMSE, Log-RMSE and
relative L2 over receiver-scene entries. Relative L2 averages each receiver's
error norm divided by its Reference norm, floored at one. Summaries report mean
and sample SD across training seeds; a single-seed run has no sample SD. LOCO
averages the three folds equally within each seed before computing seed statistics.

### Result files

| Output | Location |
| --- | --- |
| PhyGRec per-run scores | `runs/results/{main,a9,l7,na,rb,aim,gcc}/seedSEED.json` |
| PhyGRec summaries | `runs/results/{main,loco,ablation,all}_summary.json` |
| Baseline per-run scores | `runs/results/baselines/METHOD/SPLIT/seedSEED.json` |
| Baseline summaries | `runs/results/baselines/{main,loco,all}_summary.json` |

With `--structure`, these files also contain annotation and clustering results.

## Repository layout

```text
configs/                  # Fixed main, LOCO, ablation and baseline configurations
scripts/                  # Reproduction, evaluation and benchmark verification
src/phygrec/
  models/                 # Recovery solver, circle operator, AIM, GCC and observable inputs
  training/               # Lightning training, checkpoints, validation and selection
  evaluation/             # Expression/structure scoring and seed/fold summaries
  data/                   # Benchmark paths, scene stores and batching
  baselines/              # Seven comparison methods and their training/evaluation CLI
  __init__.py             # Published splits, seeds, ablations and release paths
  utils.py                # Shared file hashing and numerical transforms
tests/                    # Model, protocol and evaluation checks
```

`models/` contains the numerical method without training or evaluation dependencies.
The main method and baselines share `evaluation/`, which scores supplied model
instances. `scripts/evaluate.py` loads PhyGRec checkpoints, and `scripts/main.py`
runs training and test scoring. The callback in `training/validation.py` selects
the best checkpoint during training; `evaluation/scoring.py` summarizes seeds
and folds.
Within `baselines/`, `runner.py` keeps fixed model construction, training,
validation selection and test execution together. Shared circle geometry lives
in `models/operators.py`. Training losses and checkpoint validation live with the
Lightning module. Within `evaluation/`, `expression.py` owns expression metrics
and validation/test loops, `annotation.py` owns cell-type annotation, and
`clustering.py` owns PCA/Leiden scoring. `scoring.py` combines physical-cell
aggregation, scoring dispatch and fold/seed summaries. Annotation and clustering
modules load only inside the isolated scoring process.

Repository and companion-data paths follow the editable installation layout shown
above. Published ablation scores and original parameter counts are retained in the
separate data release.

## License

A code license has not yet been selected.
