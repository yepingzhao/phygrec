# PhyGRec benchmark data

This directory holds the prepared benchmark for PhyGRec: the main split, three
receiver-LOCO folds, and resources for cell-type annotation and clustering.
Extract the separately distributed data here and run commands from `phygrec/`.
The repository tracks this README; data files are excluded from Git.

## Directory layout

```text
data/
  readme.md
  MANIFEST.json
  benchmark/
    main/
      train.h5
      val.h5
      test.h5
      genes.txt
    loco/
      fold_a9/   # The same four files
      fold_l7/   # The same four files
      fold_na/   # The same four files
  structure/
    main.npz
    a9.npz
    l7.npz
    na.npz
    manifest.json
```

Extract the data contents directly here, so that `data/MANIFEST.json` and
`data/benchmark/` exist. Keep the paths and gene order from the data release.

## Benchmark splits

Each directory contains separate training, validation and test scene stores,
plus an ordered panel of 1,000 genes in `genes.txt`.

| CLI split | Directory | Held-out receiver chip |
| --- | --- | --- |
| `main` | `benchmark/main/` | Fixed main split |
| `a9` | `benchmark/loco/fold_a9/` | `Y40360A9` |
| `l7` | `benchmark/loco/fold_l7/` | `Y40360L7` |
| `na` | `benchmark/loco/fold_na/` | `Y40360NA` |

Receiver-LOCO excludes the held-out receiver chip from training, validation and
highly variable gene selection. The marker list is fixed; remaining genes are
selected from the two training chips. The ablations and seven comparison
baselines use these same prepared splits.

The HDF5 stores contain measured source profiles, simulated mixtures, spatial
candidate graphs, masks and Reference expression targets. Model inputs are mixed
expression, receiver/candidate indices, distances and masks. Reference expression
supports training losses and evaluation. Test scoring follows training and
validation checkpoint selection.

## Annotation and clustering resources

`structure/` supplies frozen physical-cell identities, gene order and Reference
clustering partitions. Its NPZ files contain `ids`, `genes` and `clusters`.
`structure/manifest.json` records cohort sizes, partition hashes and aggregated
Reference-expression hashes.

| Split | Physical cells | Genes |
| --- | ---: | ---: |
| Main | 6,394 | 1,000 |
| A9 | 2,027 | 1,000 |
| L7 | 2,127 | 1,000 |
| NA | 2,240 | 1,000 |

These resources are used after prediction when `--structure` is enabled.
Receiver-scene expression is averaged by physical cell identity before annotation
and clustering. Cohorts include cells with nonzero Reference expression;
zero-Reference receivers still contribute to expression evaluation.

```bash
python -m pip install -e '.[structure]'
python scripts/main.py --experiment main --seeds 20260816 --structure
```

## Verify and use the data

After installing the package and placing the complete data release here:

```bash
python scripts/verify_release.py
python scripts/main.py --experiment main --seeds 20260816 --dry-run
python scripts/main.py --experiment main --seeds 20260816
```

Provide every file listed in `data/MANIFEST.json`. Manifest paths are relative
to `data/`. Verification checks SHA-256 hashes,
HDF5 schema, gene order, split identity isolation and LOCO chip exclusion.

## Experiment outputs

Training checkpoints, logs and validation records are saved in the corresponding
experiment directories under `runs/`. Evaluation scores and summaries are saved
to `runs/results/`.

See the [repository README](../README.md#saved-runs-and-checkpoints) for checkpoint
loading and evaluation commands.
