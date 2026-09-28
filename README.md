# YearPredictionMSD: Replicating the APLR Paper

This project replicates the YearPredictionMSD experiment (Table 15) from:

> von Ottenbreit, M. and De Bin, R. (2024). *Automatic piecewise linear regression*.
> Computational Statistics, 39, 1867-1907. https://doi.org/10.1007/s00180-024-01475-4

The goal has two stages: first reproduce the paper's reported results, then look for
gaps in the paper and try to improve on them.

## Task

Predict the release year of a song (a continuous target) from 90 audio timbre features:
12 timbre averages and 78 timbre covariances. Performance is measured with test MSE and R-squared.

## Data

The dataset is YearPredictionMSD from the UCI Machine Learning Repository, a subset of the
Million Song Dataset with 515,345 songs. The data is not included in this repository because
of its size (about 450 MB).

1. Download it from https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
2. The UCI file is a headerless text file whose first column is `Year`. Convert it to a CSV with a
   header row named `Year`, `TimbreAvg1` to `TimbreAvg12`, and `TimbreCovariance1` to
   `TimbreCovariance78`:

```python
import pandas as pd
cols = ["Year"] + [f"TimbreAvg{i}" for i in range(1, 13)] + [f"TimbreCovariance{i}" for i in range(1, 79)]
df = pd.read_csv("YearPredictionMSD.txt", header=None, names=cols)
df.to_csv("YearPredictionMSD.csv", index=False)
```

3. Put `YearPredictionMSD.csv` in the project root (or set the `MSD_CSV` environment variable
   to its path).

### Train/test split

The notebook uses the official split on the original row order: the first 463,715 rows for
training and the last 51,630 for testing (about 90/10). The data is not shuffled. Songs by
the same artist are stored next to each other, so a random split would leak artist
information into the test set and inflate scores.

## Models

| Model | Implementation | Notes |
|---|---|---|
| APLR | `aplr` (`APLRTuner`, `APLRRegressor`) | The paper's grid, then a 5-fold CV refit |
| LightGBM | `lightgbm` | `num_leaves` tuned on a positional validation split |
| Random Forest | `scikit-learn` | 200 trees, all cores |
| glmboost, gamboost | R `mboost` | Run through `Rscript` |
| MARS | R `earth` | `degree=4`; run through `Rscript` |
| Linear regression | `scikit-learn` | Python stand-in for glmboost |

MARS (`py-earth`) and gamboost have no working Python install, which is why they run in R.

## Paper results (Table 15)

| Model | MSE | R-squared |
|---|---|---|
| APLR | 79.93 | 0.3213 |
| LightGBM | 80.06 | 0.3208 |
| Random Forest | 83.60 | 0.2947 |
| MARS | 86.19 | 0.2686 |
| gamboost | 87.01 | 0.2615 |
| glmboost | 90.72 | 0.2297 |

## Replication status

Results from an earlier run on a single CPU core, with reduced settings for Random Forest
(40 shallow trees, 25% subsampling) and APLR (single validation split, 15-minute time limit,
no grid search). These are not the final numbers.

| Model | Our MSE | Our R-squared | Paper MSE |
|---|---|---|---|
| LightGBM | 80.63 | 0.3153 | 80.06 |
| APLR (reduced) | 85.90 | 0.2706 | 79.93 |
| Random Forest (reduced) | 87.81 | 0.2544 | 83.60 |
| Linear regression (glmboost analogue) | 90.44 | 0.2320 | 90.72 (glmboost) |

LightGBM and the linear baseline already match the paper closely, which confirms the data
loading, the split and the evaluation. The gaps for APLR and Random Forest are expected to
shrink with the full-strength notebook. MARS and gamboost have not been run yet.

## Setup

Requires Python 3.11 or 3.12.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install pandas numpy scikit-learn lightgbm aplr matplotlib ipykernel
```

For MARS, gamboost and glmboost, also install R from https://cran.r-project.org and check that
`Rscript --version` works. The first run installs the R packages `earth` and `mboost`.

## Running

1. Open `YearPredictionMSD_Full_Replication.ipynb` in VS Code or Jupyter and select the
   `.venv` kernel.
2. Run the cells from top to bottom, one heavy model at a time (LightGBM, Random Forest, then APLR).
3. Each finished model is saved to `results/results.json`, with test predictions in
   `results/pred_*.npy`. A saved model is skipped when you rerun, so an interrupted run can resume.

To check that everything works before the long run, set `QUICK_TEST=1` in the environment
first. It runs the whole notebook on a small subsample in a few minutes. The numbers from a
quick test are meaningless; delete the `results_quick/` folder afterwards.

Expected cost on an 8-core machine: LightGBM and the linear baseline take minutes, the
Random Forest takes tens of minutes, and APLR with the full grid can take hours. Random Forest
memory use is the main risk on a 16 GB machine. If it runs out of memory, lower `RF_TREES`.

## Project structure

```
.
├── YearPredictionMSD_Full_Replication.ipynb   # main notebook
├── results/                                   # saved metrics and test predictions
├── .gitignore
└── README.md
```

The dataset CSV and the virtual environment are not tracked (see `.gitignore`).

## Known limitations

- The paper's exact hyperparameters for Random Forest are not stated in the text I could access, so the Random Forest settings are reasonable choices, not a verified copy.
- The R settings for MARS and the boosting models follow the paper's best-found values where known and are otherwise my choices. Check them if the R results differ from the paper.
- Results are single runs. Small MSE differences (a few tenths of a point) are within run-to-run noise.

## Next steps

- Run the full-strength notebook and compare every model with Table 15.
- Once the results match, analyse gaps using the saved predictions: per-decade errors, residual patterns, ensembling and repeated runs with confidence intervals.

## References

- Paper: https://doi.org/10.1007/s00180-024-01475-4
- APLR package: https://github.com/ottenbreit-data-science/aplr
- Dataset: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
