# Lab 1: Environment and first system measurements

## 1. Environment
Python 3.11.9 in an isolated venv (`.venv`). Dependencies are pinned in
`requirements.txt`; versions printed by `src/print_versions.py` are in
`results/versions.txt`.

## 2. Baseline accuracy
Breast Cancer Wisconsin, 70/30 stratified split, random_state=42.

| Model | Accuracy |
|---|---|
| LogisticRegression | 0.9415 |
| RandomForestClassifier | 0.9357 |

Note: LogisticRegression raised a ConvergenceWarning at max_iter=1000 on
unscaled data; the lab fixes this setting, so I left it unchanged.

## 3. System measurements
Training time: median of 5 runs after 1 warm-up. Inference: median of 100
single-sample predictions after 1 warm-up. Size: `joblib.dump`
(saved as `results/model_logreg.joblib` and `results/model_random_forest.joblib`
so neither overwrites the other). Peak memory: `memory_profiler`
(whole-process RSS, so it includes Python and scikit-learn overhead).

| Model | Train (s) | Inference (ms) | Size (bytes) | Size (KB) | Peak mem train (MiB) | Peak mem infer (MiB) |
|---|---|---|---|---|---|---|
| logreg | 0.5377 | 0.2951 | 1103 | 1.08 | 123.7 | 123.7 |
| random_forest | 0.3021 | 11.843 | 290889 | 284.07 | 125.0 | 125.0 |

LogisticRegression trains slower than the random forest because lbfgs hits
the 1000-iteration limit on unscaled features.

## 4. Deployment fit
| Target | Budget | logreg | random_forest |
|---|---|---|---|
| Cloud | mem ≥1 GB, ≤100 ms, ≤500 MB | Pass (124 MiB, 0.30 ms, 1.08 KB) | Pass (125 MiB, 11.8 ms, 284 KB) |
| Edge | 256–1024 MB, ≤50 ms, ≤50 MB | Pass | Pass |
| Mobile | 64–256 MB, ≤20 ms, ≤10 MB | Pass | Borderline (11.8 ms is under 20 ms, but with little margin on slower phone CPUs) |
| TinyML | ≤256 KB, ≤10 ms, ≤100 KB | Fail on memory (~124 MiB); latency and size pass | Fail on all three (125 MiB, 11.8 ms, 284 KB) |