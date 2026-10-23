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
(Paste the budget table here with Pass/Fail per criterion.)

**logreg:** fits Cloud, Edge and Mobile. Its size (1.08 KB) and latency
(0.30 ms) would satisfy TinyML, but the measured memory (~124 MiB, dominated
by the Python runtime) is far above 256 KB, so it would need to be exported
to a non-Python runtime (for example C) to run on TinyML.

**random_forest:** fits Cloud and Edge. On Mobile it is borderline: 11.8 ms
is under 20 ms on my laptop, but phone CPUs are slower. It does not fit
TinyML (284 KB > 100 KB, 11.8 ms > 10 ms, memory far above 256 KB).

Caveats: measurements are from a laptop, memory includes the Python runtime,
and I treated each memory range as the memory available on that device class.