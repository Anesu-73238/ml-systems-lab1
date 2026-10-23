import csv
import os
import time
import statistics
import warnings

import joblib
from memory_profiler import memory_usage

from baseline import get_data, get_models

warnings.filterwarnings("ignore")


def peak_mib(func):
    """Peak process memory (MiB) while func runs."""
    p = memory_usage((func, (), {}), interval=0.01, max_usage=True)
    return float(p[0] if isinstance(p, (list, tuple)) else p)


def main():
    X_train, X_test, y_train, y_test = get_data()
    sample = X_test[:1]  # single sample for inference

    rows = []
    for name, model in get_models().items():
        # Training time: 1 warm-up, then median of 5 runs
        model.fit(X_train, y_train)
        times = []
        for _ in range(5):
            t = time.perf_counter()
            model.fit(X_train, y_train)
            times.append(time.perf_counter() - t)
        train_s = statistics.median(times)

        # Single-sample inference: 1 warm-up, then median of 100 runs
        model.predict(sample)
        lat = []
        for _ in range(100):
            t = time.perf_counter()
            model.predict(sample)
            lat.append((time.perf_counter() - t) * 1000)
        infer_ms = statistics.median(lat)

        # Model size
        path = f"results/model_{name}.joblib"
        joblib.dump(model, path)
        size_b = os.path.getsize(path)

        # Peak memory during training and inference
        train_mem = peak_mib(lambda: model.fit(X_train, y_train))
        infer_mem = peak_mib(lambda: [model.predict(sample) for _ in range(100)])

        rows.append([name, round(train_s, 4), round(infer_ms, 4), size_b,
                     round(size_b / 1024, 2), round(train_mem, 1), round(infer_mem, 1)])

    with open("results/measurements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "train_median_s", "infer_median_ms", "size_bytes",
                    "size_KB", "peak_mem_train_MiB", "peak_mem_infer_MiB"])
        w.writerows(rows)

    print(open("results/measurements.csv").read())


if __name__ == "__main__":
    main()