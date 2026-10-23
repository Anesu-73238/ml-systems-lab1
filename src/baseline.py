import csv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)


def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    with open("results/baseline_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy"])
        for name, m in get_models().items():
            m.fit(X_train, y_train)
            w.writerow([name, f"{m.score(X_test, y_test):.4f}"])