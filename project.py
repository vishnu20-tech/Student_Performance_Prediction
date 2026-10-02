"""
Student Performance Prediction Using Machine Learning
Dataset : UCI Student Performance (Cortez & Silva, 2008) - student-mat.csv
Target  : G3 (final grade, 0-20)  -> regression with Linear Regression
Usage   : python student_performance_prediction.py --data data/student-mat.csv
"""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "G3"
CORE_NUMERIC = ["studytime", "absences", "failures", "G1", "G2"]
EXTRA_NUMERIC = ["age", "Medu", "Fedu", "traveltime", "famrel", "freetime", "goout", "health"]
CATEGORICAL = ["school", "sex", "address", "schoolsup", "famsup", "paid", "higher", "internet"]
EARLY_NUMERIC = ["studytime", "absences", "failures"] + EXTRA_NUMERIC   # no G1/G2
PASS_MARK = 10          # G3 >= 10 treated as "Pass" (a project-level assumption)
OUT = Path("outputs"); FIG = OUT / "figures"


def load_data(path):
    df = pd.read_csv(path, sep=";")          # UCI files are semicolon-separated
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    return df


def run_eda(df):
    FIG.mkdir(parents=True, exist_ok=True)
    print(df[CORE_NUMERIC + [TARGET]].describe().round(2))
    plt.figure(figsize=(6, 4)); sns.histplot(df[TARGET], bins=21, kde=True)
    plt.title("Distribution of final grade (G3)"); plt.tight_layout()
    plt.savefig(FIG / "g3_distribution.png", dpi=200); plt.close()
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[CORE_NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation heatmap"); plt.tight_layout()
    plt.savefig(FIG / "correlation_heatmap.png", dpi=200); plt.close()
    plt.figure(figsize=(6, 4)); sns.boxplot(x="studytime", y=TARGET, data=df)
    plt.title("G3 by weekly study-time category"); plt.tight_layout()
    plt.savefig(FIG / "studytime_vs_g3.png", dpi=200); plt.close()


def build_pipeline(numeric, categorical):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", pre), ("model", LinearRegression())])


def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    res = {"Model": name, "MAE": mean_absolute_error(y_true, y_pred),
           "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}
    print({k: (round(v, 3) if not isinstance(v, str) else v) for k, v in res.items()})
    return res


def main(path):
    df = load_data(path)
    run_eda(df)
    results = []
    for label, num, cat in [("Linear Regression (core features)", CORE_NUMERIC, []),
                            ("Linear Regression (extended features)", CORE_NUMERIC + EXTRA_NUMERIC, CATEGORICAL),
                            ("Linear Regression (without G1/G2)", EARLY_NUMERIC, CATEGORICAL)]:
        X, y = df[num + cat], df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = build_pipeline(num, cat).fit(X_tr, y_tr)
        pred = np.clip(pipe.predict(X_te), 0, 20)
        results.append(evaluate(label, y_te, pred))
        cv = cross_val_score(build_pipeline(num, cat), X, y, cv=5, scoring="r2")
        print(f"  5-fold CV R2: mean={cv.mean():.3f}, std={cv.std():.3f}")
        if "core" in label:
            joblib.dump(pipe, OUT / "model_core.joblib")
            core_y_te, core_pred = y_te, pred
            base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
            results.append(evaluate("Baseline (mean of training G3)", y_te, base.predict(X_te)))
    pd.DataFrame(results).round(3).to_csv(OUT / "metrics.csv", index=False)

    plt.figure(figsize=(5, 5)); plt.scatter(core_y_te, core_pred, alpha=0.7)
    plt.plot([0, 20], [0, 20], "r--"); plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
    plt.title("Predicted vs actual (test set)"); plt.tight_layout()
    plt.savefig(FIG / "pred_vs_actual.png", dpi=200); plt.close()
    plt.figure(figsize=(6, 4)); sns.histplot(core_y_te - core_pred, bins=15, kde=True)
    plt.xlabel("Residual (actual - predicted)"); plt.title("Residual distribution"); plt.tight_layout()
    plt.savefig(FIG / "residuals.png", dpi=200); plt.close()


def predict_one(model_path, studytime, absences, failures, g1, g2):
    model = joblib.load(model_path)
    row = pd.DataFrame([{"studytime": studytime, "absences": absences,
                         "failures": failures, "G1": g1, "G2": g2}])
    g3 = float(np.clip(model.predict(row)[0], 0, 20))
    return round(g3, 1), "Pass" if g3 >= PASS_MARK else "Fail"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/student-mat.csv")
    OUT.mkdir(exist_ok=True); main(ap.parse_args().data)