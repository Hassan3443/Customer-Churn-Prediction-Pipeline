"""Evaluate the saved LightGBM model on the 5,000-row cleaned sample and export a browser explorer.

What this does, and what it can not do:
  * The saved model (models/lgbm_model.pkl) was trained on the full 999,999-row dataset,
    which is not in this repo. Only a 5,000-row sample is. So the numbers here are
    sample numbers, not the headline test-set numbers in the README.
  * The sample is split 50/50 with a fixed seed. One half fits a probability
    calibrator, the other half is used only for evaluation.
  * The saved model's raw scores are not probabilities (they average far above the
    real churn rate), so a Platt-scaling calibrator is fitted on the first half.

    python src/model_report.py        # writes docs/data.json, docs/model.json, tests/fixtures/parity.json
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/processed/clean_data_sample.csv"
MODEL = ROOT / "models/lgbm_model.pkl"
SCALER = ROOT / "models/scaler/scaler.pk1"
CONTRACT = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load():
    df = pd.read_csv(SAMPLE)
    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    return df, model, scaler


def real_units(df, scaler):
    """Undo the standardisation on the 14 scaled columns so segments are readable."""
    out = df.copy()
    cols = list(scaler.feature_names_in_)
    out[cols] = df[cols].values * scaler.scale_ + scaler.mean_
    return out


def fit_calibrator(raw_scores, y):
    """Platt scaling: p = sigmoid(a * logit(score) + b). Returns (a, b)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(raw_scores).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_calibrator(raw_scores, a, b):
    return 1 / (1 + np.exp(-(a * logit(raw_scores) + b)))


def reliability(y, p, bins=8):
    """Equal-count bins of predicted probability vs observed churn rate."""
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append([round(float(p[chunk].mean()), 4), round(float(y[chunk].mean()), 4), int(len(chunk))])
    return rows


def top_k_table(y, score, ks=range(1, 51)):
    """If we contact the top k% by score: how many churners do we reach, how precise is it?"""
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    n, pos = len(y), int(y.sum())
    cum = np.cumsum(ys)
    rows = []
    for k in ks:
        m = max(1, int(round(n * k / 100)))
        tp = int(cum[m - 1])
        rows.append([k, m, tp, round(tp / m, 4), round(tp / pos, 4)])
    return rows


def export_trees(model):
    dump = model.booster_.dump_model()
    trees = []
    for t in dump["tree_info"]:
        f, th, l, r, v = [], [], [], [], []

        def walk(node):
            i = len(f)
            f.append(-1); th.append(0.0); l.append(-1); r.append(-1); v.append(0.0)
            if "leaf_value" in node:
                v[i] = node["leaf_value"]
            else:
                assert node["decision_type"] == "<=", node["decision_type"]
                f[i], th[i] = node["split_feature"], node["threshold"]
                l[i] = walk(node["left_child"])
                r[i] = walk(node["right_child"])
            return i

        walk(t["tree_structure"])
        trees.append({"f": f, "t": th, "l": l, "r": r, "v": v})
    return trees


def build():
    df, model, scaler = load()
    feats = [c for c in df.columns if c != "churn"]
    X, y = df[feats], df["churn"].values
    raw = model.predict_proba(X)[:, 1]
    idx_a, idx_b = train_test_split(np.arange(len(df)), test_size=0.5, stratify=y, random_state=42)
    a, b = fit_calibrator(raw[idx_a], y[idx_a])
    cal_b = apply_calibrator(raw[idx_b], a, b)
    yb, rb = y[idx_b], raw[idx_b]

    # baseline: logistic regression trained on half A, scored on half B
    base = LogisticRegression(max_iter=2000).fit(X.iloc[idx_a], y[idx_a])
    base_b = base.predict_proba(X.iloc[idx_b])[:, 1]

    fpr, tpr, _ = roc_curve(yb, rb)
    keep = np.linspace(0, len(fpr) - 1, 80).astype(int)
    ru = real_units(df, scaler)
    seg = {}
    bins = {
        "contract": (ru["contract"].map(CONTRACT), list(CONTRACT.values())),
        "customer_satisfaction": (pd.cut(ru["customer_satisfaction"], [-99, 3, 5, 7, 99], labels=["0-3", "3-5", "5-7", "7+"]), ["0-3", "3-5", "5-7", "7+"]),
        "num_complaints": (pd.cut(ru["num_complaints"].round(), [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"]), ["0", "1", "2", "3+"]),
        "num_service_calls": (pd.cut(ru["num_service_calls"].round(), [-1, 1, 3, 5, 99], labels=["0-1", "2-3", "4-5", "6+"]), ["0-1", "2-3", "4-5", "6+"]),
        "tenure": (pd.cut(ru["tenure"], [-99, 12, 24, 48, 999], labels=["0-12", "12-24", "24-48", "48+"]), ["0-12", "12-24", "24-48", "48+"]),
    }
    for name, (cats, order) in bins.items():
        g = pd.DataFrame({"c": cats.astype(str), "y": y}).groupby("c")["y"].agg(["mean", "count"])
        seg[name] = [[k, round(float(g.loc[k, "mean"]) * 100, 1), int(g.loc[k, "count"])] for k in order if k in g.index]

    gain = model.booster_.feature_importance("gain")
    imp = sorted(zip(feats, (gain / gain.sum() * 100).round(2).tolist()), key=lambda x: -x[1])

    defaults = {}
    for c in feats:
        s = ru[c]
        defaults[c] = float(s.mode().iloc[0]) if set(s.unique()) <= {0, 1} or c in ("contract", "education") else round(float(s.mean()), 3)
    zcols = list(scaler.feature_names_in_)
    data = {
        "meta": {"rows": int(len(df)), "churners": int(y.sum()), "base_rate": round(float(y.mean()) * 100, 2),
                 "eval_rows": int(len(idx_b)), "eval_churners": int(yb.sum()), "calib_rows": int(len(idx_a))},
        "metrics": {
            "auc_saved": round(float(roc_auc_score(yb, rb)), 3), "auc_baseline_lr": round(float(roc_auc_score(yb, base_b)), 3),
            "pr_auc_saved": round(float(average_precision_score(yb, rb)), 3),
            "brier_raw": round(float(brier_score_loss(yb, rb)), 4), "brier_calibrated": round(float(brier_score_loss(yb, cal_b)), 4),
            "brier_base_rate": round(float(brier_score_loss(yb, np.full(len(yb), y[idx_a].mean()))), 4),
            "mean_raw_score": round(float(rb.mean()), 3), "mean_calibrated": round(float(cal_b.mean()), 4),
            "readme_auc_full_test": 0.635,
        },
        "calibrator": {"a": a, "b": b},
        "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep],
        "reliability_raw": reliability(yb, rb), "reliability_calibrated": reliability(yb, cal_b),
        "top_k": top_k_table(yb, rb), "importance": imp[:15], "segments": seg,
        "feature_labels_note": "Numeric columns were standardised before training; the explorer converts back to real units with the saved scaler.",
    }
    model_json = {
        "features": feats, "zcols": zcols, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "defaults": defaults, "trees": export_trees(model), "calibrator": {"a": a, "b": b},
        "ranges": {c: [round(float(ru[c].quantile(.01)), 2), round(float(ru[c].quantile(.99)), 2)] for c in
                   ("tenure", "monthlycharges", "customer_satisfaction", "num_complaints", "num_service_calls", "late_payments", "num_services", "age")},
    }
    # parity fixtures: z-space rows, expected raw score and calibrated probability
    rng = np.random.RandomState(0)
    pick = rng.choice(len(df), 120, replace=False)
    parity = [{"x": X.iloc[i].tolist(), "raw": float(raw[i]), "cal": float(apply_calibrator(raw[i:i + 1], a, b)[0])} for i in pick]
    return data, model_json, parity


if __name__ == "__main__":
    data, mj, parity = build()
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "tests/fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/data.json").write_text(json.dumps(data, separators=(",", ":")))
    (ROOT / "docs/model.json").write_text(json.dumps(mj, separators=(",", ":")))
    (ROOT / "tests/fixtures/parity.json").write_text(json.dumps(parity))
    print(json.dumps({"meta": data["meta"], "metrics": data["metrics"], "calibrator": data["calibrator"]}, indent=1))
    print("model.json KB:", round((ROOT / "docs/model.json").stat().st_size / 1024))
