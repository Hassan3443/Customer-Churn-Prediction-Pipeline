import json, subprocess, sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model_report as mr

@pytest.fixture(scope="module")
def built():
    return mr.build()

def test_meta_matches_csv(built):
    d, _, _ = built
    assert d["meta"]["rows"] == 5000 and d["meta"]["churners"] == 483

def test_split_disjoint_sizes(built):
    d, _, _ = built
    assert d["meta"]["eval_rows"] + d["meta"]["calib_rows"] == 5000

def test_auc_in_range(built):
    m = built[0]["metrics"]
    assert 0.5 < m["auc_saved"] < 1 and 0.5 < m["auc_baseline_lr"] < 1

def test_calibration_improves_brier(built):
    m = built[0]["metrics"]
    assert m["brier_calibrated"] < m["brier_raw"]

def test_calibrated_mean_close_to_base_rate(built):
    d = built[0]
    assert abs(d["metrics"]["mean_calibrated"] - d["meta"]["base_rate"] / 100) < 0.02

def test_top_k_monotone(built):
    rows = built[0]["top_k"]
    recalls = [r[4] for r in rows]
    assert recalls == sorted(recalls) and rows[-1][4] > 0.3

def test_top_k_beats_random(built):
    d = built[0]
    assert d["top_k"][9][3] > d["meta"]["base_rate"] / 100

def test_importance_sums_below_100(built):
    assert sum(v for _, v in built[0]["importance"]) <= 100.01

def test_exported_trees_match_lightgbm(built):
    _, mj, parity = built
    import joblib, warnings
    warnings.filterwarnings("ignore")
    model = joblib.load(mr.MODEL)
    for row in parity[:40]:
        s = sum(_walk(t, row["x"]) for t in mj["trees"])
        assert abs(1 / (1 + np.exp(-s)) - row["raw"]) < 1e-9

def _walk(t, x):
    i = 0
    while t["f"][i] >= 0:
        i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
    return t["v"][i]

def test_node_parity():
    subprocess.run(["node", str(Path(__file__).with_name("parity.test.js"))], check=True)
