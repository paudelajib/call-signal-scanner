"""End-to-end code-path tests on the synthetic fixture (see fixture.py). Run: python -m pytest -q"""
import numpy as np
import pandas as pd

from src import engine
from src.backtest import run_backtest, rowwise_spearman
from src.scoring import load_weights, score_horizon, xs_z
from tests.fixture import make_fixture


def test_live_and_rescore(tmp_path):
    cfg, paths = make_fixture(tmp_path)
    out = engine.compute_live(cfg, paths)
    assert out["n"] == 60
    rk = engine.load_ranking(paths)
    for h in cfg["horizons"]:
        s = rk[f"score_{h}"].dropna()
        assert len(s) >= 55
        assert s.between(-1, 1).all()
    # P/E is sector-relative and positive where shown
    assert (rk["pe"].dropna() > 0).all()
    assert rk["sector_pe"].notna().sum() > 40
    # re-weighting changes the ranking without downloads
    w = load_weights()
    w["medium"] = {"mom_3m": 1.0}
    engine.rescore(cfg, paths, weights=w)
    rk2 = engine.load_ranking(paths)
    assert not np.allclose(rk["score_medium"].fillna(0), rk2["score_medium"].fillna(0))
    assert paths.scores_history.exists()


def test_score_properties():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=10)
    cols = [f"S{i}" for i in range(100)]
    a = pd.DataFrame(rng.normal(size=(10, 100)), idx, cols)
    za = xs_z(a)
    assert np.allclose(za.mean(axis=1), 0, atol=1e-9)
    s = score_horizon({"a": za, "b": -za}, {"a": 1.0, "b": 0.5}, 0.9)
    # b = -a with half the weight -> score must still rank like a
    ic = rowwise_spearman(s, a)
    assert (ic > 0.99).all()
    assert s.abs().max().max() <= 1


def test_backtest_runs(tmp_path):
    cfg, paths = make_fixture(tmp_path)
    res = run_backtest(cfg=cfg)
    assert set(res["horizons"]) == set(cfg["horizons"])
    for h, v in res["horizons"].items():
        assert v["ic"]["n"] > 0
        assert len(v["quantile_excess"]) == cfg["backtest"]["quantiles"]
        # random data -> no real signal; IC should be near zero
        assert abs(v["ic"]["mean"]) < 0.1


def test_no_lookahead(tmp_path):
    """Features on date d must be identical whether or not later data exists."""
    from src.engine import build_inputs
    from src.features import compute_features
    cfg, paths = make_fixture(tmp_path)
    inp = build_inputs(cfg, paths)
    full = compute_features(inp["panel"], inp["bench"], inp["macro"], inp["sectors"], inp["ey"],
                            inp["news"], cfg)
    cut = inp["panel"]["close"].index[-300]
    tr = {k: v.loc[:cut] for k, v in inp["panel"].items()}
    ey = {k: v.loc[:cut] for k, v in inp["ey"].items()}
    part = compute_features(tr, inp["bench"].loc[:cut], inp["macro"].loc[:cut], inp["sectors"], ey,
                            inp["news"].loc[:cut], cfg)
    for k in part:
        if k.startswith("_ema"):
            continue
        a, b = full[k].loc[cut], part[k].loc[cut]
        assert np.allclose(a.fillna(-999), b.fillna(-999), rtol=1e-9, atol=1e-9), k
