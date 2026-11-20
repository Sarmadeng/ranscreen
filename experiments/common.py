"""Shared experiment helpers: sweep grids, replicate loops and summary with Wilson intervals."""
from __future__ import annotations
import sys, os, time
import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from ranscreen import build_environment, Config, probs, sample_log, CANDIDATES, estimators as E
from ranscreen.paths import RESULTS

RESULTS = str(RESULTS) + ''
TEMPS = (0.2, 0.4, 0.8, 1.2, 2.0)
SIZES = (1_000, 5_000, 20_000)
N_REP = int(os.environ.get('RANSCREEN_NREP', '500'))


def out_dir(dataset: str) -> str:
    d = os.path.join(RESULTS, dataset)
    os.makedirs(d, exist_ok=True)
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return float(c - h), float(c + h)


def coverage_table(df: pd.DataFrame, by, lb_col='lb', truth='d_true') -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        k = int((g[lb_col] <= g[truth]).sum())
        lo, hi = wilson(k, len(g))
        rec = dict(zip(by, keys))
        rec.update(reps=len(g), d_true=g[truth].mean(), coverage=k/len(g),
                   cov_lo=lo, cov_hi=hi, acceptance=float((g[lb_col] > 0).mean()))
        rows.append(rec)
    return pd.DataFrame(rows)


def env_for(dataset: str):
    return build_environment(Config(dataset=dataset), verbose=True)


def timed(label: str):
    t0 = time.time()
    class _T:
        def __enter__(self): return self
        def __exit__(self, *a): print(f'[{label}] {time.time()-t0:.0f}s', flush=True)
    return _T()
