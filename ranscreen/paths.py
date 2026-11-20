"""Locate the results tree without assuming where the repository was cloned.

The experiment scripts previously hard-coded the authors' own absolute path, which meant a
reader who cloned the repository could not run them. This resolves the root in three steps:
an explicit RANSCREEN_RESULTS environment variable, then the two layouts the repository has
been published with, then the authors' path as a last resort.
"""
from __future__ import annotations
import os, pathlib

_HERE = pathlib.Path(__file__).resolve().parent.parent


def results_root() -> pathlib.Path:
    env = os.environ.get('RANSCREEN_RESULTS')
    if env:
        return pathlib.Path(env)
    for cand in (_HERE/'results', _HERE):
        if (cand/'4g').is_dir() or (cand/'revision').is_dir():
            return cand
    return _HERE/'results'


RESULTS = results_root()


def read_records(name: str):
    """Read a result file whether it is stored plain or gzipped.

    The large per-replicate sweeps are published gzipped to keep the repository small;
    pandas reads either form, but only if it is given the name that exists.
    """
    import pandas as pd
    base = RESULTS/name
    for cand in (base, base.with_suffix(base.suffix + '.gz')):
        if cand.exists():
            return pd.read_csv(cand)
    raise FileNotFoundError(f'{base} (nor its .gz form) is present under {RESULTS}')
