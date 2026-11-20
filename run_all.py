#!/usr/bin/env python3
"""Reproduce every number, table and figure in the paper.

    python run_all.py --dataset 4g          # the primary corpus
    python run_all.py --dataset 5g          # external validation
    python run_all.py --all                 # both, then the figures
    python run_all.py --quick               # a reduced-replicate pass for a smoke test

Each experiment writes its raw per-replicate records and a summary to results/<dataset>/, so any
table in the paper can be recomputed without rerunning the simulations.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, 'experiments')
STAGES = [
    ('audit_main.py',            'main sweep: validity and power, 500 replicates per cell'),
    ('exp_frontier.py',          'certifiability frontier and its validation'),
    ('exp_estimator_family.py',  'the charge across the published estimator family'),
    ('exp_sequential.py',        'sequential control with windowed importance sampling'),
    ('exp_transfer.py',          'cross-regime transfer and the diagnostic comparison'),
    ('exp_anytime.py',           'anytime-valid screening under continuous monitoring'),
]


def run(script: str, dataset: str, quick: bool) -> float:
    t0 = time.time()
    env = dict(os.environ, RANSCREEN_QUICK='1' if quick else '0')
    print(f'\n>>> {script} [{dataset}]', flush=True)
    p = subprocess.run([sys.executable, os.path.join(EXP, script), dataset], env=env)
    if p.returncode != 0:
        print(f'!!! {script} failed with code {p.returncode}')
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='4g', choices=['4g', '5g'])
    ap.add_argument('--all', action='store_true', help='run both datasets, then build the figures')
    ap.add_argument('--quick', action='store_true', help='reduced replicates, for a smoke test')
    ap.add_argument('--only', nargs='*', help='run only these scripts')
    a = ap.parse_args()

    datasets = ['4g', '5g'] if a.all else [a.dataset]
    stages = [s for s in STAGES if not a.only or s[0] in a.only]
    total = 0.0
    for ds in datasets:
        for script, what in stages:
            print(f'\n{"="*78}\n{what}\n{"="*78}')
            total += run(script, ds, a.quick)
    subprocess.run([sys.executable, os.path.join(EXP, 'claim_audit.py')])
    if a.all:
        subprocess.run([sys.executable, os.path.join(EXP, 'make_figures.py')])
    print(f'\ntotal wall time {total/60:.1f} min; results under {os.path.join(HERE, "results")}')


if __name__ == '__main__':
    main()
