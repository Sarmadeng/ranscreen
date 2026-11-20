# ranscreen — certified counterfactual screening of handover control policies

Code and records for *Certified Counterfactual Screening of AI-Driven Handover
Control Policies in Self-Optimizing Radio Access Networks*
(F. A. Rawdhan, S. K. Ibrahim), under review at **Network** (MDPI),
manuscript `network-4683582`.

Every number in the paper is reproduced from the records stored here. Nothing is
hand-copied: the tables and figures are regenerated from these files.

## Layout

```
ranscreen/        the package: data, policies, estimators, bounds, path resolution
experiments/      one script per experiment in the paper
tests/            numerical identity tests for the propositions
results/4g/       per-replicate records and summaries, 4G corpus
results/5g/       per-replicate records and summaries, 5G corpus
results/revision/ records produced for the revision
results/*.log     run logs from the sweeps
run_all.py        reproduces everything from scratch
```

Large per-replicate sweeps are stored gzipped. The helper
`ranscreen.paths.read_records` reads either form, so nothing needs unpacking.
The results root is located automatically relative to the package; set
`RANSCREEN_RESULTS` to point somewhere else.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q              # the propositions, checked numerically
python experiments/claim_audit.py       # every claim in the paper, pass/fail
```

Both run from any working directory and need no datasets: they read the stored
records. `claim_audit.py` prints two tables — the paper's internal claims and,
separately, every claim made in the **abstract** — each as a pass/fail predicate
evaluated against those records. All 20 and all 6 currently pass.

## Reproducing from scratch

The datasets are not redistributed here. Both are public:

- 4G: Raca et al., ACM MMSys 2018 — <http://cs1dev.ucc.ie/misl/4G_Dataset/LTE_Dataset.zip>
- 5G: Raca et al., ACM MMSys 2020 — <https://github.com/uccmisl/5Gdataset>

Unpack them and point the package at them, then run everything:

```bash
export RANSCREEN_4G_ROOT=/path/to/4G_Dataset
export RANSCREEN_5G_ROOT=/path/to/5G-production-dataset
python run_all.py                       # several hours on a CPU runtime
```

Without those variables the loaders look under `data/4G_Dataset` and
`data/5G-production-dataset` relative to the working directory, and say so
explicitly if the files are absent.

## What each experiment answers

| Script | Produces |
|---|---|
| `audit_main.py` | the 60-point coverage sweep, both corpora |
| `exp_frontier.py` | the certifiability frontier |
| `exp_estimator_family.py` | the generalised charge across published estimators |
| `exp_sequential.py` | windowed per-decision importance sampling |
| `exp_transfer.py` | deployment-regime shift |
| `exp_anytime.py` | monitoring and the betting confidence sequence |
| `ess_raw_check.py` | Kish ESS on **raw** weights (revision) |
| `exp_revision.py` | operational failure, out-of-sample thresholds, retention bias (revision) |
| `exp_postselection.py` | Proposition 7 evaluated (revision) |
| `exp_cqi_sensitivity.py` | the 5G audit repeated on the NR CQI table (revision) |
| `exp_bound_compare.py` | certificate comparison, full 4G and 5G columns (revision) |
| `make_figures.py` | all eight figures, 600 dpi plus vector PDF and EPS |

## Notes on the implementation

- `ranscreen/data.py` carries **both** CQI tables — LTE (TS 36.213) and NR
  table 2 (TS 38.214) — with `use_cqi_table()` to switch between them. The
  spectral efficiency values are normative; the SINR thresholds at which each
  index becomes reportable are not specified in either release and are an
  engineering mapping, so both vectors are given explicitly here rather than
  cited.
- `ranscreen/estimators.py` computes the charge in closed form,
  `sum_a (pi - M*pi0)_+`, with no division. The abstention branch of `gate()`
  keys on the charge with `TAU_DEFAULT = 6e-3`. Abstention is **not** applied in
  the coverage audit, so no abstain outcome enters any number reported in the
  paper.
- Each experiment is seeded per replicate and writes its raw per-replicate
  records beside its summary, so any table can be recomputed without rerunning
  the simulations.

## Licence

MIT. See `LICENSE`.
