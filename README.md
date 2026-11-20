# ranscreen — certified counterfactual screening of handover control policies

Code and data for *Certified Counterfactual Screening of AI-Driven Handover
Control Policies in Self-Optimizing Radio Access Networks*
(F. A. Rawdhan, S. K. Ibrahim), under review at **Network** (MDPI),
manuscript `network-4683582`.

Every number in the paper is reproduced by the scripts here. Nothing is
hand-copied: tables and figures are regenerated from the files in `results/`.

## Layout

```
ranscreen/        the package: data, policies, estimators, bounds
experiments/      one script per experiment in the paper
tests/            identity tests for the propositions
results/          per-replicate records and summaries, by corpus
paper/            the compiled manuscript
run_all.py        reproduces every result from scratch
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # the propositions, checked numerically
python experiments/claim_audit.py   # every claim in the paper, pass/fail
```

`claim_audit.py` prints two tables: the paper's internal claims and, separately,
every claim made in the **abstract**, each as a pass/fail predicate evaluated
against the stored results. All of them currently pass.

## Reproducing from scratch

```bash
python run_all.py                   # several hours on a CPU runtime
```

The datasets are not redistributed here. Both are public:

- 4G: Raca et al., ACM MMSys 2018 — <http://cs1dev.ucc.ie/misl/4G_Dataset/LTE_Dataset.zip>
- 5G: Raca et al., ACM MMSys 2020 — <https://github.com/uccmisl/5Gdataset>

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
| `exp_bound_compare.py` | certificate comparison with full 4G and 5G columns (revision) |
| `make_figures.py` | all eight figures, 600 dpi plus vector PDF and EPS |

## Notes on the implementation

- `ranscreen/data.py` carries **both** CQI tables — LTE (TS 36.213) and NR
  table 2 (TS 38.214) — with `use_cqi_table()` to switch. The spectral
  efficiency values are normative; the SINR thresholds at which each index
  becomes reportable are an engineering mapping and are not specified in either
  release, so both vectors are given explicitly here rather than cited.
- `ranscreen/estimators.py` computes the charge in closed form,
  `sum_a (pi - M*pi0)_+`, with no division. The abstention branch of `gate()`
  keys on the charge with `TAU_DEFAULT = 6e-3`; abstention is **not** applied in
  the coverage audit, so no abstain outcome enters any reported number.
- Large per-replicate records are stored gzipped. `pandas.read_csv` reads them
  directly without unpacking.

## Reproducibility

Each experiment is seeded per replicate and writes its raw per-replicate
records next to its summary, so any table in the paper can be recomputed
without rerunning simulations.

## Licence

MIT. See `LICENSE`.
