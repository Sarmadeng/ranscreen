# Certified counterfactual screening of RAN control policies

Reproduces every number, table and figure in the paper from raw measurement traces.

An autonomous controller proposes a handover policy. Should the operator deploy it? This code
answers that question from logged decisions alone, and — more to the point — measures whether the
confidence bound that licenses the decision actually holds.

## Headline results

| | 4G corpus | 5G corpus |
|---|---|---|
| samples after quality control | 92,126 (132 traces) | 144,333 (83 traces) |
| handover cost, calibrated from real cell changes | 0.043 (2,292 events) | 0.232 same-RAT, 0.432 across RAT |
| worst-case coverage, standard certificate | **0.000** [0.000, 0.008] | **0.000** [0.000, 0.008] |
| worst-case coverage, charged certificate | 0.926 [0.900, 0.946] | 0.936 [0.911, 0.954] |
| power at the same operating points | 0.997 → 0.997 | 1.000 → 1.000 |
| coverage as the log grows (weak exploration) | 0.358 → 0.164 → 0.014 | 0.310 → 0.038 → 0.000 |

Nominal level 0.95; 500 independent logs per operating point; intervals are Wilson.

## Install and run

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 10 identity tests, ~5 s
python run_all.py --all             # both corpora end to end
python run_all.py --dataset 4g --only audit_main.py
```

Point `ranscreen/data.py: DATASETS` at the extracted corpora. Both are public:

* 4G — *4G LTE speed dataset and bandwidth*, 135 traces across five mobility classes.
* 5G — *5G production dataset*, 83 traces, `github.com/uccmisl/5Gdataset`.

The two share a column schema, so one pipeline serves both.

## Layout

```
ranscreen/
  data.py          loading, quality control, the SINR model, the trace-driven environment
  policies.py      the policy family, logging, and the offline agent that searches it
  estimators.py    estimators, charges, confidence bounds, the gate
experiments/
  audit_main.py            validity and power across 60 operating points
  exp_frontier.py          the certifiability frontier, computable before any data
  exp_estimator_family.py  the charge applied to IPS, SNIPS, DR, Switch-DR, shrinkage
  exp_sequential.py        windowed per-decision importance sampling
  exp_transfer.py          deployment-regime transfer, and which diagnostic predicts failure
  exp_anytime.py           betting confidence sequences under continuous monitoring
  claim_audit.py           every headline claim as a pass/fail predicate
tests/                     the identities, checked numerically
```

## What the code implements

**Proposition 1 (bias identity).** With exact propensities and `w = min(w_raw, M)`, the truncated
doubly-robust contribution satisfies `E[DR] = Δ − E[(w_raw − M)₊ (r − q̂)]`.

**Proposition 2 (closed-form charge).** `Σ_a π₀(a|x)(π/π₀ − M)₊ = Σ_a (π − M π₀)₊ =: c(x)`.
The charge needs no division, lies in `[0, 1]`, handles contexts without support by tending to
`π(a|x)` rather than diverging, and is the Rao–Blackwell projection of the sampled correction.

**Theorem 3 (any weight transform).** For any `φ`, the charge is `Σ_a |π − π₀ φ(π/π₀)|`; hard
truncation recovers Proposition 2, so one correction covers the published estimator family.

**Certifiability frontier.** `Δ > 2 r_max · E[c(X)]` is sufficient for eventual acceptance, and
`E[c(X)]` depends only on the two policies — so a candidate's screenability is known in advance.

## Reproducibility notes

* Every experiment is seeded per replicate; rerunning reproduces the recorded CSVs bit for bit.
* `results/<dataset>/` holds raw per-replicate records alongside the summaries, so any table can
  be recomputed without rerunning the simulations.
* Two oracle-construction pitfalls are documented in `ranscreen/data.py`: applying the nonlinear
  CQI map to the mean SINR, and applying the handover cost or the clipping outside the
  expectation. Each produced an apparent certificate failure larger than the effects under study.
* Public traces do not record the propensities of the logged handovers, which were taken by a
  deterministic rule. We follow the standard supervised-to-bandit conversion: the measurements
  supply contexts and calibrate the reward; the logging policy is superimposed with known
  propensities. This is a stated limitation, not a property of the data.
