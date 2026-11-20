# Revision package — network-4683582

## What to upload to MDPI

| File | Purpose |
|---|---|
| `main_clean.pdf` / `main_clean.tex` | the revised manuscript, clean copy |
| `main_mdpi.pdf` / `main_mdpi.tex` | the same manuscript with changes coloured by reviewer |
| `refs.bib` | 51 references, every one with a DOI or a URL and access date |
| `figures/` | all eight figures regenerated, panel labels now below the panels |
| `Response_Reviewer_1.txt` … `_3.txt` | point-by-point responses, one per reviewer |

## Colour key in the marked copy

| Colour | Meaning |
|---|---|
| **Blue** | changed for Reviewer 1 |
| **Orange** | changed for Reviewer 2 |
| **Green** | changed for Reviewer 3 |
| **Purple** | the same point raised by more than one reviewer |

A legend box appears on the first page of the marked copy. Black text is
unchanged from the submitted version.

## One source, two outputs

`main_mdpi.tex` carries `\newif\ifMARKUP\MARKUPtrue`. Running

```
python3 build_both.py
```

writes `main_clean.tex` with the flag flipped and compiles both files, then
reports errors, overfull boxes and unresolved references for each. Both
currently compile with zero of all three.

Edit **only** `main_mdpi.tex`; `main_clean.tex` is generated.

## Scripts that produced the new results

| Script | Answers |
|---|---|
| `experiments/ess_raw_check.py` | R1-10: Kish ESS on raw weights |
| `experiments/exp_revision.py` | R1-7 operational failure, R1-11 out-of-sample thresholds, R1-21 retention by mobility |
| `experiments/exp_postselection.py` | R1-14: Proposition 7 evaluated |
| `experiments/exp_cqi_sensitivity.py` | R1-13: 5G audit on the NR CQI table |
| `experiments/exp_bound_compare.py` | R1-24: full 4G and 5G columns for Table 7 |

Raw per-replicate records for all of them are under `results/revision/`.

## Code changes made for the revision

- `ranscreen/data.py` — both CQI tables (TS 36.213 and TS 38.214 table 2) with
  `use_cqi_table()` to switch; the SINR threshold vectors are documented as an
  engineering mapping rather than a specification value.
- `ranscreen/estimators.py` — `gate()` now keys abstention on the charge with
  `TAU_DEFAULT = 6e-3`, matching Algorithm 1; it previously used the ESS and the
  maximum weight, which did not match the paper. `certifiable()` docstring now
  states that the frontier is sufficient and not necessary.
- `experiments/make_figures.py` — `panel_label()` places (a)/(b)/(c) below the
  axes.

## Still to do before resubmission

1. **Mint the archival DOI.** The Data Availability Statement now says a fixed
   release is archived with a permanent identifier. Create a GitHub release and
   let Zenodo mint the DOI, then put it in `refs.bib` and in the statement.
   Until that is done the sentence promises something that does not exist.
2. **Check the abstract length.** It is 212 words against MDPI's guideline of
   about 200. Trim if the editorial office objects.
3. **Confirm the O-RAN release.** Reference 2 now cites O-RAN.WG1.TS.OAD-R004
   v14.00 (ETSI TS 103 982 V14.0.0). Confirm that is the version you consulted.
