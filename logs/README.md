# Logs index

Models: **LB** = `ar_minimax.py` (lower bound), **MILO** = `ar_milo.py` (exact AR), **UB** = `ar_minimax_ub.py` (upper bound).
In every kept result, LB <= MILO <= UB holds (tolerance 1e-4). No run has a lower-bound violation.

## Layout

Every script writes to its own folder, one timestamped `run_<ts>/` per execution. Summaries and CSVs sit at the top of a run; the raw per-instance solver logs are in `instances/`. Everything in `logs/` except this README is git-ignored.

```
logs/
  benchmark/n<tasks>_m<machines>/run_<ts>/     benchmark.py      (LB vs MILO, Gamma sweep)
      summary.log  details.log  results.csv  config.json
      instances/ref_<k>/{minimax,milo,milo_gurobi}.log
      instances/round_<r>/gamma_<G>_{minimax,milo,milo_gurobi}.log
  benchmark/aggregate_summary.md               aggregate_results.py (all benchmark runs)
  bounds/gamma_<G>/run_<ts>/                   compare_bounds.py (LB / MILO / UB)
      summary_all.log  results_all.csv
      n<n>/{summary.log, results.csv, ub_results.json, instances/round_<r>_ub.log}
  single/{minimax,milo,minimax_ub}_<ts>.log    python ar_*.py --reference/--seed
  pruning_experiment/                          one-off Gamma = 2.5 experiment (own scripts)
  _archive/                                    old or broken runs
```

`summary.log` holds only the final tables; `details.log` has the per-instance comparison (first decisions, event order, gap, runtime) in run order.

## Current results

| Folder | What it is | Main files |
|---|---|---|
| `bounds/gamma_2p5/run_20261030_162143/` | **3-way comparison LB / MILO / UB** (n = 4, 5; Γ = 2.5; 50 instances each), made by `compare_bounds.py`. It reuses the LB and MILO results from `pruning_experiment/` and adds the UB. | `summary_all.log` (n = 4 and 5 side by side), `results_all.csv`, `n4/`, `n5/` (`summary.log`, `results.csv`, `ub_results.json`, `instances/` with one UB log per round) |
| `pruning_experiment/` | **LB vs MILO at Γ = 2.5** (n = 4, 5, 6; 50 instances; seed 2026 + n), the source data of `docs/minimax_pruning_experiment_gamma2p5.md`. **Two MILO n = 6 solves (rounds 2, 5) are still running and write here: do not move this folder.** | `res_n{4,5,6}.json` (LB, plus MILO for n = 4), `milo_n5.json` + `milo_n5_rerun.json` (MILO n = 5), `milo_n6_*` (MILO n = 6), `run_exp.py`, `run_milo.py`, `agg.py` |
| `benchmark/n4_m2/run_20261024_164929/` | **LB vs MILO Γ sweep** (n = 4; Γ = 0, 0.5, ..., 4; 50 instances; 450 comparisons plus 2 reference instances), made by `benchmark.py`. All solved: 437 equal, 13 with LB below MILO. Made before the layout change and moved into it afterwards, so it has no `config.json`. | `summary.log`, `details.log`, `results.csv`, `instances/` |

Summarize all benchmark runs with `python aggregate_results.py` (writes `logs/benchmark/aggregate_summary.md`), or a single run with `--logs-dir logs/benchmark/n4_m2/run_20261024_164929`.

## `_archive/` (not used in any result)

| Folder | Why archived |
|---|---|
| `gamma_sweep_n4_duplicate_run_20261024_152553/` | Exact duplicate of `gamma_sweep_n4/run_20261024_164929` (same 452 rows, no value differences). |
| `batch_n5_milo_license_errors_and_aborted/` | n = 5 Γ sweeps from before the full Gurobi license: in `run_20261024_152912` MILO failed with "Model too large for size-limited license" on 345 of 450 instances; `run_20261024_161652` and `run_20261024_170738` were aborted (no `results.csv`). |
| `batch_n6_aborted_license_errors/` | n = 6 Γ sweep aborted at round 8, MILO license errors (816 MB). |
| `run_20261020_195131_milo_license_errors/` | 3-round n = 5 test, MILO license errors on all rounds. |
| `run_20261020_195746_old_code_smoke_test/`, `run_20261020_212255_old_code_smoke_test/` | 10-round n = 5 smoke tests from 2026-10-20, run on minimax code from before the 2026-10-24 changes. |
| `minimax_ub_reference_test_runs/` | `python ar_minimax_ub.py --reference 0/1` checks made while writing the UB solver. |
