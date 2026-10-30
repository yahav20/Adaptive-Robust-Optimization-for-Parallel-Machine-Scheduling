# Adaptive Robust Optimization for Parallel Machine Scheduling

This project compares two approaches to adaptive robust scheduling on identical parallel machines. The objective is to minimize the worst-case makespan when task durations are uncertain:

\[
d_i = d_i^0 + \bar d_i u_i, \qquad 0 \le u_i \le 1, \qquad \sum_i u_i \le \Gamma.
\]

## Approaches

- **Hybrid Minimax** ([`ar_minimax.py`](ar_minimax.py)) searches the scheduler/adversary decision tree recursively and evaluates completed paths with linear programs. At scheduler decisions after tasks finish, it restricts shared completed-task durations to a finite set of feasible candidate realizations from the search. This discretized adversary is weaker than the continuous adversary in the exact MILO, so Minimax is a lower bound on the exact adaptive-robust makespan.
- **Hybrid Minimax upper bound** ([`ar_minimax_ub.py`](ar_minimax_ub.py)) searches the same tree, but solves every leaf LP independently, so the adversary may pick different durations for already-finished tasks in sibling branches. Relaxing non-anticipativity this way makes it an upper bound on the exact makespan; it equals the MILO with `enforce_non_anticipativity=false`. It is standalone and does not import `ar_minimax.py`.
- **Exact MILO** ([`ar_milo.py`](ar_milo.py)) represents the game as a monolithic mixed-integer linear optimization model and enforces non-anticipativity across shared histories.
- **Benchmark** ([`benchmark.py`](benchmark.py)) runs both methods on reference and seeded random instances, writes per-instance logs and `results.csv`, and checks their values and decisions.

Shared instance, uncertainty, and logging utilities are in [`ar_common.py`](ar_common.py). Benchmark parameters and reference instances are configured in [`config/config.json`](config/config.json).

## Additional documentation

- [Paper formalism and MILO deep dive](docs/paper_formalism_and_milo_deep_dive.md)
- [Bounding architecture](docs/bounding_architecture_lb_and_ub.md)
- [Iterative bound-tightening roadmap](docs/iterative_bound_tightening_roadmap.md)
- [Benchmarking and conclusions guide](docs/benchmarking_and_conclusions_guide.md)

## Requirements

- Python 3.9 or newer
- `gurobipy` and a working Gurobi license

Install the Python package with:

```sh
python3 -m pip install gurobipy
```

Gurobi licensing and setup are managed separately; see the [Gurobi installation documentation](https://www.gurobi.com/documentation/). When `GRB_LICENSE_FILE` is unset, the solver automatically uses `config/gurobi.lic` if that local file exists. An existing `GRB_LICENSE_FILE` setting takes precedence; set it to your license path if your license is stored elsewhere.

## Run

Run the configured benchmark from the project directory:

```sh
python3 benchmark.py --config config/config.json
```

Each execution creates a timestamped directory under `logs/`, including `results.csv`, a readable `summary.log`, and solver trace logs. The summary compares per-instance runtimes and reports which solver was faster, then compares the complete event order (grouped task starts and task finishes) and identifies the first event where the paths differ. It also records both full event sequences in `results.csv`; identical event sequences can still have different makespans if the solvers select different feasible duration realizations. The random instances are reproducible using the seed in `config/config.json`; each generated task-duration instance is evaluated at every value in `benchmark.gamma_values`. The default configuration runs 50 random instances across Gamma values from 0 to 4 in 0.5 increments, in addition to the reference checks. Each random instance uses the same task durations across its Gamma sweep.

Aggregate all available run CSVs into a Markdown report:

```sh
python3 aggregate_results.py
```

By default, the aggregator reads `logs/run_*/results.csv` and writes `logs/aggregate_summary.md`. To summarize one specific benchmark run without mixing in earlier runs, point `--logs-dir` directly at that run folder. Paths and the comparison tolerance can be overridden:

```sh
python3 aggregate_results.py --logs-dir logs --output logs/aggregate_summary.md --tolerance 0.0001
```

The current aggregate report is [`logs/aggregate_summary.md`](logs/aggregate_summary.md).

Run an individual solver on a configured reference instance with:

```sh
python3 ar_minimax.py --reference 0
python3 ar_minimax_ub.py --reference 0
python3 ar_milo.py --reference 0
```

Compare all three models (LB <= MILO <= UB) by gap, first decision, and runtime with [`compare_bounds.py`](compare_bounds.py). It can reuse saved LB/MILO results and run only the UB, or run all three models when no saved data is given:

```sh
python3 compare_bounds.py --n 4 5 --gamma 2.5 --rounds 50 \
    --lb-json 4=logs/pruning_experiment/res_n4.json 5=logs/pruning_experiment/res_n5.json \
    --milo-json 4=logs/pruning_experiment/res_n4.json 5=logs/pruning_experiment/milo_n5.json,logs/pruning_experiment/milo_n5_rerun.json \
    --recover-decisions
python3 compare_bounds.py --n 4 --gamma 1.5 --rounds 20     # runs LB, MILO and UB
```

It writes `logs/bounds/gamma_<g>/run_<ts>/summary_all.log` (all n side by side), `results_all.csv`, and one folder per n with `results.csv`, `summary.log`, `ub_results.json`, and UB logs.

## Interpreting the comparison

The reported gap is `Minimax - MILO`. It should be non-positive, up to numerical tolerance: Minimax is a lower bound on the exact MILO. The benchmark also reports whether the two methods choose the same first decision and checks reference instances when an expected value is provided.