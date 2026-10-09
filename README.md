# Adaptive Robust Optimization for Parallel Machine Scheduling

This project compares two approaches to adaptive robust scheduling on identical parallel machines. The objective is to minimize the worst-case makespan when task durations are uncertain:

\[
d_i = d_i^0 + \bar d_i u_i, \qquad 0 \le u_i \le 1, \qquad \sum_i u_i \le \Gamma.
\]

## Approaches

- **Hybrid Minimax** ([`ar_minimax.py`](ar_minimax.py)) searches the scheduler/adversary decision tree recursively and evaluates completed paths with linear programs. Each leaf LP is independent, so the adversary can choose different durations in different futures. Its result is therefore an upper bound on the exact adaptive-robust makespan.
- **Exact MILO** ([`ar_milo.py`](ar_milo.py)) represents the game as a monolithic mixed-integer linear optimization model and enforces non-anticipativity across shared histories.
- **Benchmark** ([`benchmark.py`](benchmark.py)) runs both methods on reference and seeded random instances, writes per-instance logs and `results.csv`, and checks their values and decisions.

Shared instance, uncertainty, and logging utilities are in [`ar_common.py`](ar_common.py). Benchmark parameters and reference instances are configured in [`config.json`](config.json).

## Requirements

- Python 3.9 or newer
- `gurobipy` and a working Gurobi license

Install the Python package with:

```sh
python3 -m pip install gurobipy
```

Gurobi licensing and setup are managed separately; see the [Gurobi installation documentation](https://www.gurobi.com/documentation/).

## Run

Run the configured benchmark from the project directory:

```sh
python3 benchmark.py --config config.json
```

Each execution creates a timestamped directory under `logs/`, including `results.csv`, a readable `summary.log`, and solver trace logs. The random instances are reproducible using the seed in `config.json`.

Aggregate all available run CSVs into a Markdown report:

```sh
python3 aggregate_results.py
```

By default, the aggregator reads `logs/run_*/results.csv` and writes `logs/aggregate_summary.md`. Paths and the comparison tolerance can be overridden:

```sh
python3 aggregate_results.py --logs-dir logs --output logs/aggregate_summary.md --tolerance 0.0001
```

The current aggregate report is [`logs/aggregate_summary.md`](logs/aggregate_summary.md).

Run an individual solver on a configured reference instance with:

```sh
python3 ar_minimax.py --reference 0
python3 ar_milo.py --reference 0
```

## Interpreting the comparison

The reported gap is `Minimax - MILO`. A positive gap means the exact MILO found a lower makespan. Since the independent-leaf Minimax relaxation gives the adversary more freedom, its value should be at least the exact MILO value, up to numerical tolerance. The benchmark also reports whether the two methods choose the same first decision and checks reference instances when an expected value is provided.