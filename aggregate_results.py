"""Aggregate benchmark results.csv files into a Markdown report.

Usage: python3 aggregate_results.py [--logs-dir logs/benchmark] [--output logs/benchmark/aggregate_summary.md]

Finds every run_*/results.csv below --logs-dir (e.g. logs/benchmark/n4_m2/run_<ts>/results.csv),
or reads --logs-dir/results.csv directly when it points at one run folder.
"""

import argparse
import csv
from pathlib import Path
from statistics import mean


def as_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def as_bool(value):
    if value is None or value == "":
        return None
    return value.strip().lower() in {"true", "1", "yes"}


def read_results(logs_dir):
    logs_path = Path(logs_dir)
    direct_results = logs_path / "results.csv"
    if direct_results.is_file():
        with direct_results.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        return {logs_path.name: rows} if rows else {}

    runs = {}
    for csv_path in sorted(logs_path.rglob("run_*/results.csv")):
        with csv_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if rows:
            runs[csv_path.parent.relative_to(logs_path).as_posix()] = rows
    return runs


def mean_field(rows, field):
    values = [as_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def summarize_rows(rows, tolerance):
    rounds = [row for row in rows if row.get("tag", "").startswith("round")]
    solved = [row for row in rounds if as_float(row.get("minimax")) is not None
              and as_float(row.get("milo")) is not None]
    gaps = [as_float(row.get("gap")) for row in solved]
    gaps = [gap for gap in gaps if gap is not None]
    return {
        "rounds": rounds,
        "solved": solved,
        "gaps": gaps,
        "milo_better": sum(gap > tolerance for gap in gaps),
        "approximately_equal": sum(abs(gap) <= tolerance for gap in gaps),
        "minimax_below_milo": sum(gap < -tolerance for gap in gaps),
        "different_decisions": sum(as_bool(row.get("same_decision")) is False for row in solved),
        "mean_regret": mean_field(solved, "regret"),
        "mean_minimax_time": mean_field(solved, "minimax_time"),
        "mean_milo_time": mean_field(solved, "milo_time"),
    }


def render_report(runs, tolerance):
    all_rows = [row for run_rows in runs.values() for row in run_rows]
    total = summarize_rows(all_rows, tolerance)
    gamma_values = sorted({
        gamma for row in all_rows
        if row.get("tag", "").startswith("round")
        and (gamma := as_float(row.get("gamma"))) is not None
    })
    baselines = {}
    for run_name, run_rows in runs.items():
        for row in run_rows:
            if row.get("tag", "").startswith("round") and as_float(row.get("gamma")) == 0:
                key = (run_name, row["tag"].partition("_gamma_")[0])
                baselines[key] = row

    lines = [
        "# Aggregated Benchmark Summary",
        "",
        f"- Run folders: {len(runs)}",
        f"- Random rounds solved by both methods: {len(total['solved'])}/{len(total['rounds'])}",
        f"- Gap tolerance: {tolerance:g}",
        "",
        "## Random-Round Results",
        "",
        f"- Mean gap (Minimax - MILO): {fmt(mean(total['gaps']) if total['gaps'] else None)}",
        f"- Mean gap percentage: {fmt(mean_field(total['solved'], 'gap_pct'), 4)}%",
        f"- MILO lower by more than tolerance: {total['milo_better']}",
        f"- Approximately equal: {total['approximately_equal']}",
        f"- Minimax below MILO by more than tolerance: {total['minimax_below_milo']}",
        f"- Different selected decisions (excluding ties as marked by the benchmark): {total['different_decisions']}",
        f"- Mean regret of Minimax decision: {fmt(total['mean_regret'])}",
        f"- Mean solve time: Minimax {fmt(total['mean_minimax_time'], 3)} s; MILO {fmt(total['mean_milo_time'], 3)} s",
        "",
        "## Per-Run Results",
        "",
        "| Run | Solved / rounds | Mean gap | Mean gap % | MILO better | Equal | Theory violations | Mean Minimax (s) | Mean MILO (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run_name, rows in runs.items():
        summary = summarize_rows(rows, tolerance)
        run_gap_mean = mean(summary["gaps"]) if summary["gaps"] else None
        lines.append(
            f"| {run_name} | {len(summary['solved'])} / {len(summary['rounds'])} "
            f"| {fmt(run_gap_mean)} | {fmt(mean_field(summary['solved'], 'gap_pct'), 4)}% "
            f"| {summary['milo_better']} | {summary['approximately_equal']} "
            f"| {summary['minimax_below_milo']} | {fmt(summary['mean_minimax_time'], 3)} "
            f"| {fmt(summary['mean_milo_time'], 3)} |"
        )

    lines.extend([
        "",
        "## Results by Gamma",
        "",
        "| Gamma | Solved / comparisons | Mean Minimax | Mean MILO | Mean gap | Paired Δ Minimax vs Γ=0 | Paired Δ MILO vs Γ=0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for gamma in gamma_values:
        gamma_rows = [
            (run_name, row) for run_name, run_rows in runs.items() for row in run_rows
            if row.get("tag", "").startswith("round") and as_float(row.get("gamma")) == gamma
        ]
        solved = [row for _, row in gamma_rows if as_float(row.get("minimax")) is not None
                  and as_float(row.get("milo")) is not None]
        paired = []
        if gamma != 0:
            for run_name, row in gamma_rows:
                baseline = baselines.get((run_name, row["tag"].partition("_gamma_")[0]))
                current_mm, current_milo = as_float(row.get("minimax")), as_float(row.get("milo"))
                base_mm = as_float(baseline.get("minimax")) if baseline else None
                base_milo = as_float(baseline.get("milo")) if baseline else None
                if current_mm is not None and current_milo is not None and base_mm is not None and base_milo is not None:
                    paired.append((current_mm - base_mm, current_milo - base_milo))
        delta_mm = mean([delta[0] for delta in paired]) if paired else (0.0 if gamma == 0 and solved else None)
        delta_milo = mean([delta[1] for delta in paired]) if paired else (0.0 if gamma == 0 and solved else None)
        lines.append(
            f"| {gamma:g} | {len(solved)} / {len(gamma_rows)} "
            f"| {fmt(mean_field(solved, 'minimax'))} | {fmt(mean_field(solved, 'milo'))} "
            f"| {fmt(mean_field(solved, 'gap'))} | {fmt(delta_mm)} | {fmt(delta_milo)} |"
        )

    references = [row for row in all_rows if row.get("tag", "").startswith("ref")]
    reference_checks = [as_bool(row.get("reference_ok")) for row in references]
    checked_references = [result for result in reference_checks if result is not None]
    lines.extend([
        "",
        "## Reference Checks",
        "",
        f"- Passed: {sum(checked_references)} / {len(checked_references)} checked reference instances",
        "",
        "## Notes",
        "",
        "Gap is calculated as Minimax minus MILO. A positive value means MILO found a lower makespan; "
        "a negative value beyond tolerance is flagged as a theory-check violation.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs/benchmark", help="directory searched for run_* folders")
    parser.add_argument("--output", default="logs/benchmark/aggregate_summary.md", help="Markdown output path")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="gap threshold for comparisons")
    args = parser.parse_args()

    runs = read_results(args.logs_dir)
    if not runs:
        parser.error(f"No run_*/results.csv files found below {args.logs_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(runs, args.tolerance), encoding="utf-8")
    print(f"Aggregated {len(runs)} run(s) into {output_path}")


if __name__ == "__main__":
    main()