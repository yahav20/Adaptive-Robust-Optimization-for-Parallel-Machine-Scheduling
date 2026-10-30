"""
Benchmark: our Minimax idea (ar_minimax.py) vs the paper's exact MILO (ar_milo.py).

    python3 benchmark.py [--config config/config.json]

1. Reference checks: runs both models on config.reference_instances (the scheduling.py
   example must give 1.8296) and reports PASS/FAIL.
2. n_rounds random task-duration instances, each evaluated at every configured Gamma.

Output: logs/benchmark/n<n_tasks>_m<n_machines>/run_<ts>/
    summary.log    final summary tables only
    details.log    per-instance comparison (first decisions, event order, gap, runtime)
    results.csv    one row per instance
    config.json    copy of the config used for the run
    instances/ref_<k>/{minimax,milo,milo_gurobi}.log
    instances/round_<r>/gamma_<G>_{minimax,milo,milo_gurobi}.log
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import replace

from ar_common import close_logger, fmt_val, load_config, random_instance, reference_instances, setup_logger, tasks_str
from ar_milo import solve_milo
from ar_minimax import solve_minimax


def fmt_pct(p):
    return "n/a" if p is None else f"{0.0 if abs(p) < 5e-3 else p:.2f}"


def fmt_event(event):
    kind, payload = event
    if kind == "S":
        return f"start {tasks_str(payload)}"
    return f"T{payload + 1} finishes"


def compare_event_orders(minimax_path, milo_path):
    if minimax_path is None or milo_path is None:
        return None, "", "", ""
    minimax_events = [fmt_event(event) for event in minimax_path]
    milo_events = [fmt_event(event) for event in milo_path]
    same = minimax_events == milo_events
    difference = ""
    if not same:
        for index in range(max(len(minimax_events), len(milo_events))):
            minimax_event = minimax_events[index] if index < len(minimax_events) else "<end>"
            milo_event = milo_events[index] if index < len(milo_events) else "<end>"
            if minimax_event != milo_event:
                difference = f"event {index + 1}: Minimax [{minimax_event}] vs MILO [{milo_event}]"
                break
    return same, difference, " -> ".join(minimax_events), " -> ".join(milo_events)


def instance_log_prefix(run_dir, tag):
    """instances/ref_0/ for references, instances/round_01/gamma_0p5_ for random rounds."""
    group, _, gamma_part = tag.partition("_gamma_")
    folder = os.path.join(run_dir, "instances", group)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"gamma_{gamma_part}_" if gamma_part else "")


def run_instance(inst, cfg, run_dir, tag, out):
    """Run both models on one instance and log the comparison to `out`. Returns a result row."""
    bcfg = cfg["benchmark"]
    tol = bcfg.get("tolerance", 1e-4)
    prefix = instance_log_prefix(run_dir, tag)
    gurobi_log = f"{prefix}milo_gurobi.log" if cfg["milo"].get("write_gurobi_log") else None

    lg = setup_logger(f"{tag}_minimax", f"{prefix}minimax.log")
    mm = solve_minimax(inst, cfg, lg)
    close_logger(lg)

    lg = setup_logger(f"{tag}_milo", f"{prefix}milo.log")
    milo = solve_milo(inst, cfg, lg, gurobi_log=gurobi_log)
    close_logger(lg)

    # ------------------------------------------------------------- comparison
    out.info(inst.describe())
    out.info("")
    out.info(f"  {'first decision':<16} | {'Minimax (ours)':>14} | {'MILO (exact)':>12} | {'gap':>8}")
    out.info(f"  {'-' * 16}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 8}")
    for combo, mv in mm["root_choices"].items():
        lv = milo["root_choices"].get(combo)
        gap = mv - lv if lv is not None else None
        marks = ("M" if combo == mm["root_choice"] else " ") + ("P" if combo == milo["root_choice"] else " ")
        out.info(f"  {marks} start {tasks_str(combo):<8} | {fmt_val(mv):>14} | {fmt_val(lv):>12} | {fmt_val(gap):>8}")
    out.info("  (M = minimax's first decision, P = MILO's first decision)")
    out.info("")

    mv, lv = mm["value"], milo["value"]
    order_same, order_difference, minimax_order, milo_order = compare_event_orders(
        mm["worst_leaf"]["path"] if mm.get("worst_leaf") else None,
        milo.get("worst_path"),
    )
    if order_same is None:
        out.info("  Whole event order comparison: unavailable (one solver has no complete path)")
    elif order_same:
        out.info(f"  Whole event order comparison: SAME ({len(mm['worst_leaf']['path'])} events)")
        out.info(f"    Order: {minimax_order}")
    else:
        out.info(f"  Whole event order comparison: DIFFERENT; first divergence at {order_difference}")
        out.info(f"    Minimax: {minimax_order}")
        out.info(f"    MILO   : {milo_order}")
    same_order_makespan_diff = order_same and lv is not None and abs(mv - lv) > tol
    if same_order_makespan_diff:
        out.info("  Makespans differ despite the same event order; the duration/uncertainty decisions differ.")

    if abs(mm["time"] - milo["time"]) <= 1e-6:
        faster, speedup = "tie", 1.0
    elif mm["time"] < milo["time"]:
        faster, speedup = "Minimax", milo["time"] / mm["time"] if mm["time"] else None
    else:
        faster, speedup = "MILO", mm["time"] / milo["time"] if milo["time"] else None
    out.info(f"  Runtime comparison: Minimax {mm['time']:.3f}s, MILO {milo['time']:.3f}s; "
             f"{faster} faster" + (f" ({speedup:.2f}x)" if speedup is not None else ""))
    out.info(f"  Minimax (ours) : {fmt_val(mv)}  start {tasks_str(mm['root_choice'])}  time {mm['time']:.3f}s  "
             f"leaves {mm['stats']['leaves']}  Gurobi LPs {mm['stats']['gurobi_calls']} "
             f"({mm['stats']['gurobi_time']:.3f}s in Gurobi)")
    if lv is None:
        out.info(f"  MILO (exact)   : not solved -> {milo['status']}")
    else:
        rc = tasks_str(milo["root_choice"]) if milo["root_choice"] else "?"
        out.info(f"  MILO (exact)   : {fmt_val(lv)}  start {rc}  time {milo['time']:.3f}s  "
                 f"size {milo['size']}  tree-build LPs {milo['stats'].get('feasibility_lps')}  status {milo['status']}")
        if not milo.get("optimal", True):
            out.info("  WARNING: MILO did not prove optimality (time limit?) -> do not average this row")

    row = dict(tag=tag, name=inst.name, n_tasks=inst.n_tasks, n_machines=inst.n_machines, gamma=inst.gamma,
               d0=inst.d0, d_bar=inst.d_bar, minimax=mv, milo=lv, gap=None, gap_pct=None,
               minimax_choice=tasks_str(mm["root_choice"]),
               milo_choice=tasks_str(milo["root_choice"]) if milo["root_choice"] else "",
               minimax_event_order=minimax_order, milo_event_order=milo_order,
               event_order_same=order_same, first_event_difference=order_difference,
               same_order_makespan_diff=same_order_makespan_diff,
               same_decision=None, regret=None, anticipation_cases=len(mm["anticipation"]),
               minimax_time=mm["time"], milo_time=milo["time"], faster=faster, speedup=speedup,
               minimax_gurobi_calls=mm["stats"]["gurobi_calls"],
               milo_tree_lps=milo["stats"].get("feasibility_lps"),
               milo_status=milo["status"],
               milo_optimal=milo.get("optimal"),
               bound_ok=None,
               reference=inst.expected_makespan, reference_ok=None)

    if lv is not None:
        gap = mv - lv
        exact_of_mm = milo["root_choices"].get(mm["root_choice"])
        # a different first decision with the same exact AR value is a tie, not a disagreement
        same = mm["root_choice"] == milo["root_choice"] or (exact_of_mm is not None and abs(exact_of_mm - lv) <= tol)
        row.update(gap=gap, gap_pct=100.0 * gap / lv if lv > 0 else 0.0, same_decision=same, bound_ok=gap <= tol)
        if gap < -tol:
            verdict = "Minimax is LOWER (valid lower bound on exact AR)"
        elif gap > tol:
            verdict = "!!! Minimax exceeds MILO - lower-bound violation"
        else:
            verdict = "EQUAL (discretization gap is within tolerance)"
        out.info(f"  Gap minimax - MILO = {fmt_val(gap)} ({fmt_pct(row['gap_pct'])}%) -> {verdict}")
        if mm["root_choice"] == milo["root_choice"]:
            out.info("  Same first decision? YES")
        else:
            out.info(f"  Same first decision? {'TIE (different task set, same exact AR value)' if same else 'NO'}")
        if exact_of_mm is not None:
            row["regret"] = exact_of_mm - lv
            out.info(f"  Minimax's first decision evaluated by the exact AR model: {exact_of_mm:.4f} "
                     f"(regret vs optimal AR = {row['regret']:.4f})")

    if mm["anticipation"]:
        out.info(f"  Anticipation: the minimax adversary 'saw the future' at {len(mm['anticipation'])} "
                 f"scheduler node(s), e.g.")
        for a in mm["anticipation"][:3]:
            opts = "; ".join(f"start {tasks_str(c)} -> d{a['task'] + 1}={dv:.4f}" for c, dv, _ in a["per_option"])
            out.info(f"    at {a['node']}: T{a['task'] + 1} already finished but assumed {opts}")
        out.info("    The MILO forces ONE value per revealed duration, so its adversary is weaker.")
        if row["gap"] is not None and abs(row["gap"]) <= tol:
            out.info("    Here it did NOT change the result: those nodes are off the critical path, or another")
            out.info("    consistent duration vector reaches the same worst case.")
        elif row["gap"] is not None:
            out.info("    A nonzero lower-bound gap is consistent with the finite candidate restriction.")
    elif lv is not None:
        out.info("  No anticipation detected in the minimax worst cases.")

    if inst.expected_makespan is not None:
        ok_mm = abs(mv - inst.expected_makespan) <= tol
        ok_milo = lv is not None and abs(lv - inst.expected_makespan) <= tol
        row["reference_ok"] = ok_mm and ok_milo
        out.info(f"  Reference check (expected {inst.expected_makespan:.4f}): minimax {'PASS' if ok_mm else 'FAIL'}, "
                 f"MILO {'PASS' if ok_milo else 'FAIL'}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["benchmark"]
    log_cfg = cfg.get("logging", {})
    prob = cfg["problem"]
    run_dir = os.path.join(log_cfg.get("log_dir", "logs"), "benchmark",
                           f"n{prob['n_tasks']}_m{prob['n_machines']}", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    console_level = log_cfg.get("console_level", "INFO")
    out = setup_logger("benchmark", os.path.join(run_dir, "details.log"), console=True, console_level=console_level)
    rows = []

    if bcfg.get("run_reference_checks", True):
        for k, inst in enumerate(reference_instances(cfg)):
            if inst.is_discrete:
                out.info(f"REFERENCE {k}: {inst.name} skipped (discrete U is MILO-only: "
                         f"python ar_milo.py --reference {k})")
                continue
            out.info("#" * 78)
            out.info(f"REFERENCE {k}: {inst.name}")
            out.info("#" * 78)
            rows.append(run_instance(inst, cfg, run_dir, f"ref_{k}", out))
            out.info("")

    rng = random.Random(cfg["random_instance"]["seed"])
    n_rounds = bcfg["n_rounds"]
    gamma_values = bcfg.get("gamma_values", [prob["gamma"]])
    if (not gamma_values
            or any(not isinstance(gamma, (int, float)) or not 0 <= gamma <= prob["n_tasks"]
                   for gamma in gamma_values)):
        raise ValueError(f"benchmark.gamma_values must contain values between 0 and {prob['n_tasks']}")
    if len(set(gamma_values)) != len(gamma_values):
        raise ValueError("benchmark.gamma_values must not contain duplicates")

    for r in range(1, n_rounds + 1):
        base_inst = random_instance(cfg, rng, name=f"round {r}")
        for gamma in gamma_values:
            inst = replace(base_inst, name=f"round {r} Gamma={gamma:g}", gamma=gamma)
            gamma_tag = str(gamma).replace(".", "p")
            tag = f"round_{r:02d}_gamma_{gamma_tag}"
            out.info("#" * 78)
            out.info(f"ROUND {r}/{n_rounds}, Gamma={gamma:g} "
                     f"(n_tasks={prob['n_tasks']}, n_machines={prob['n_machines']})")
            out.info("#" * 78)
            rows.append(run_instance(inst, cfg, run_dir, tag, out))
            out.info("")

    # ------------------------------------------------------------- summary
    close_logger(out)
    out = setup_logger("benchmark_summary", os.path.join(run_dir, "summary.log"), console=True,
                       console_level=console_level)
    rnd_rows = [x for x in rows if x["tag"].startswith("round")]
    rnd = [x for x in rnd_rows if x["minimax"] is not None and x["milo"] is not None]
    tw = max([len("instance")] + [len(x["tag"]) for x in rows])
    out.info("=" * 78)
    out.info("BENCHMARK SUMMARY")
    out.info("=" * 78)
    out.info(f"  {'instance':<{tw}} | {'minimax':>8} | {'MILO':>8} | {'gap':>7} | {'gap%':>6} | {'same 1st':>8} | "
             f"{'t_mm':>6} | {'t_milo':>6}")
    for x in rows:
        gap_pct = fmt_pct(x["gap_pct"])
        out.info(f"  {x['tag']:<{tw}} | {fmt_val(x['minimax']):>8} | {fmt_val(x['milo']):>8} | "
                 f"{fmt_val(x['gap']):>7} | {gap_pct:>6} | "
                 f"{str(x['same_decision']):>8} | {x['minimax_time']:6.2f} | {x['milo_time']:6.2f}")
    if rnd:
        tol = bcfg.get("tolerance", 1e-4)
        above = sum(1 for x in rnd if x["gap"] < -tol)
        equal = sum(1 for x in rnd if abs(x["gap"]) <= tol)
        violations = sum(1 for x in rnd if x["gap"] > tol)
        out.info("")
        out.info(f"  Random comparisons solved by both: {len(rnd)}/{len(rnd_rows)}")
        out.info(f"  MILO above Minimax (valid lower-bound gap): {above}, equal: {equal}, "
                 f"Minimax above MILO (violations): {violations}")
        out.info(f"  Mean gap: {sum(x['gap'] for x in rnd) / len(rnd):.4f}  "
                 f"({sum(x['gap_pct'] for x in rnd) / len(rnd):.2f}%), max gap: {max(x['gap'] for x in rnd):.4f}")
        out.info(f"  Different first decision (ties excluded): {sum(1 for x in rnd if not x['same_decision'])} round(s); "
                 f"mean regret of minimax decision: {fmt_val(sum(x['regret'] or 0 for x in rnd) / len(rnd))}")
        out.info(f"  Mean time: minimax {sum(x['minimax_time'] for x in rnd) / len(rnd):.3f}s, "
                 f"MILO {sum(x['milo_time'] for x in rnd) / len(rnd):.3f}s")
        out.info("  Results by Gamma:")
        baseline_by_round = {
            x["tag"].partition("_gamma_")[0]: x
            for x in rnd_rows if x["gamma"] == 0
        }
        for gamma in gamma_values:
            gamma_rows = [x for x in rnd_rows if x["gamma"] == gamma]
            gamma_solved = [x for x in gamma_rows if x["minimax"] is not None and x["milo"] is not None]
            gaps = [x["gap"] for x in gamma_solved if x["gap"] is not None]
            mean_gap = sum(gaps) / len(gaps) if gaps else None
            paired = [
                (x, baseline_by_round.get(x["tag"].partition("_gamma_")[0]))
                for x in gamma_solved
            ]
            paired = [
                (x, baseline) for x, baseline in paired
                if baseline is not None and baseline["minimax"] is not None and baseline["milo"] is not None
            ]
            delta_mm = (
                sum(x["minimax"] - baseline["minimax"] for x, baseline in paired) / len(paired)
                if paired else None
            )
            delta_milo = (
                sum(x["milo"] - baseline["milo"] for x, baseline in paired) / len(paired)
                if paired else None
            )
            out.info(f"    Gamma={gamma:g}: solved {len(gamma_solved)}/{len(gamma_rows)}, "
                     f"mean makespan Minimax/MILO={fmt_val(sum(x['minimax'] for x in gamma_solved) / len(gamma_solved) if gamma_solved else None)}/"
                     f"{fmt_val(sum(x['milo'] for x in gamma_solved) / len(gamma_solved) if gamma_solved else None)}, "
                     f"mean gap={fmt_val(mean_gap)}, paired delta vs Gamma=0 Minimax/MILO="
                     f"{fmt_val(delta_mm)}/{fmt_val(delta_milo)}")
    out.info("")
    out.info("PER-INSTANCE RUNTIME AND EVENT-ORDER COMPARISON")
    out.info(f"  {'instance':<{tw}} | {'Minimax (s)':>11} | {'MILO (s)':>8} | {'faster':>8} | "
             f"{'speedup':>7} | {'same order':>10} | {'same-order gap':>14} | first event difference")
    for x in rows:
        speedup_text = f"{x['speedup']:.2f}x" if x["speedup"] is not None else "n/a"
        difference = x["first_event_difference"] or ("same" if x["event_order_same"] else "unavailable")
        out.info(f"  {x['tag']:<{tw}} | {x['minimax_time']:11.3f} | {x['milo_time']:8.3f} | "
                 f"{x['faster']:>8} | {speedup_text:>7} | {str(x['event_order_same']):>10} | "
                 f"{str(x['same_order_makespan_diff']):>14} | {difference}")
    if rows:
        minimax_wins = sum(x["faster"] == "Minimax" for x in rows)
        milo_wins = sum(x["faster"] == "MILO" for x in rows)
        time_ties = sum(x["faster"] == "tie" for x in rows)
        compared_orders = [x for x in rows if x["event_order_same"] is not None]
        same_orders = sum(x["event_order_same"] for x in compared_orders)
        same_order_gaps = sum(x["same_order_makespan_diff"] is True for x in rows)
        out.info(f"  Faster counts: Minimax {minimax_wins}, MILO {milo_wins}, ties {time_ties}")
        out.info(f"  Mean runtime (all instances): Minimax {sum(x['minimax_time'] for x in rows) / len(rows):.3f}s, "
                 f"MILO {sum(x['milo_time'] for x in rows) / len(rows):.3f}s")
        out.info(f"  Identical whole event orders: {same_orders}/{len(compared_orders)} "
                 f"({len(rows) - len(compared_orders)} unavailable)")
        out.info(f"  Same order but makespan differs beyond tolerance: {same_order_gaps}")
    out.info(f"  Theory check Minimax <= exact MILO: "
             f"{'OK' if all(x['bound_ok'] in (True, None) for x in rows) else 'VIOLATED'}")
    refs = [x for x in rows if x["reference_ok"] is not None]
    if refs:
        out.info(f"  Reference checks: {'PASS' if all(x['reference_ok'] for x in refs) else 'FAIL'}")

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out.info(f"  summary.log, details.log, results.csv and instances/ in {run_dir}")
    close_logger(out)


if __name__ == "__main__":
    main()
