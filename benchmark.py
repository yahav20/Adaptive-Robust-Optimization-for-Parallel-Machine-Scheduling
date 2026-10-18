"""
Benchmark: our Minimax idea (ar_minimax.py) vs the paper's exact MILO (ar_milo.py).

    python3 benchmark.py [--config config.json]

1. Reference checks: runs both models on config.reference_instances (the scheduling.py
   example must give 1.8296) and reports PASS/FAIL.
2. n_rounds random instances (n_tasks, n_machines, gamma from config). Each round prints
   a side-by-side comparison and writes full logs:
       logs/run_<time>/round_XX_minimax.log      full minimax tree trace + leaf LPs
       logs/run_<time>/round_XX_milo.log         MILO size, solves, worst-case paths, tree
       logs/run_<time>/round_XX_milo_gurobi.log  raw Gurobi output
       logs/run_<time>/round_XX_milo_relaxed.log MILO without non-anticipativity (cross-check)
       logs/run_<time>/summary.log, results.csv
"""

import argparse
import csv
import os
import random
import time

from ar_common import close_logger, fmt_val, load_config, random_instance, reference_instances, setup_logger, tasks_str
from ar_milo import solve_milo
from ar_minimax import solve_minimax


def fmt_pct(p):
    return "n/a" if p is None else f"{0.0 if abs(p) < 5e-3 else p:.2f}"


def run_instance(inst, cfg, run_dir, tag, out):
    """Run both models on one instance and log the comparison to `out`. Returns a result row."""
    bcfg = cfg["benchmark"]
    tol = bcfg.get("tolerance", 1e-4)
    gurobi_log = os.path.join(run_dir, f"{tag}_milo_gurobi.log") if cfg["milo"].get("write_gurobi_log") else None

    lg = setup_logger(f"{tag}_minimax", os.path.join(run_dir, f"{tag}_minimax.log"))
    mm = solve_minimax(inst, cfg, lg)
    close_logger(lg)

    lg = setup_logger(f"{tag}_milo", os.path.join(run_dir, f"{tag}_milo.log"))
    milo = solve_milo(inst, cfg, lg, gurobi_log=gurobi_log)
    close_logger(lg)

    relaxed = None
    if bcfg.get("cross_check_relaxed_milo", True):
        lg = setup_logger(f"{tag}_relaxed", os.path.join(run_dir, f"{tag}_milo_relaxed.log"))
        relaxed = solve_milo(inst, cfg, lg, enforce_na=False)
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
    out.info(f"  Minimax (ours) : {fmt_val(mv)}  start {tasks_str(mm['root_choice'])}  time {mm['time']:.3f}s  "
             f"leaves {mm['stats']['leaves']}  LPs {mm['stats']['leaf_lps'] + mm['stats']['feasibility_lps']}")
    if lv is None:
        out.info(f"  MILO (exact)   : not solved -> {milo['status']}")
    else:
        rc = tasks_str(milo["root_choice"]) if milo["root_choice"] else "?"
        out.info(f"  MILO (exact)   : {fmt_val(lv)}  start {rc}  time {milo['time']:.3f}s  "
                 f"size {milo['size']}  status {milo['status']}")

    row = dict(tag=tag, name=inst.name, n_tasks=inst.n_tasks, n_machines=inst.n_machines, gamma=inst.gamma,
               d0=inst.d0, d_bar=inst.d_bar, minimax=mv, milo=lv, gap=None, gap_pct=None,
               minimax_choice=tasks_str(mm["root_choice"]),
               milo_choice=tasks_str(milo["root_choice"]) if milo["root_choice"] else "",
               same_decision=None, regret=None, anticipation_cases=len(mm["anticipation"]),
               minimax_time=mm["time"], milo_time=milo["time"], milo_status=milo["status"],
               relaxed_milo=relaxed["value"] if relaxed else None, relaxed_bound_ok=None, bound_ok=None,
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

    if relaxed is not None and relaxed["value"] is not None:
        row["relaxed_bound_ok"] = relaxed["value"] >= mv - tol
        out.info(f"  Relaxed-MILO bound check: Minimax = {mv:.4f} <= MILO WITHOUT non-anticipativity "
                 f"= {relaxed['value']:.4f} "
                 f"{'OK' if row['relaxed_bound_ok'] else 'VIOLATED'}")

    if inst.expected_makespan is not None:
        ok_mm = abs(mv - inst.expected_makespan) <= tol
        ok_milo = lv is not None and abs(lv - inst.expected_makespan) <= tol
        row["reference_ok"] = ok_mm and ok_milo
        out.info(f"  Reference check (expected {inst.expected_makespan:.4f}): minimax {'PASS' if ok_mm else 'FAIL'}, "
                 f"MILO {'PASS' if ok_milo else 'FAIL'}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bcfg = cfg["benchmark"]
    log_cfg = cfg.get("logging", {})
    run_dir = os.path.join(log_cfg.get("log_dir", "logs"), f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    out = setup_logger("benchmark", os.path.join(run_dir, "summary.log"), console=True,
                       console_level=log_cfg.get("console_level", "INFO"))
    rows = []

    if bcfg.get("run_reference_checks", True):
        for k, inst in enumerate(reference_instances(cfg)):
            out.info("#" * 78)
            out.info(f"REFERENCE {k}: {inst.name}")
            out.info("#" * 78)
            rows.append(run_instance(inst, cfg, run_dir, f"ref_{k}", out))
            out.info("")

    prob = cfg["problem"]
    rng = random.Random(cfg["random_instance"]["seed"])
    n_rounds = bcfg["n_rounds"]
    for r in range(1, n_rounds + 1):
        inst = random_instance(cfg, rng, name=f"round {r}")
        out.info("#" * 78)
        out.info(f"ROUND {r}/{n_rounds}  (n_tasks={prob['n_tasks']}, n_machines={prob['n_machines']}, "
                 f"Gamma={prob['gamma']})")
        out.info("#" * 78)
        rows.append(run_instance(inst, cfg, run_dir, f"round_{r:02d}", out))
        out.info("")

    # ------------------------------------------------------------- summary
    rnd = [x for x in rows if x["tag"].startswith("round") and x["milo"] is not None]
    out.info("=" * 78)
    out.info("BENCHMARK SUMMARY")
    out.info("=" * 78)
    out.info(f"  {'instance':<10} | {'minimax':>8} | {'MILO':>8} | {'gap':>7} | {'gap%':>6} | {'same 1st':>8} | "
             f"{'t_mm':>6} | {'t_milo':>6}")
    for x in rows:
        gap_pct = fmt_pct(x["gap_pct"])
        out.info(f"  {x['tag']:<10} | {fmt_val(x['minimax']):>8} | {fmt_val(x['milo']):>8} | "
                 f"{fmt_val(x['gap']):>7} | {gap_pct:>6} | "
                 f"{str(x['same_decision']):>8} | {x['minimax_time']:6.2f} | {x['milo_time']:6.2f}")
    if rnd:
        tol = bcfg.get("tolerance", 1e-4)
        above = sum(1 for x in rnd if x["gap"] < -tol)
        equal = sum(1 for x in rnd if abs(x["gap"]) <= tol)
        violations = sum(1 for x in rnd if x["gap"] > tol)
        out.info("")
        out.info(f"  Random rounds solved by both: {len(rnd)}/{n_rounds}")
        out.info(f"  MILO above Minimax (valid lower-bound gap): {above}, equal: {equal}, "
                 f"Minimax above MILO (violations): {violations}")
        out.info(f"  Mean gap: {sum(x['gap'] for x in rnd) / len(rnd):.4f}  "
                 f"({sum(x['gap_pct'] for x in rnd) / len(rnd):.2f}%), max gap: {max(x['gap'] for x in rnd):.4f}")
        out.info(f"  Different first decision (ties excluded): {sum(1 for x in rnd if not x['same_decision'])} round(s); "
                 f"mean regret of minimax decision: {fmt_val(sum(x['regret'] or 0 for x in rnd) / len(rnd))}")
        out.info(f"  Mean time: minimax {sum(x['minimax_time'] for x in rnd) / len(rnd):.3f}s, "
                 f"MILO {sum(x['milo_time'] for x in rnd) / len(rnd):.3f}s")
    out.info(f"  Theory check Minimax <= exact MILO: "
             f"{'OK' if all(x['bound_ok'] in (True, None) for x in rows) else 'VIOLATED'}")
    out.info(f"  Bound check Minimax <= relaxed MILO: "
             f"{'OK' if all(x['relaxed_bound_ok'] in (True, None) for x in rows) else 'VIOLATED'}")
    refs = [x for x in rows if x["reference_ok"] is not None]
    if refs:
        out.info(f"  Reference checks: {'PASS' if all(x['reference_ok'] for x in refs) else 'FAIL'}")

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out.info(f"  Logs and results.csv in {run_dir}")
    close_logger(out)


if __name__ == "__main__":
    main()
