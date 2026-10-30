"""
Three-way bounds benchmark on the same random instances:
    LB   = ar_minimax.py     (non-anticipativity over a finite candidate set)  -> lower bound
    MILO = ar_milo.py        (paper's exact AR model)
    UB   = ar_minimax_ub.py  (independent leaf LPs, non-anticipativity relaxed) -> upper bound
Theory: LB <= MILO <= UB. Compared by value gap, first decision and runtime.

    python compare_bounds.py --n 4 5 --gamma 2.5 --rounds 50 \\
        --lb-json   4=logs/pruning_experiment/res_n4.json 5=logs/pruning_experiment/res_n5.json \\
        --milo-json 4=logs/pruning_experiment/res_n4.json \\
                    5=logs/pruning_experiment/milo_n5.json,logs/pruning_experiment/milo_n5_rerun.json \\
        --recover-decisions

    python compare_bounds.py --n 4 --gamma 1.5 --rounds 20      # no saved data: runs all three models

Instances are generated like logs/pruning_experiment/run_exp.py: config with n_tasks and gamma overridden,
random.Random(random_instance.seed + n), one random_instance per round (checked against saved d0 / d_bar).
  * UB always runs.
  * LB / MILO values, times and stats come from saved JSON rows when given (several files per n are
    merged, later files override earlier ones by round). Without a saved file for n the model runs in full.
  * --recover-decisions re-runs a saved model only where its first decision was not saved, to fill in the
    decision columns; the saved values and times are kept.
Outputs logs/bounds/gamma_<g>/run_<ts>/: n<n>/{results.csv, ub_results.json, summary.log, instances/round_XX_ub.log}
and the cross-n results_all.csv and summary_all.log.
"""

import argparse
import copy
import csv
import json
import logging
import os
import random
import statistics
import time

from ar_common import close_logger, fmt_val, load_config, path_str, random_instance, setup_logger, tasks_str
from ar_milo import solve_milo
from ar_minimax import solve_minimax
from ar_minimax_ub import solve_minimax_ub
from benchmark import fmt_pct


def null_logger(name):
    lg = logging.getLogger(name)
    for h in list(lg.handlers):
        lg.removeHandler(h)
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg


def parse_paths(items):
    """['4=a.json', '5=b.json,c.json'] -> {4: ['a.json'], 5: ['b.json', 'c.json']}"""
    out = {}
    for item in items or []:
        n, _, paths = item.partition("=")
        out[int(n)] = [p for p in paths.split(",") if p]
    return out


def as_choice(c):
    return tuple(c) if c is not None else None


def load_lb(paths):
    """Saved minimax (LB) rows in the run_exp.py format -> {round: row}."""
    rows = {}
    for p in paths:
        for r in json.load(open(p)):
            if r.get("minimax") is None:
                continue
            rows[r["round"]] = dict(value=r["minimax"], time=r.get("mm_time"), stats=r.get("stats") or {},
                                    choice=as_choice(r.get("root_choice")), d0=r.get("d0"), d_bar=r.get("d_bar"),
                                    source=p)
    return rows


def load_milo(paths):
    """Saved MILO rows (run_exp.py or run_milo.py format) -> {round: row}."""
    rows = {}
    for p in paths:
        for r in json.load(open(p)):
            if r.get("milo") is None or r.get("milo_status") is None:
                continue
            status = r["milo_status"]
            rows[r["round"]] = dict(value=r["milo"], time=r.get("milo_time"), status=status,
                                    optimal=r.get("optimal", status == "OPTIMAL"),
                                    tree_lps=r.get("tree_lps", r.get("milo_tree_lps")),
                                    choice=as_choice(r.get("root_choice")), root_choices=None,
                                    d0=r.get("d0"), d_bar=r.get("d_bar"), source=p)
    return rows


def lb_row(res):
    return dict(value=res["value"], time=res["time"], stats=res["stats"], choice=res["root_choice"], source="run")


def milo_row(res):
    return dict(value=res["value"], time=res["time"], status=res["status"], optimal=res.get("optimal"),
                tree_lps=res["stats"].get("feasibility_lps"), choice=as_choice(res["root_choice"]),
                root_choices=res.get("root_choices") or None, source="run")


def compare_choice(choice, ref_choice, milo_rc, ub_rc, tol):
    """'same' / 'tie(milo)' / 'tie(ub)' / 'diff'; a tie is a different task set with the same value,
    judged by the exact MILO values per first decision when available, else by the UB values."""
    if choice is None or ref_choice is None:
        return ""
    if choice == ref_choice:
        return "same"
    for basis, rc in (("milo", milo_rc), ("ub", ub_rc)):
        if rc and choice in rc and ref_choice in rc:
            return f"tie({basis})" if abs(rc[choice] - rc[ref_choice]) <= tol else "diff"
    return "diff"


def pct(a, ref):
    return 100.0 * a / ref if a is not None and ref else None


def stats_of(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return dict(mean=sum(v) / len(v), median=statistics.median(v), max=max(v), min=min(v), count=len(v))


def run_n(n, args, cfg0, lb_paths, milo_paths, run_dir, tol):
    cfg = copy.deepcopy(cfg0)
    cfg["problem"]["n_tasks"] = n
    cfg["problem"]["gamma"] = args.gamma
    cfg["logging"]["trace_tree"] = args.trace
    n_dir = os.path.join(run_dir, f"n{n}")
    os.makedirs(n_dir, exist_ok=True)
    nlog = setup_logger(f"bounds_n{n}", os.path.join(n_dir, "summary.log"))
    quiet = null_logger(f"bounds_quiet_{n}")

    lb_saved = load_lb(lb_paths) if lb_paths else {}
    milo_saved = load_milo(milo_paths) if milo_paths else {}
    rng = random.Random(cfg["random_instance"]["seed"] + n)
    rows, ub_raw = [], []

    for r in range(1, args.rounds + 1):
        inst = random_instance(cfg, rng, f"n{n} r{r}")
        for saved in (lb_saved.get(r), milo_saved.get(r)):
            if saved and saved.get("d0") is not None:
                assert saved["d0"] == inst.d0 and saved["d_bar"] == inst.d_bar, \
                    f"n={n} round {r}: regenerated instance differs from {saved['source']}"

        # UB (new solver, always run)
        lg = setup_logger(f"ub_n{n}_r{r}", os.path.join(n_dir, "instances", f"round_{r:02d}_ub.log"))
        ub = solve_minimax_ub(inst, cfg, lg)
        close_logger(lg)

        # LB: saved values, re-run for decisions only if asked
        notes = []
        lb = lb_saved.get(r)
        if lb is None:
            lb = lb_row(solve_minimax(inst, cfg, quiet))
        elif lb["choice"] is None and args.recover_decisions:
            rerun = solve_minimax(inst, cfg, quiet)
            lb["choice"] = rerun["root_choice"]
            if abs(rerun["value"] - lb["value"]) > tol:
                notes.append(f"LB re-run value {rerun['value']:.6f} != saved {lb['value']:.6f}")

        # MILO: saved values, re-run for decisions only if asked
        milo = milo_saved.get(r)
        if milo is None:
            milo = milo_row(solve_milo(inst, cfg, quiet))
        elif milo["choice"] is None and args.recover_decisions:
            rerun = solve_milo(inst, cfg, quiet)
            milo["choice"] = as_choice(rerun["root_choice"])
            milo["root_choices"] = rerun.get("root_choices") or None
            if rerun["value"] is None or abs(rerun["value"] - milo["value"]) > tol:
                notes.append(f"MILO re-run value {fmt_val(rerun['value'])} != saved {milo['value']:.6f}")

        lv, mv, uv = lb["value"], milo["value"], ub["value"]
        milo_ok = bool(milo.get("optimal"))
        milo_rc, ub_rc = milo.get("root_choices"), ub["root_choices"]
        regret = lambda c: (milo_rc[c] - mv) if milo_rc and c in milo_rc and milo_ok else None
        ub_t, lb_t, milo_t = ub["time"], lb.get("time"), milo.get("time")
        row = dict(
            n=n, round=r, gamma=args.gamma, d0=inst.d0, d_bar=inst.d_bar,
            lb=lv, milo=mv, ub=uv, milo_status=milo.get("status"), milo_optimal=milo_ok,
            gap_milo_lb=mv - lv, gap_milo_lb_pct=pct(mv - lv, mv),
            gap_ub_milo=uv - mv, gap_ub_milo_pct=pct(uv - mv, mv),
            bracket_ub_lb=uv - lv, bracket_pct=pct(uv - lv, mv),
            lb_le_milo=(lv <= mv + tol) if milo_ok else None,
            milo_le_ub=(mv <= uv + tol) if milo_ok else None,
            lb_le_ub=lv <= uv + tol,
            lb_choice=tasks_str(lb["choice"]) if lb["choice"] else "",
            milo_choice=tasks_str(milo["choice"]) if milo["choice"] else "",
            ub_choice=tasks_str(ub["root_choice"]),
            ub_vs_milo=compare_choice(ub["root_choice"], milo["choice"], milo_rc, ub_rc, tol),
            lb_vs_milo=compare_choice(lb["choice"], milo["choice"], milo_rc, ub_rc, tol),
            ub_vs_lb=compare_choice(ub["root_choice"], lb["choice"], milo_rc, ub_rc, tol),
            ub_regret=regret(ub["root_choice"]), lb_regret=regret(lb["choice"]),
            lb_time=lb_t, milo_time=milo_t, ub_time=ub_t,
            milo_over_ub_time=milo_t / ub_t if milo_t and ub_t else None,
            lb_over_ub_time=lb_t / ub_t if lb_t and ub_t else None,
            lb_gurobi_calls=lb.get("stats", {}).get("gurobi_calls"),
            ub_gurobi_calls=ub["stats"]["gurobi_calls"], milo_tree_lps=milo.get("tree_lps"),
            ub_leaves=ub["stats"]["leaves"], ub_na_relaxed_nodes=ub["stats"]["na_relaxed_nodes"],
            ub_event_order=path_str(ub["worst_leaf"]["path"]),
            lb_source=lb.get("source"), milo_source=milo.get("source"), notes="; ".join(notes),
        )
        rows.append(row)
        ub_raw.append(dict(round=r, value=uv, root_choice=list(ub["root_choice"]),
                           root_choices=[[list(c), v] for c, v in ub_rc.items()],
                           worst_path=row["ub_event_order"], worst_d=ub["worst_leaf"]["d"],
                           stats=ub["stats"], time=ub_t, anticipation=len(ub["anticipation"])))

        line = (f"n={n} round {r:2d}: LB {fmt_val(lv)}  MILO {fmt_val(mv)}{'' if milo_ok else '*'}  UB {fmt_val(uv)}  "
                f"| 1st LB {row['lb_choice']} MILO {row['milo_choice']} UB {row['ub_choice']} ({row['ub_vs_milo']})  "
                f"| t_UB {ub_t:.2f}s  LPs {row['ub_gurobi_calls']}" + (f"  !! {row['notes']}" if notes else ""))
        print(line, flush=True)
        nlog.info(line)

    with open(os.path.join(n_dir, "ub_results.json"), "w") as f:
        json.dump(ub_raw, f)
    write_csv(os.path.join(n_dir, "results.csv"), rows)
    summary = summarize(rows, tol)
    nlog.info("")
    for line in instance_table(rows) + detail_lines(n, rows, summary, tol):
        nlog.info(line)
    close_logger(nlog)
    return rows, summary


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, tol):
    ok = [x for x in rows if x["milo_optimal"]]
    s = dict(total=len(rows), solved=len(ok), unproven=[x for x in rows if not x["milo_optimal"]])
    s["lb_eq"] = sum(abs(x["gap_milo_lb"]) <= tol for x in ok)
    s["lb_below"] = sum(x["gap_milo_lb"] > tol for x in ok)
    s["lb_viol"] = sum(x["gap_milo_lb"] < -tol for x in ok)
    s["ub_eq"] = sum(abs(x["gap_ub_milo"]) <= tol for x in ok)
    s["ub_above"] = sum(x["gap_ub_milo"] > tol for x in ok)
    s["ub_viol"] = sum(x["gap_ub_milo"] < -tol for x in ok)
    s["both_tight"] = sum(abs(x["gap_ub_milo"]) <= tol and abs(x["gap_milo_lb"]) <= tol for x in ok)
    s["lb_le_ub_viol"] = sum(not x["lb_le_ub"] for x in rows)
    for key in ("gap_milo_lb", "gap_milo_lb_pct", "gap_ub_milo", "gap_ub_milo_pct", "bracket_ub_lb", "bracket_pct"):
        s[key] = stats_of([x[key] for x in ok])
    for pair in ("ub_vs_milo", "lb_vs_milo", "ub_vs_lb"):
        vals = [x[pair] for x in ok if x[pair]]
        s[pair] = dict(compared=len(vals), same=vals.count("same"), tie_milo=vals.count("tie(milo)"),
                       tie_ub=vals.count("tie(ub)"), diff=vals.count("diff"))
    s["ub_regret"] = stats_of([x["ub_regret"] for x in ok])
    s["lb_regret"] = stats_of([x["lb_regret"] for x in ok])
    for key in ("lb_time", "milo_time", "ub_time", "lb_gurobi_calls", "ub_gurobi_calls", "milo_tree_lps",
                "milo_over_ub_time", "lb_over_ub_time", "ub_na_relaxed_nodes"):
        s[key] = stats_of([x[key] for x in ok])
    s["ub_faster_than_milo"] = sum(1 for x in ok if x["ub_time"] < (x["milo_time"] or 0))
    s["ub_faster_than_lb"] = sum(1 for x in ok if x["ub_time"] < (x["lb_time"] or 0))
    s["ub_fewer_lps_than_lb"] = sum(1 for x in ok if x["lb_gurobi_calls"] is not None
                                    and x["ub_gurobi_calls"] < x["lb_gurobi_calls"])
    s["ub_fewer_lps_than_milo"] = sum(1 for x in ok if x["milo_tree_lps"] is not None
                                      and x["ub_gurobi_calls"] < x["milo_tree_lps"])
    s["bounds_ok"] = s["lb_viol"] == 0 and s["ub_viol"] == 0 and s["lb_le_ub_viol"] == 0
    return s


def fs(st, key="mean", digits=4, suffix=""):
    if st is None:
        return "n/a"
    v = st[key]
    return f"{0.0 if abs(v) < 0.5 * 10 ** -digits else v:.{digits}f}{suffix}"   # no solver-noise -0.0000


def instance_table(rows):
    lines = ["PER-INSTANCE RESULTS (* = MILO not proven optimal)",
             f"  {'rnd':>3} | {'LB':>7} | {'MILO':>8} | {'UB':>7} | {'MILO-LB':>7} | {'UB-MILO':>7} | "
             f"{'LB 1st':>8} | {'MILO 1st':>8} | {'UB 1st':>8} | {'UB/MILO':>8} | {'t_LB':>6} | {'t_MILO':>8} | "
             f"{'t_UB':>6} | {'LPs LB':>6} | {'LPs UB':>6}"]
    for x in rows:
        lines.append(
            f"  {x['round']:>3} | {fmt_val(x['lb']):>7} | {fmt_val(x['milo']) + ('' if x['milo_optimal'] else '*'):>8} | "
            f"{fmt_val(x['ub']):>7} | {fmt_val(x['gap_milo_lb']):>7} | {fmt_val(x['gap_ub_milo']):>7} | "
            f"{x['lb_choice']:>8} | {x['milo_choice']:>8} | {x['ub_choice']:>8} | {x['ub_vs_milo']:>8} | "
            f"{x['lb_time'] or 0:6.2f} | {x['milo_time'] or 0:8.2f} | {x['ub_time']:6.2f} | "
            f"{str(x['lb_gurobi_calls']):>6} | {x['ub_gurobi_calls']:>6}")
    lines.append("")
    return lines


def detail_lines(n, rows, s, tol):
    lines = [f"DETAILS n={n}"]
    gaps = [x for x in rows if x["milo_optimal"] and (abs(x["gap_milo_lb"]) > tol or abs(x["gap_ub_milo"]) > tol)]
    lines.append(f"  Instances with a nonzero gap (MILO optimal): {len(gaps)}")
    for x in gaps:
        lines.append(f"    round {x['round']:2d}: LB {x['lb']:.4f}  MILO {x['milo']:.4f}  UB {x['ub']:.4f}  "
                     f"MILO-LB {fmt_pct(x['gap_milo_lb_pct'])}%  UB-MILO {fmt_pct(x['gap_ub_milo_pct'])}%  "
                     f"1st LB/MILO/UB {x['lb_choice']}/{x['milo_choice']}/{x['ub_choice']}")
    lines.append(f"  MILO not proven optimal (excluded from the statistics): {len(s['unproven'])}")
    for x in s["unproven"]:
        lo = max(x["lb"], x["milo"])
        lines.append(f"    round {x['round']:2d}: LB {x['lb']:.4f}, MILO incumbent {x['milo']:.4f} ({x['milo_status']}), "
                     f"UB {x['ub']:.4f} -> exact AR in [{lo:.4f}, {x['ub']:.4f}] (width {fmt_val(x['ub'] - lo)})")
    for model in ("ub", "lb"):
        key = f"{model}_vs_milo"
        diffs = [x for x in rows if x["milo_optimal"] and x[key] in ("diff", "tie(ub)")]
        lines.append(f"  {model.upper()} first decision different from MILO (exact-AR ties excluded): {len(diffs)}")
        for x in diffs:
            reg = f", regret under exact AR {x[f'{model}_regret']:.4f}" if x[f"{model}_regret"] is not None else ""
            kind = " (equal UB value; exact-AR value of this decision not available)" if x[key] == "tie(ub)" else ""
            lines.append(f"    round {x['round']:2d}: {model.upper()} {x[f'{model}_choice']} vs MILO {x['milo_choice']}{reg}{kind}")
    notes = [x for x in rows if x["notes"]]
    if notes:
        lines.append("  Re-run notes:")
        lines += [f"    round {x['round']:2d}: {x['notes']}" for x in notes]
    lines.append("")
    return lines


def summary_all_lines(results, args, tol):
    ns = list(results)
    w = 22
    head = lambda label: f"  {label:<44}" + "".join(f" | {'n=' + str(n):>{w}}" for n in ns)
    row = lambda label, f: f"  {label:<44}" + "".join(f" | {f(results[n][1]):>{w}}" for n in ns)
    dec = lambda pair: (lambda s: f"{s[pair]['same']} / {s[pair]['tie_milo']} / {s[pair]['tie_ub']} / {s[pair]['diff']}")
    L = ["=" * 78,
         f"BOUNDS SUMMARY: LB (ar_minimax) <= MILO (exact AR) <= UB (ar_minimax_ub), Gamma={args.gamma:g}, "
         f"{args.rounds} instances per n",
         "=" * 78,
         "  gap columns are relative to the exact MILO value; statistics use the instances where MILO is optimal",
         "", "VALUES", head("metric"),
         row("instances (MILO proven optimal / total)", lambda s: f"{s['solved']} / {s['total']}"),
         row("LB = MILO / LB < MILO / LB > MILO (viol.)", lambda s: f"{s['lb_eq']} / {s['lb_below']} / {s['lb_viol']}"),
         row("UB = MILO / UB > MILO / UB < MILO (viol.)", lambda s: f"{s['ub_eq']} / {s['ub_above']} / {s['ub_viol']}"),
         row("LB = MILO = UB (bracket closed)", lambda s: f"{s['both_tight']}"),
         row("mean / max  MILO - LB", lambda s: f"{fs(s['gap_milo_lb'])} / {fs(s['gap_milo_lb'], 'max')}"),
         row("mean / max  MILO - LB  (%)", lambda s: f"{fs(s['gap_milo_lb_pct'], digits=3)} / {fs(s['gap_milo_lb_pct'], 'max', 2)}"),
         row("mean / max  UB - MILO", lambda s: f"{fs(s['gap_ub_milo'])} / {fs(s['gap_ub_milo'], 'max')}"),
         row("mean / max  UB - MILO  (%)", lambda s: f"{fs(s['gap_ub_milo_pct'], digits=3)} / {fs(s['gap_ub_milo_pct'], 'max', 2)}"),
         row("mean / max  bracket UB - LB  (%)", lambda s: f"{fs(s['bracket_pct'], digits=3)} / {fs(s['bracket_pct'], 'max', 2)}"),
         row("theory check LB <= MILO <= UB", lambda s: "OK" if s["bounds_ok"] else "VIOLATED"),
         "", "FIRST DECISIONS: same / tie(milo) / tie(ub) / diff",
         "  tie(milo) = different task set, same exact AR value; tie(ub) = same UB value, exact-AR value unknown",
         head("pair"),
         row("UB vs MILO", dec("ub_vs_milo")),
         row("LB vs MILO", dec("lb_vs_milo")),
         row("UB vs LB", dec("ub_vs_lb")),
         row("mean / max regret of UB decision (exact AR)", lambda s: f"{fs(s['ub_regret'])} / {fs(s['ub_regret'], 'max')}"),
         row("mean / max regret of LB decision (exact AR)", lambda s: f"{fs(s['lb_regret'])} / {fs(s['lb_regret'], 'max')}"),
         "", "RUNTIME (s) AND LP CALLS", head("metric"),
         row("LB   time mean / median / max", lambda s: f"{fs(s['lb_time'], digits=2)} / {fs(s['lb_time'], 'median', 2)} / {fs(s['lb_time'], 'max', 2)}"),
         row("MILO time mean / median / max", lambda s: f"{fs(s['milo_time'], digits=1)} / {fs(s['milo_time'], 'median', 1)} / {fs(s['milo_time'], 'max', 1)}"),
         row("UB   time mean / median / max", lambda s: f"{fs(s['ub_time'], digits=2)} / {fs(s['ub_time'], 'median', 2)} / {fs(s['ub_time'], 'max', 2)}"),
         row("MILO time / UB time  mean / median", lambda s: f"{fs(s['milo_over_ub_time'], digits=1, suffix='x')} / {fs(s['milo_over_ub_time'], 'median', 1, 'x')}"),
         row("LB time / UB time  mean / median", lambda s: f"{fs(s['lb_over_ub_time'], digits=2, suffix='x')} / {fs(s['lb_over_ub_time'], 'median', 2, 'x')}"),
         row("UB faster than MILO / than LB", lambda s: f"{s['ub_faster_than_milo']} / {s['ub_faster_than_lb']} of {s['solved']}"),
         row("Gurobi calls LB / UB (mean)", lambda s: f"{fs(s['lb_gurobi_calls'], digits=1)} / {fs(s['ub_gurobi_calls'], digits=1)}"),
         row("MILO tree-build LPs (mean)", lambda s: fs(s["milo_tree_lps"], digits=1)),
         row("UB fewer LP calls than LB / than MILO", lambda s: f"{s['ub_fewer_lps_than_lb']} / {s['ub_fewer_lps_than_milo']} of {s['solved']}"),
         row("UB sched. nodes relaxing NA (mean)", lambda s: fs(s["ub_na_relaxed_nodes"], digits=1)),
         ""]
    for n in ns:
        L += detail_lines(n, results[n][0], results[n][1], tol)
    if args.lb_json or args.milo_json:
        L += ["NOTES",
              "  * LB and MILO values, times and LP counts are the saved ones (" +
              ", ".join(sorted({p for v in parse_paths(args.lb_json).values() for p in v} |
                               {p for v in parse_paths(args.milo_json).values() for p in v})) + ").",
              "    Those runs shared the machine with other runs, while the UB ran alone now, so compare the",
              "    LP-call counts for a machine-independent cost comparison.",
              "  * Missing first decisions were recovered by re-running the model only for the decision"
              + (" (--recover-decisions)." if args.recover_decisions else " (not requested: shown empty)."),
              "  * Ties use the exact MILO value of each first decision when available, else the UB values.",
              ""]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--n", type=int, nargs="+", required=True, help="numbers of tasks")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--lb-json", nargs="*", default=[], help="N=path[,path...] saved minimax (LB) rows")
    ap.add_argument("--milo-json", nargs="*", default=[], help="N=path[,path...] saved MILO rows")
    ap.add_argument("--recover-decisions", action="store_true",
                    help="re-run saved models whose first decision was not saved (values and times stay saved)")
    ap.add_argument("--out-dir", default=None, help="default: <log_dir>/bounds/gamma_<g>")
    ap.add_argument("--trace", action="store_true", help="write the full UB tree trace to instances/round_XX_ub.log")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tol = cfg.get("benchmark", {}).get("tolerance", 1e-4)
    gamma_tag = f"{args.gamma:g}".replace(".", "p")
    base = args.out_dir or os.path.join(cfg.get("logging", {}).get("log_dir", "logs"), "bounds", f"gamma_{gamma_tag}")
    run_dir = os.path.join(base, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    lb_paths, milo_paths = parse_paths(args.lb_json), parse_paths(args.milo_json)

    results = {}
    for n in args.n:
        results[n] = run_n(n, args, cfg, lb_paths.get(n), milo_paths.get(n), run_dir, tol)

    write_csv(os.path.join(run_dir, "results_all.csv"), [x for n in results for x in results[n][0]])
    out = setup_logger("bounds_all", os.path.join(run_dir, "summary_all.log"), console=True)
    for line in summary_all_lines(results, args, tol):
        out.info(line)
    out.info(f"Logs, results_all.csv and per-n folders in {run_dir}")
    close_logger(out)


if __name__ == "__main__":
    main()
