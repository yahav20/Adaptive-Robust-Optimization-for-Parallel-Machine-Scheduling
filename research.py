"""
Research experiment: Hybrid Minimax (upper bound) vs exact MILO across many instances.

For every instance (grid over n_tasks, n_machines, Gamma and random seeds) we record
    * values: Minimax, MILO, nominal (Gamma=0), gap, first decisions, regret
    * where the gap comes from: number of anticipation conflicts
    * the structure of the leaf that locks the Minimax value (L*):
        - its rank among all leaves (is the adversary simply choosing the worst LP leaf?)
        - budget use, fractional u's, tight sequence constraints (ties)
        - how much of the budget sits on the critical chain (tasks summed in the makespan)
    * every scheduler decision in the tree, to test simple dispatching rules (LPT, SPT, ...)

Output: logs/research_<stamp>/{instances.csv, decisions.csv, report.html}

    python3 research.py              # full grid (~170 instances, a few minutes)
    python3 research.py --focus      # ~420 instances where gaps occur (n=4, m=2, Gamma 1..2.5)
    python3 research.py --quick      # 2 seeds per cell, to test the pipeline
"""

import argparse
import csv
import dataclasses
import html
import math
import os
import random
import statistics as st
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from ar_common import load_config, make_env, random_instance, setup_logger, task_name, tasks_str
from ar_milo import MiloSolver
from ar_minimax import NEG_INF, MinimaxSolver
from visualize_tree import eval_minimax, find_anticipation, snapshot_milo, walk

TOL = 1e-4
# (family label, n_tasks, n_machines, d_bar range, gammas, seeds per gamma).
# n=5 is left out: with 2-3 machines most MILOs exceed the size-limited Gurobi license, and
# keeping only the ones that fit would bias the sample towards small trees.
GRID = [
    ("n=3, m=2", 3, 2, (0.0, 1.0), [0.5, 1.0, 1.5, 2.0, 3.0], 8),
    ("n=4, m=2", 4, 2, (0.0, 1.0), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], 12),
    ("n=4, m=2, wide d̄", 4, 2, (0.0, 3.0), [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], 8),
    ("n=4, m=3", 4, 3, (0.0, 1.0), [0.5, 1.0, 1.5, 2.0, 3.0], 8),
]
# Focused grid: the only region where gaps appeared in GRID (n - m >= 2, middle Gamma), with many seeds.
FOCUS = [
    ("n=4, m=2", 4, 2, (0.0, 1.0), [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], 40),
    ("n=4, m=2, wide d̄", 4, 2, (0.0, 3.0), [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], 20),
]
RULES = {
    "LPT (largest d0+d_bar)": lambda inst, i: inst.d0[i] + inst.d_bar[i],
    "largest nominal d0": lambda inst, i: inst.d0[i],
    "largest deviation d_bar": lambda inst, i: inst.d_bar[i],
    "SPT (smallest d0+d_bar)": lambda inst, i: -(inst.d0[i] + inst.d_bar[i]),
}


# ---------------------------------------------------------------------- certificate
def binding_conflicts(root):
    """Conflicts on the Minimax strategy tree only: keep every option at scheduler nodes, only the
    chosen child at adversary nodes. If there are none, the locked leaves' durations can be merged
    into one non-anticipative scenario, so MILO >= Minimax, i.e. Minimax is exact."""
    count, stack = 0, [root]
    while stack:
        x = stack.pop()
        if x.kind == "LEAF":
            continue
        if x.kind == "ADVERSARY":
            stack.append(x.mm_best)
            continue
        stack.extend(x.children)
        locked = [c.mm_leaf.mm_d for c in x.children if c.mm_leaf.mm_d]
        for i in x.finished:
            vals = [d[i] for d in locked]
            if vals and max(vals) - min(vals) > 1e-6:
                count += 1
    return count


def strategy_lower_bound(root, inst, env):
    """Fix the Minimax adversary strategy (chosen child at adversary nodes, every option at scheduler
    nodes) and solve ONE LP with non-anticipative durations (one u per task per finish edge, as in
    the MILO):  max t  s.t. every strategy-tree leaf is feasible and has makespan >= t.
    Any feasible point is a feasible MILO solution, so the optimum is a lower bound on the AR value."""
    n = inst.n_tasks
    m = gp.Model("strategy_LB", env=env)
    t = m.addVar(lb=-GRB.INFINITY, name="t")
    stack = [(root, {})]
    while stack:
        x, u_map = stack.pop()
        if x.finished_task is not None:
            u_map = dict(u_map)
            u_map[x.finished_task] = m.addVar(lb=0.0, ub=1.0)
        if x.kind == "LEAF":
            d = [inst.d0[i] + inst.d_bar[i] * u_map[i] for i in range(n)]
            m.addConstr(gp.quicksum(u_map[i] for i in range(n)) <= inst.gamma)
            for row in x.E:
                m.addConstr(gp.quicksum(row[i] * d[i] for i in range(n) if row[i]) <= 0)
            m.addConstr(t <= gp.quicksum(x.e[i] * d[i] for i in range(n) if x.e[i]))
        elif x.kind == "ADVERSARY":
            stack.append((x.mm_best, u_map))
        else:
            stack.extend((c, u_map) for c in x.children)
    m.setObjective(t, GRB.MAXIMIZE)
    m.optimize()
    val = m.ObjVal if m.Status == GRB.OPTIMAL else NEG_INF
    size = m.NumVars
    m.dispose()
    return val, size


# ---------------------------------------------------------------------- one instance
def analyze(inst, cfg, logger, family):
    n = inst.n_tasks
    t0 = time.perf_counter()
    mm_res = MinimaxSolver(inst, cfg.get("minimax", {}), logger, trace=False).solve()
    t_mm = time.perf_counter() - t0
    nominal = MinimaxSolver(dataclasses.replace(inst, gamma=0.0), {}, logger, trace=False).solve()["value"]

    milo = MiloSolver(inst, cfg.get("milo", {}), logger, trace=False)
    t0 = time.perf_counter()
    root = milo.build_tree({}, tuple(range(n)), (), [], [0] * n, min(inst.n_machines, n), [])
    for k, leaf in enumerate(x for x in walk(root) if x.kind == "LEAF"):
        leaf.lid = f"L{k + 1}"
    eval_minimax(root, MinimaxSolver(inst, {}, logger, trace=False), [])
    assert abs(root.mm - mm_res["value"]) < 1e-6
    try:
        milo.build_model(root)
        ml_val, ml_status, _ = milo._optimize(root.val, "full")
    except gp.GurobiError as e:
        return None, [], f"MILO skipped: {e}"
    t_ml = time.perf_counter() - t0
    t0 = time.perf_counter()
    lb, lb_vars = strategy_lower_bound(root, inst, milo.env)
    t_lb = time.perf_counter() - t0
    for c in root.children:
        c.ml_root = milo._optimize(c.val, "choice")[0]
        c.ml_leaf = snapshot_milo(c, milo)
    for x in walk(root):
        x.anticip = False

    leaves = [x for x in walk(root) if x.kind == "LEAF"]
    finite = [x.mm for x in leaves if x.mm != NEG_INF]
    mm_child = root.mm_best
    ml_child = min(root.children, key=lambda c: c.ml_root)
    Ls = root.mm_leaf
    u, d = Ls.mm_u, Ls.mm_d
    chain = [i for i in range(n) if Ls.e[i]]
    sum_u = sum(u)
    tight = sum(1 for r in Ls.E if abs(sum(r[i] * d[i] for i in range(n))) < 1e-6)
    sub_leaves = [x.mm for x in walk(mm_child) if x.kind == "LEAF" and x.mm != NEG_INF]
    anti_chosen = len(find_anticipation(mm_child))
    anti_total = anti_chosen + sum(len(find_anticipation(c)) for c in root.children if c is not mm_child)

    adv = [x for x in walk(root) if x.kind == "ADVERSARY" and len(x.children) > 1]
    adv_worst = [abs(x.mm - max(y.mm for y in walk(x) if y.kind == "LEAF")) < 1e-6 for x in adv]
    last = Ls.path[-1][1]
    longest = max(range(n), key=lambda i: inst.d0[i] + inst.d_bar[i])

    row = dict(
        family=family, n=n, m=inst.n_machines, gamma=inst.gamma, name=inst.name,
        d0=" ".join(f"{x:.4f}" for x in inst.d0), d_bar=" ".join(f"{x:.4f}" for x in inst.d_bar),
        minimax=root.mm, milo=ml_val, milo_status=ml_status, nominal=nominal,
        gap=root.mm - ml_val, gap_pct=100 * (root.mm - ml_val) / ml_val,
        has_gap=root.mm - ml_val > TOL,
        mm_choice=tasks_str(mm_child.path[-1][1]), milo_choice=tasks_str(ml_child.path[-1][1]),
        regret=mm_child.ml_root - ml_child.ml_root,
        anticip_chosen=anti_chosen, anticip_total=anti_total,
        binding_conflicts=binding_conflicts(root), certified_exact=binding_conflicts(root) == 0,
        strategy_lb=lb, lb_feasible=lb != NEG_INF, lb_gap=ml_val - lb, lb_certified=root.mm - lb <= TOL, lb_vars=lb_vars, t_lb=t_lb,
        n_leaves=len(leaves), n_scheduler=milo.stats["scheduler_nodes"], n_adversary=milo.stats["adversary_nodes"],
        max_leaf=max(finite), min_leaf=min(finite),
        lstar_share_above=sum(1 for v in finite if v > Ls.mm + 1e-6) / len(finite),
        lstar_is_global_max=abs(Ls.mm - max(finite)) < 1e-6,
        lstar_is_subtree_max=abs(Ls.mm - max(sub_leaves)) < 1e-6,
        adv_picks_worst_share=sum(adv_worst) / len(adv) if adv else 1.0,
        sum_u=sum_u, budget_saturated=sum_u >= min(inst.gamma, n) - 1e-6,
        n_u_frac=sum(1 for x in u if 1e-6 < x < 1 - 1e-6), n_u_full=sum(1 for x in u if x >= 1 - 1e-6),
        n_rows=len(Ls.E), n_tight=tight,
        budget_on_chain=sum(u[i] for i in chain) / sum_u if sum_u > 1e-9 else 1.0,
        chain_len=len(chain), last_task=task_name(last), last_is_longest=(last == longest),
        milo_leaf_is_lstar=(ml_child is mm_child and ml_child.ml_leaf is Ls),
        t_minimax=t_mm, t_milo=t_ml,
    )

    decisions = []
    for x in walk(root):
        if x.kind != "SCHEDULER" or len(x.children) < 2:
            continue
        best = min(c.mm for c in x.children)
        optimal = {c.path[-1][1] for c in x.children if c.mm - best < 1e-6}
        rec = dict(family=family, n=n, m=inst.n_machines, gamma=inst.gamma, name=inst.name, depth=len(x.path),
                   is_root=not x.path, n_options=len(x.children), n_optimal=len(optimal),
                   margin=sorted(c.mm for c in x.children)[1] - best)
        k = len(x.children[0].path[-1][1])
        for rule, key in RULES.items():
            pick = tuple(sorted(sorted(x.pending, key=lambda i: -key(inst, i))[:k]))
            rec[rule] = pick in optimal
            if rule.startswith("LPT"):
                rec["lpt_loss"] = next(c.mm for c in x.children if c.path[-1][1] == pick) - best
        decisions.append(rec)
    return row, decisions, None


def run(cfg, grid, quick, logger, progress):
    rows, decisions = [], []
    seed_base = 100_000 if grid is FOCUS else 0      # never reuse the main grid's instances
    for ci, (fam, n, m, dbar, gammas, reps) in enumerate(grid):
        for g in gammas:
            for r in range(2 if quick else reps):
                c = dict(cfg)
                c["problem"] = dict(n_tasks=n, n_machines=m, gamma=g)
                c["random_instance"] = dict(cfg["random_instance"], d_bar_range=list(dbar))
                seed = seed_base + 10_000 * ci + 1000 * int(g * 4) + r
                inst = random_instance(c, random.Random(seed), name=f"F{ci}-n{n}m{m}G{g}s{seed}")
                row, dec, err = analyze(inst, cfg, logger, fam)
                if err:
                    progress(f"  {inst.name}: {err}")
                    continue
                rows.append(row)
                decisions += dec
                progress(f"  {inst.name}: minimax={row['minimax']:.4f} milo={row['milo']:.4f} "
                         f"gap={row['gap']:+.4f} anticip={row['anticip_chosen']}")
    return rows, decisions


# ---------------------------------------------------------------------- SVG charts
INK, INK2, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAY = "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def esc(s):
    return html.escape(str(s))


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    step = 10 ** math.floor(math.log10(raw))
    for f in (1, 2, 2.5, 5, 10):
        if raw <= f * step:
            step *= f
            break
    t = math.floor(lo / step) * step
    out = []
    while t <= hi + step * 1e-9:
        out.append(round(t, 10))
        t += step
    if out[-1] < hi:
        out.append(round(out[-1] + step, 10))
    return out


def fmt_tick(v):
    return f"{v:g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.0e}"


class Plot:
    """Minimal XY plot: linear or log axes, hairline grid, one y axis."""

    def __init__(self, xlo, xhi, ylo, yhi, w=560, h=330, xlog=False, xlabel="", ylabel="", xticks=None):
        self.w, self.h, self.xlog = w, h, xlog
        self.ml, self.mr, self.mt, self.mb = 58, 110, 14, 46
        if xlog:
            self.xt = xticks or [10 ** k for k in range(math.floor(math.log10(xlo)), math.ceil(math.log10(xhi)) + 1)]
        else:
            self.xt = xticks or nice_ticks(xlo, xhi)
        self.yt = nice_ticks(ylo, yhi)
        self.x0, self.x1 = (self.xt[0], self.xt[-1])
        self.y0, self.y1 = self.yt[0], self.yt[-1]
        self.parts = []
        self.xlabel, self.ylabel = xlabel, ylabel
        self.label_ys = []

    def X(self, x):
        pw = self.w - self.ml - self.mr
        if self.xlog:
            return self.ml + pw * (math.log10(x) - math.log10(self.x0)) / (math.log10(self.x1) - math.log10(self.x0))
        return self.ml + pw * (x - self.x0) / (self.x1 - self.x0)

    def Y(self, y):
        ph = self.h - self.mt - self.mb
        return self.mt + ph * (1 - (y - self.y0) / (self.y1 - self.y0))

    def add(self, s):
        self.parts.append(s)

    def svg(self):
        g = []
        for t in self.yt:
            y = self.Y(t)
            g.append(f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRIDC}"/>'
                     f'<text x="{self.ml - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt_tick(t)}</text>')
        for t in self.xt:
            x = self.X(t)
            g.append(f'<text x="{x:.1f}" y="{self.h - self.mb + 16}" text-anchor="middle" class="tick">{fmt_tick(t)}</text>')
        g.append(f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.h - self.mb}" y2="{self.h - self.mb}" stroke="{MUTED}"/>')
        g.append(f'<text x="{(self.ml + self.w - self.mr) / 2}" y="{self.h - 8}" text-anchor="middle" class="axl">{esc(self.xlabel)}</text>')
        g.append(f'<text transform="translate(14,{(self.mt + self.h - self.mb) / 2}) rotate(-90)" text-anchor="middle" class="axl">{esc(self.ylabel)}</text>')
        return f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">{"".join(g)}{"".join(self.parts)}</svg>'


def dot(p, x, y, color, tip, r=4.5):
    p.add(f'<circle cx="{p.X(x):.1f}" cy="{p.Y(y):.1f}" r="{r}" fill="{color}" stroke="{SURF}" stroke-width="2" '
          f'class="hov"><title>{esc(tip)}</title></circle>')


def line(p, pts, color, label, tips):
    d = " ".join(f"{'M' if k == 0 else 'L'}{p.X(x):.1f},{p.Y(y):.1f}" for k, (x, y) in enumerate(pts))
    p.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    for (x, y), t in zip(pts, tips):
        dot(p, x, y, color, t, r=4)
    x, y = pts[-1]
    ly = p.Y(y) + 4
    while any(abs(ly - o) < 13 for o in p.label_ys):    # nudge direct labels apart
        ly += 13
    p.label_ys.append(ly)
    p.add(f'<text x="{p.X(x) + 10:.1f}" y="{ly:.1f}" class="dl">{esc(label)}</text>')


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{esc(t)}</span>' for t, c in items) + "</div>"


def hbar_chart(items, w=620, unit="%", color=SERIES[0], vmax=100):
    """items: (label, value, tooltip). One series -> one color."""
    bh, gap, lw = 22, 8, 270
    h = len(items) * (bh + gap) + 10
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    pw = w - lw - 60
    for k, (lab, v, tip) in enumerate(items):
        y = 5 + k * (bh + gap)
        bw = max(2, pw * v / vmax)
        out.append(f'<text x="{lw - 8}" y="{y + bh / 2 + 4}" text-anchor="end" class="lab">{esc(lab)}</text>'
                   f'<rect x="{lw}" y="{y}" width="{pw}" height="{bh}" fill="{GRIDC}" opacity=".45" rx="4"/>'
                   f'<rect x="{lw}" y="{y}" width="{bw:.1f}" height="{bh}" fill="{color}" rx="4" class="hov"><title>{esc(tip)}</title></rect>'
                   f'<text x="{lw + bw + 6:.1f}" y="{y + bh / 2 + 4}" class="val">{v:.0f}{unit}</text>')
    out.append("</svg>")
    return "".join(out)


def heatmap(rows, cols, cell, w=620):
    """cell(r, c) -> (value 0..1 or None, label, tooltip)."""
    cw, ch, lw, top = (w - 130) / len(cols), 40, 130, 24
    h = top + len(rows) * ch + 8
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for j, c in enumerate(cols):
        out.append(f'<text x="{lw + cw * (j + .5):.1f}" y="16" text-anchor="middle" class="tick">Γ={c:g}</text>')
    for i, r in enumerate(rows):
        out.append(f'<text x="{lw - 8}" y="{top + ch * i + ch / 2 + 4}" text-anchor="end" class="lab">{esc(r)}</text>')
        for j, c in enumerate(cols):
            v, lab, tip = cell(r, c)
            x, y = lw + cw * j, top + ch * i
            if v is None:
                out.append(f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{cw - 2:.1f}" height="{ch - 2}" fill="#f1f0ec" rx="3"/>'
                           f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4}" text-anchor="middle" class="tick">-</text>')
                continue
            fill = SEQ[min(len(SEQ) - 1, int(v * (len(SEQ) - 1) + 0.5))]
            ink = "#ffffff" if v > 0.5 else INK
            out.append(f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{cw - 2:.1f}" height="{ch - 2}" fill="{fill}" rx="3" class="hov">'
                       f'<title>{esc(tip)}</title></rect><text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4}" text-anchor="middle" '
                       f'style="fill:{ink};font-size:12px">{esc(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_cols(cats, parts, w=560, h=300, ylabel=""):
    """parts: list of (label, color, [value per cat]) - shares summing to 100."""
    ml, mb, mt = 50, 40, 10
    pw, ph = w - ml - 20, h - mb - mt
    bw = pw / len(cats) * 0.62
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for t in (0, 25, 50, 75, 100):
        y = mt + ph * (1 - t / 100)
        out.append(f'<line x1="{ml}" x2="{w - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRIDC}"/>'
                   f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{t}%</text>')
    for j, c in enumerate(cats):
        x = ml + pw / len(cats) * (j + 0.19)
        base = 0
        for lab, color, vals in parts:
            v = vals[j]
            y1, y0 = mt + ph * (1 - (base + v) / 100), mt + ph * (1 - base / 100)
            if v > 0:
                out.append(f'<rect x="{x:.1f}" y="{y1 + 1:.1f}" width="{bw:.1f}" height="{max(0, y0 - y1 - 2):.1f}" fill="{color}" '
                           f'rx="3" class="hov"><title>{esc(f"{c}: {lab} {v:.0f}%")}</title></rect>')
            base += v
        out.append(f'<text x="{x + bw / 2:.1f}" y="{h - mb + 16}" text-anchor="middle" class="tick">{esc(c)}</text>')
    out.append(f'<text x="{ml + pw / 2}" y="{h - 6}" text-anchor="middle" class="axl">{esc(ylabel)}</text></svg>')
    return "".join(out)


# ---------------------------------------------------------------------- report
def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def pct(xs):
    xs = list(xs)
    return 100 * sum(1 for x in xs if x) / len(xs) if xs else float("nan")


def report(rows, decisions, path, elapsed, grid):
    configs = [g[0] for g in grid]
    gammas = sorted({r["gamma"] for r in rows})
    gapped = [r for r in rows if r["has_gap"]]
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[(r["family"], r["gamma"])].append(r)

    # --- KPI tiles
    same = pct(abs(r["regret"]) <= TOL for r in rows)
    kpis = [("instances", f"{len(rows)}", f"{len(configs)} (n,m) configs × Γ grid"),
            ("instances with a gap", f"{pct(r['has_gap'] for r in rows):.0f}%", f"{len(gapped)} of {len(rows)} (gap > {TOL})"),
            ("mean gap when > 0", f"{mean(r['gap_pct'] for r in gapped):.2f}%" if gapped else "-",
             f"max {max((r['gap_pct'] for r in gapped), default=0):.2f}%"),
            ("first decision is AR-optimal", f"{same:.0f}%", f"max regret {max(r['regret'] for r in rows):.4f}"),
            ("Minimax ≥ MILO", f"{pct(r['gap'] >= -TOL for r in rows):.0f}%", "upper-bound property")]
    kpi_html = '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kl">{esc(a)}</div><div class="kv">{esc(b)}</div><div class="ks">{esc(c)}</div></div>'
        for a, b, c in kpis) + "</div>"

    # --- 1 scatter MILO vs Minimax
    lo = min(min(r["milo"] for r in rows), min(r["minimax"] for r in rows))
    hi = max(max(r["milo"] for r in rows), max(r["minimax"] for r in rows))
    p = Plot(lo, hi, lo, hi, xlabel="MILO (exact AR) makespan", ylabel="Minimax makespan", w=520, h=400)
    p.mr = 20
    p.add(f'<line x1="{p.X(p.x0):.1f}" y1="{p.Y(p.x0):.1f}" x2="{p.X(min(p.x1, p.y1)):.1f}" y2="{p.Y(min(p.x1, p.y1)):.1f}" stroke="{MUTED}"/>')
    for r in sorted(rows, key=lambda r: r["has_gap"]):
        dot(p, r["milo"], r["minimax"], SERIES[0] if r["has_gap"] else GRAY,
            f"{r['name']}\nMILO {r['milo']:.4f}  Minimax {r['minimax']:.4f}  gap {r['gap']:+.4f} ({r['gap_pct']:.2f}%)")
    chart1 = p.svg() + legend([("gap > 1e-4", SERIES[0]), ("equal (on the diagonal)", GRAY)])

    # --- 2 heatmap share of instances with gap
    def cell(rc, g):
        rs = by_cfg.get((rc, g))
        if not rs:
            return None, "", ""
        k = sum(r["has_gap"] for r in rs)
        mg = mean(r["gap_pct"] for r in rs)
        return k / len(rs), f"{k}/{len(rs)}", f"{rc}, Γ={g}: {k} of {len(rs)} with gap, mean gap {mg:.3f}%"
    chart2 = heatmap(configs, gammas, cell)

    # --- 3 mean gap% vs gamma per config
    ymax = max(mean(max(0, r["gap_pct"]) for r in rs) for rs in by_cfg.values())
    p = Plot(min(gammas), max(gammas), 0, max(ymax * 1.1, 0.01), xlabel="Γ (uncertainty budget)", ylabel="mean gap (% of MILO)")
    p.mr = 130
    for k, fam in enumerate(configs):
        pts = [(g, mean(max(0, r["gap_pct"]) for r in by_cfg[(fam, g)])) for g in gammas if by_cfg.get((fam, g))]
        line(p, pts, SERIES[k], fam, [f"{fam}, Γ={g}: mean gap {v:.3f}%" for g, v in pts])
    chart3 = p.svg() + legend([(fam, SERIES[k]) for k, fam in enumerate(configs)])

    # --- 4 gap vs anticipation conflicts
    amax = max(r["anticip_chosen"] for r in rows)
    p = Plot(0, max(1, amax), 0, max(max(r["gap_pct"] for r in rows) * 1.1, 0.01),
             xlabel="anticipation conflicts in the chosen first-decision subtree", ylabel="gap (% of MILO)", w=520,
             xticks=list(range(max(1, amax) + 1)))
    p.mr = 20
    for r in rows:
        jit = (hash(r["name"]) % 100 - 50) / 250
        dot(p, max(0, r["anticip_chosen"] + jit), max(0, r["gap_pct"]), SERIES[0] if r["has_gap"] else GRAY,
            f"{r['name']}: {r['anticip_chosen']} conflicts, gap {r['gap_pct']:.3f}%", r=4)
    chart4 = p.svg() + legend([("gap > 1e-4", SERIES[0]), ("no gap", GRAY)])
    # --- 4b sandwich: Minimax - MILO vs MILO - LB per instance (both in % of MILO)
    fin = [r for r in rows if r["lb_feasible"]]
    p = Plot(0, max(max((r["minimax"] - r["milo"]) / r["milo"] * 100 for r in rows), 0.01) * 1.05,
             0, max(max((r["milo"] - r["strategy_lb"]) / r["milo"] * 100 for r in fin), 0.01) * 1.05,
             xlabel="Minimax − MILO (% of MILO)", ylabel="MILO − strategy LB (% of MILO)", w=520)
    p.mr = 20
    for r in fin:
        up, lo_ = max(0, r["gap_pct"]), max(0, (r["milo"] - r["strategy_lb"]) / r["milo"] * 100)
        dot(p, up, lo_, SERIES[0] if r["lb_certified"] else SERIES[1],
            f"{r['name']}: Minimax {r['minimax']:.4f}, MILO {r['milo']:.4f}, LB {r['strategy_lb']:.4f}", r=4)
    n_cert = sum(r["lb_certified"] for r in fin)
    chart4b = p.svg() + legend([(f"LB = Minimax, certified exact ({n_cert}, all at 0,0)", SERIES[0]),
                                (f"not certified ({len(fin) - n_cert})", SERIES[1])])

    # --- 5 hypotheses about L*
    hyp = [
        ("certificate: no conflict on strategy tree", pct(r["certified_exact"] for r in rows)),
        ("certificate: strategy LP bound = Minimax", pct(r["lb_certified"] for r in rows)),
        ("L* is the worst leaf of the whole tree", pct(r["lstar_is_global_max"] for r in rows)),
        ("L* is the worst leaf of its first-decision subtree", pct(r["lstar_is_subtree_max"] for r in rows)),
        ("adversary node = worst leaf below it (avg)", 100 * mean(r["adv_picks_worst_share"] for r in rows)),
        ("budget fully used at L* (Σu = min(Γ,n))", pct(r["budget_saturated"] for r in rows)),
        ("≥1 order constraint tight at L* (a tie)", pct(r["n_tight"] > 0 for r in rows)),
        ("#fractional u ≤ 1 + #ties at L*", pct(r["n_u_frac"] <= 1 + r["n_tight"] for r in rows)),
        ("all budget on the critical chain", pct(r["budget_on_chain"] > 1 - 1e-6 for r in rows)),
        ("task finishing last is the longest task", pct(r["last_is_longest"] for r in rows)),
        ("MILO worst-case leaf is L* too", pct(r["milo_leaf_is_lstar"] for r in rows)),
    ]
    chart5 = hbar_chart([(a, v, f"{a}: {v:.1f}% of {len(rows)} instances") for a, v in hyp])

    # --- 6 histogram: share of leaves above L*
    bins = [0, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
    names = ["0 (worst)", "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", ">60%"]
    counts = [sum(1 for r in rows if bins[k] <= r["lstar_share_above"] < bins[k + 1]) for k in range(len(names))]
    chart6 = hbar_chart([(f"{nm} of leaves above L*", 100 * c / len(rows), f"{c} instances") for nm, c in zip(names, counts)])

    # --- 7 budget on/off critical chain per gamma (n=4, m=2)
    cats, on, off = [], [], []
    for g in gammas:
        rs = by_cfg.get(("n=4, m=2", g))
        if rs:
            cats.append(f"Γ={g:g}")
            v = 100 * mean(r["budget_on_chain"] for r in rs)
            on.append(v)
            off.append(100 - v)
    chart7 = stacked_cols(cats, [("on critical chain", SERIES[0], on), ("off chain (forces the order)", SERIES[1], off)],
                          ylabel="share of the used budget Σu at L*  (n=4, m=2)") + \
        legend([("on the critical chain (tasks in the makespan sum)", SERIES[0]), ("off the chain: delays other tasks to force a bad order", SERIES[1])])

    # --- 8 dispatching rules
    ditems = []
    chance = mean(d["n_optimal"] / d["n_options"] for d in decisions)
    for rule in RULES:
        v = pct(d[rule] for d in decisions)
        v_root = pct(d[rule] for d in decisions if d["is_root"])
        ditems.append((rule, v, f"{rule}: optimal in {v:.1f}% of {len(decisions)} decisions (root only: {v_root:.1f}%)"))
    ditems.append(("random pick (chance)", 100 * chance, "expected share of optimal options"))
    chart8 = hbar_chart(ditems)
    root_rule = {rule: pct(d[rule] for d in decisions if d["is_root"]) for rule in RULES}

    # --- 9 price of robustness
    p = Plot(min(gammas), max(gammas), 1, max(max(r["milo"] / r["nominal"] for r in rows), 1.1),
             xlabel="Γ (uncertainty budget)", ylabel="MILO makespan / nominal makespan")
    p.mr = 130
    for k, fam in enumerate(configs):
        pts = [(g, mean(r["milo"] / r["nominal"] for r in by_cfg[(fam, g)])) for g in gammas if by_cfg.get((fam, g))]
        line(p, pts, SERIES[k], fam, [f"{fam}, Γ={g}: MILO/nominal {v:.3f}" for g, v in pts])
    chart9 = p.svg() + legend([(fam, SERIES[k]) for k, fam in enumerate(configs)])

    # --- 10 run time
    tmax = max(max(r["t_milo"] for r in rows), max(r["t_minimax"] for r in rows))
    p = Plot(min(r["n_leaves"] for r in rows), max(r["n_leaves"] for r in rows), 0, tmax * 1.1, xlog=True,
             xlabel="number of leaves in the tree (log scale)", ylabel="time (s)")
    for r in rows:
        dot(p, r["n_leaves"], r["t_minimax"], SERIES[0], f"{r['name']}: Minimax {r['t_minimax']:.3f}s", r=3.5)
        dot(p, r["n_leaves"], r["t_milo"], SERIES[1], f"{r['name']}: MILO {r['t_milo']:.3f}s", r=3.5)
    chart10 = p.svg() + legend([("Minimax (tree + leaf LPs)", SERIES[0]), ("MILO (tree + model + all solves)", SERIES[1])])

    # --- findings (computed text)
    gap_g = {g: pct(r["has_gap"] for r in rows if r["gamma"] == g) for g in gammas}
    corr_rows = [r for r in rows]
    gap_when_conf = pct(r["has_gap"] for r in rows if r["anticip_chosen"] > 0)
    def frac(cond, pool):
        pool = list(pool)
        k = sum(1 for r in pool if cond(r))
        return f"{k}/{len(pool)} ({100 * k / len(pool):.1f}%)" if pool else "0/0"

    adaptive = [r for r in rows if r["n"] - r["m"] >= 2]
    nonadaptive = [r for r in rows if r["n"] - r["m"] <= 1]
    lpt_wrong = [d for d in decisions if not d["LPT (largest d0+d_bar)"]]
    findings = [
        f"<b>Gap frequency.</b> Minimax = MILO (within {TOL}) in {frac(lambda r: not r['has_gap'], rows)} instances. "
        + (f"Gapped instances: mean gap {mean(r['gap_pct'] for r in gapped):.2f}%, max {max(r['gap_pct'] for r in gapped):.2f}% of the MILO value."
           if gapped else "No gap found."),
        "" if not nonadaptive else f"<b>No adaptive choice &rArr; no gap.</b> When n − m ≤ 1, only one task is left after the first decision, so the scheduler never "
        f"chooses again and the leaf LPs cannot anticipate anything: gap in {frac(lambda r: r['has_gap'], nonadaptive)}, "
        f"anticipation conflicts in {frac(lambda r: r['anticip_total'] > 0, nonadaptive)}. With n − m ≥ 2: gap in "
        f"{frac(lambda r: r['has_gap'], adaptive)}, conflicts in {frac(lambda r: r['anticip_total'] > 0, adaptive)}.",
        "<b>Gap vs Γ</b> (instances with gap / instances): " + ", ".join(
            f"Γ={g:g}: {frac(lambda r: r['has_gap'], [r for r in rows if r['gamma'] == g])}" for g in gammas) + ".",
        f"<b>Conflicts are necessary but not sufficient.</b> Gapped instances with a conflict in the chosen subtree: "
        f"{frac(lambda r: r['anticip_chosen'] > 0, gapped)}. Instances with a conflict that have a gap: "
        f"{frac(lambda r: r['has_gap'], [r for r in rows if r['anticip_chosen'] > 0])} - most conflicts are on options the scheduler "
        f"would not pick anyway, so they do not move the MIN.",
        f"<b>A cheap exactness certificate.</b> Count conflicts only on the Minimax strategy tree (all options at scheduler nodes, only the "
        f"chosen child at adversary nodes). With no such conflict the leaf durations merge into one non-anticipative scenario, so "
        f"Minimax = MILO without solving the MILO. Certified: {frac(lambda r: r['certified_exact'], rows)}; certified instances with a gap "
        f"(should be 0): {frac(lambda r: r['has_gap'], [r for r in rows if r['certified_exact']])}; uncertified instances that are exact anyway "
        f"(alternative LP optima can resolve a conflict): {frac(lambda r: not r['has_gap'], [r for r in rows if not r['certified_exact']])}. "
        f"Gapped instances whose conflict lies in another first decision (MILO picks a different first move): "
        f"{frac(lambda r: r['anticip_chosen'] == 0, gapped)}.",
        f"<b>Sandwich with one extra LP.</b> Fixing the Minimax adversary strategy and solving one LP with shared durations gives a "
        f"lower bound LB ≤ MILO ≤ Minimax. LB ≤ MILO + {TOL} holds in {frac(lambda r: r['strategy_lb'] <= r['milo'] + TOL, rows)} (validity check). "
        f"LB = Minimax (certified exact) in {frac(lambda r: r['lb_certified'], rows)}; certified but gapped (should be 0): "
        f"{frac(lambda r: r['has_gap'], [r for r in rows if r['lb_certified']])}. LB = MILO in {frac(lambda r: r['lb_gap'] <= TOL, rows)}. "
        f"The fixed strategy cannot be realized with shared durations (LP infeasible, LB = −∞) in "
        f"{frac(lambda r: not r['lb_feasible'], rows)}. Otherwise the mean bracket width Minimax − LB is "
        f"{mean(r['minimax'] - r['strategy_lb'] for r in rows if r['lb_feasible']):.4f} "
        f"(max {max((r['minimax'] - r['strategy_lb'] for r in rows if r['lb_feasible']), default=0):.4f}); LP size ≤ {max(r['lb_vars'] for r in rows)} variables, "
        f"mean time {1000 * mean(r['t_lb'] for r in rows):.1f} ms.",
        f"<b>The first decision is robust to the relaxation.</b> Minimax's first decision is also AR-optimal in "
        f"{frac(lambda r: abs(r['regret']) <= TOL, rows)} instances; max regret {max(r['regret'] for r in rows):.5f}.",
        f"<b>The adversary does not simply get the worst LP leaf.</b> L* is the worst leaf of the whole tree in "
        f"{frac(lambda r: r['lstar_is_global_max'], rows)} and of its own first-decision subtree in {frac(lambda r: r['lstar_is_subtree_max'], rows)}. "
        f"On average {100 * mean(r['lstar_share_above'] for r in rows):.0f}% of the leaves have a larger LP value than L*; "
        f"Minimax / worst leaf = {mean(r['minimax'] / r['max_leaf'] for r in rows):.3f}. That ratio is the value of adapting: the scheduler's "
        f"later MIN choices cut off the worst sequences.",
        f"<b>Shape of the locked leaf.</b> Budget fully used (Σu = min(Γ,n)) in {frac(lambda r: r['budget_saturated'], rows)}; when it is not, "
        f"the remaining budget cannot raise the makespan without breaking the event order. ≥1 tight order constraint (a tie of two "
        f"completion times) in {frac(lambda r: r['n_tight'] > 0, rows)}. Every fractional u is paid for by a tight constraint "
        f"(#fractional u ≤ 1 + #ties, the LP-vertex count) in {frac(lambda r: r['n_u_frac'] <= 1 + r['n_tight'], rows)}: the adversary pushes "
        f"tasks to their max (u=1) and uses fractional u only to land exactly on a tie.",
        f"<b>Budget off the critical chain.</b> All of Σu on the tasks summed in the makespan in {frac(lambda r: r['budget_on_chain'] > 1 - 1e-6, rows)}; "
        f"on average {100 - 100 * mean(r['budget_on_chain'] for r in rows):.0f}% of the used budget lengthens other tasks, which only "
        f"matters through the order constraints: it forces the scheduler to start the next task on the worse machine.",
        f"<b>LPT is almost the optimal scheduler.</b> Starting the pending task(s) with the largest d0 + d̄ is optimal at "
        f"{frac(lambda d: d['LPT (largest d0+d_bar)'], decisions)} scheduler decisions (chance: {100 * mean(d['n_optimal'] / d['n_options'] for d in decisions):.0f}%). "
        + (f"When it is wrong, it loses on average {mean(d['lpt_loss'] for d in lpt_wrong):.4f} (max {max(d['lpt_loss'] for d in lpt_wrong):.4f}) "
           f"of worst-case makespan." if lpt_wrong else "")
        + " Other rules: " + ", ".join(f"{a} {v:.0f}%" for a, v, _ in ditems if not a.startswith("LPT")) + ".",
    ]

    table_cols = ["family", "name", "gamma", "minimax", "milo", "gap_pct", "mm_choice", "milo_choice", "regret", "anticip_chosen",
                  "lstar_share_above", "sum_u", "n_tight", "budget_on_chain", "last_task"]
    trows = "".join("<tr>" + "".join(f"<td>{r[c]:.4f}</td>" if isinstance(r[c], float) else f"<td>{esc(r[c])}</td>"
                                     for c in table_cols) + "</tr>" for r in rows)

    findings = [f for f in findings if f]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Minimax vs MILO - research report</title>
<style>
.viz-root{{color-scheme:light;background:{SURF};color:{INK};font-family:Helvetica,Arial,sans-serif;margin:0;padding:24px 32px;max-width:1300px}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:18px;margin:34px 0 4px}}p.sub{{color:{INK2};margin:0 0 10px;font-size:14px;max-width:820px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}}.kpi{{border:1px solid {GRIDC};border-radius:8px;padding:10px 14px;min-width:200px;background:#fff}}
.kl{{font-size:12px;color:{INK2}}}.kv{{font-size:30px;font-weight:bold;margin:2px 0}}.ks{{font-size:12px;color:{MUTED}}}
.grid{{display:flex;flex-wrap:wrap;gap:26px}}.card{{background:#fff;border:1px solid {GRIDC};border-radius:8px;padding:12px 14px}}
.card h3{{font-size:15px;margin:0 0 2px}}.card p{{font-size:13px;color:{INK2};margin:0 0 8px;max-width:600px}}
.tick{{font-size:11px;fill:{INK2}}}.axl{{font-size:12px;fill:{INK2}}}.lab{{font-size:12px;fill:{INK}}}.val{{font-size:12px;fill:{INK2}}}.dl{{font-size:12px;fill:{INK}}}
.hov:hover{{opacity:.75}}.legend{{font-size:12px;color:{INK2};margin-top:4px}}.legend span{{margin-right:14px}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}}
ol li{{margin:8px 0;font-size:14px;max-width:1000px}}table{{border-collapse:collapse;font-size:11px}}td,th{{border:1px solid {GRIDC};padding:2px 5px}}
</style></head><body class="viz-root">
<h1>Hybrid Minimax vs exact MILO &mdash; {len(rows)} experiments</h1>
<p class="sub">Durations d<sub>i</sub> = d0<sub>i</sub> + d&#772;<sub>i</sub>u<sub>i</sub>, d0, d&#772; ~ U(0,1), Σu ≤ Γ. Gap = Minimax − MILO (≥ 0 by theory).
L* = the leaf whose LP value is passed up to the root by the Minimax (the "locked" leaf). Run time {elapsed:.0f}s.
Hover any mark for its numbers.</p>
{kpi_html}
<h2>Findings</h2><ol>{"".join(f"<li>{f}</li>" for f in findings)}</ol>

<h2>1. How far apart are the two models?</h2>
<div class="grid">
<div class="card"><h3>Minimax vs MILO, one dot per instance</h3><p>On the diagonal = identical. Above it = Minimax is pessimistic.</p>{chart1}</div>
<div class="card"><h3>Instances with a gap, by size and Γ</h3><p>Cell = instances with gap / instances run. Darker = more often.</p>{chart2}
<h3 style="margin-top:18px">Mean gap vs Γ</h3><p>Average over all instances in the cell (zeros included).</p>{chart3}</div>
</div>

<h2>2. Where does the gap come from?</h2>
<div class="grid">
<div class="card"><h3>Gap vs number of anticipation conflicts</h3><p>A conflict = a scheduler node where the options' locked leaves give an
already-finished task different durations. Small horizontal jitter added.</p>{chart4}</div>
<div class="card"><h3>Sandwich: LB ≤ MILO ≤ Minimax</h3><p>LB = one LP over the Minimax strategy tree with shared durations.
Infeasible strategy LPs (LB = −∞) are left out. Dot at (0,0) = all three equal. x = how pessimistic Minimax is, y = how far the cheap lower bound is below the truth.</p>{chart4b}</div>
<div class="card"><h3>Hypotheses about the locked leaf L*</h3><p>Share of instances where the statement holds.</p>{chart5}</div>
</div>

<h2>3. What is common to the locked leaves?</h2>
<div class="grid">
<div class="card"><h3>Rank of L* among all leaves</h3><p>Share of leaves whose LP value is larger than L*. "0" would mean the adversary simply
gets the worst LP leaf.</p>{chart6}</div>
<div class="card"><h3>Where the adversary spends its budget at L*</h3><p>Critical chain = the tasks whose durations add up to the makespan.</p>{chart7}</div>
</div>

<h2>4. Does the optimal scheduler follow a simple rule?</h2>
<div class="grid"><div class="card"><h3>Share of scheduler decisions where the rule's pick is optimal</h3>
<p>Over all {len(decisions)} scheduler nodes with ≥2 options (ties count as optimal). Chance = picking an option at random.</p>{chart8}</div>
<div class="card"><h3>Price of robustness (MILO / nominal makespan)</h3><p>Nominal = optimal makespan with Γ=0.</p>{chart9}</div></div>

<h2>5. Cost</h2><div class="card"><h3>Run time vs tree size</h3><p>Same axis for both (seconds).</p>{chart10}</div>

<h2>Data</h2><details><summary>all instances (also in instances.csv)</summary><table><tr>{"".join(f"<th>{c}</th>" for c in table_cols)}</tr>{trows}</table></details>
</body></html>"""
    with open(path, "w") as f:
        f.write(page)
    return findings


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--focus", action="store_true", help="many seeds in the region where gaps occur (n=4, m=2)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["milo"] = dict(cfg.get("milo", {}), write_gurobi_log=False)
    tag = "focus" if args.focus else "main"
    out_dir = os.path.join(cfg.get("logging", {}).get("log_dir", "logs"), f"research_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger("research", os.path.join(out_dir, "solvers.log"))
    t0 = time.perf_counter()
    grid = FOCUS if args.focus else GRID
    rows, decisions = run(cfg, grid, args.quick, logger, print)
    write_csv(os.path.join(out_dir, "instances.csv"), rows)
    write_csv(os.path.join(out_dir, "decisions.csv"), decisions)
    findings = report(rows, decisions, os.path.join(out_dir, "report.html"), time.perf_counter() - t0, grid)
    print()
    for f in findings:
        print(" *", html.unescape(f.replace("<b>", "").replace("</b>", "")))
    print(f"\nReport: {os.path.join(out_dir, 'report.html')}")


if __name__ == "__main__":
    main()
