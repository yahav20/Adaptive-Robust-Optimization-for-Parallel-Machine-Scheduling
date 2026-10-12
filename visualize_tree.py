"""
Visual comparison of the Hybrid Minimax tree (ar_minimax.py) and the exact MILO (ar_milo.py).

Both models share the same scenario tree (MiloSolver.build_tree). For every node we show:
    * Minimax value  : backward induction over independent leaf LPs, plus the leaf that
                       "locked" the value (the leaf whose LP value propagated up to the node).
    * MILO value     : t_sigma in the MILO solution when committing to that first decision.
                       Only nodes on the worst-case path are tight; others are just bounds.
Every leaf lists its sequence constraints E d <= 0, the makespan expression and the
durations chosen by each model, so the math behind each number can be checked by hand.

Output: logs/viz_<tag>_<stamp>/report.html (+ one SVG per first decision, needs Graphviz `dot`).

    python3 visualize_tree.py --round 6        # benchmark round 6 (same seed as benchmark.py)
    python3 visualize_tree.py --reference 1    # reference instance #1
    python3 visualize_tree.py --seed 7         # any random instance
"""

import argparse
import html
import os
import random
import subprocess
import time

from ar_common import (
    close_logger, fmt_d, fmt_expr, fmt_row, fmt_val, load_config, path_str, random_instance,
    reference_instances, setup_logger, task_name, tasks_str,
)
from ar_milo import MiloSolver
from ar_minimax import NEG_INF, MinimaxSolver

EPS = 1e-4


def pick_instance(cfg, args):
    if args.reference is not None:
        return reference_instances(cfg)[args.reference], f"ref{args.reference}"
    if args.round is not None:
        rng = random.Random(cfg["random_instance"]["seed"])   # same sequence as benchmark.py
        for r in range(1, args.round + 1):
            inst = random_instance(cfg, rng, name=f"round {r}")
        return inst, f"round{args.round:02d}"
    seed = cfg["random_instance"]["seed"] if args.seed is None else args.seed
    return random_instance(cfg, random.Random(seed), name=f"random(seed={seed})"), f"seed{seed}"


def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)


def walk_post(node):
    for c in node.children:
        yield from walk_post(c)
    yield node


def edge_label(node):
    kind, payload = node.path[-1]
    return f"start {tasks_str(payload)}" if kind == "S" else f"{task_name(payload)} first"


def fmt_dv(d):
    return "(" + ", ".join(f"{v:.4f}" for v in d) + ")" if d else "n/a"


# ---------------------------------------------------------------------- evaluation
def eval_minimax(node, mm, steps):
    """Same backward induction as MinimaxSolver, but stored on the explicit tree.
    `steps` collects the evaluation order (post-order = the order the recursion finishes nodes)."""
    if node.kind == "LEAF":
        node.mm, node.mm_d, node.mm_u = mm.solve_leaf_lp(node.E, node.e)
        node.mm_leaf, node.mm_best = node, None
    else:
        for c in node.children:
            eval_minimax(c, mm, steps)
        pick = min if node.kind == "SCHEDULER" else max
        node.mm_best = pick(node.children, key=lambda c: c.mm)
        node.mm, node.mm_leaf = node.mm_best.mm, node.mm_best.mm_leaf
    steps.append(node)
    node.step = len(steps)
    return node.mm


def step_text(n):
    if n.kind == "LEAF":
        how = (f"LP infeasible -> -inf" if n.mm == NEG_INF else
               f"LP: max {fmt_expr(n.e)} = {fmt_val(n.mm)} with d={fmt_dv(n.mm_d)}")
        return f"step {n.step:3d}  LEAF {n.lid:<4} {path_str(n.path)}\n           {how}"
    op = "MIN" if n.kind == "SCHEDULER" else "MAX"
    who = "scheduler picks" if n.kind == "SCHEDULER" else "adversary picks"
    opts = ", ".join(f"{edge_label(c)}={fmt_val(c.mm)}" for c in n.children)
    if n.kind == "ADVERSARY":
        opts += "".join(f", {task_name(j)} first=-inf" for j in sorted(set(n.running) - {c.finished_task for c in n.children}))
    return (f"step {n.step:3d}  {n.kind:<9} {path_str(n.path)}\n"
            f"           {op}({opts}) = {fmt_val(n.mm)}  -> {who} '{edge_label(n.mm_best)}' (locked by {n.mm_leaf.lid})")


def snapshot_milo(top, solver):
    """Store the current MILO solution on every node of `top` and mark the worst-case path."""
    v = solver._v
    for n in walk(top):
        n.ml = v(n.val)
        n.ml_path = False
        if n.kind == "LEAF":
            n.ml_d = solver._leaf_d(n)
            n.ml_z = n.z.X > 0.5
        elif n.kind == "ADVERSARY" and n.w:
            n.ml_kept = [c for c, w in zip(n.children, n.w) if w.X < 0.5]
        else:
            n.ml_kept = list(n.children)
    node = top
    node.ml_path = True
    while node.kind != "LEAF":   # same rule as MiloSolver._report_subtree
        if len(node.children) == 1:
            node = node.children[0]
        elif node.kind == "SCHEDULER":
            node = min(node.children, key=lambda c: c.ml)
        else:
            node = min(node.ml_kept or node.children, key=lambda c: c.ml)
        node.ml_path = True
    return node


def mark_minimax_path(top):
    node = top
    node.mm_path = True
    while node.kind != "LEAF":
        node = node.mm_best
        node.mm_path = True


def find_anticipation(top):
    """Scheduler nodes where the locked leaves of different options disagree on an already
    revealed duration (impossible in reality, allowed by the independent leaf LPs)."""
    out = []
    for n in walk(top):
        if n.kind != "SCHEDULER" or len(n.children) < 2 or not n.finished:
            continue
        for i in n.finished:
            vals = [(edge_label(c), c.mm_leaf.mm_d[i], c.mm) for c in n.children if c.mm_leaf.mm_d]
            if vals and max(x[1] for x in vals) - min(x[1] for x in vals) > EPS:
                out.append(dict(node=n, task=i, vals=vals))
                n.anticip = True
    return out


# ---------------------------------------------------------------------- drawing
def node_text(n, inst):
    lines = []
    if n.kind == "LEAF":
        lines.append(f"LEAF {n.lid}   [step {n.step}: solve LP]")
        lines.append(f"max {fmt_expr(n.e)}  s.t.")
        lines.extend(f"   {fmt_row(r)}" for r in n.E)
        lines.append(f"Minimax LP = {fmt_val(n.mm)}")
        lines.append(f"  d = {fmt_dv(n.mm_d)}")
        if n.ml_z:
            lines.append("MILO: z=1 (disabled)")
        else:
            lines.append(f"MILO = {fmt_val(n.ml)}")
        lines.append(f"  d = {fmt_dv(n.ml_d)}")
        return "\n".join(lines)
    title = "SCHEDULER (min)" if n.kind == "SCHEDULER" else "ADVERSARY (max)"
    lines.append(title)
    if n.running:
        lines.append("running: " + ", ".join(f"{task_name(j)}:{fmt_expr(e)}" for j, e in sorted(n.running.items())))
    if n.kind == "SCHEDULER" and n.pending:
        lines.append(f"pending {tasks_str(n.pending)}")
    if n.finished:
        lines.append(f"finished {tasks_str(n.finished)}")
    op = "MIN" if n.kind == "SCHEDULER" else "MAX"
    lines.append(f"[step {n.step}] Minimax {op} over options:")
    for c in n.children:
        mark = "  <== chosen" if c is n.mm_best else ""
        lines.append(f"   {edge_label(c):<12} = {fmt_val(c.mm)}{mark}")
    if n.kind == "ADVERSARY":
        for j in sorted(set(n.running) - {c.finished_task for c in n.children}):
            lines.append(f"   {task_name(j) + ' first':<12} = -inf (impossible)")
    lines.append(f"=> {fmt_val(n.mm)}  (locked by leaf {n.mm_leaf.lid})")
    lines.append(f"MILO t = {fmt_val(n.ml)}" + ("" if n.ml_path else " (bound)"))
    if n.finished and n.u_map:
        rev = ", ".join(f"{task_name(i)}={inst.d0[i] + inst.d_bar[i] * n.u_map[i].X:.4f}" for i in n.finished)
        lines.append(f"MILO revealed: {rev}")
    if getattr(n, "anticip", False):
        lines.append("!! ANTICIPATION (see table)")
    return "\n".join(lines)


def dot_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\l") + "\\l"


def subtree_dot(top, inst, title):
    ids = {}
    out = ['digraph G {', 'rankdir=LR; nodesep=0.15; ranksep=0.5;',
           f'labelloc=t; label="{title}"; fontname="Helvetica"; fontsize=18;',
           'node [fontname="Courier", fontsize=10, style=filled];',
           'edge [fontname="Helvetica", fontsize=10];']
    for k, n in enumerate(walk(top)):
        ids[id(n)] = f"n{k}"
        fill = {"SCHEDULER": "#dbe9ff", "ADVERSARY": "#ffe1dc", "LEAF": "#eeeeee"}[n.kind]
        if n.kind == "LEAF" and n.ml_z:
            fill = "#f7f7f7"
        if getattr(n, "anticip", False):
            fill = "#ffd27f"
        shape = {"SCHEDULER": "box", "ADVERSARY": "box", "LEAF": "note"}[n.kind]
        style = "filled,rounded" if n.kind == "ADVERSARY" else "filled"
        pen = "3" if (n.mm_path or n.ml_path) else "1"
        color = "#7b2cbf" if (n.mm_path and n.ml_path) else "#1f5fbf" if n.mm_path else "#c0392b" if n.ml_path else "#666666"
        out.append(f'{ids[id(n)]} [shape={shape}, style="{style}", fillcolor="{fill}", color="{color}", '
                   f'penwidth={pen}, label="{dot_escape(node_text(n, inst))}"];')
    for n in walk(top):
        for c in n.children:
            mm_e, ml_e = c.mm_path, c.ml_path
            chosen = c is n.mm_best
            color = ("#7b2cbf" if (mm_e and ml_e) else "#1f5fbf" if mm_e else "#c0392b" if ml_e
                     else "#2e9e44" if chosen else "#999999")
            pen = "3" if (mm_e or ml_e) else "2" if chosen else "1"
            style = "dashed" if (n.kind == "ADVERSARY" and c not in n.ml_kept) else "solid"
            label = f"{edge_label(c)}\\n{fmt_val(c.mm)}" + (" ✔" if chosen else "")
            out.append(f'{ids[id(n)]} -> {ids[id(c)]} [label="{label}", color="{color}", '
                       f'fontcolor="{"#2e9e44" if chosen else "#555555"}", penwidth={pen}, style={style}];')
    out.append("}")
    return "\n".join(out)


def render_svg(dot_src, path):
    with open(path.replace(".svg", ".dot"), "w") as f:
        f.write(dot_src)
    try:
        subprocess.run(["dot", "-Tsvg", "-o", path], input=dot_src.encode(), check=True)
        with open(path) as f:
            svg = f.read()
        return svg[svg.find("<svg"):]
    except (OSError, subprocess.CalledProcessError) as e:
        return f"<p>Graphviz failed ({html.escape(str(e))}); DOT source saved next to the report.</p>"


# ---------------------------------------------------------------------- report
CSS = """body{font-family:Helvetica,Arial,sans-serif;margin:24px;max-width:1800px}
table{border-collapse:collapse;margin:8px 0 18px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}
th{background:#f2f2f2}code,.m{font-family:Menlo,Courier,monospace;font-size:12px}
.diff{background:#ffe9a8}.chosen{font-weight:bold;background:#e4f5e4}.svgbox{overflow:auto;border:1px solid #ddd}
.legend span{display:inline-block;padding:2px 8px;margin-right:6px;border:2px solid}"""


def esc(s):
    return html.escape(str(s))


def leaf_table(top, inst):
    rows = ["<tr><th>leaf</th><th>event sequence</th><th>E d &le; 0 (sequence constraints)</th>"
            "<th>makespan</th><th>Minimax LP value</th><th>Minimax d</th><th>&Sigma;u</th>"
            "<th>MILO value</th><th>MILO d (shared u)</th><th>d differs?</th></tr>"]
    for n in walk(top):
        if n.kind != "LEAF":
            continue
        diff = n.mm_d and any(abs(a - b) > EPS for a, b in zip(n.mm_d, n.ml_d))
        cls = []
        if n.mm_path:
            cls.append("chosen")
        if diff:
            cls.append("diff")
        ml = "disabled (z=1)" if n.ml_z else fmt_val(n.ml)
        rows.append(f"<tr class='{' '.join(cls)}'><td>{n.lid}</td><td class=m>{esc(path_str(n.path))}</td>"
                    f"<td class=m>{esc('; '.join(fmt_row(r) for r in n.E))}</td><td class=m>{esc(fmt_expr(n.e))}</td>"
                    f"<td>{fmt_val(n.mm)}</td><td class=m>{fmt_dv(n.mm_d)}</td>"
                    f"<td>{sum(n.mm_u):.3f}</td><td>{ml}</td>"
                    f"<td class=m>{fmt_dv(n.ml_d)}</td><td>{'yes' if diff else ''}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def worked_math(leaf, inst):
    """Show by hand why the locked leaf of the chosen first decision has its value."""
    return "<pre>" + esc(worked_math_text(leaf, inst)) + "</pre>"


def worked_math_text(leaf, inst):
    n = len(inst.d0)
    d, u = leaf.mm_d, leaf.mm_u
    terms = [i for i in range(n) if leaf.e[i]]
    parts = [f"Leaf {leaf.lid}: {path_str(leaf.path)}",
             f"maximize  C_last = {fmt_expr(leaf.e)}",
             "subject to",
             *[f"   {fmt_row(r)}" for r in leaf.E],
             *[f"   d{i + 1} = {inst.d0[i]:.4f} + {inst.d_bar[i]:.4f}*u{i + 1}" for i in range(n)],
             f"   u1+...+u{n} <= Gamma = {inst.gamma},  0 <= u_i <= 1"]
    if d is None:
        return "\n".join(parts + ["LP infeasible -> -inf (the adversary cannot make this sequence happen)"])
    parts += ["LP optimum:",
             "   u = (" + ", ".join(f"{x:.4f}" for x in u) + f"),  sum u = {sum(u):.4f}",
             "   d = " + fmt_d(d),
             "   C_last = " + " + ".join(f"{d[i]:.4f}" for i in terms) + f" = {leaf.mm:.4f}",
             "check each sequence constraint (lhs <= rhs):"]
    for r in leaf.E:
        lhs = sum(r[i] * d[i] for i in range(n) if r[i] > 0)
        rhs = sum(-r[i] * d[i] for i in range(n) if r[i] < 0)
        parts.append(f"   {fmt_row(r)}:  {lhs:.4f} <= {rhs:.4f}  {'OK' if lhs <= rhs + 1e-6 else 'VIOLATED'}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--reference", type=int, default=None)
    ap.add_argument("--round", type=int, default=None, help="benchmark round (1-based)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst, tag = pick_instance(cfg, args)

    out_dir = os.path.join(cfg.get("logging", {}).get("log_dir", "logs"), f"viz_{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger("viz", os.path.join(out_dir, "solvers.log"), console=False)

    # reference results from the real solvers (sanity check of the re-implementation below)
    mm_res = MinimaxSolver(inst, cfg.get("minimax", {}), logger, trace=False).solve()

    milo = MiloSolver(inst, cfg.get("milo", {}), logger, trace=False)
    root = milo.build_tree({}, tuple(range(inst.n_tasks)), (), [], [0] * inst.n_tasks,
                           min(inst.n_machines, inst.n_tasks), [])
    milo.build_model(root)
    mm = MinimaxSolver(inst, cfg.get("minimax", {}), logger, trace=False)
    for k, leaf in enumerate(n for n in walk(root) if n.kind == "LEAF"):
        leaf.lid = f"L{k + 1}"
    steps = []
    eval_minimax(root, mm, steps)
    assert abs(root.mm - mm_res["value"]) < 1e-6, (root.mm, mm_res["value"])
    for n in walk(root):
        n.mm_path = n.ml_path = False
        n.anticip = False

    ml_full, _, _ = milo._optimize(root.val, "full MILO")
    sections, summary = [], []
    for c in root.children:
        combo = c.path[-1][1]
        ml_val, status, _ = milo._optimize(c.val, f"start {tasks_str(combo)}")
        ml_leaf = snapshot_milo(c, milo)
        mark_minimax_path(c)
        anti = find_anticipation(c)
        summary.append(dict(combo=combo, mm=c.mm, mm_leaf=c.mm_leaf, ml=ml_val, ml_leaf=ml_leaf, anti=anti,
                            n_leaves=sum(1 for n in walk(c) if n.kind == "LEAF")))
        title = (f"First decision start {tasks_str(combo)}:  Minimax {fmt_val(c.mm)} (locked by {c.mm_leaf.lid})"
                 f"   vs   MILO {fmt_val(ml_val)} (worst leaf {ml_leaf.lid})")
        svg = render_svg(subtree_dot(c, inst, title), os.path.join(out_dir, f"tree_{''.join(task_name(t) for t in combo)}.svg"))
        anti_html = ""
        if anti:
            items = "".join(
                f"<li>at <code>{esc(path_str(a['node'].path))}</code> {task_name(a['task'])} has already finished, "
                f"but the locked leaves use " + ", ".join(f"<b>{esc(lbl)}</b>: d{a['task'] + 1}={dv:.4f} (value {fmt_val(v)})"
                                                     for lbl, dv, v in a["vals"]) + "</li>" for a in anti)
            anti_html = f"<h4>Anticipation cases (orange nodes)</h4><ul>{items}</ul>"
        sections.append(dict(combo=combo, html=(
            f"<h3>{esc(title)}</h3>"
            f"<h4>Worked math for the Minimax locked leaf {c.mm_leaf.lid}</h4>{worked_math(c.mm_leaf, inst)}"
            f"{anti_html}<div class=svgbox>{svg}</div>"
            f"<h4>Algorithm run (backward induction, in the order the recursion finishes each node)</h4>"
            f"<pre>{esc(chr(10).join(step_text(n) for n in walk_post(c)))}</pre>"
            f"<h4>All leaves</h4>{leaf_table(c, inst)}")))

    mm_best = min(summary, key=lambda s: s["mm"])
    ml_best = min(summary, key=lambda s: s["ml"])
    root_rows = "".join(
        f"<tr><td>start {tasks_str(s['combo'])}</td><td{' class=chosen' if s is mm_best else ''}>{fmt_val(s['mm'])}</td>"
        f"<td>{s['mm_leaf'].lid}: <code>{esc(path_str(s['mm_leaf'].path))}</code><br><code>{fmt_d(s['mm_leaf'].mm_d)}</code></td>"
        f"<td{' class=chosen' if s is ml_best else ''}>{fmt_val(s['ml'])}</td>"
        f"<td>{s['ml_leaf'].lid}: <code>{fmt_d(s['ml_leaf'].ml_d)}</code></td>"
        f"<td{' class=diff' if s['mm'] - s['ml'] > EPS else ''}>{s['mm'] - s['ml']:+.4f}</td>"
        f"<td>{len(s['anti'])}</td><td>{s['n_leaves']}</td></tr>" for s in summary)
    inst_rows = "".join(f"<tr><td>{task_name(i)}</td><td>{inst.d0[i]:.4f}</td><td>{inst.d_bar[i]:.4f}</td>"
                        f"<td>{inst.d0[i] + inst.d_bar[i]:.4f}</td></tr>" for i in range(inst.n_tasks))
    order = sorted(sections, key=lambda s: s["combo"] != mm_best["combo"])
    body = "".join(f"<details{' open' if s['combo'] == mm_best['combo'] else ''}><summary><b>Tree for start "
                   f"{tasks_str(s['combo'])}</b></summary>{s['html']}</details>" for s in order)

    page = f"""<!doctype html><html><head><meta charset=utf-8><title>Minimax vs MILO - {esc(inst.name)}</title>
<style>{CSS}</style></head><body>
<h1>Minimax tree vs MILO &mdash; {esc(inst.name)}</h1>
<p>n_tasks={inst.n_tasks}, n_machines={inst.n_machines}, &Gamma;={inst.gamma}.
Durations d<sub>i</sub> = d0<sub>i</sub> + d&#772;<sub>i</sub>&middot;u<sub>i</sub>, 0&le;u<sub>i</sub>&le;1, &Sigma;u<sub>i</sub> &le; &Gamma;.</p>
<table><tr><th>task</th><th>d0</th><th>d_bar</th><th>max d</th></tr>{inst_rows}</table>
<h2>Result</h2>
<p><b>Minimax (upper bound)</b> = {fmt_val(root.mm)} &nbsp; | &nbsp; <b>MILO (exact AR)</b> = {fmt_val(ml_full)}
&nbsp; | &nbsp; gap = {root.mm - ml_full:+.4f}</p>
<table><tr><th>first decision</th><th>Minimax value</th><th>Minimax locked leaf (path, d)</th>
<th>MILO value</th><th>MILO worst leaf d</th><th>gap</th><th>anticipation cases</th><th>leaves</th></tr>{root_rows}</table>
<pre>{esc(step_text(root))}</pre>
<h2>How to read the trees</h2>
<ul>
<li>Blue box = SCHEDULER (min over options), red rounded box = ADVERSARY (max over who finishes first), grey note = LEAF (one LP).</li>
<li>Every internal node shows <i>Minimax = value &larr; locked by Lk</i>: Lk is the leaf whose LP value propagated up to this node.</li>
<li>Edge label = option and the value it returned; green &#10004; = the option this node picked (min for scheduler, max for adversary).
<i>[step k]</i> = order in which the recursion finished the node (leaves first, then their parents).</li>
<li class=legend><span style="border-color:#1f5fbf">Minimax worst-case path</span><span style="border-color:#c0392b">MILO worst-case path</span>
<span style="border-color:#7b2cbf">both</span> dashed edge = MILO adversary dropped that branch (w=1); orange node = anticipation.</li>
<li>Leaf LP (Minimax): max e&middot;d s.t. E d &le; 0, budget. Each leaf picks its own d.
MILO: one u variable per task per "finish edge", shared by every leaf below it (non-anticipativity); MILO values off the red path are bounds only.</li>
<li>Highlighted leaf rows: green = Minimax locked leaf, yellow = Minimax and MILO durations differ.</li>
</ul>
{body}
</body></html>"""
    report = os.path.join(out_dir, "report.html")
    with open(report, "w") as f:
        f.write(page)
    close_logger(logger)

    print(f"{inst.name}: Minimax = {fmt_val(root.mm)}, MILO = {fmt_val(ml_full)}, gap = {root.mm - ml_full:+.4f}")
    for s in summary:
        print(f"  start {tasks_str(s['combo'])}: Minimax {fmt_val(s['mm'])} locked by {s['mm_leaf'].lid} "
              f"[{path_str(s['mm_leaf'].path)}] d={fmt_d(s['mm_leaf'].mm_d)} | MILO {fmt_val(s['ml'])} "
              f"| anticipation {len(s['anti'])}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
