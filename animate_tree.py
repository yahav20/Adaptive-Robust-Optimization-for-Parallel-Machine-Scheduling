"""
Step-by-step animation ("game replay") of the Hybrid Minimax algorithm, compared with the MILO.

Writes one self-contained HTML player (no internet needed):
    1. the tree is drawn with every value unknown ("?")
    2. every leaf LP is solved, one leaf per step (the math is shown in the side panel)
    3. internal nodes are evaluated bottom-up: the node highlights its children, compares their
       values with a bar chart and keeps the MIN (scheduler) or MAX (adversary)
    4. the worst-case path is highlighted, anticipation conflicts are shown, and finally the
       exact MILO (non-anticipative) values are overlaid

Controls: play/pause, step forward/back (arrow keys), speed, which subtree to show, evaluation
order (all leaves first, or the real recursion order of ar_minimax.py). Mouse wheel zooms, drag pans.

    python3 animate_tree.py --round 6        # benchmark round 6 (same seed as benchmark.py)
    python3 animate_tree.py --reference 1
    python3 animate_tree.py --seed 7
"""

import argparse
import json
import os
import time

from ar_common import close_logger, fmt_expr, load_config, path_str, setup_logger, task_name, tasks_str
from ar_milo import MiloSolver
from ar_minimax import NEG_INF, MinimaxSolver
from visualize_tree import (
    edge_label, eval_minimax, find_anticipation, mark_minimax_path, pick_instance, snapshot_milo, walk,
    worked_math_text,
)


def num(v):
    return None if v is None or v == NEG_INF else round(v, 6)


def short_edge(node):
    kind, payload = node.path[-1]
    return "start " + ",".join(task_name(t) for t in payload) if kind == "S" else f"{task_name(payload)} done"


def export(root, inst, ml_full):
    ids = {id(n): k for k, n in enumerate(walk(root))}
    nodes = []
    for top in root.children:
        for n in walk(top):
            n.top = ids[id(top)]
    root.top = ids[id(root)]
    for n in walk(root):
        d = dict(id=ids[id(n)], kind=n.kind[0], ch=[ids[id(c)] for c in n.children],
                 edge=short_edge(n) if n.path else "", edgeLong=edge_label(n) if n.path else "",
                 path=path_str(n.path), top=n.top, mm=num(n.mm), mlPath=bool(getattr(n, "ml_path", False)),
                 ml=num(getattr(n, "ml", None)))
        if n.kind == "LEAF":
            d.update(lid=n.lid, math=worked_math_text(n, inst), obj=fmt_expr(n.e),
                     d=[round(x, 4) for x in n.mm_d] if n.mm_d else None,
                     mlZ=bool(n.ml_z), mlD=[round(x, 4) for x in n.ml_d])
        else:
            d.update(best=ids[id(n.mm_best)], lock=n.mm_leaf.lid,
                     running=", ".join(f"{task_name(j)}: C={fmt_expr(e)}" for j, e in sorted(n.running.items())),
                     pending=tasks_str(n.pending), finished=tasks_str(n.finished))
            if n.kind == "ADVERSARY":
                d["pruned"] = [f"{task_name(j)} done" for j in sorted(set(n.running) - {c.finished_task for c in n.children})]
            if n.finished and n.u_map and n is not root:
                d["revealed"] = ", ".join(f"d{i + 1}={inst.d0[i] + inst.d_bar[i] * n.u_map[i].X:.4f}" for i in n.finished)
        nodes.append(d)
    nodes[ids[id(root)]]["ml"] = num(ml_full)
    return nodes


def anticipation_export(root, ids_of):
    out = {}
    for top in root.children:
        for a in find_anticipation(top):
            out.setdefault(ids_of[id(a["node"])], []).append(
                dict(task=a["task"] + 1, vals=[[lbl, round(dv, 4), num(v)] for lbl, dv, v in a["vals"]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--reference", type=int, default=None)
    ap.add_argument("--round", type=int, default=None, help="benchmark round (1-based)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst, tag = pick_instance(cfg, args)

    out_dir = cfg.get("logging", {}).get("log_dir", "logs")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger("anim", None)
    milo = MiloSolver(inst, cfg.get("milo", {}), logger, trace=False)
    root = milo.build_tree({}, tuple(range(inst.n_tasks)), (), [], [0] * inst.n_tasks,
                           min(inst.n_machines, inst.n_tasks), [])
    milo.build_model(root)
    mm = MinimaxSolver(inst, cfg.get("minimax", {}), logger, trace=False)
    for k, leaf in enumerate(n for n in walk(root) if n.kind == "LEAF"):
        leaf.lid = f"L{k + 1}"
    eval_minimax(root, mm, [])
    for n in walk(root):
        n.mm_path = n.ml_path = False
        n.anticip = False
    ml_full, _, _ = milo._optimize(root.val, "full MILO")
    for c in root.children:
        milo._optimize(c.val, "first decision")
        snapshot_milo(c, milo)
        c.ml = milo._v(c.val)
        mark_minimax_path(c)
    root.ml, root.ml_path = ml_full, False
    close_logger(logger)

    ids_of = {id(n): k for k, n in enumerate(walk(root))}
    data = dict(
        name=inst.name, n=inst.n_tasks, m=inst.n_machines, gamma=inst.gamma,
        d0=inst.d0, dbar=inst.d_bar, root=ids_of[id(root)],
        best=ids_of[id(root.mm_best)], nodes=export(root, inst, ml_full),
        anticip=anticipation_export(root, ids_of),
    )
    path = os.path.join(out_dir, f"animation_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.html")
    with open(path, "w") as f:
        f.write(PAGE.replace("/*DATA*/null", json.dumps(data)))
    print(f"Minimax = {root.mm:.4f}, MILO = {ml_full:.4f}")
    print(f"Animation: {path}")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Minimax tree - step by step</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Helvetica,Arial,sans-serif;background:#f6f7fb;color:#222;height:100vh;display:flex;flex-direction:column}
#top{padding:8px 14px;background:#1e2a44;color:#fff;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
#top h1{font-size:17px;margin:0 12px 0 0}#top button{font-size:16px;padding:4px 10px;border:0;border-radius:6px;cursor:pointer;background:#e8ecf5}
#top button.play{background:#36b37e;color:#fff;min-width:80px}#top label{font-size:13px}#top select{font-size:13px}
#main{flex:1;display:flex;min-height:0}#stage{flex:1;position:relative;min-width:0}
svg{width:100%;height:100%;cursor:grab;background:#fff}
#panel{width:470px;padding:14px 16px;overflow:auto;background:#fff;border-left:1px solid #ddd;font-size:14px}
#phase{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#888}#title{font-size:19px;font-weight:bold;margin:4px 0 10px}
pre{background:#f3f4f8;padding:8px;font-size:12px;border-radius:6px;white-space:pre-wrap}
.bar{display:flex;align-items:center;margin:6px 0;font-size:13px}.bar .lab{width:120px}.bar .b{height:22px;border-radius:4px;background:#c9cfdc;color:#222;
padding-left:6px;line-height:22px;font-weight:bold;transition:all .4s}.bar.win .b{background:#36b37e;color:#fff}.bar.inf .b{background:#eee;color:#999;font-weight:normal}
.verdict{margin-top:10px;padding:8px;border-radius:6px;background:#e9f7ef;font-weight:bold}
.warn{margin-top:10px;padding:8px;border-radius:6px;background:#fff1d6}
#progress{height:5px;background:#36b37e;width:0;transition:width .3s}
g.node rect{stroke:#888;stroke-width:1.5;transition:all .35s}g.node text{font-size:12px;pointer-events:none}
g.node.S rect{fill:#e3edff}g.node.A rect{fill:#ffe5e0}g.node.L rect{fill:#f1f1f1}
g.node.unknown text.val{fill:#aaa}g.node .val{font-weight:bold;font-size:14px}g.node .ml{fill:#c0392b;font-size:11px;font-weight:bold}
g.node.active rect{stroke:#f39c12;stroke-width:5}g.node.cmp rect{stroke:#2980b9;stroke-width:3.5}
g.node.loser{opacity:.35}g.node.path rect{stroke:#1f5fbf;stroke-width:4}g.node.mlpath rect{stroke:#c0392b;stroke-width:4}
g.node.path.mlpath rect{stroke:#7b2cbf}g.node.anti rect{fill:#ffd27f}g.node.justset rect{fill:#c8f3d9}
path.edge{fill:none;stroke:#bbb;stroke-width:1.5;transition:all .35s}path.edge.cmp{stroke:#2980b9;stroke-width:3}
path.edge.win{stroke:#36b37e;stroke-width:3.5}path.edge.lose{stroke:#ddd;stroke-dasharray:4 4}
path.edge.path{stroke:#1f5fbf;stroke-width:5}path.edge.mlpath{stroke:#c0392b;stroke-width:5}path.edge.path.mlpath{stroke:#7b2cbf}
text.elab{font-size:10px;fill:#666;paint-order:stroke;stroke:#fff;stroke-width:3px}text.elab.win{fill:#2e9e44;font-weight:bold}
#legend{position:absolute;left:10px;bottom:10px;background:rgba(255,255,255,.92);padding:6px 10px;border-radius:6px;font-size:12px;border:1px solid #ddd}
#legend span{display:inline-block;width:12px;height:12px;border-radius:3px;margin:0 4px -2px 10px;border:1px solid #888}
</style></head><body>
<div id="top"><h1>Minimax tree &mdash; step by step</h1>
<button id="first" title="first step">&#9198;</button><button id="prev" title="previous (&larr;)">&#9664;</button>
<button id="play" class="play">&#9654; Play</button><button id="next" title="next (&rarr;)">&#9654;&#124;</button><button id="last" title="last step">&#9197;</button>
<input id="slider" type="range" min="0" value="0" style="width:220px"><span id="count"></span>
<label>speed <input id="speed" type="range" min="150" max="3000" value="1300" style="width:90px;direction:rtl"></label>
<label>tree <select id="tree"></select></label>
<label>order <select id="order"><option value="levels">all leaves first, then level by level</option>
<option value="dfs">real recursion order (ar_minimax.py)</option></select></label>
<label><input id="follow" type="checkbox"> follow</label><button id="fit">fit</button></div>
<div id="progress"></div>
<div id="main"><div id="stage"><svg id="svg"><g id="vp"></g></svg>
<div id="legend"><span style="background:#e3edff"></span>Scheduler = MIN<span style="background:#ffe5e0"></span>Adversary = MAX
<span style="background:#f1f1f1"></span>Leaf = LP<span style="background:#fff;border:3px solid #f39c12"></span>now
<span style="background:#fff;border:3px solid #2980b9"></span>compared<span style="background:#fff;border:3px solid #36b37e"></span>chosen
<span style="background:#fff;border:3px solid #1f5fbf"></span>Minimax path<span style="background:#fff;border:3px solid #c0392b"></span>MILO path</div></div>
<div id="panel"><div id="phase"></div><div id="title"></div><div id="body"></div></div></div>
<script>
const D = /*DATA*/null;
const N = D.nodes, W = 118, H = 46, DX = 240, DY = 56;
const fmt = v => v === null || v === undefined ? "−∞" : v.toFixed(4);
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let viewRoot, vis, steps, cur = 0, timer = null, view = {x:0, y:0, k:1};
const svg = document.getElementById("svg"), vp = document.getElementById("vp");

// ---------------------------------------------------------------- tree select
const sel = document.getElementById("tree");
N[D.root].ch.forEach(c => { const o = document.createElement("option"); o.value = c;
  o.textContent = "first decision: " + N[c].edge + (c === D.best ? "  (Minimax choice)" : ""); sel.appendChild(o); });
const full = document.createElement("option"); full.value = D.root; full.textContent = "FULL tree (root decides among all first decisions)"; sel.appendChild(full);
sel.value = D.best;

function title(n) {
  if (n.kind === "L") return n.lid;
  return (n.kind === "S" ? "▼ MIN scheduler" : "▲ MAX adversary");
}

// ---------------------------------------------------------------- layout + drawing
function build() {
  viewRoot = +sel.value; vis = []; let leafY = 0;
  (function lay(id, depth) {
    const n = N[id]; vis.push(id); n.x = depth * DX;
    if (!n.ch.length) { n.y = leafY++ * DY; return; }
    n.ch.forEach(c => lay(c, depth + 1));
    n.y = (N[n.ch[0]].y + N[n.ch[n.ch.length - 1]].y) / 2;
  })(viewRoot, 0);
  // path of the Minimax worst case inside this view
  vis.forEach(id => { N[id].onPath = false; N[id].onMl = false; });
  for (let id = viewRoot; ; id = N[id].best) { N[id].onPath = true; if (N[id].kind === "L") break; }
  const mlTop = viewRoot === D.root ? N[D.root].ch.reduce((a, b) => N[a].ml <= N[b].ml ? a : b) : viewRoot;
  vis.forEach(id => { N[id].onMl = N[id].mlPath && N[id].top === N[mlTop].top; });
  if (viewRoot === D.root) N[D.root].onMl = true;

  let h = "";
  vis.forEach(id => { const n = N[id]; n.ch.forEach(c => { const m = N[c];
    const x1 = n.x + W, y1 = n.y + H / 2, x2 = m.x, y2 = m.y + H / 2, xm = (x1 + x2) / 2;
    h += `<path class="edge" id="e${c}" d="M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}"/>`;
    h += `<text class="elab" id="t${c}" x="${xm}" y="${(y1 + y2) / 2 - 4}" text-anchor="middle">${esc(m.edge)}</text>`; }); });
  vis.forEach(id => { const n = N[id];
    h += `<g class="node ${n.kind}" id="n${id}" transform="translate(${n.x},${n.y})"><rect width="${W}" height="${H}" rx="${n.kind === "A" ? 12 : 3}"/>
      <text x="6" y="14">${title(n)}</text><text class="val" x="6" y="31">?</text><text class="ml" x="6" y="43"></text></g>`; });
  vp.innerHTML = h;
  vis.forEach(id => document.getElementById("n" + id).onclick = () => { const k = steps.findIndex(s => s.n === id && s.t !== "cmp"); if (k >= 0) go(k); });
  steps = makeSteps(); cur = 0;
  document.getElementById("slider").max = steps.length - 1;
  fit(); render();
}

// ---------------------------------------------------------------- steps
function height(id) { const n = N[id]; return n.ch.length ? 1 + Math.max(...n.ch.map(height)) : 0; }
function makeSteps() {
  const S = [{t: "intro"}];
  const internal = vis.filter(id => N[id].kind !== "L");
  const push = id => { const n = N[id];
    if (n.kind === "L") S.push({t: "leaf", n: id});
    else if (n.ch.length === 1 && !(n.pruned && n.pruned.length)) S.push({t: "pass", n: id});
    else { S.push({t: "cmp", n: id}); S.push({t: "pick", n: id}); } };
  if (document.getElementById("order").value === "levels") {
    vis.filter(id => N[id].kind === "L").forEach(push);
    internal.map(id => [height(id), N[id].y, id]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).forEach(x => push(x[2]));
  } else {
    (function post(id) { N[id].ch.forEach(post); push(id); })(viewRoot);
  }
  S.push({t: "path"});
  vis.filter(id => D.anticip[id]).forEach(id => S.push({t: "anti", n: id}));
  S.push({t: "milo"}); S.push({t: "end"});
  return S;
}

function stateAt(k) {
  const st = {known: new Set(), picked: new Set(), active: null, cmp: false, path: false, anti: new Set(), milo: false, justset: null};
  for (let i = 0; i <= k; i++) { const s = steps[i]; st.active = null; st.cmp = false; st.justset = null;
    if (s.t === "leaf") { st.known.add(s.n); st.active = s.n; st.justset = s.n; }
    else if (s.t === "cmp") { st.active = s.n; st.cmp = true; }
    else if (s.t === "pick" || s.t === "pass") { st.known.add(s.n); st.picked.add(s.n); st.active = s.n; st.justset = s.n; }
    else if (s.t === "path") st.path = true;
    else if (s.t === "anti") { st.anti.add(s.n); st.active = s.n; }
    else if (s.t === "milo" || s.t === "end") st.milo = true; }
  return st;
}

// ---------------------------------------------------------------- render
function render() {
  const st = stateAt(cur), s = steps[cur];
  vis.forEach(id => { const n = N[id], g = document.getElementById("n" + id);
    const par = vis.find(p => N[p].ch.includes(id));
    const loser = par !== undefined && st.picked.has(par) && N[par].best !== id && !st.path;
    g.setAttribute("class", ["node", n.kind, st.known.has(id) ? "" : "unknown", st.active === id ? "active" : "",
      st.cmp && N[st.active].ch.includes(id) ? "cmp" : "", loser ? "loser" : "", st.path && n.onPath ? "path" : "",
      st.milo && n.onMl ? "mlpath" : "", st.anti.has(id) ? "anti" : "", st.justset === id ? "justset" : ""].join(" "));
    g.querySelector(".val").textContent = st.known.has(id) ? "= " + fmt(n.mm) + (n.kind !== "L" ? "  ←" + n.lock : "") : "?";
    g.querySelector(".ml").textContent = st.milo && n.onMl ? "MILO " + fmt(n.ml) : "";
    if (par === undefined) return;
    const e = document.getElementById("e" + id), t = document.getElementById("t" + id), p = N[par];
    const win = st.picked.has(par) && p.best === id;
    e.setAttribute("class", ["edge", st.cmp && st.active === par ? "cmp" : "", win ? "win" : "",
      st.picked.has(par) && !win ? "lose" : "", st.path && n.onPath && p.onPath ? "path" : "",
      st.milo && n.onMl && p.onMl ? "mlpath" : ""].join(" "));
    t.setAttribute("class", "elab" + (win ? " win" : ""));
    t.textContent = n.edge + (st.known.has(id) ? " = " + fmt(n.mm) : "") + (win ? " ✔" : ""); });
  panel(s);
  document.getElementById("slider").value = cur;
  document.getElementById("count").textContent = `step ${cur} / ${steps.length - 1}`;
  document.getElementById("progress").style.width = (100 * cur / (steps.length - 1)) + "%";
  if (document.getElementById("follow").checked && s.n !== undefined) center(s.n);
}

function bars(n, picked) {
  const opts = n.ch.map(c => ({lab: N[c].edge, v: N[c].mm, id: c}));
  (n.pruned || []).forEach(p => opts.push({lab: p, v: null}));
  const fin = opts.filter(o => o.v !== null).map(o => o.v), lo = Math.min(...fin), hi = Math.max(...fin);
  return opts.map(o => { const w = o.v === null ? 25 : 35 + 60 * (hi > lo ? (o.v - lo) / (hi - lo) : 1);
    const cls = o.v === null ? "inf" : (picked && o.id === n.best ? "win" : "");
    return `<div class="bar ${cls}"><div class="lab">${esc(o.lab)}</div><div class="b" style="width:${w}%">${o.v === null ? "−∞ impossible under Γ" : fmt(o.v)}${cls === "win" ? " ✔" : ""}</div></div>`; }).join("");
}

function panel(s) {
  const P = document.getElementById("phase"), T = document.getElementById("title"), B = document.getElementById("body");
  const n = s.n !== undefined ? N[s.n] : null, nl = vis.filter(id => N[id].kind === "L").length;
  if (s.t === "intro") {
    P.textContent = "phase 0 - build the tree"; T.textContent = "Nobody knows any value yet";
    B.innerHTML = `<p>Instance <b>${esc(D.name)}</b>: ${D.n} tasks, ${D.m} machines, Γ = ${D.gamma}.</p>
      <pre>${D.d0.map((x, i) => `T${i + 1}: d = ${x.toFixed(4)} + ${D.dbar[i].toFixed(4)}·u${i + 1}`).join("\n")}\n0 ≤ u_i ≤ 1,  Σ u_i ≤ Γ</pre>
      <p>The tree shown has <b>${vis.length}</b> nodes and <b>${nl}</b> leaves. Each leaf is one full sequence of start/finish events.</p>
      <p><b>Plan:</b> (1) solve one LP per leaf &rarr; its worst-case makespan. (2) go up: a <b>scheduler</b> node keeps the <b>MIN</b> of its options,
      an <b>adversary</b> node keeps the <b>MAX</b>. (3) the root value is the guaranteed worst-case makespan.</p><p>Press ▶ Play or →.</p>`;
  } else if (s.t === "leaf") {
    P.textContent = "phase 1 - solve the leaf LPs"; T.textContent = `Leaf ${n.lid}: makespan = ${fmt(n.mm)}`;
    B.innerHTML = `<p>Event sequence:<br><code>${esc(n.path)}</code></p><p>The adversary picks durations inside the uncertainty set that make
      <b>this exact order of events</b> happen and make the last completion time as large as possible:</p><pre>${esc(n.math)}</pre>`;
  } else if (s.t === "cmp" || s.t === "pick" || s.t === "pass") {
    const isS = n.kind === "S", picked = s.t !== "cmp";
    P.textContent = "phase 2 - back up the values";
    T.textContent = (isS ? "Scheduler" : "Adversary") + (s.t === "cmp" ? " compares its options" : s.t === "pass" ? ": only one option" : (isS ? " keeps the MIN" : " keeps the MAX"));
    const who = isS ? `A machine is idle. The scheduler chooses which task to start and wants the <b>smallest</b> worst-case makespan.`
                    : `All machines are busy. The adversary chooses which running task finishes first and wants the <b>largest</b> makespan.`;
    B.innerHTML = `<p><code>${esc(n.path)}</code></p><p>${who}</p><pre>running : ${esc(n.running || "-")}\nfinished: ${esc(n.finished)}${isS ? "\npending : " + esc(n.pending) : ""}</pre>
      ${bars(n, picked)}${picked ? `<div class="verdict">${isS ? "MIN" : "MAX"}(${n.ch.map(c => fmt(N[c].mm)).concat((n.pruned || []).map(() => "−∞")).join(", ")}) = ${fmt(n.mm)}
      &rarr; ${isS ? "start" : ""} <b>${esc(N[n.best].edgeLong)}</b><br><span style="font-weight:normal">value locked by leaf ${n.lock}</span></div>` : ""}`;
  } else if (s.t === "path") {
    const r = N[viewRoot]; let p = [], id = viewRoot; while (N[id].kind !== "L") { id = N[id].best; p.push(N[id].edgeLong); }
    P.textContent = "phase 3 - result"; T.textContent = `Minimax worst case = ${fmt(r.mm)}`;
    B.innerHTML = `<p>Following the chosen option at every node from the top gives the <b style="color:#1f5fbf">blue path</b>:</p>
      <pre>${esc(p.join("\n  -> "))}</pre><p>It ends at leaf <b>${r.lock}</b> with durations <code>(${N[id].d.map(x => x.toFixed(4)).join(", ")})</code>.</p>
      ${viewRoot !== D.root ? `<p>Other first decisions (root MIN):</p>${bars(N[D.root], true)}` : ""}`;
  } else if (s.t === "anti") {
    P.textContent = "phase 4 - where Minimax cheats"; T.textContent = "Anticipation conflict";
    B.innerHTML = `<p><code>${esc(n.path)}</code></p>` + D.anticip[s.n].map(a => `<p>Task T${a.task} has <b>already finished</b>, so d${a.task} is one known number here.
      But the leaf LPs behind each option used different values:</p><pre>${a.vals.map(v => `${v[0].padEnd(12)} uses d${a.task} = ${v[1].toFixed(4)}  -> value ${fmt(v[2])}`).join("\n")}</pre>`).join("")
      + `<div class="warn">A real adversary cannot do this: it would have to know the scheduler's next choice before T${D.anticip[s.n][0].task} finished.
      This is why Minimax is only an <b>upper bound</b>. MILO forces one shared value${n.revealed ? ` (here MILO uses <b>${esc(n.revealed)}</b>)` : ""}.</div>`;
  } else {
    const r = N[viewRoot], gap = r.mm - r.ml;
    P.textContent = s.t === "milo" ? "phase 5 - compare with the exact MILO" : "done"; T.textContent = `Minimax ${fmt(r.mm)}  vs  MILO ${fmt(r.ml)}`;
    B.innerHTML = `<p>The <b style="color:#c0392b">red values/path</b> are the exact adaptive-robust solution (MILO): every task duration is a single value shared by
      all futures that look identical when it finishes (non-anticipativity).</p>
      <div class="${gap > 1e-4 ? "warn" : "verdict"}">gap = Minimax − MILO = ${(gap).toFixed(4)} ${gap > 1e-4 ? "&rarr; Minimax is pessimistic here because of the anticipation conflicts above."
      : "&rarr; no gap: anticipation did not change the value here."}</div>
      <p style="color:#777;font-size:12px">MILO values shown only on its worst-case path; elsewhere they are just bounds.</p>`;
  }
}

// ---------------------------------------------------------------- controls
function go(k) { cur = Math.max(0, Math.min(steps.length - 1, k)); render(); }
function stop() { clearInterval(timer); timer = null; document.getElementById("play").innerHTML = "&#9654; Play"; }
function play() { if (timer) return stop(); if (cur === steps.length - 1) cur = 0;
  document.getElementById("play").innerHTML = "&#10074;&#10074; Pause";
  timer = setInterval(() => { if (cur >= steps.length - 1) return stop(); go(cur + 1); }, +document.getElementById("speed").value); }
document.getElementById("play").onclick = play;
document.getElementById("next").onclick = () => { stop(); go(cur + 1); };
document.getElementById("prev").onclick = () => { stop(); go(cur - 1); };
document.getElementById("first").onclick = () => { stop(); go(0); };
document.getElementById("last").onclick = () => { stop(); go(steps.length - 1); };
document.getElementById("slider").oninput = e => { stop(); go(+e.target.value); };
document.getElementById("speed").onchange = () => { if (timer) { stop(); play(); } };
sel.onchange = () => { stop(); document.getElementById("follow").checked = +sel.value === D.root; build(); };
document.getElementById("order").onchange = () => { stop(); steps = makeSteps(); go(0); };
document.getElementById("fit").onclick = () => fit();
document.addEventListener("keydown", e => { if (e.key === "ArrowRight") { stop(); go(cur + 1); } if (e.key === "ArrowLeft") { stop(); go(cur - 1); } if (e.key === " ") { e.preventDefault(); play(); } });

// ---------------------------------------------------------------- pan / zoom
function apply() { vp.setAttribute("transform", `translate(${view.x},${view.y}) scale(${view.k})`); }
function fit() { const b = vp.getBBox(), r = svg.getBoundingClientRect();
  view.k = Math.min(r.width / (b.width + 40), r.height / (b.height + 40), 1.4);
  view.x = (r.width - b.width * view.k) / 2 - b.x * view.k; view.y = (r.height - b.height * view.k) / 2 - b.y * view.k; apply(); }
function center(id) { const n = N[id], r = svg.getBoundingClientRect(); view.k = Math.max(view.k, 0.8);
  view.x = r.width / 2 - (n.x + W / 2) * view.k; view.y = r.height / 2 - (n.y + H / 2) * view.k; apply(); }
svg.addEventListener("wheel", e => { e.preventDefault(); const r = svg.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const f = e.deltaY < 0 ? 1.12 : 1 / 1.12; view.x = mx - (mx - view.x) * f; view.y = my - (my - view.y) * f; view.k *= f; apply(); }, {passive: false});
let drag = null;
svg.addEventListener("mousedown", e => drag = {x: e.clientX - view.x, y: e.clientY - view.y});
window.addEventListener("mousemove", e => { if (drag) { view.x = e.clientX - drag.x; view.y = e.clientY - drag.y; apply(); } });
window.addEventListener("mouseup", () => drag = null);
window.addEventListener("resize", fit);
build();
</script></body></html>"""


if __name__ == "__main__":
    main()
