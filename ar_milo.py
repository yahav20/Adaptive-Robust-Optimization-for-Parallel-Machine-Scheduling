"""
PAPER MODEL: exact Adaptive Robust (AR) policy as ONE monolithic MILO.

Generic version of milo_schedulig.py (any number of tasks, machines and Gamma).

The scenario tree is the same as in ar_minimax.py (same node names in the logs):
    SCHEDULER node sigma in D :  t_sigma <= t_delta                 for every child delta  (min)
    ADVERSARY node sigma in N :  t_sigma <= t_delta + M_w*w_delta,  sum_delta w_delta <= k-1  (max)
    LEAF sigma in L           :  E_sigma d_sigma <= M_z*z_sigma
                                 t_sigma  = e_sigma' d_sigma - M_z*z_sigma
                                 d_sigma  = d0 + d_bar*u_sigma,  u in [0,1],  sum u <= Gamma
    objective                 :  max t_root

Non-anticipativity:
    d_sigma(i) = d_delta(i) for every i in F at A(sigma, delta) (lowest common ancestor).
    Equivalent compact form used here: when task i finishes on the edge into node A, ONE
    variable u[i]@A is created and every leaf below A uses it. Leaves that split before
    i finished get different variables. This removes the equality constraints entirely
    (important for the size-limited Gurobi license) and is mathematically identical.

    Bounds hierarchy:
        Minimax (discretized NA) <= MILO (exact AR)

Run standalone:
    python3 ar_milo.py                  # random instance from config/config.json
    python3 ar_milo.py --reference 0    # reference instance #0 (scheduling.py example)
"""

import argparse
import itertools
import os
import time

import gurobipy as gp
from gurobipy import GRB

from ar_common import (
    TOL, Instance, close_logger, fmt_d, fmt_expr, fmt_row, fmt_val, instance_from_args,
    load_config, lp_feasible, make_env, path_str, setup_logger, task_name, tasks_str,
)


def sub_expr(expr1, expr2):
    return [expr1[i] - expr2[i] for i in range(len(expr1))]


class Node:
    def __init__(self, kind, path, running, pending, finished, E_matrix):
        self.kind = kind            # 'SCHEDULER' | 'ADVERSARY' | 'LEAF'
        self.path = path
        self.running = running
        self.pending = pending
        self.finished = finished
        self.E = E_matrix
        self.children = []
        self.finished_task = None   # task that finished on the edge into this node
        self.e = None               # leaf: makespan expression (C of last task)
        # model objects
        self.val = None             # Var / LinExpr = t_sigma
        self.w = []                 # adversary binaries (one per child)
        self.u_map = None           # task -> u variable visible at this node
        self.z = None               # leaf binary


class MiloSolver:
    def __init__(self, inst: Instance, cfg: dict, logger, trace=True, trace_max_depth=1000, gurobi_log=None):
        self.inst = inst
        self.n = inst.n_tasks
        self.m = inst.n_machines
        self.cfg = cfg
        self.log = logger
        self.trace = trace
        self.trace_max_depth = trace_max_depth
        self.gurobi_log = gurobi_log
        self.prune = cfg.get("prune_infeasible_branches", True)
        self.env = make_env()
        self.D_max = inst.max_total_duration
        # Safe Big-M constants to avoid numerical trickling in Gurobi when z=1 or w=1
        self.M_z = 2.0 * self.D_max + 1.0           # ensures E_sigma d <= M_z is always slack & t_sigma <= -D_max - 1
        self.M_w = self.D_max + self.M_z + 1.0      # >= t_sigma - t_delta for any pair
        self.stats = dict(scheduler_nodes=0, adversary_nodes=0, leaves=0, pruned_branches=0)

    def _t(self, depth, msg):
        if self.trace and depth <= self.trace_max_depth:
            self.log.debug("|  " * depth + msg)

    # ------------------------------------------------------------------ tree
    def build_tree(self, running, pending, finished, E_matrix, start_expr, n_slots, path):
        """Scheduler node (mirrors MinimaxSolver.scheduler_turn)."""
        node = Node("SCHEDULER", path, running, pending, finished, E_matrix)
        self.stats["scheduler_nodes"] += 1
        k = min(n_slots, len(pending))
        for combo in itertools.combinations(pending, k):
            new_running = dict(running)
            for j in combo:
                expr = list(start_expr)
                expr[j] += 1
                new_running[j] = expr
            rest = tuple(t for t in pending if t not in combo)
            child = self._adversary(new_running, rest, finished, E_matrix, path + [("S", combo)])
            if child is not None:
                node.children.append(child)
        return node if node.children else None

    def _adversary(self, running, pending, finished, E_matrix, path):
        node = Node("ADVERSARY", path, running, pending, finished, E_matrix)
        self.stats["adversary_nodes"] += 1
        order = sorted(running)
        for f in order:
            new_rows = [sub_expr(running[f], running[r]) for r in order if r != f]
            E2 = E_matrix + new_rows
            if self.prune and new_rows and not lp_feasible(self.env, self.inst, E2):
                self.stats["pruned_branches"] += 1
                continue
            rest_running = {r: e for r, e in running.items() if r != f}
            new_finished = finished + (f,)
            child_path = path + [("F", f)]
            if not rest_running and not pending:
                child = Node("LEAF", child_path, {}, (), new_finished, E2)
                child.e = running[f]
                self.stats["leaves"] += 1
            elif pending:
                child = self.build_tree(rest_running, pending, new_finished, E2, running[f], 1, child_path)
            else:
                child = self._adversary(rest_running, pending, new_finished, E2, child_path)
            if child is not None:
                child.finished_task = f
                node.children.append(child)
        return node if node.children else None

    # ------------------------------------------------------------------ model
    def build_model(self, root):
        m = gp.Model("MILO_Adaptive_Robust", env=self.env)
        self.model = m
        self._build(root, {})
        m.setObjective(root.val, GRB.MAXIMIZE)
        m.update()
        return m

    def _build(self, node, u_map):
        m, inst, n = self.model, self.inst, self.n
        if node.finished_task is not None:
            f = node.finished_task
            # duration of f is revealed here -> shared by every leaf below this node
            u_map = dict(u_map)
            u_map[f] = m.addVar(lb=0.0, ub=1.0, name=f"u[{task_name(f)}]@{path_str(node.path)}")
        node.u_map = u_map

        if node.kind == "LEAF":
            u = [u_map[i] for i in range(n)]
            d = [inst.d0[i] + inst.d_bar[i] * u[i] for i in range(n)]
            z = m.addVar(vtype=GRB.BINARY, name=f"z@{path_str(node.path)}")
            node.z = z
            m.addConstr(gp.quicksum(u) <= inst.gamma)
            for row in node.E:
                m.addConstr(gp.quicksum(row[i] * d[i] for i in range(n) if row[i] != 0) <= self.M_z * z)
            node.val = gp.quicksum(node.e[i] * d[i] for i in range(n) if node.e[i] != 0) - self.M_z * z
            return node.val

        child_vals = [self._build(c, u_map) for c in node.children]
        if len(child_vals) == 1:
            node.val = child_vals[0]      # no decision here -> same value as the only child
            return node.val
        t = m.addVar(lb=-self.M_z, ub=self.D_max, name=f"t@{path_str(node.path)}")
        if node.kind == "SCHEDULER":
            for cv in child_vals:
                m.addConstr(t <= cv)
        else:
            for c, cv in zip(node.children, child_vals):
                w = m.addVar(vtype=GRB.BINARY, name=f"w@{path_str(c.path)}")
                node.w.append(w)
                m.addConstr(t <= cv + self.M_w * w)
            m.addConstr(gp.quicksum(node.w) <= len(node.children) - 1)
        node.val = t
        return t

    # ------------------------------------------------------------------ solve
    def _optimize(self, objective, tag):
        m = self.model
        m.setObjective(objective, GRB.MAXIMIZE)
        m.Params.TimeLimit = self.cfg.get("time_limit_sec", 300)
        m.Params.MIPGap = self.cfg.get("mip_gap", 0.0)
        m.Params.Threads = self.cfg.get("threads", 0)
        if self.gurobi_log:
            m.Params.LogToConsole = 0
            m.Params.LogFile = self.gurobi_log
            m.Params.OutputFlag = 1
        t0 = time.perf_counter()
        m.optimize()
        elapsed = time.perf_counter() - t0
        status = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT"}.get(m.Status, f"STATUS_{m.Status}")
        val = m.ObjVal if m.SolCount > 0 else None
        bound = m.ObjBound if m.SolCount > 0 else None
        self.log.info(f"  [{tag}] status={status} value={fmt_val(val)} bound={fmt_val(bound)} "
                      f"nodes={int(m.NodeCount)} time={elapsed:.3f}s")
        return val, status, elapsed

    @staticmethod
    def _v(x):
        return x.X if isinstance(x, gp.Var) else x.getValue()

    def solve(self):
        inst = self.inst
        self.log.info("=" * 78)
        self.log.info("PAPER MODEL - monolithic MILO for the exact Adaptive Robust policy")
        self.log.info("=" * 78)
        self.log.info(inst.describe())
        self.log.info(f"prune_infeasible_branches={self.prune}, M_z={self.M_z:.4f}, M_w={self.M_w:.4f}")

        t0 = time.perf_counter()
        root = self.build_tree({}, tuple(range(self.n)), (), [], [0] * self.n, min(self.m, self.n), [])
        try:
            self.build_model(root)
        except gp.GurobiError as e:
            return self._error(str(e), t0)
        m = self.model
        size = dict(vars=m.NumVars, binaries=m.NumBinVars, constraints=m.NumConstrs)
        self.log.info(f"Scenario tree: {self.stats}")
        self.log.info(f"MILO size: {size}  (build {time.perf_counter() - t0:.3f}s)")
        self.log.info("-" * 78)

        try:
            value, status, _ = self._optimize(root.val, "full MILO, max t_root")
            root_choices, root_status = {}, {}
            if self.cfg.get("solve_each_root_choice", True) and len(root.children) > 1:
                self.log.info("Solving the MILO once per FIRST decision (value of committing to it):")
                for c in root.children:
                    combo = c.path[-1][1]
                    v, s, _ = self._optimize(c.val, f"first decision start {tasks_str(combo)}")
                    root_choices[combo], root_status[combo] = v, s
                    self._report_subtree(c, f"first decision start {tasks_str(combo)}")
            else:
                combo = root.children[0].path[-1][1] if len(root.children) == 1 else None
                if combo is not None:
                    root_choices[combo] = value
                self._report_subtree(root, "full MILO")
        except gp.GurobiError as e:
            return self._error(str(e), t0)
        elapsed = time.perf_counter() - t0

        if root_choices and all(v is not None for v in root_choices.values()):
            root_choice = min(root_choices, key=lambda c: root_choices[c])
            best = root_choices[root_choice]
            if value is not None and abs(best - value) > 1e-4:
                self.log.info(f"  WARNING: min over first decisions ({best:.4f}) != full MILO ({value:.4f})")
        else:
            root_choice = None

        self.log.info("-" * 78)
        self.log.info("MILO RESULT")
        for combo, v in root_choices.items():
            mark = "  <== chosen" if combo == root_choice else ""
            self.log.info(f"  first decision start {tasks_str(combo)} -> AR worst-case makespan {fmt_val(v)}{mark}")
        self.log.info(f"  Optimal AR worst-case makespan = {fmt_val(value)}  (status {status})")
        self.log.info(f"  total time = {elapsed:.3f}s")
        return dict(model="milo", value=value, status=status, root_choice=root_choice,
                    root_choices=root_choices, stats=dict(self.stats), size=size, time=elapsed)

    def _error(self, msg, t0):
        self.log.info(f"  GUROBI ERROR: {msg}")
        if "size-limited" in msg:
            self.log.info("  The free pip license allows 2000 variables/constraints. Reduce n_tasks or "
                          "install a full (academic) license.")
        return dict(model="milo", value=None, status=f"ERROR: {msg}", root_choice=None, root_choices={},
                    stats=dict(self.stats), size=None, time=time.perf_counter() - t0)

    # ------------------------------------------------------------------ explain solution
    def _report_subtree(self, top, title):
        """Worst-case path of the current solution + (trace) the whole subtree with values."""
        leaves = list(self._leaves(top))
        disabled = sum(1 for l in leaves if l.z.X > 0.5)
        self.log.info(f"    {title}: {len(leaves)} leaves, {disabled} with z=1 (sequence impossible with the "
                      f"durations the adversary committed to, or a branch the adversary does not need)")
        node = top
        self.log.info(f"    worst-case path under the AR policy:")
        while node.kind != "LEAF":
            if len(node.children) == 1:
                nxt = node.children[0]
            elif node.kind == "SCHEDULER":
                nxt = min(node.children, key=lambda c: self._v(c.val))
            else:
                active = [c for c, w in zip(node.children, node.w) if w.X < 0.5] or node.children
                # Adversary is a maximizer; among active branches (w=0), pick the binding max child
                nxt = max(active, key=lambda c: self._v(c.val))
            if len(node.children) > 1:
                opts = ", ".join(f"{self._edge(c)}:{self._v(c.val):.4f}" for c in node.children)
                revealed = self._revealed(node)
                self.log.info(f"      [{node.kind}] {path_str(node.path)}  options {opts} -> {self._edge(nxt)}"
                              + (f"  | revealed (shared by all options): {revealed}" if revealed else ""))
            node = nxt
        d = self._leaf_d(node)
        self.log.info(f"      [LEAF] {path_str(node.path)}  makespan {self._v(node.val):.4f}  d = {fmt_d(d)}")
        if self.trace:
            self._t(0, f"---- full tree for '{title}' (values of non-binding branches are only bounds) ----")
            self._dump(top, 0)

    def _dump(self, node, depth):
        val = self._v(node.val)
        if node.kind == "LEAF":
            d = self._leaf_d(node)
            z = node.z.X > 0.5
            self._t(depth, f"[LEAF] {path_str(node.path)} makespan={fmt_expr(node.e)} "
                           + ("DISABLED (z=1)" if z else f"= {val:.4f}") + f" d={fmt_d(d)}")
            return
        extra = ""
        if node.kind == "ADVERSARY" and node.w:
            picked = [self._edge(c) for c, w in zip(node.children, node.w) if w.X < 0.5]
            extra = f" adversary keeps {picked}"
        revealed = self._revealed(node)
        self._t(depth, f"[{node.kind}] {path_str(node.path)} t={val:.4f}{extra}"
                       + (f" revealed: {revealed}" if revealed else ""))
        for c in node.children:
            self._dump(c, depth + 1)

    def _revealed(self, node):
        if not node.finished:
            return ""
        parts = []
        for i in node.finished:
            u = node.u_map.get(i) if node.u_map else None
            if u is not None:
                parts.append(f"{task_name(i)}={self.inst.d0[i] + self.inst.d_bar[i] * u.X:.4f}")
        return ", ".join(parts)

    def _leaf_d(self, leaf):
        return [self.inst.d0[i] + self.inst.d_bar[i] * leaf.u_map[i].X for i in range(self.n)]

    def _leaves(self, node):
        if node.kind == "LEAF":
            yield node
        for c in node.children:
            yield from self._leaves(c)

    @staticmethod
    def _edge(node):
        kind, payload = node.path[-1]
        return f"start {tasks_str(payload)}" if kind == "S" else f"{task_name(payload)} first"


def solve_milo(inst, cfg, logger, gurobi_log=None):
    milo_cfg = dict(cfg.get("milo", {}))
    log_cfg = cfg.get("logging", {})
    solver = MiloSolver(inst, milo_cfg, logger, trace=log_cfg.get("trace_tree", True),
                        trace_max_depth=log_cfg.get("trace_max_depth", 1000), gurobi_log=gurobi_log)
    return solver.solve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--reference", type=int, default=None, help="index into reference_instances")
    ap.add_argument("--seed", type=int, default=None, help="seed for a random instance")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst = instance_from_args(cfg, args.reference, args.seed)
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"milo_{stamp}.log")
    logger = setup_logger("milo", log_path, console=True)
    glog = os.path.join(log_dir, f"milo_{stamp}_gurobi.log") if cfg.get("milo", {}).get("write_gurobi_log") else None
    res = solve_milo(inst, cfg, logger, gurobi_log=glog)
    if inst.expected_makespan is not None and res["value"] is not None:
        ok = abs(res["value"] - inst.expected_makespan) <= cfg.get("benchmark", {}).get("tolerance", 1e-4)
        logger.info(f"Reference check: expected {inst.expected_makespan:.4f}, got {res['value']:.4f} -> "
                    f"{'PASS' if ok else 'FAIL'}")
    logger.info(f"Full log written to {log_path}")
    close_logger(logger)


if __name__ == "__main__":
    main()