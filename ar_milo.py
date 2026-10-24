"""
PAPER MODEL: exact Adaptive Robust (AR) policy as ONE monolithic MILO (model (P), paper Sec. 6.1).

Generic version of milo_schedulig.py (any number of tasks, machines and Gamma).

The scenario tree is the same as in ar_minimax.py (same node names in the logs):
    SCHEDULER node sigma in D :  t_sigma <= t_delta                 for every child delta  (min)
    ADVERSARY node sigma in N :  t_sigma <= t_delta + M_w*w_delta,  sum_delta w_delta <= k-1  (max)
    LEAF sigma in L           :  E_sigma d_sigma <= M_z*z_sigma
                                 t_sigma  = e_sigma' d_sigma - M_z*z_sigma
                                 d_sigma in U
    objective                 :  max t_root

Uncertainty set U:
    budgeted (default) : d = d0 + d_bar*u,  u in [0,1],  sum u <= Gamma
    discrete           : d = sum_s lambda_s * scenario_s,  lambda binary, sum_s lambda_s = 1
                         (Instance.scenarios, paper Sec. 4.2.2 / 7.2)

Non-anticipativity (paper eq. 4):
    d_sigma(i) = d_delta(i) for every i in F at A(sigma, delta), where A is the last common
    ancestor IN D, i.e. the last common REAL scheduling decision (scheduler node with more
    than one option). Compact form used here: at every real scheduler node A, ONE variable
    per task of F(A) that is not shared yet is created (u[i]@A for budgeted U, d[i]@A for
    discrete U) and every leaf below A uses it. Tasks that finish below the last real
    decision get leaf-private variables, exactly as in the paper. For budgeted U this removes
    the equality constraints entirely (important for the size-limited Gurobi license).

Final phase (paper Table 4/7, Fig. 2):
    Below the last real decision, the adversary node (n-1, n-m-1) has every remaining
    finishing order as a DIRECT leaf child (one t, one w per leaf). The recursive builder
    produces nested adversary/forced-scheduler chains there; flatten_final_phase() collapses
    them so the MILO has the paper's size.

Differences from the paper that remain (all value-preserving, see docs/ar_milo_vs_paper_review.md):
    * prune_infeasible_branches=true drops finishing orders impossible for every d in U
      (the paper keeps them and disables them with z=1). Set it to false for an exact replica.
    * Single-child nodes are pass-throughs (no t, no w); with pruning off these are only the
      forced root (n <= m) that the paper omits as well.
    * Two tighter Big-M constants and bounded t instead of one large M and free t.

Rolling horizon (paper Sec. 3 / 7.1): solve_milo_from_state() re-solves the MILO from a
mid-execution state (S, F, D) over U_{S,F,D,I,D_bar} and returns the next task to start.

    Bounds hierarchy:
        Minimax (discretized NA) <= MILO (exact AR) <= MILO without NA

Run standalone:
    python3 ar_milo.py                  # random instance from config/config.json
    python3 ar_milo.py --reference 0    # reference instance #0 (scheduling.py example)
    python3 ar_milo.py --reference 0 --started 1 2 --finished 1 --durations 0.763
                                        # rolling-horizon decision after T1 finished at 0.763
"""

import argparse
import itertools
import math
import os
import time

import gurobipy as gp
from gurobipy import GRB

from ar_common import (
    TOL, Instance, close_logger, discrete_instance, fmt_d, fmt_expr, fmt_val, instance_from_args,
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
        self.e = None               # leaf: makespan expression (C of last task)
        # model objects
        self.val = None             # Var / LinExpr = t_sigma
        self.w = []                 # adversary binaries (one per child)
        self.u_map = None           # task -> shared variable visible at this node (u budgeted / d discrete)
        self.d = None               # leaf: duration expressions d_sigma
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
        self.enforce_na = cfg.get("enforce_non_anticipativity", True)
        self.flatten = cfg.get("flatten_final_phase", True)
        self.tie_tol = cfg.get("tie_tol", 1e-6)
        self.env = make_env()
        # max_total_duration = sum(d0 + d_bar) >= sum(d) for every d in U, so it bounds every
        # completion time, every |row . d| of E and every makespan -> the Big-Ms below are valid.
        self.D_max = inst.max_total_duration
        # Safe Big-M constants to avoid numerical trickling in Gurobi when z=1 or w=1
        self.M_z = 2.0 * self.D_max + 1.0           # ensures E_sigma d <= M_z is always slack & t_sigma <= -D_max - 1
        self.M_w = self.D_max + self.M_z + 1.0      # >= t_sigma - t_delta for any pair
        self.stats = dict(scheduler_nodes=0, real_decisions=0, adversary_nodes=0, leaves=0,
                          pruned_branches=0, feasibility_lps=0)

    def _t(self, depth, msg):
        if self.trace and depth <= self.trace_max_depth:
            self.log.debug("|  " * depth + msg)

    # ------------------------------------------------------------------ tree
    def build_tree(self, running, pending, finished, E_matrix, start_expr, n_slots, path):
        """Scheduler node (mirrors MinimaxSolver.scheduler_turn)."""
        node = Node("SCHEDULER", path, running, pending, finished, E_matrix)
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
        order = sorted(running)
        for f in order:
            new_rows = [sub_expr(running[f], running[r]) for r in order if r != f]
            E2 = E_matrix + new_rows
            if self.prune and new_rows:
                self.stats["feasibility_lps"] += 1
                if not lp_feasible(self.env, self.inst, E2):
                    self.stats["pruned_branches"] += 1
                    continue
            rest_running = {r: e for r, e in running.items() if r != f}
            new_finished = finished + (f,)
            child_path = path + [("F", f)]
            if not rest_running and not pending:
                child = Node("LEAF", child_path, {}, (), new_finished, E2)
                child.e = running[f]
            elif pending:
                child = self.build_tree(rest_running, pending, new_finished, E2, running[f], 1, child_path)
            else:
                child = self._adversary(rest_running, pending, new_finished, E2, child_path)
            if child is not None:
                node.children.append(child)
        return node if node.children else None

    def tree_from_state(self, started, finished):
        """Scheduler node at the decision state (S, F) reached by replaying the event history.

        `started` lists the tasks in start order (the first min(m, n) start at time 0, then one
        task starts each time a task finishes), `finished` lists the tasks in finish order. The
        E rows of the history are kept: with the finished durations fixed (condition_instance),
        "C_last <= C_i" for every running task i is exactly the elapsed-time condition
        d_i >= D_bar_i of U_{S,F,D,I,D_bar}.
        """
        n, m, inst = self.n, self.m, self.inst
        started, finished = tuple(started), tuple(finished)
        if len(set(started)) != len(started) or not set(finished) <= set(started):
            raise ValueError("started must be distinct tasks and contain every finished task")
        if started and len(started) < min(m, n):
            raise ValueError(f"the first decision starts min(m, n) = {min(m, n)} tasks")
        running, E, path, last = {}, [], [], [0] * n
        first = tuple(sorted(started[:min(m, n)]))
        for j in first:
            running[j] = [1 if i == j else 0 for i in range(n)]
        if first:
            path.append(("S", first))
        nxt = len(first)
        for k, f in enumerate(finished):
            if f not in running:
                raise ValueError(f"{task_name(f)} finishes but is not running")
            E += [sub_expr(running[f], running[r]) for r in sorted(running) if r != f]
            last = running.pop(f)
            path.append(("F", f))
            if nxt < len(started):
                j = started[nxt]
                nxt += 1
                expr = list(last)
                expr[j] += 1
                running[j] = expr
                path.append(("S", (j,)))
            elif k < len(finished) - 1 and len(started) < n:
                raise ValueError("history skips a scheduling decision (a machine stayed idle)")
        if nxt < len(started):
            raise ValueError("more tasks started than machines became free")
        pending = tuple(t for t in range(n) if t not in started)
        if not pending or len(running) >= m:
            raise ValueError("(S, F) is not a decision state (no pending task or no idle machine)")
        # rows among finished tasks only are constants now -> check them and drop them
        fixed = set(finished)
        kept = []
        for row in E:
            if any(row[i] != 0 for i in range(n) if i not in fixed):
                kept.append(row)
            elif sum(row[i] * inst.d0[i] for i in range(n)) > 1e-6:
                raise ValueError("realized durations contradict the finishing order")
        return self.build_tree(running, pending, tuple(finished), kept, last, m - len(running), path)

    def flatten_final_phase(self, node):
        """Paper Table 4/7: below the last real decision, the adversary node (n-1, n-m-1) has every
        remaining finishing order as a direct leaf child. Returns True if the subtree contains a
        real scheduler decision (scheduler node with more than one option)."""
        if node.kind == "LEAF":
            return False
        has_real = [self.flatten_final_phase(c) for c in node.children]
        if any(has_real) or (node.kind == "SCHEDULER" and len(node.children) > 1):
            return True
        if node.kind == "ADVERSARY":
            node.children = list(self._leaves(node))
        return False

    def _count_nodes(self, root):
        for key in ("scheduler_nodes", "real_decisions", "adversary_nodes", "leaves"):
            self.stats[key] = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node.kind == "SCHEDULER":
                self.stats["scheduler_nodes"] += 1
                self.stats["real_decisions"] += len(node.children) > 1
            elif node.kind == "ADVERSARY":
                self.stats["adversary_nodes"] += 1
            else:
                self.stats["leaves"] += 1
            stack.extend(node.children)

    # ------------------------------------------------------------------ model
    def build_model(self, root):
        m = gp.Model("MILO_Adaptive_Robust", env=self.env)
        self.model = m
        self._build(root, {})
        m.setObjective(root.val, GRB.MAXIMIZE)
        m.update()
        return m

    def _shared_var(self, i, path):
        inst = self.inst
        if inst.is_discrete:
            return self.model.addVar(lb=inst.d0[i], ub=inst.d0[i] + inst.d_bar[i],
                                     name=f"d[{task_name(i)}]@{path_str(path)}")
        return self.model.addVar(lb=0.0, ub=1.0, name=f"u[{task_name(i)}]@{path_str(path)}")

    def _build(self, node, u_map):
        m, n = self.model, self.n
        if self.enforce_na and node.kind == "SCHEDULER" and len(node.children) > 1:
            # node is in D: the durations of F(node) are shared by every leaf below it (paper eq. 4)
            new = [f for f in node.finished if f not in u_map]
            if new:
                u_map = dict(u_map)
                for f in new:
                    u_map[f] = self._shared_var(f, node.path)
        node.u_map = u_map

        if node.kind == "LEAF":
            node.val = self._build_leaf(node, u_map)
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

    def _build_leaf(self, node, u_map):
        m, inst, n = self.model, self.inst, self.n
        tag = path_str(node.path)
        if inst.is_discrete:
            scen = inst.scenarios
            lam = m.addVars(len(scen), vtype=GRB.BINARY, name=f"lam@{tag}")
            m.addConstr(lam.sum() == 1)
            d = [gp.quicksum(scen[s][i] * lam[s] for s in range(len(scen))) for i in range(n)]
            for i, shared in u_map.items():
                m.addConstr(d[i] == shared)
        else:
            u = [u_map[i] if i in u_map else m.addVar(lb=0.0, ub=1.0, name=f"u[{task_name(i)}]@{tag}")
                 for i in range(n)]
            m.addConstr(gp.quicksum(u) <= inst.gamma)
            d = [inst.d0[i] + inst.d_bar[i] * u[i] for i in range(n)]
        node.d = d
        z = m.addVar(vtype=GRB.BINARY, name=f"z@{tag}")
        node.z = z
        for row in node.E:
            m.addConstr(gp.quicksum(row[i] * d[i] for i in range(n) if row[i] != 0) <= self.M_z * z)
        return gp.quicksum(node.e[i] * d[i] for i in range(n) if node.e[i] != 0) - self.M_z * z

    # ------------------------------------------------------------------ solve
    def _optimize(self, objective, tag):
        m = self.model
        m.setObjective(objective, GRB.MAXIMIZE)
        m.Params.TimeLimit = self.cfg.get("time_limit_sec", 300)
        m.Params.MIPGap = self.cfg.get("mip_gap", 0.0)
        m.Params.Threads = self.cfg.get("threads", 0)
        # Big-M safety: a z "equal to 0" may be IntFeasTol, relaxing E-rows by M_z*IntFeasTol
        m.Params.IntFeasTol = self.cfg.get("int_feas_tol", 1e-9)
        m.Params.NumericFocus = self.cfg.get("numeric_focus", 1)
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
        if isinstance(x, gp.Var):
            return x.X
        if isinstance(x, (int, float)):
            return float(x)
        return x.getValue()

    def solve(self, state=None):
        """Solve the t = 0 problem, or (state = (started, finished)) the re-solve at that state."""
        inst = self.inst
        self.log.info("=" * 78)
        self.log.info("PAPER MODEL - monolithic MILO for the exact Adaptive Robust policy")
        self.log.info("=" * 78)
        self.log.info(inst.describe())
        self.log.info(f"prune_infeasible_branches={self.prune}, enforce_non_anticipativity={self.enforce_na}, "
                      f"flatten_final_phase={self.flatten}, M_z={self.M_z:.4f}, M_w={self.M_w:.4f}")
        if not self.enforce_na:
            self.log.info("NOTE: non-anticipativity is OFF -> this is a relaxation (upper bound), not the paper's model")

        t0 = time.perf_counter()
        if state is None:
            root = self.build_tree({}, tuple(range(self.n)), (), [], [0] * self.n, min(self.m, self.n), [])
        else:
            started, finished = state
            self.log.info(f"Rolling-horizon re-solve from state S={tasks_str(started)} F={tasks_str(finished)}")
            root = self.tree_from_state(started, finished)
        if self.flatten:
            self.flatten_final_phase(root)
        self._count_nodes(root)
        if state is None and not self.prune:
            expected = math.factorial(self.n) * self.m ** max(0, self.n - self.m)
            if self.stats["leaves"] != expected:
                self.log.info(f"  WARNING: {self.stats['leaves']} leaves, expected n!*m^(n-m) = {expected}")
        try:
            self.build_model(root)
        except gp.GurobiError as e:
            return self._error(str(e), t0)
        m = self.model
        size = dict(vars=m.NumVars, binaries=m.NumBinVars, constraints=m.NumConstrs)
        build_time = time.perf_counter() - t0
        self.log.info(f"Scenario tree: {self.stats}")
        self.log.info(f"MILO size: {size}  (build {build_time:.3f}s)")
        self.log.info("-" * 78)

        # The paper's method is ONE solve of max t_root; per-first-decision solves are a diagnostic
        # (and the decision rule of the rolling horizon) and are timed separately.
        worst_path = None
        try:
            value, status, solve_time = self._optimize(root.val, "full MILO, max t_root")
            t_diag = time.perf_counter()
            root_choices, root_status, root_paths = {}, {}, {}
            per_choice = self.cfg.get("solve_each_root_choice", True) or state is not None
            if per_choice and len(root.children) > 1:
                self.log.info("Solving the MILO once per FIRST decision (value of committing to it):")
                for c in root.children:
                    combo = c.path[-1][1]
                    v, s, _ = self._optimize(c.val, f"first decision start {tasks_str(combo)}")
                    root_choices[combo], root_status[combo] = v, s
                    root_paths[combo] = self._report_subtree(c, f"first decision start {tasks_str(combo)}")
            else:
                combo = root.children[0].path[-1][1] if len(root.children) == 1 else None
                if combo is not None:
                    root_choices[combo], root_status[combo] = value, status
                worst_path = self._report_subtree(root, "full MILO")
            root_choice_time = time.perf_counter() - t_diag
        except gp.GurobiError as e:
            return self._error(str(e), t0)
        total_time = time.perf_counter() - t0

        if root_choices and all(v is not None for v in root_choices.values()):
            best = min(root_choices.values())
            # paper Sec. 7.1: lexicographically smallest decision among ties (robust to float noise)
            root_choice = min(c for c, v in root_choices.items() if v <= best + self.tie_tol)
            if value is not None and abs(best - value) > 1e-4:
                self.log.info(f"  WARNING: min over first decisions ({best:.4f}) != full MILO ({value:.4f})")
            worst_path = root_paths.get(root_choice, worst_path)
        else:
            root_choice = None
        optimal = status == "OPTIMAL" and all(s == "OPTIMAL" for s in root_status.values())

        self.log.info("-" * 78)
        self.log.info("MILO RESULT")
        for combo, v in root_choices.items():
            mark = "  <== chosen" if combo == root_choice else ""
            self.log.info(f"  first decision start {tasks_str(combo)} -> AR worst-case makespan {fmt_val(v)}"
                          f"  ({root_status[combo]}){mark}")
        self.log.info(f"  Optimal AR worst-case makespan = {fmt_val(value)}  (status {status})")
        if not optimal:
            self.log.info("  WARNING: not every solve reached OPTIMAL -> value / decision are not certified")
        self.log.info(f"  time = {build_time + solve_time:.3f}s (build {build_time:.3f}s + solve {solve_time:.3f}s); "
                      f"per-first-decision solves {root_choice_time:.3f}s; total {total_time:.3f}s")
        return dict(model="milo" if self.enforce_na else "milo_no_na", value=value, status=status,
                    optimal=optimal, root_choice=root_choice, root_choices=root_choices,
                    root_status=root_status, worst_path=worst_path, stats=dict(self.stats), size=size,
                    time=build_time + solve_time, build_time=build_time, solve_time=solve_time,
                    root_choice_time=root_choice_time, total_time=total_time)

    def _error(self, msg, t0):
        self.log.info(f"  GUROBI ERROR: {msg}")
        if "size-limited" in msg:
            self.log.info("  The free pip license allows 2000 variables/constraints. Reduce n_tasks or "
                          "install a full (academic) license.")
        elapsed = time.perf_counter() - t0
        return dict(model="milo", value=None, status=f"ERROR: {msg}", optimal=False, root_choice=None,
                    root_choices={}, root_status={}, worst_path=None, stats=dict(self.stats), size=None,
                    time=elapsed, build_time=elapsed, solve_time=0.0, root_choice_time=0.0, total_time=elapsed)

    # ------------------------------------------------------------------ explain solution
    def _report_subtree(self, top, title):
        """Worst-case path of the current solution + (trace) the whole subtree with values.
        Returns the event path of the worst-case leaf."""
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
                opts = ", ".join(f"{self._edge(c, node)}:{self._v(c.val):.4f}" for c in node.children)
                revealed = self._revealed(node)
                self.log.info(f"      [{node.kind}] {path_str(node.path)}  options {opts} -> {self._edge(nxt, node)}"
                              + (f"  | revealed (shared by all options): {revealed}" if revealed else ""))
            node = nxt
        d = self._leaf_d(node)
        self.log.info(f"      [LEAF] {path_str(node.path)}  makespan {self._v(node.val):.4f}  d = {fmt_d(d)}")
        if self.trace:
            self._t(0, f"---- full tree for '{title}' (values of non-binding branches are only bounds) ----")
            self._dump(top, 0)
        return list(node.path)

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
            picked = [self._edge(c, node) for c, w in zip(node.children, node.w) if w.X < 0.5]
            extra = f" adversary keeps {picked}"
        revealed = self._revealed(node)
        self._t(depth, f"[{node.kind}] {path_str(node.path)} t={val:.4f}{extra}"
                       + (f" revealed: {revealed}" if revealed else ""))
        for c in node.children:
            self._dump(c, depth + 1)

    def _revealed(self, node):
        if not node.finished or not node.u_map:
            return ""
        parts = []
        for i in node.finished:
            var = node.u_map.get(i)
            if var is not None:
                d = var.X if self.inst.is_discrete else self.inst.d0[i] + self.inst.d_bar[i] * var.X
                parts.append(f"{task_name(i)}={d:.4f}")
        return ", ".join(parts)

    def _leaf_d(self, leaf):
        return [self._v(x) for x in leaf.d]

    def _leaves(self, node):
        if node.kind == "LEAF":
            yield node
        for c in node.children:
            yield from self._leaves(c)

    @staticmethod
    def _edge(child, parent):
        events = child.path[len(parent.path):]
        if len(events) == 1:
            kind, payload = events[0]
            return f"start {tasks_str(payload)}" if kind == "S" else f"{task_name(payload)} first"
        # flattened final phase: the whole remaining finishing order
        return "order " + ">".join(task_name(p) for k, p in events if k == "F")


def condition_instance(inst, finished, durations, tol=1e-6):
    """U_{S,F,D} (paper Sec. 3): d in U with d_k = D_k for every finished task k.

    `durations` maps task -> realized duration (a dict, or a list indexed by task). For budgeted U
    the finished tasks become constants (d_bar = 0) and their deviations u_k are taken out of
    Gamma; for discrete U only the scenarios that agree with the realized durations remain.
    """
    D = {k: float(durations[k]) for k in finished}
    name = f"{inst.name} | F={tasks_str(finished)}"
    if inst.is_discrete:
        scen = [s for s in inst.scenarios if all(abs(s[k] - v) <= tol for k, v in D.items())]
        if not scen:
            raise ValueError("no scenario of U matches the realized durations")
        return discrete_instance(name, scen, inst.n_machines)
    d0, d_bar, used = list(inst.d0), list(inst.d_bar), 0.0
    for k, v in D.items():
        u = (v - d0[k]) / d_bar[k] if d_bar[k] > 0 else (0.0 if abs(v - d0[k]) <= tol else -1.0)
        if u < -tol or u > 1 + tol:
            raise ValueError(f"realized duration of {task_name(k)} = {v} is outside U")
        used += min(max(u, 0.0), 1.0)
        d0[k], d_bar[k] = v, 0.0
    if used > inst.gamma + tol:
        raise ValueError(f"realized durations use {used:.6f} > Gamma = {inst.gamma} of the budget")
    return Instance(name=name, d0=d0, d_bar=d_bar, gamma=max(inst.gamma - used, 0.0),
                    n_machines=inst.n_machines)


def _make_solver(inst, cfg, logger, gurobi_log=None):
    milo_cfg = dict(cfg.get("milo", {}))
    log_cfg = cfg.get("logging", {})
    return MiloSolver(inst, milo_cfg, logger, trace=log_cfg.get("trace_tree", True),
                      trace_max_depth=log_cfg.get("trace_max_depth", 1000), gurobi_log=gurobi_log)


def solve_milo(inst, cfg, logger, gurobi_log=None):
    return _make_solver(inst, cfg, logger, gurobi_log).solve()


def solve_milo_from_state(inst, cfg, logger, started, finished, durations, gurobi_log=None):
    """Rolling-horizon step: re-solve the MILO at decision state (S, F) with the realized durations
    of F fixed. `next_tasks` in the result is the task(s) to start on the idle machine(s)."""
    cond = condition_instance(inst, finished, durations)
    res = _make_solver(cond, cfg, logger, gurobi_log).solve(state=(tuple(started), tuple(finished)))
    res["next_tasks"] = res["root_choice"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--reference", type=int, default=None, help="index into reference_instances")
    ap.add_argument("--seed", type=int, default=None, help="seed for a random instance")
    ap.add_argument("--started", type=int, nargs="*", default=None,
                    help="rolling horizon: started tasks in start order (1-based, T1 = 1)")
    ap.add_argument("--finished", type=int, nargs="*", default=[],
                    help="rolling horizon: finished tasks in finish order (1-based)")
    ap.add_argument("--durations", type=float, nargs="*", default=[],
                    help="rolling horizon: realized durations of --finished, same order")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst = instance_from_args(cfg, args.reference, args.seed)
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"milo_{stamp}.log")
    logger = setup_logger("milo", log_path, console=True)
    glog = os.path.join(log_dir, f"milo_{stamp}_gurobi.log") if cfg.get("milo", {}).get("write_gurobi_log") else None
    if args.started is not None:
        if len(args.durations) != len(args.finished):
            ap.error("--durations needs one value per --finished task")
        started = [t - 1 for t in args.started]
        finished = [t - 1 for t in args.finished]
        res = solve_milo_from_state(inst, cfg, logger, started, finished,
                                    dict(zip(finished, args.durations)), gurobi_log=glog)
        nxt = tasks_str(res["next_tasks"]) if res["next_tasks"] else "?"
        logger.info(f"Rolling-horizon decision: start {nxt} next")
    else:
        res = solve_milo(inst, cfg, logger, gurobi_log=glog)
        if inst.expected_makespan is not None and res["value"] is not None:
            ok = abs(res["value"] - inst.expected_makespan) <= cfg.get("benchmark", {}).get("tolerance", 1e-4)
            logger.info(f"Reference check: expected {inst.expected_makespan:.4f}, got {res['value']:.4f} -> "
                        f"{'PASS' if ok else 'FAIL'}")
    logger.info(f"Full log written to {log_path}")
    close_logger(logger)


if __name__ == "__main__":
    main()
