"""
OUR IDEA: Hybrid Minimax Tree Search for Adaptive Robust parallel machine scheduling.

Generic version of scheduling.py (any number of tasks, machines and Gamma).

Game tree (built recursively, never stored):
    * SCHEDULER node (minimizer): machines are idle and tasks are pending. The
      scheduler picks which pending task(s) to start. At the root all m machines are
      idle, so it picks a set of min(m, n) tasks; afterwards exactly one machine frees
      up at a time, so it picks one task.
    * ADVERSARY node (maximizer): all machines are busy (or nothing is left to start).
      The adversary picks which running task finishes first. Choosing task f adds the
      E-rows C_f - C_r <= 0 for every other running task r.
    * LEAF: all tasks finished. We solve ONE independent LP:
          max  C_last(d)   s.t.  E d <= 0,  d = d0 + d_bar*u,  u in [0,1],  sum u <= Gamma
      Infeasible LP -> -inf (the adversary cannot make that sequence happen).

Backward induction with Alpha-Beta Pruning: adversary = max over children, scheduler = min over children.

Known relaxation: by forcing the adversary to choose a discrete completion order
and ignoring cross-branch non-anticipativity, we restrict the adversary's full 
continuous strategy space. The result is therefore a LOWER BOUND on the true AR
makespan computed by ar_milo.py (as discussed).

Run standalone:
    python3 ar_minimax.py                  # random instance from config.json
    python3 ar_minimax.py --reference 0    # reference instance #0 (scheduling.py example)
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

NEG_INF = float("-inf")


def sub_expr(expr1, expr2):
    """Helper: difference of two linear expressions (to build E-matrix rows)."""
    return [expr1[i] - expr2[i] for i in range(len(expr1))]


class MinimaxSolver:
    def __init__(self, inst: Instance, cfg: dict, logger, trace=True, trace_max_depth=1000):
        self.inst = inst
        self.n = inst.n_tasks
        self.m = inst.n_machines
        self.log = logger
        self.trace = trace
        self.trace_max_depth = trace_max_depth
        self.prune = cfg.get("prune_infeasible_branches", True)
        self.env = make_env()
        self.stats = dict(scheduler_nodes=0, adversary_nodes=0, leaves=0, leaf_lps=0,
                          infeasible_leaves=0, pruned_branches=0, feasibility_lps=0)
        self.anticipation = []   # detected inconsistent "revealed" durations
        self.root_choices = {}   # first decision -> worst-case makespan

    # ------------------------------------------------------------------ logging
    def _t(self, depth, msg):
        if self.trace and depth <= self.trace_max_depth:
            self.log.debug("|  " * depth + msg)

    # ------------------------------------------------------------------ LPs
    def solve_leaf_lp(self, E_matrix, e_vector):
        """Continuous worst case for one fixed sequence of events (one leaf)."""
        n = self.n
        model = gp.Model("Leaf_LP", env=self.env)
        u = model.addVars(n, lb=0.0, ub=1.0, name="u")
        d = model.addVars(n, name="d")
        model.addConstr(gp.quicksum(u[i] for i in range(n)) <= self.inst.gamma)
        for i in range(n):
            model.addConstr(d[i] == self.inst.d0[i] + self.inst.d_bar[i] * u[i])
        for row in E_matrix:
            model.addConstr(gp.quicksum(row[i] * d[i] for i in range(n)) <= 0)
        model.setObjective(gp.quicksum(e_vector[i] * d[i] for i in range(n)), GRB.MAXIMIZE)
        model.optimize()
        self.stats["leaf_lps"] += 1
        if model.Status == GRB.OPTIMAL:
            val, d_vals, u_vals = model.ObjVal, [d[i].X for i in range(n)], [u[i].X for i in range(n)]
        else:
            val, d_vals, u_vals = NEG_INF, None, None
        model.dispose()
        return val, d_vals, u_vals

    def _feasible(self, E_matrix):
        self.stats["feasibility_lps"] += 1
        return lp_feasible(self.env, self.inst, E_matrix)

    # ------------------------------------------------------------------ tree
    def solve(self):
        inst = self.inst
        self.log.info("=" * 78)
        self.log.info("OUR IDEA - Hybrid Minimax tree + independent LP at every leaf")
        self.log.info("=" * 78)
        self.log.info(inst.describe())
        self.log.info(f"prune_infeasible_branches={self.prune}")
        self.log.info("Reading the trace: [SCHEDULER] = min node, [ADVERSARY] = max node, [LEAF] = LP.")
        self.log.info("Completion times C(Ti) are written as sums of durations d_i.")
        self.log.info("-" * 78)

        t0 = time.perf_counter()
        zero = [0] * self.n
        
        # Initiate Minimax with alpha = -infinity, beta = infinity
        value, leaf = self.scheduler_turn(running={}, pending=tuple(range(self.n)), finished=(),
                                          E_matrix=[], start_expr=zero, n_slots=min(self.m, self.n),
                                          path=[], depth=0, alpha=NEG_INF, beta=float("inf"))
        
        elapsed = time.perf_counter() - t0
        root_choice = min(self.root_choices, key=lambda c: self.root_choices[c])

        self.log.info("-" * 78)
        self.log.info("MINIMAX RESULT")
        for combo, val in self.root_choices.items():
            mark = "  <== chosen" if combo == root_choice else ""
            self.log.info(f"  first decision start {tasks_str(combo)} -> worst-case makespan {fmt_val(val)}{mark}")
        self.log.info(f"  Guaranteed worst-case makespan (lower bound on true AR) = {fmt_val(value)}")
        self.log.info(f"  Worst-case scenario under optimal play: {path_str(leaf['path'])}")
        self.log.info(f"    durations chosen by adversary at that leaf: {fmt_d(leaf['d'])}")
        self.log.info(f"    budget used sum(u) = {sum(leaf['u']):.4f} / Gamma = {inst.gamma}")
        self.log.info(f"  Stats: {self.stats}, time = {elapsed:.3f}s")
        self.log.info(f"  Anticipation cases detected (adversary used different durations for an "
                      f"already-finished task in different futures): {len(self.anticipation)}")
        for a in self.anticipation[:10]:
            self.log.info("    " + self._fmt_anticipation(a))
        if len(self.anticipation) > 10:
            self.log.info(f"    ... {len(self.anticipation) - 10} more (see trace above)")
        return dict(model="minimax", value=value, root_choice=root_choice,
                    root_choices=dict(self.root_choices), worst_leaf=leaf, stats=dict(self.stats),
                    anticipation=list(self.anticipation), time=elapsed, status="OPTIMAL")

    def scheduler_turn(self, running, pending, finished, E_matrix, start_expr, n_slots, path, depth, alpha, beta):
        """Scheduler (min): start k = min(idle machines, pending) tasks at time start_expr."""
        self.stats["scheduler_nodes"] += 1
        k = min(n_slots, len(pending))
        node = path_str(path)
        self._t(depth, f"[SCHEDULER] {node}")
        self._t(depth, f"   running={self._fmt_running(running)} pending={tasks_str(pending)} "
                       f"finished={tasks_str(finished)}; must start {k} task(s) at time {fmt_expr(start_expr)}")

        results = []
        best_val = float('inf')
        best_combo = None
        best_leaf = None

        for combo in itertools.combinations(pending, k):
            new_running = dict(running)
            for j in combo:
                expr = list(start_expr)
                expr[j] += 1                       # C_j = start time + d_j
                new_running[j] = expr
            rest = tuple(t for t in pending if t not in combo)
            self._t(depth + 1, f"option: start {tasks_str(combo)} -> "
                               + ", ".join(f"C({task_name(j)}) = {fmt_expr(new_running[j])}" for j in combo))
            
            val, leaf = self.adversary_turn(new_running, rest, finished, E_matrix, path + [("S", combo)], depth + 1, alpha, beta)
            results.append((combo, val, leaf))

            if val < best_val:
                best_val = val
                best_combo = combo
                best_leaf = leaf
            
            # Alpha-Beta Pruning: Minimizer updates beta
            beta = min(beta, best_val)
            if best_val <= alpha:
                self.stats["pruned_branches"] += 1
                self._t(depth + 1, f"   Alpha pruning! best_val {best_val:.4f} <= alpha {alpha:.4f} (pruned remaining combinations)")
                break

        if depth == 0:
            self.root_choices = {c: v for c, v, _ in results}
        if len(results) > 1:
            self._check_anticipation(node, finished, results)
        
        self._t(depth, f"[SCHEDULER] {node} -> MIN of explored options => start {tasks_str(best_combo)} (value {fmt_val(best_val)})")
        return best_val, best_leaf

    def adversary_turn(self, running, pending, finished, E_matrix, path, depth, alpha, beta):
        """Adversary (max): decide which running task finishes first."""
        self.stats["adversary_nodes"] += 1
        node = path_str(path)
        self._t(depth, f"[ADVERSARY] {node}")
        self._t(depth, f"   running={self._fmt_running(running)}; who finishes first?")

        results = []
        best_val = NEG_INF
        best_f = None
        best_leaf = None

        order = sorted(running)
        for f in order:
            new_rows = [sub_expr(running[f], running[r]) for r in order if r != f]
            E2 = E_matrix + new_rows
            child_path = path + [("F", f)]
            cons = "; ".join(fmt_row(r) for r in new_rows) if new_rows else "(only task left)"
            self._t(depth + 1, f"option: {task_name(f)} finishes first  [adds {cons}]")
            
            if self.prune and new_rows and not self._feasible(E2):
                self.stats["pruned_branches"] += 1
                self._t(depth + 1, "   impossible sequence under the budget Gamma -> -inf (pruned)")
                val, leaf = NEG_INF, None
            else:
                rest_running = {r: e for r, e in running.items() if r != f}
                new_finished = finished + (f,)
                if not rest_running and not pending:
                    val, leaf = self.leaf(E2, running[f], child_path, depth + 1)
                elif pending:
                    # the machine that ran f is idle at time C_f -> scheduler decides
                    val, leaf = self.scheduler_turn(rest_running, pending, new_finished, E2, running[f], 1,
                                                    child_path, depth + 1, alpha, beta)
                else:
                    val, leaf = self.adversary_turn(rest_running, pending, new_finished, E2, child_path, depth + 1, alpha, beta)
            
            results.append((f, val, leaf))
            
            if val > best_val:
                best_val = val
                best_f = f
                best_leaf = leaf
            
            # Alpha-Beta Pruning: Maximizer updates alpha
            alpha = max(alpha, best_val)
            if best_val >= beta:
                self.stats["pruned_branches"] += 1
                self._t(depth + 1, f"   Beta pruning! best_val {best_val:.4f} >= beta {beta:.4f} (pruned remaining permutations)")
                break

        self._t(depth, f"[ADVERSARY] {node} -> MAX of explored options => {task_name(best_f)} finishes first (value {fmt_val(best_val)})")
        return best_val, best_leaf

    def leaf(self, E_matrix, e_vector, path, depth):
        self.stats["leaves"] += 1
        val, d_vals, u_vals = self.solve_leaf_lp(E_matrix, e_vector)
        self._t(depth, f"[LEAF] {path_str(path)}")
        self._t(depth, f"   makespan = C(last) = {fmt_expr(e_vector)}; sequence constraints: "
                       + ("; ".join(fmt_row(r) for r in E_matrix) or "none"))
        if val == NEG_INF:
            self.stats["infeasible_leaves"] += 1
            self._t(depth, "   LP infeasible -> -inf")
        else:
            self._t(depth, f"   LP -> makespan {val:.4f} with d = {fmt_d(d_vals)} (sum u = {sum(u_vals):.4f})")
        return val, dict(path=path, d=d_vals, u=u_vals, value=val)

    # ------------------------------------------------------------------ analysis
    def _check_anticipation(self, node, finished, results):
        """At a scheduler node the durations of `finished` tasks are already revealed.
        In reality they must be identical in every option the scheduler compares. In the
        minimax each option's worst case comes from its own LP, so they may differ."""
        for i in finished:
            per_option = [(c, leaf["d"][i], v) for c, v, leaf in results if leaf is not None and leaf["d"] is not None]
            if len(per_option) < 2:
                continue
            vals = [x[1] for x in per_option]
            if max(vals) - min(vals) > 1e-4:
                a = dict(node=node, task=i, per_option=per_option)
                self.anticipation.append(a)
                self._t(0, "   [ANTICIPATION] " + self._fmt_anticipation(a))

    @staticmethod
    def _fmt_anticipation(a):
        opts = ", ".join(f"if start {tasks_str(c)}: d{a['task'] + 1}={dv:.4f} (makespan {fmt_val(v)})"
                         for c, dv, v in a["per_option"])
        return (f"at {a['node']}: {task_name(a['task'])} already finished, but the adversary's worst cases "
                f"use different durations for it -> {opts}")

    @staticmethod
    def _fmt_running(running):
        return "{" + ", ".join(f"{task_name(j)}: C={fmt_expr(e)}" for j, e in sorted(running.items())) + "}"


def solve_minimax(inst, cfg, logger):
    log_cfg = cfg.get("logging", {})
    solver = MinimaxSolver(inst, cfg.get("minimax", {}), logger,
                           trace=log_cfg.get("trace_tree", True),
                           trace_max_depth=log_cfg.get("trace_max_depth", 1000))
    return solver.solve()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--reference", type=int, default=None, help="index into reference_instances")
    ap.add_argument("--seed", type=int, default=None, help="seed for a random instance")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inst = instance_from_args(cfg, args.reference, args.seed)
    log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    log_path = os.path.join(log_dir, f"minimax_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger("minimax", log_path, console=True)
    res = solve_minimax(inst, cfg, logger)
    if inst.expected_makespan is not None:
        ok = abs(res["value"] - inst.expected_makespan) <= cfg.get("benchmark", {}).get("tolerance", 1e-4)
        logger.info(f"Reference check: expected {inst.expected_makespan:.4f}, got {res['value']:.4f} -> "
                    f"{'PASS' if ok else 'FAIL'}")
    logger.info(f"Full trace written to {log_path}")
    close_logger(logger)


if __name__ == "__main__":
    main()