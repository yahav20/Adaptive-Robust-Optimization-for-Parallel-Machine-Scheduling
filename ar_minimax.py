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
    * LEAF: all tasks finished. We solve ONE LP over the state-induced uncertainty set U_{S,F,D}:
          max  C_last(d)   s.t.  E d <= 0,  d_k = D_k (for k in F),
                                 d = d0 + d_bar*u,  u in [0,1],  sum u <= Gamma
      If no d in U satisfies E d <= 0, we fall back to the lowest value in U (d = d0).

Discretization & Lower Bound guarantee:
    At any scheduler node (S, F) with F != empty, the continuous durations of completed
    tasks k in F are discretized using the candidate realizations d* propagated from all
    feasible leaves in the subtree. Enforcing non-anticipativity (d_sigma(k) = d_delta(k)
    for all k in F) over this discrete subset of U restricts the adversary compared to
    the full continuous MILO, guaranteeing a true LOWER BOUND on the exact AR makespan:
        Minimax <= MILO (exact AR).

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
    def _lowest_in_U(self, e_vector, fixed_d=None):
        """Fallback when E_sigma d <= 0 has no feasible solution:
        pick the lowest value that still keeps d inside U (respecting fixed_d if possible)."""
        n = self.n
        fixed_d = fixed_d or {}
        u_vals = [0.0] * n
        d_vals = list(self.inst.d0)
        for k, val_k in fixed_d.items():
            d_vals[k] = val_k
            if self.inst.d_bar[k] > 0:
                u_vals[k] = max(0.0, min(1.0, (val_k - self.inst.d0[k]) / self.inst.d_bar[k]))
        if sum(u_vals) > self.inst.gamma + TOL:
            u_vals = [0.0] * n
            d_vals = list(self.inst.d0)
        val = sum(e_vector[i] * d_vals[i] for i in range(n))
        return val, d_vals, u_vals

    def solve_leaf_lp(self, E_matrix, e_vector, fixed_d=None):
        """Continuous worst case for one fixed sequence of events (one leaf) over U_{S,F,D}."""
        n = self.n
        fixed_d = fixed_d or {}
        model = gp.Model("Leaf_LP", env=self.env)
        u = model.addVars(n, lb=0.0, ub=1.0, name="u")
        d = model.addVars(n, name="d")
        model.addConstr(gp.quicksum(u[i] for i in range(n)) <= self.inst.gamma)
        for i in range(n):
            model.addConstr(d[i] == self.inst.d0[i] + self.inst.d_bar[i] * u[i])
        for k, val_k in fixed_d.items():
            model.addConstr(d[k] == val_k)
        for row in E_matrix:
            model.addConstr(gp.quicksum(row[i] * d[i] for i in range(n)) <= 0)
        model.setObjective(gp.quicksum(e_vector[i] * d[i] for i in range(n)), GRB.MAXIMIZE)
        model.optimize()
        self.stats["leaf_lps"] += 1
        if model.Status == GRB.OPTIMAL:
            val = model.ObjVal
            d_vals = [d[i].X for i in range(n)]
            u_vals = [u[i].X for i in range(n)]
            feasible = True
        else:
            val, d_vals, u_vals = self._lowest_in_U(e_vector, fixed_d)
            feasible = False
        model.dispose()
        return val, d_vals, u_vals, feasible

    def _feasible(self, E_matrix, fixed_d=None):
        self.stats["feasibility_lps"] += 1
        if not fixed_d:
            return lp_feasible(self.env, self.inst, E_matrix)
        n = self.n
        model = gp.Model("Feas_LP", env=self.env)
        u = model.addVars(n, lb=0.0, ub=1.0)
        d = model.addVars(n)
        model.addConstr(gp.quicksum(u[i] for i in range(n)) <= self.inst.gamma)
        for i in range(n):
            model.addConstr(d[i] == self.inst.d0[i] + self.inst.d_bar[i] * u[i])
        for k, val_k in fixed_d.items():
            model.addConstr(d[k] == val_k)
        for row in E_matrix:
            model.addConstr(gp.quicksum(row[i] * d[i] for i in range(n)) <= 0)
        model.optimize()
        ok = (model.Status == GRB.OPTIMAL)
        model.dispose()
        return ok

    # ------------------------------------------------------------------ tree
    def solve(self):
        inst = self.inst
        self.log.info("=" * 78)
        self.log.info("OUR IDEA - Hybrid Minimax tree + Leaf LP + Discretized Non-Anticipativity")
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
                                          path=[], depth=0, alpha=NEG_INF, beta=float("inf"),
                                          fixed_d={}, candidate_pool=None)

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

    def scheduler_turn(self, running, pending, finished, E_matrix, start_expr, n_slots,
                       path, depth, alpha, beta, fixed_d=None, candidate_pool=None):
        """Scheduler (min): start k = min(idle machines, pending) tasks at time start_expr."""
        self.stats["scheduler_nodes"] += 1
        fixed_d = fixed_d or {}
        k = min(n_slots, len(pending))
        node = path_str(path)
        self._t(depth, f"[SCHEDULER] {node}")
        self._t(depth, f"   running={self._fmt_running(running)} pending={tasks_str(pending)} "
                       f"finished={tasks_str(finished)}; must start {k} task(s) at time {fmt_expr(start_expr)}")

        combos = list(itertools.combinations(pending, k))
        unfixed_finished = [i for i in finished if i not in fixed_d]

        # Case 1: Only 1 option OR all finished tasks are already fixed in fixed_d (e.g. root node F=())
        if len(combos) <= 1 or not unfixed_finished:
            results = []
            best_val = float("inf")
            best_combo = None
            best_leaf = None

            for combo in combos:
                new_running = dict(running)
                for j in combo:
                    expr = list(start_expr)
                    expr[j] += 1
                    new_running[j] = expr
                rest = tuple(t for t in pending if t not in combo)
                self._t(depth + 1, f"option: start {tasks_str(combo)} -> "
                                   + ", ".join(f"C({task_name(j)}) = {fmt_expr(new_running[j])}" for j in combo))

                val, leaf = self.adversary_turn(new_running, rest, finished, E_matrix,
                                                path + [("S", combo)], depth + 1, alpha, beta,
                                                fixed_d, candidate_pool)
                results.append((combo, val, leaf))

                if val < best_val:
                    best_val = val
                    best_combo = combo
                    best_leaf = leaf

                beta = min(beta, best_val)
                if best_val <= alpha:
                    self.stats["pruned_branches"] += 1
                    self._t(depth + 1, f"   Alpha pruning! best_val {best_val:.4f} <= alpha {alpha:.4f}")
                    break

            if depth == 0:
                self.root_choices = {c: v for c, v, _ in results}
            self._t(depth, f"[SCHEDULER] {node} -> MIN of explored options => start {tasks_str(best_combo)} (value {fmt_val(best_val)})")
            return best_val, best_leaf

        # Case 2: Multiple scheduler options AND newly finished tasks in F.
        # Step 2a: Probe children to collect all candidate discrete realizations d* from feasible leaves.
        probe_results = []
        local_pool = []

        for combo in combos:
            new_running = dict(running)
            for j in combo:
                expr = list(start_expr)
                expr[j] += 1
                new_running[j] = expr
            rest = tuple(t for t in pending if t not in combo)
            val, leaf = self.adversary_turn(new_running, rest, finished, E_matrix,
                                            path + [("S", combo)], depth + 1, NEG_INF, float("inf"),
                                            fixed_d, local_pool)
            probe_results.append((combo, val, leaf))

        if candidate_pool is not None:
            candidate_pool.extend(local_pool)

        # Deduplicate discrete candidate realizations of D_F = {d_i : i in finished}
        candidate_D_list = []
        seen_D = set()
        # Prioritize the winning leaves of each branch first for better Alpha-Beta move ordering
        priority_d = [leaf["d"] for _, _, leaf in probe_results
                      if leaf is not None and leaf.get("feasible", False) and leaf.get("d") is not None]
        for d_vec in priority_d + local_pool:
            key = tuple(round(d_vec[i], 6) for i in finished)
            if key not in seen_D:
                seen_D.add(key)
                candidate_D_list.append({i: d_vec[i] for i in finished})

        # Also include nominal durations d0 for finished tasks as a guaranteed feasible baseline in U
        nom_key = tuple(round(self.inst.d0[i], 6) for i in finished)
        if nom_key not in seen_D:
            seen_D.add(nom_key)
            candidate_D_list.append({i: self.inst.d0[i] for i in finished})

        # Check if the winning leaves of all branches already agree on the durations of finished tasks
        winning_keys = {tuple(round(d_vec[i], 6) for i in finished) for d_vec in priority_d}
        if len(winning_keys) == 1:
            best_combo, best_val, best_leaf = min(probe_results, key=lambda x: x[1])
            self._t(depth, f"[SCHEDULER] {node} -> consistent D_F across branches => start {tasks_str(best_combo)} (value {fmt_val(best_val)})")
            return best_val, best_leaf

        # Step 2b: Discretized Non-Anticipative evaluation.
        # For each discrete candidate realization D_F of the completed tasks F,
        # the scheduler minimizes over combos, and the adversary picks the candidate D_F that maximizes that minimum.
        best_adv_val = NEG_INF
        best_adv_combo = None
        best_adv_leaf = None
        best_adv_results = []

        for cand_D in candidate_D_list:
            merged_fixed = dict(fixed_d)
            merged_fixed.update(cand_D)
            if not self._feasible(E_matrix, merged_fixed):
                continue

            sched_best_val = float("inf")
            sched_best_combo = None
            sched_best_leaf = None
            cand_results = []
            local_beta = beta

            for combo in combos:
                new_running = dict(running)
                for j in combo:
                    expr = list(start_expr)
                    expr[j] += 1
                    new_running[j] = expr
                rest = tuple(t for t in pending if t not in combo)
                val, leaf = self.adversary_turn(new_running, rest, finished, E_matrix,
                                                path + [("S", combo)], depth + 1,
                                                best_adv_val, local_beta, merged_fixed, candidate_pool)
                cand_results.append((combo, val, leaf))

                if val < sched_best_val:
                    sched_best_val = val
                    sched_best_combo = combo
                    sched_best_leaf = leaf

                local_beta = min(local_beta, sched_best_val)
                if sched_best_val <= max(alpha, best_adv_val):
                    self.stats["pruned_branches"] += 1
                    break

            if sched_best_val > best_adv_val:
                best_adv_val = sched_best_val
                best_adv_combo = sched_best_combo
                best_adv_leaf = sched_best_leaf
                best_adv_results = cand_results

            if best_adv_val >= beta:
                self.stats["pruned_branches"] += 1
                break

        if best_adv_leaf is None:
            best_adv_combo, best_adv_val, best_adv_leaf = min(probe_results, key=lambda x: x[1])
        elif len(best_adv_results) > 1:
            self._check_anticipation(node, finished, best_adv_results)

        self._t(depth, f"[SCHEDULER] {node} -> MIN under discretized D_F => start {tasks_str(best_adv_combo)} (value {fmt_val(best_adv_val)})")
        return best_adv_val, best_adv_leaf

    def adversary_turn(self, running, pending, finished, E_matrix, path, depth,
                       alpha, beta, fixed_d=None, candidate_pool=None):
        """Adversary (max): decide which running task finishes first."""
        self.stats["adversary_nodes"] += 1
        fixed_d = fixed_d or {}
        node = path_str(path)
        self._t(depth, f"[ADVERSARY] {node}")
        self._t(depth, f"   running={self._fmt_running(running)}; who finishes first?")

        order = sorted(running)
        feasible_results = []
        infeasible_results = []
        best_val = NEG_INF
        best_f = None
        best_leaf = None

        for f in order:
            new_rows = [sub_expr(running[f], running[r]) for r in order if r != f]
            E2 = E_matrix + new_rows
            child_path = path + [("F", f)]
            cons = "; ".join(fmt_row(r) for r in new_rows) if new_rows else "(only task left)"
            self._t(depth + 1, f"option: {task_name(f)} finishes first  [adds {cons}]")

            if self.prune and new_rows and not self._feasible(E2, fixed_d):
                self.stats["pruned_branches"] += 1
                self._t(depth + 1, "   impossible sequence under the budget Gamma -> fallback to lowest in U")
                low_val, low_d, low_u = self._lowest_in_U(running[f], fixed_d)
                infeasible_results.append((f, low_val, dict(path=child_path, d=low_d, u=low_u, value=low_val, feasible=False)))
                continue

            rest_running = {r: e for r, e in running.items() if r != f}
            new_finished = finished + (f,)
            if not rest_running and not pending:
                val, leaf = self.leaf(E2, running[f], child_path, depth + 1, fixed_d, candidate_pool)
            elif pending:
                val, leaf = self.scheduler_turn(rest_running, pending, new_finished, E2, running[f], 1,
                                                child_path, depth + 1, alpha, beta, fixed_d, candidate_pool)
            else:
                val, leaf = self.adversary_turn(rest_running, pending, new_finished, E2,
                                                child_path, depth + 1, alpha, beta, fixed_d, candidate_pool)

            if leaf is not None and leaf.get("feasible", True):
                feasible_results.append((f, val, leaf))
                if val > best_val:
                    best_val = val
                    best_f = f
                    best_leaf = leaf

                # Alpha-Beta Pruning: Maximizer updates alpha over feasible branches
                alpha = max(alpha, best_val)
                if best_val >= beta:
                    self.stats["pruned_branches"] += 1
                    self._t(depth + 1, f"   Beta pruning! best_val {best_val:.4f} >= beta {beta:.4f} (pruned remaining permutations)")
                    break
            else:
                infeasible_results.append((f, val, leaf))

        # Only if NO feasible branch exists at all, pick the lowest value that keeps us in U
        if best_leaf is None and infeasible_results:
            best_f, best_val, best_leaf = min(infeasible_results, key=lambda x: x[1])

        self._t(depth, f"[ADVERSARY] {node} -> MAX of explored options => {task_name(best_f)} finishes first (value {fmt_val(best_val)})")
        return best_val, best_leaf

    def leaf(self, E_matrix, e_vector, path, depth, fixed_d=None, candidate_pool=None):
        self.stats["leaves"] += 1
        val, d_vals, u_vals, feasible = self.solve_leaf_lp(E_matrix, e_vector, fixed_d)
        self._t(depth, f"[LEAF] {path_str(path)}")
        self._t(depth, f"   makespan = C(last) = {fmt_expr(e_vector)}; sequence constraints: "
                       + ("; ".join(fmt_row(r) for r in E_matrix) or "none"))
        if not feasible:
            self.stats["infeasible_leaves"] += 1
            self._t(depth, f"   LP infeasible -> fallback to lowest in U: {val:.4f} with d = {fmt_d(d_vals)}")
        else:
            if candidate_pool is not None and d_vals is not None:
                candidate_pool.append(d_vals)
            self._t(depth, f"   LP -> makespan {val:.4f} with d = {fmt_d(d_vals)} (sum u = {sum(u_vals):.4f})")
        return val, dict(path=path, d=d_vals, u=u_vals, value=val, feasible=feasible)

    # ------------------------------------------------------------------ analysis
    def _check_anticipation(self, node, finished, results):
        """At a scheduler node the durations of `finished` tasks are already revealed."""
        for i in finished:
            per_option = [(c, leaf["d"][i], v) for c, v, leaf in results
                          if leaf is not None and leaf.get("feasible", False) and leaf["d"] is not None]
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