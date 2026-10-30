"""
UPPER BOUND: Hybrid Minimax Tree Search with independent leaf LPs (non-anticipativity relaxed).
(Standalone copy of the ar_minimax.py machinery: persistent incremental LP, closed-form / witness
 fast paths, memoization, tail collapse, alpha-beta & move ordering)

Same game tree as ar_minimax.py (built recursively, never stored):
    * SCHEDULER node (minimizer): machines are idle and tasks are pending. The
      scheduler picks which pending task(s) to start. At the root all m machines are
      idle, so it picks a set of min(m, n) tasks; afterwards exactly one machine frees
      up at a time, so it picks one task.
    * ADVERSARY node (maximizer): all machines are busy (or nothing is left to start).
      The adversary picks which running task finishes first. Choosing task f adds the
      E-rows C_f - C_r <= 0 for every other running task r.
    * LEAF: all tasks finished. We solve ONE independent LP over U:
          max  C_last(d)   s.t.  E d <= 0,  d = d0 + d_bar*u,  u in [0,1],  sum u <= Gamma
      If no d in U satisfies E d <= 0, we fall back to the lowest value in U (d = d0).
    * TAIL: once nothing is pending there is no scheduler decision left, so the max over all finish
      orders of the running tasks is  max_r max{C_r(d) : E d <= 0, d in U}  -> one LP per running task.

Difference to the lower bound (ar_minimax.py):
    The LB fixes the durations of finished tasks F to shared candidate values D_F at every scheduler
    node (non-anticipativity over a finite candidate set). Here every leaf LP picks its own d, so in
    sibling branches of a scheduler node the adversary may "re-decide" the durations of tasks that
    already finished. Dropping the non-anticipativity constraints d_sigma(i) = d_delta(i) of the exact
    MILO enlarges the adversary's feasible set, which guarantees a true UPPER BOUND:
        Minimax LB <= MILO (exact AR) <= Minimax UB.
    Leaf values depend only on their own path, so alpha-beta pruning is exact for this value.
    The UB equals the MILO with milo.enforce_non_anticipativity=false (the same relaxed game).

Options (config section "minimax_ub", falls back to "minimax"):
    prune_infeasible_branches  drop finish orders that are impossible under Gamma (default true)
    alpha_beta                 false = solve every leaf LP (same value, for checking)
    exact_root_values          search every first decision with an open window, so root_choices holds
                               the exact UB value of each first decision (default true)
    tie_tol                    first-decision tie tolerance (lexicographic among ties, like ar_milo.py)

Run standalone:
    python3 ar_minimax_ub.py                  # random instance from config/config.json
    python3 ar_minimax_ub.py --reference 0    # reference instance #0 (scheduling.py example)
"""

import argparse
import itertools
import os
import time
from collections import deque

import gurobipy as gp
from gurobipy import GRB

from ar_common import (
    TOL, Instance, close_logger, fmt_d, fmt_expr, fmt_row, fmt_val, instance_from_args,
    load_config, make_env, path_str, setup_logger, task_name, tasks_str,
)

NEG_INF = float("-inf")
POS_INF = float("inf")


def sub_expr(expr1, expr2):
    """Helper: difference of two linear expressions (to build E-matrix rows)."""
    return tuple(expr1[i] - expr2[i] for i in range(len(expr1)))


class MinimaxUBSolver:
    def __init__(self, inst: Instance, cfg: dict, logger, trace=True, trace_max_depth=1000):
        self.inst = inst
        self.n = inst.n_tasks
        self.m = inst.n_machines
        self.log = logger
        self.trace = trace and logger.isEnabledFor(10)  # 10 = logging.DEBUG
        self.trace_max_depth = trace_max_depth
        self.prune = cfg.get("prune_infeasible_branches", True)
        self.alpha_beta = cfg.get("alpha_beta", True)
        self.exact_root_values = cfg.get("exact_root_values", True)
        self.tie_tol = cfg.get("tie_tol", 1e-6)
        self.env = make_env()
        self._trace_lines = []

        self.stats = dict(
            scheduler_nodes=0,
            adversary_nodes=0,
            leaves=0,
            leaf_lps=0,
            infeasible_leaves=0,
            pruned_branches=0,
            prune_alpha=0,
            prune_beta=0,
            prune_tail_bound=0,
            prune_infeasible=0,
            feasibility_lps=0,
            fast_feas_hits=0,
            greedy_hits=0,
            witness_hits=0,
            cache_hits=0,
            na_relaxed_nodes=0,
            gurobi_calls=0,
            gurobi_time=0.0,
        )
        self.anticipation = []   # scheduler nodes where sibling worst cases disagree on a finished task
        self.root_choices = {}   # first decision -> worst-case makespan (upper bound)
        self.root_leaves = {}    # first decision -> worst-case leaf

        # Caches for memoization (no fixed durations here, so E alone is the key)
        self._feas_cache = {}
        self._leaf_cache = {}
        # Recently seen feasible duration vectors: a cheap feasibility certificate before calling Gurobi
        self._witnesses = deque(maxlen=32)

        # Precompute task bounds for fast arithmetic checks
        self.d0 = list(inst.d0)
        self.d_bar = list(inst.d_bar)
        self.d_max = [self.d0[i] + self.d_bar[i] for i in range(self.n)]
        self.gamma = inst.gamma

        # Build ONE persistent Gurobi LP model for the entire search
        self._init_persistent_lp()

    def _init_persistent_lp(self):
        """Initialize a single reusable LP model with base variables u, d and budget constraint."""
        self.lp = gp.Model("Persistent_Leaf_LP_UB", env=self.env)
        self.lp.Params.OutputFlag = 0
        # Dual simplex (Method=1) is fastest when modifying constraints incrementally
        self.lp.Params.Method = 1

        self.u_vars = [self.lp.addVar(lb=0.0, ub=1.0, name=f"u_{i}") for i in range(self.n)]
        self.d_vars = [
            self.lp.addVar(lb=self.d0[i], ub=self.d_max[i], name=f"d_{i}")
            for i in range(self.n)
        ]
        self.lp.addConstr(gp.quicksum(self.u_vars) <= self.gamma, name="budget")
        for i in range(self.n):
            self.lp.addConstr(
                self.d_vars[i] - self.d_bar[i] * self.u_vars[i] == self.d0[i],
                name=f"def_d_{i}",
            )
        self.lp.update()
        self._active_E_constrs = []
        self._synced_E = ()

    def _dispose_lp(self):
        if getattr(self, "lp", None) is not None:
            self.lp.dispose()
            self.lp = None

    # ------------------------------------------------------------------ logging
    def _do_trace(self, depth):
        return self.trace and depth <= self.trace_max_depth

    def _t(self, depth, msg):
        # buffered: written once after the timed search (one logger call per line costs more than the search)
        if self.trace and depth <= self.trace_max_depth:
            self._trace_lines.append("|  " * depth + msg)

    # ------------------------------------------------------------------ Fast Math & LP
    def _lowest_in_U(self, e_vector):
        """Fallback when E_sigma d <= 0 has no feasible solution: the lowest value in U (d = d0)."""
        val = sum(e_vector[i] * self.d0[i] for i in range(self.n))
        return val, list(self.d0), [0.0] * self.n

    def _fast_check_E(self, E_matrix):
        """
        O(n * |E|) pure-Python arithmetic filter before calling Gurobi:
          Returns:
            False -> Proven INFEASIBLE (at least one row in E cannot be <= 0 under Gamma)
            True  -> Proven FEASIBLE (the baseline d0 satisfies E d <= 0)
            None  -> Undetermined (requires Gurobi LP check)
        """
        n = self.n
        baseline_feasible = True
        for row in E_matrix:
            # 1. Check baseline (u_i = 0 for all tasks)
            base_val = 0.0
            neg_gains = []
            for i in range(n):
                r_i = row[i]
                if r_i == 0:
                    continue
                base_val += r_i * self.d0[i]
                if r_i < 0 and self.d_bar[i] > 0:
                    neg_gains.append(-r_i * self.d_bar[i])

            if base_val > TOL:
                baseline_feasible = False
                # 2. Compute minimum possible value of row * d subject to sum u_i <= Gamma
                if not neg_gains or self.gamma <= 0.0:
                    return False
                neg_gains.sort(reverse=True)
                budget = self.gamma
                max_reduction = 0.0
                for gain in neg_gains:
                    if budget >= 1.0:
                        max_reduction += gain
                        budget -= 1.0
                    else:
                        max_reduction += gain * budget
                        break
                if base_val - max_reduction > TOL:
                    return False

        if baseline_feasible:
            return True
        return None

    def _sync_lp_state(self, E_matrix):
        """Update the E-matrix constraints on the persistent LP model.
        The search is depth-first, so consecutive E matrices share a prefix: only the rows after the
        common prefix are removed / added. Variable bounds never change (no fixed durations)."""
        prev = self._synced_E
        k, lim = 0, min(len(prev), len(E_matrix))
        while k < lim and prev[k] == E_matrix[k]:
            k += 1
        if k < len(self._active_E_constrs):
            self.lp.remove(self._active_E_constrs[k:])
            del self._active_E_constrs[k:]
        for row in E_matrix[k:]:
            expr = gp.LinExpr(
                [(row[i], self.d_vars[i]) for i in range(self.n) if row[i] != 0]
            )
            self._active_E_constrs.append(self.lp.addConstr(expr <= 0.0))
        self._synced_E = E_matrix

    def _optimize(self):
        self.stats["gurobi_calls"] += 1
        t0 = time.perf_counter()
        self.lp.optimize()
        self.stats["gurobi_time"] += time.perf_counter() - t0

    def _greedy_max(self, e_vector):
        """Closed-form max of e.d over U, ignoring E: spend the budget on the largest e_i * d_bar_i.
        It is an upper bound on the leaf LP, and its optimum whenever the maximizer also satisfies E."""
        n = self.n
        d_vals = list(self.d0)
        u_vals = [0.0] * n
        rem = self.gamma
        gains = sorted((i for i in range(n) if e_vector[i] > 0 and self.d_bar[i] > 0),
                       key=lambda i: -e_vector[i] * self.d_bar[i])
        for i in gains:
            if rem <= 0.0:
                break
            a = min(1.0, rem)
            u_vals[i] = a
            d_vals[i] += a * self.d_bar[i]
            rem -= a
        val = sum(e_vector[i] * d_vals[i] for i in range(n) if e_vector[i] != 0)
        return val, d_vals, u_vals

    def _satisfies(self, E_matrix, d_vals):
        """Is the full duration vector d_vals (assumed in U) a witness for E d <= 0?"""
        n = self.n
        for row in E_matrix:
            if sum(row[i] * d_vals[i] for i in range(n) if row[i] != 0) > TOL:
                return False
        return True

    def _feasible(self, E_matrix, hint=None):
        """Is {d in U : E d <= 0} non-empty? `hint` is an optional known-good vector."""
        fast_res = self._fast_check_E(E_matrix)
        if fast_res is not None:
            self.stats["fast_feas_hits"] += 1
            return fast_res

        if E_matrix in self._feas_cache:
            self.stats["cache_hits"] += 1
            return self._feas_cache[E_matrix]

        if (hint is not None and self._satisfies(E_matrix, hint)) or \
                any(self._satisfies(E_matrix, w) for w in self._witnesses):
            self.stats["witness_hits"] += 1
            self._feas_cache[E_matrix] = True
            return True

        self.stats["feasibility_lps"] += 1
        self._sync_lp_state(E_matrix)
        self.lp.setObjective(0.0, GRB.MAXIMIZE)
        self._optimize()
        ok = (self.lp.Status == GRB.OPTIMAL)
        if ok:
            self._witnesses.append([v.X for v in self.d_vars])
        self._feas_cache[E_matrix] = ok
        return ok

    def solve_leaf_lp(self, E_matrix, e_vector):
        """Continuous worst case for one fixed sequence of events (one leaf) over U, independently."""
        # Fast-path infeasibility check before touching Gurobi
        if self._fast_check_E(E_matrix) is False:
            self.stats["fast_feas_hits"] += 1
            val, d_vals, u_vals = self._lowest_in_U(e_vector)
            return val, d_vals, u_vals, False

        key = (E_matrix, tuple(e_vector))
        if key in self._leaf_cache:
            self.stats["cache_hits"] += 1
            return self._leaf_cache[key]

        # Fast path: the relaxation maximizer (ignoring E) already respects E -> it is the LP optimum
        greedy = self._greedy_max(e_vector)
        if self._satisfies(E_matrix, greedy[1]):
            self.stats["greedy_hits"] += 1
            res = (*greedy, True)
            self._witnesses.append(greedy[1])
            self._leaf_cache[key] = res
            self._feas_cache[E_matrix] = True
            return res

        self.stats["leaf_lps"] += 1
        self._sync_lp_state(E_matrix)
        obj_expr = gp.LinExpr(
            [(e_vector[i], self.d_vars[i]) for i in range(self.n) if e_vector[i] != 0]
        )
        self.lp.setObjective(obj_expr, GRB.MAXIMIZE)
        self._optimize()

        if self.lp.Status == GRB.OPTIMAL:
            val = self.lp.ObjVal
            d_vals = [self.d_vars[i].X for i in range(self.n)]
            u_vals = [self.u_vars[i].X for i in range(self.n)]
            feasible = True
            self._witnesses.append(d_vals)
        else:
            val, d_vals, u_vals = self._lowest_in_U(e_vector)
            feasible = False

        res = (val, d_vals, u_vals, feasible)
        self._leaf_cache[key] = res
        self._feas_cache[E_matrix] = feasible
        return res

    # ------------------------------------------------------------------ tree
    def solve(self):
        inst = self.inst
        self.log.info("=" * 78)
        self.log.info("UPPER BOUND - Hybrid Minimax tree + independent Leaf LPs (non-anticipativity relaxed)")
        self.log.info("=" * 78)
        self.log.info(inst.describe())
        self.log.info(f"prune_infeasible_branches={self.prune}, alpha_beta={self.alpha_beta}, "
                      f"exact_root_values={self.exact_root_values}")
        self.log.info("Reading the trace: [SCHEDULER] = min node, [ADVERSARY] = max node, [LEAF] = LP.")
        self.log.info("Completion times C(Ti) are written as sums of durations d_i.")
        self.log.info("-" * 78)

        t0 = time.perf_counter()
        zero = (0,) * self.n

        try:
            value, leaf = self.scheduler_turn(
                running={},
                pending=tuple(range(self.n)),
                finished=(),
                E_matrix=(),
                start_expr=zero,
                n_slots=min(self.m, self.n),
                path=[],
                depth=0,
                alpha=NEG_INF,
                beta=POS_INF,
            )
        finally:
            self._dispose_lp()

        elapsed = time.perf_counter() - t0      # search only; the buffered trace is written below
        self.stats["gurobi_time"] = round(self.stats["gurobi_time"], 4)
        if self._trace_lines:
            self.log.debug("\n".join(self._trace_lines))
            self._trace_lines = []
        # Same first-decision rule as ar_milo.py: min value, lexicographically smallest among ties
        best = min(self.root_choices.values())
        root_choice = min(c for c, v in self.root_choices.items() if v <= best + self.tie_tol)
        leaf = self.root_leaves.get(root_choice, leaf)

        self.log.info("-" * 78)
        self.log.info("MINIMAX UPPER BOUND RESULT")
        for combo, val in self.root_choices.items():
            mark = "  <== chosen" if combo == root_choice else ""
            self.log.info(f"  first decision start {tasks_str(combo)} -> worst-case makespan {fmt_val(val)}{mark}")
        self.log.info(f"  Worst-case makespan of the relaxed game (upper bound on true AR) = {fmt_val(value)}")
        self.log.info(f"  Worst-case scenario under optimal play: {path_str(leaf['path'])}")
        self.log.info(f"    durations chosen by adversary at that leaf: {fmt_d(leaf['d'])}")
        self.log.info(f"    budget used sum(u) = {sum(leaf['u']):.4f} / Gamma = {inst.gamma}")
        self.log.info(f"  Stats: {self.stats}, time = {elapsed:.3f}s")
        self.log.info(f"  Scheduler nodes where the relaxed adversary used different durations for an "
                      f"already-finished task in sibling branches: {len(self.anticipation)}")
        for a in self.anticipation[:10]:
            self.log.info("    " + self._fmt_anticipation(a))
        if len(self.anticipation) > 10:
            self.log.info(f"    ... {len(self.anticipation) - 10} more")
        return dict(
            model="minimax_ub",
            value=value,
            root_choice=root_choice,
            root_choices=dict(self.root_choices),
            worst_leaf=leaf,
            stats=dict(self.stats),
            anticipation=list(self.anticipation),
            time=elapsed,
            status="OPTIMAL",
        )

    def scheduler_turn(self, running, pending, finished, E_matrix, start_expr, n_slots,
                       path, depth, alpha, beta):
        """Scheduler (min): start k = min(idle machines, pending) tasks at time start_expr."""
        self.stats["scheduler_nodes"] += 1
        k = min(n_slots, len(pending))
        do_t = self._do_trace(depth)
        do_t1 = self._do_trace(depth + 1)

        if do_t:
            node = path_str(path)
            self._t(depth, f"[SCHEDULER] {node}")
            self._t(depth, f"   running={self._fmt_running(running)} pending={tasks_str(pending)} "
                           f"finished={tasks_str(finished)}; must start {k} task(s) at time {fmt_expr(start_expr)}")
        else:
            node = None

        combos = list(itertools.combinations(pending, k))
        # Move ordering at inner nodes: try larger max-duration tasks first to tighten beta early
        if depth > 0 and len(combos) > 1:
            combos.sort(key=lambda c: sum(self.d_max[j] for j in c), reverse=True)
        open_window = not self.alpha_beta or (depth == 0 and self.exact_root_values)

        results = []
        best_val = POS_INF
        best_combo = None
        best_leaf = None

        for combo in combos:
            new_running = dict(running)
            for j in combo:
                expr = list(start_expr)
                expr[j] += 1
                new_running[j] = tuple(expr)
            rest = tuple(t for t in pending if t not in combo)

            if do_t1:
                self._t(depth + 1, f"option: start {tasks_str(combo)} -> "
                                   + ", ".join(f"C({task_name(j)}) = {fmt_expr(new_running[j])}" for j in combo))

            a, b = (NEG_INF, POS_INF) if open_window else (alpha, min(beta, best_val))
            val, leaf = self.adversary_turn(
                new_running, rest, finished, E_matrix,
                path + [("S", combo)], depth + 1, a, b,
            )
            results.append((combo, val, leaf))

            if val < best_val:
                best_val = val
                best_combo = combo
                best_leaf = leaf

            if self.alpha_beta and best_val <= alpha:
                self.stats["pruned_branches"] += 1
                self.stats["prune_alpha"] += 1
                if do_t1:
                    self._t(depth + 1, f"   Alpha pruning! best_val {best_val:.4f} <= alpha {alpha:.4f}")
                break

        if depth == 0:
            self.root_choices = {c: v for c, v, _ in results}
            self.root_leaves = {c: lf for c, _, lf in results}
        if finished and len(results) > 1:
            self._check_anticipation(node if node is not None else path, finished, results)
        if do_t:
            self._t(depth, f"[SCHEDULER] {node} -> MIN of explored options => start {tasks_str(best_combo)} (value {fmt_val(best_val)})")
        return best_val, best_leaf

    def tail_turn(self, running, E_matrix, path, depth):
        """Adversary node with nothing left to start. No scheduler decision remains below, and the
        finish-order regions of the running tasks cover the whole polytope P = {d in U : E d <= 0},
        so the max over all finish orders equals
            max_r  max_{d in P} C_r(d)        -> one LP per running task, no ordering rows.
        The reported leaf path lists the remaining finishes in the order induced by the optimal d."""
        self.stats["adversary_nodes"] += 1
        do_t = self._do_trace(depth)
        if do_t:
            self._t(depth, f"[ADVERSARY/TAIL] {path_str(path)}")
            self._t(depth, f"   running={self._fmt_running(running)}; nothing pending -> "
                           f"worst case = max over r of LP max C(r)")
            self._t(depth, "   sequence constraints: " + ("; ".join(fmt_row(r) for r in E_matrix) or "none"))

        # Upper bound per task (relaxation without E), try the most promising task first
        bounds = sorted(((self._greedy_max(running[r])[0], r) for r in sorted(running)), key=lambda x: -x[0])

        best_val, best_r, best = NEG_INF, None, None
        infeasible = []
        for ub, r in bounds:
            if ub <= best_val + TOL:
                self.stats["pruned_branches"] += 1
                self.stats["prune_tail_bound"] += 1
                if do_t:
                    self._t(depth + 1, f"skip {task_name(r)} last: bound {ub:.4f} <= current max {best_val:.4f}")
                continue
            self.stats["leaves"] += 1
            val, d_vals, u_vals, feasible = self.solve_leaf_lp(E_matrix, running[r])
            if not feasible:
                self.stats["infeasible_leaves"] += 1
                infeasible.append((val, r, d_vals, u_vals))
                if do_t:
                    self._t(depth + 1, f"{task_name(r)} last: LP infeasible -> lowest in U {val:.4f}")
                continue
            if do_t:
                self._t(depth + 1, f"{task_name(r)} last: max C({task_name(r)}) = {fmt_expr(running[r])} "
                                   f"-> {val:.4f} with d = {fmt_d(d_vals)}")
            if val > best_val:
                best_val, best_r, best = val, r, (d_vals, u_vals)

        if best is None:
            # P is empty (only possible with prune_infeasible_branches=false): lowest value in U
            val, r, d_vals, u_vals = min(infeasible, key=lambda x: x[0])
            leaf_path = path + [("F", t) for t in sorted(running, key=lambda t: t == r)]
            return val, dict(path=leaf_path, d=d_vals, u=u_vals, value=val, feasible=False)

        d_vals, u_vals = best
        C = {t: sum(e[i] * d_vals[i] for i in range(self.n)) for t, e in running.items()}
        order = sorted(running, key=lambda t: (C[t], t == best_r))
        leaf_path = path + [("F", t) for t in order]
        if do_t:
            self._t(depth, f"[ADVERSARY/TAIL] {path_str(path)} -> MAX => {path_str(leaf_path[len(path):])} "
                           f"(value {fmt_val(best_val)})")
        return best_val, dict(path=leaf_path, d=d_vals, u=u_vals, value=best_val, feasible=True)

    def adversary_turn(self, running, pending, finished, E_matrix, path, depth, alpha, beta):
        """Adversary (max): decide which running task finishes first."""
        if not pending:
            return self.tail_turn(running, E_matrix, path, depth)
        self.stats["adversary_nodes"] += 1
        do_t = self._do_trace(depth)
        do_t1 = self._do_trace(depth + 1)

        if do_t:
            node = path_str(path)
            self._t(depth, f"[ADVERSARY] {node}")
            self._t(depth, f"   running={self._fmt_running(running)}; who finishes first?")
        else:
            node = None

        order = sorted(running)
        infeasible_results = []
        best_val = NEG_INF
        best_f = None
        best_leaf = None

        for f in order:
            new_rows = tuple(sub_expr(running[f], running[r]) for r in order if r != f)
            E2 = E_matrix + new_rows
            child_path = path + [("F", f)]

            if do_t1:
                cons = "; ".join(fmt_row(r) for r in new_rows) if new_rows else "(only task left)"
                self._t(depth + 1, f"option: {task_name(f)} finishes first  [adds {cons}]")

            if self.prune and new_rows and not self._feasible(E2):
                self.stats["pruned_branches"] += 1
                self.stats["prune_infeasible"] += 1
                if do_t1:
                    self._t(depth + 1, "   impossible sequence under the budget Gamma -> fallback to lowest in U")
                low_val, low_d, low_u = self._lowest_in_U(running[f])
                infeasible_results.append(
                    (f, low_val, dict(path=child_path, d=low_d, u=low_u, value=low_val, feasible=False))
                )
                continue

            rest_running = {r: e for r, e in running.items() if r != f}
            # pending is non-empty here (empty pending is handled by tail_turn): a machine frees up
            a, b = (alpha, beta) if self.alpha_beta else (NEG_INF, POS_INF)
            val, leaf = self.scheduler_turn(
                rest_running, pending, finished + (f,), E2, running[f], 1,
                child_path, depth + 1, a, b,
            )

            if leaf is not None and leaf.get("feasible", True):
                if val > best_val:
                    best_val = val
                    best_f = f
                    best_leaf = leaf

                # Alpha-Beta Pruning: exact here, leaf values depend only on their own path
                if self.alpha_beta:
                    alpha = max(alpha, best_val)
                    if best_val >= beta:
                        self.stats["pruned_branches"] += 1
                        self.stats["prune_beta"] += 1
                        if do_t1:
                            self._t(depth + 1, f"   Beta pruning! best_val {best_val:.4f} >= beta {beta:.4f} (pruned remaining permutations)")
                        break
            else:
                infeasible_results.append((f, val, leaf))

        # Only if NO feasible branch exists at all, pick the lowest value that keeps us in U
        if best_leaf is None and infeasible_results:
            best_f, best_val, best_leaf = min(infeasible_results, key=lambda x: x[1])

        if do_t:
            self._t(depth, f"[ADVERSARY] {node} -> MAX of explored options => {task_name(best_f)} finishes first (value {fmt_val(best_val)})")
        return best_val, best_leaf

    # ------------------------------------------------------------------ analysis
    def _check_anticipation(self, node, finished, results):
        """The durations of `finished` tasks are already revealed at a scheduler node; count the nodes where
        the independent leaf LPs of sibling branches disagree on them (where the UB relaxes the exact MILO)."""
        found = False
        for i in finished:
            per_option = [
                (c, leaf["d"][i], v) for c, v, leaf in results
                if leaf is not None and leaf.get("feasible", False) and leaf["d"] is not None
            ]
            if len(per_option) < 2:
                continue
            vals = [x[1] for x in per_option]
            if max(vals) - min(vals) > 1e-4:
                found = True
                if len(self.anticipation) < 1000:
                    node_name = node if isinstance(node, str) else path_str(node)
                    self.anticipation.append(dict(node=node_name, task=i, per_option=per_option))
        if found:
            self.stats["na_relaxed_nodes"] += 1

    @staticmethod
    def _fmt_anticipation(a):
        opts = ", ".join(
            f"if start {tasks_str(c)}: d{a['task'] + 1}={dv:.4f} (makespan {fmt_val(v)})"
            for c, dv, v in a["per_option"]
        )
        return (f"at {a['node']}: {task_name(a['task'])} already finished, but the sibling worst cases "
                f"use different durations for it -> {opts}")

    @staticmethod
    def _fmt_running(running):
        return "{" + ", ".join(f"{task_name(j)}: C={fmt_expr(e)}" for j, e in sorted(running.items())) + "}"


def solve_minimax_ub(inst, cfg, logger):
    if inst.is_discrete:
        raise NotImplementedError("Minimax UB supports only the budgeted uncertainty set (use ar_milo.py)")
    log_cfg = cfg.get("logging", {})
    solver = MinimaxUBSolver(
        inst,
        cfg.get("minimax_ub", cfg.get("minimax", {})),
        logger,
        trace=log_cfg.get("trace_tree", False),
        trace_max_depth=log_cfg.get("trace_max_depth", 1000),
    )
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
    log_path = os.path.join(log_dir, "single", f"minimax_ub_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger("minimax_ub", log_path, console=True)
    res = solve_minimax_ub(inst, cfg, logger)
    if inst.expected_makespan is not None:
        tol = cfg.get("benchmark", {}).get("tolerance", 1e-4)
        gap = res["value"] - inst.expected_makespan
        logger.info(f"Reference check: exact AR {inst.expected_makespan:.4f}, UB {res['value']:.4f} -> "
                    f"{'PASS' if gap >= -tol else 'FAIL'} (UB must be >= exact AR; "
                    f"{'tight' if abs(gap) <= tol else f'gap {gap:.4f}'})")
    logger.info(f"Full trace written to {log_path}")
    close_logger(logger)


if __name__ == "__main__":
    main()
