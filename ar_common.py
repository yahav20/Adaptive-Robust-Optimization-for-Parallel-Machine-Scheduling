"""
Shared helpers for the Adaptive Robust (AR) parallel machine scheduling experiments.

Used by:
    ar_minimax.py  - our idea: Hybrid Minimax tree search + independent LP per leaf
    ar_milo.py     - the paper's exact monolithic MILO (with non-anticipativity)
    benchmark.py   - runs both on the same instances and compares them

Conventions (same as scheduling.py):
    * Tasks are indexed 0..n-1 internally and printed as T1..Tn.
    * A completion time C_j is a linear expression in the durations d, stored as a
      list of coefficients (C_j = sum_i expr[i] * d_i).
    * A sequence-validity constraint (row of the E matrix) is a coefficient list
      `row` meaning  sum_i row[i] * d_i <= 0.
    * Durations: d_i = d0_i + d_bar_i * u_i,  u_i in [0, 1],  sum_i u_i <= Gamma.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import gurobipy as gp
from gurobipy import GRB

TOL = 1e-6


# ----------------------------------------------------------------------------
# Instance / config
# ----------------------------------------------------------------------------

@dataclass
class Instance:
    name: str
    d0: List[float]
    d_bar: List[float]
    gamma: float
    n_machines: int
    expected_makespan: Optional[float] = None

    @property
    def n_tasks(self):
        return len(self.d0)

    @property
    def max_total_duration(self):
        """Upper bound on any completion time (all tasks back to back, fully perturbed)."""
        return sum(a + b for a, b in zip(self.d0, self.d_bar))

    def describe(self):
        lines = [
            f"Instance '{self.name}': n_tasks={self.n_tasks}, n_machines={self.n_machines}, Gamma={self.gamma}",
            "  task |    d0    |  d_bar   | max d = d0+d_bar",
        ]
        for i in range(self.n_tasks):
            lines.append(f"  {task_name(i):>4} | {self.d0[i]:8.4f} | {self.d_bar[i]:8.4f} | {self.d0[i] + self.d_bar[i]:8.4f}")
        if self.expected_makespan is not None:
            lines.append(f"  expected optimal AR makespan (reference) = {self.expected_makespan:.4f}")
        return "\n".join(lines)


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def random_instance(cfg, rng: random.Random, name: str) -> Instance:
    """Random instance using the `problem` and `random_instance` sections of the config."""
    prob = cfg["problem"]
    rnd = cfg["random_instance"]
    n = prob["n_tasks"]
    dec = rnd.get("decimals", 4)
    lo0, hi0 = rnd["d0_range"]
    lob, hib = rnd["d_bar_range"]
    d0 = [round(rng.uniform(lo0, hi0), dec) for _ in range(n)]
    d_bar = [round(rng.uniform(lob, hib), dec) for _ in range(n)]
    return Instance(name=name, d0=d0, d_bar=d_bar, gamma=prob["gamma"], n_machines=prob["n_machines"])


def reference_instances(cfg) -> List[Instance]:
    out = []
    for ref in cfg.get("reference_instances", []):
        out.append(Instance(
            name=ref["name"], d0=list(ref["d0"]), d_bar=list(ref["d_bar"]),
            gamma=ref["gamma"], n_machines=ref["n_machines"],
            expected_makespan=ref.get("expected_makespan"),
        ))
    return out


def instance_from_args(cfg, reference_index=None, seed=None) -> Instance:
    """Used by the standalone `python3 ar_minimax.py` / `python3 ar_milo.py` entry points."""
    if reference_index is not None:
        return reference_instances(cfg)[reference_index]
    if seed is None:
        seed = cfg["random_instance"]["seed"]
    return random_instance(cfg, random.Random(seed), name=f"random(seed={seed})")


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def setup_logger(name, file_path=None, console=False, console_level="INFO"):
    """Logger writing everything (DEBUG) to `file_path` and optionally INFO+ to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(message)s")
    if file_path:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        fh = logging.FileHandler(file_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def close_logger(logger):
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


# ----------------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------------

def task_name(i):
    return f"T{i + 1}"


def tasks_str(tasks):
    return "{" + ",".join(task_name(t) for t in tasks) + "}"


def fmt_expr(expr):
    """[1, 0, 1] -> 'd1 + d3'"""
    terms = []
    for i, c in enumerate(expr):
        if c == 0:
            continue
        name = f"d{i + 1}"
        terms.append(name if c == 1 else f"{c:g}*{name}")
    return " + ".join(terms) if terms else "0"


def fmt_row(row):
    """E row (row . d <= 0) as 'positive part <= negative part', e.g. 'd1 <= d2'."""
    pos = [c if c > 0 else 0 for c in row]
    neg = [-c if c < 0 else 0 for c in row]
    return f"{fmt_expr(pos)} <= {fmt_expr(neg)}"


def fmt_d(d):
    if d is None:
        return "n/a"
    return "[" + ", ".join(f"{task_name(i)}={v:.4f}" for i, v in enumerate(d)) + "]"


def fmt_val(v):
    if v is None:
        return "n/a"
    if v == float("-inf"):
        return "-inf"
    if v == float("inf"):
        return "+inf"
    if abs(v) < 5e-5:
        v = 0.0                         # avoid printing solver noise as -0.0000
    return f"{v:.4f}"


def path_str(path):
    """Event path -> 'S{T1,T2} > F:T1 > S{T3} > ...'. Same naming in both models."""
    if not path:
        return "ROOT"
    parts = []
    for kind, payload in path:
        parts.append(f"S{tasks_str(payload)}" if kind == "S" else f"F:{task_name(payload)}")
    return " > ".join(parts)


# ----------------------------------------------------------------------------
# Gurobi helpers
# ----------------------------------------------------------------------------

def make_env():
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def lp_feasible(env, inst: Instance, E_matrix):
    """Is there a duration vector in the budgeted uncertainty set satisfying E d <= 0 ?"""
    n = inst.n_tasks
    model = gp.Model("Feasibility_LP", env=env)
    u = model.addVars(n, lb=0.0, ub=1.0, name="u")
    model.addConstr(gp.quicksum(u[i] for i in range(n)) <= inst.gamma)
    d = [inst.d0[i] + inst.d_bar[i] * u[i] for i in range(n)]
    for row in E_matrix:
        model.addConstr(gp.quicksum(row[i] * d[i] for i in range(n) if row[i] != 0) <= 0)
    model.setObjective(0.0, GRB.MAXIMIZE)
    model.optimize()
    ok = model.Status == GRB.OPTIMAL
    model.dispose()
    return ok
