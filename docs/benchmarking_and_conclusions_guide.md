# Benchmarking, Experimental Analysis, and Conclusions Guide for AI Agents

This document instructs the AI agent on how to run experiments, log metrics, analyze behavioral differences in schedules, and draw rigorous scientific conclusions when comparing our **Minimax Bounding Framework ($LB$, $UB$, and Iterative Refinements)** against the paper's benchmarks (**Exact AR MILO, 2SSA, SL, SA, and PH**).

An analysis that only reports objective values (makespan numbers) is incomplete. The AI agent must investigate **what** scheduling decisions changed, **why** they changed mathematically, and **how** the algorithms behave computationally.

---

## 1. Core Evaluation Dimensions

When evaluating any problem instance $k$ (defined by number of machines $m$, tasks $n$, nominal durations $d^0$, maximum perturbations $\bar{d}$, and uncertainty budget $\Gamma$), the agent must analyze four distinct dimensions:

1. **Objective Quality & Bound Tightness:** Promised worst-case makespans and the optimality gap between $LB$, $T^{AR*}$, and $UB$.
2. **Scheduling Policy & Decision Divergence:** Differences in the actual task allocations (both at $t=0$ and dynamically along the scenario tree) and the root causes of these differences.
3. **Adversarial Behavior & Uncertainty Exploitation:** How the adversary distributes the budget $\Gamma$ across tasks under different models.
4. **Computational Efficiency & Scaling:** Runtime breakdowns, Gurobi LP/MILO solver calls, and Alpha-Beta pruning effectiveness.

---

## 2. Analyzing Scheduling Decisions (Why Schedules Differ)

A central finding of the paper is that the true value of Adaptive Robust (AR) optimization lies in making superior **here-and-now (first-stage) decisions** at $t=0$ because it anticipates future adaptations.

### 2.1. First-Stage Decision Tracking ($t=0$)
For every instance, the AI agent must log the initial set of $m$ tasks scheduled at $t=0$ by each method:
* $S_0^{AR}$: Initial tasks chosen by Exact MILO.
* $S_0^{LB}$: Initial tasks chosen by our Lower Bound Minimax model.
* $S_0^{UB}$: Initial tasks chosen by our Upper Bound Minimax model.
* $S_0^{SA}, S_0^{2SSA}$: Initial tasks chosen by static/2-stage benchmarks.

**Key Metric — Suboptimal Initial Decision Rate:**
Track the percentage of instances where $S_0^{\text{Model}} \neq S_0^{AR}$. 
* *Important Nuance (Tie-Breaking):* Before flagging a decision $S_0^{\text{Model}} \neq S_0^{AR}$ as suboptimal, the agent must plug $S_0^{\text{Model}}$ into the Exact AR evaluator (or check the root children values in the MILO). Did $S_0^{\text{Model}}$ actually yield a strictly worse guaranteed makespan, or is it an alternative optimal first-stage decision?

### 2.2. Mid-Tree Policy Divergence (Wait-and-See Decisions)
Even when two models agree at $t=0$ ($S_0^{LB} = S_0^{UB} = S_0^{AR}$), their policies may diverge at deeper Scheduler Nodes $\sigma = (S, F) \in \mathcal{D}$ (e.g., after the first task finishes).

When comparing trees, the agent should identify **Critical Divergence Nodes** where Model A schedules task $k_1$ while Model B schedules task $k_2$, and diagnose the **Root Cause** using the following taxonomy:

#### Cause 1: The "Clairvoyant Adversary" Distortion (Why $UB$ diverges from $AR$)
* **Mechanism:** In $UB$, non-anticipativity is relaxed. Suppose at $t=0$, tasks $(1, 2)$ are running, and task $1$ finishes first. The scheduler must choose between task $3$ and task $4$.
* **Diagnostic:** Check the leaf LP solutions $d_{\sigma_3}^*$ (under branch 3) and $d_{\sigma_4}^*$ (under branch 4). If $d_{\sigma_3}^*(1) \neq d_{\sigma_4}^*(1)$, the $UB$ adversary is *changing the past* (assigning a different duration to already-finished task 1 depending on whether the scheduler picks 3 or 4).
* **Impact on Schedule:** This artificial adversarial power can make a great hedging task look artificially risky in $UB$, causing the $UB$ scheduler to pick a conservative, suboptimal task compared to $AR$.

#### Cause 2: Candidate Blind Spots (Why $LB$ diverges from $AR$)
* **Mechanism:** In $LB$, the adversary only tests durations from a finite candidate pool $\hat{\mathcal{D}}$ collected from extreme leaf vertices.
* **Diagnostic:** Suppose the $LB$ scheduler chooses task $3$ because under all candidate vectors in $\hat{\mathcal{D}}$, task $3$ yields a makespan $\le 10$. However, in the exact MILO, choosing task $3$ yields a makespan of $11.5$.
* **Impact on Schedule:** Extract the exact duration vector $d^{AR*}$ from the MILO for that branch. You will find that $d^{AR*}$ is an **interior compromise vector** (not an extreme point of a single leaf) that simultaneously stresses multiple downstream branches—a scenario that was missing from $\hat{\mathcal{D}}$.

#### Cause 3: Information-Gathering vs. Load-Balancing Tasks
* **Mechanism:** Under Budgeted Uncertainty ($\Gamma$), when a task $i$ finishes with a large perturbation ($u_i \to 1$), the remaining uncertainty budget for all other tasks drops to $\Gamma - u_i$.
* **Diagnostic:** Adaptive models ($AR, LB$) often prefer scheduling tasks with **high perturbation variance ($\bar{d}_i$)** early. Why? Because observing their realization reveals how much budget $\Gamma$ the adversary has left, allowing the scheduler to perfectly balance the remaining tasks in later stages! Static models ($SA$) or weak bounds fail to value this "information revelation" property.

---

## 3. Evaluating Promised vs. Realized Performance (Rolling Horizon)

As established in Section 7 of the paper, we must distinguish between two performance phases:

1. **Promised Makespan (Planning / Contract Stage at $t=0$):**
   * What worst-case makespan guarantee does the algorithm compute at the root node?
   * Compare: $LB \le T^{AR*} \le UB$.
   * **Regret Metric:** $\text{Gap}_{LB-UB} = \frac{UB - LB}{LB} \times 100\%$, and comparison against Perfect Hindsight ($\max_{s} T_{PH, s}$).

2. **Realized Makespan (Execution Stage via Rolling Horizon):**
   * How does the policy actually perform when simulated step-by-step against sampled scenarios $d^{(s)} \in \mathcal{U}$?
   * Even if $LB$ or $UB$ has a slight gap in the *promised* makespan at $t=0$, **if it selects the exact same first-stage tasks ($S_0$) as $AR$**, its actual Rolling-Horizon execution will be 100% identical to $AR$ in the first step!
   * Therefore, the agent must report both **Promised Suboptimality (%)** and **Decision Agreement Rate (%)**.

---

## 4. Computational & Algorithmic Profiling

To justify our Minimax + LP bounding approach over the monolithic MILO, the AI agent must log detailed computational telemetry for every run:

### 4.1. Runtime & Solver Telemetry
* **Total Wall-Clock Time (sec):** Broken down into:
  * Tree traversal & Python overhead time.
  * Total time spent inside Gurobi (`model.optimize()`).
* **Number of Leaf LPs Solved:** Compare total theoretical leaves $|\mathcal{L}| = n! m^{n-m-1}$ against actual LPs solved.
* **Infeasible Leaf Ratio:** Percentage of reached leaves where $E_\sigma d_\sigma \le 0$ conflicted with $\mathcal{U}$ (returned $-\infty$).

### 4.2. Alpha-Beta Pruning Efficiency
* **Pruning Rate (%):** Percentage of tree branches/leaves skipped due to $\alpha \ge \beta$ cutoffs.
* **Move-Ordering Impact:** How much faster Alpha-Beta runs when child nodes are ordered heuristically (e.g., evaluating shortest/longest nominal tasks first, or using cached $LB/UB$ values from previous iterations).
* **Bound-Window Speedup:** In iterative tightening, measure how initializing the root window to $[\alpha_0, \beta_0] = [LB, UB]$ reduces the number of visited nodes compared to $[-\infty, +\infty]$.

---

## 5. Sensitivity Analysis Across Problem Parameters

When summarizing experiments across a test suite, the AI agent should group results by the structural regimes identified in the paper:

1. **Effect of Budget Ratio ($\Gamma / n$):**
   * **Low ($\Gamma/n \le 0.1$) & High ($\Gamma/n \ge 0.8$):** All policies ($SA, LB, AR, UB$) should converge to nearly identical makespans and identical scheduling decisions. If $LB < UB$ by a wide margin when $\Gamma=0$ or $\Gamma=n$, **flag a bug immediately!** (At $\Gamma=0$ and $\Gamma=n$, uncertainty is deterministic/box-like, so non-anticipativity has zero effect and $LB = AR = UB$ must hold).
   * **Intermediate ($\Gamma/n \in [0.25, 0.5]$):** This is the critical adaptive regime. Here, the gap between static $SA$ and $AR$ is largest, and this is where $UB - LB$ tightening works hardest.
2. **Effect of Perturbation Scale ($\alpha = \bar{d}_i / d_i^0$):**
   * Higher perturbation ratios increase the number of feasible completion orderings (fewer $-\infty$ leaves), making the game tree wider in practice and increasing the importance of adaptive re-scheduling.
3. **Scaling with $n$ ($n=4, 5, 6, \dots$):**
   * Track the crossover point where the monolithic MILO runs out of memory or exceeds time limits, while the Alpha-Beta $UB/LB$ tree search continues to return tight bounds and high-quality initial decisions $S_0$.

---

## 6. Automated Reporting Template for the AI Agent

Whenever the AI agent finishes running a benchmark script or analyzing a batch of instances, it should structure its findings using the following diagnostic template:

```text
================================================================================
BENCHMARK DIAGNOSTIC REPORT
Instance Config: n={n}, m={m}, Gamma={Gamma} (Ratio={Gamma/n:.2f})
================================================================================

1. MAKESPAN & BOUND TIGHTNESS SUMMARY
   - Perfect Hindsight (PH) Max : {ph_val:.4f}
   - Lower Bound (LB - Minimax) : {lb_val:.4f}
   - Exact AR (Monolithic MILO) : {ar_val:.4f}  <-- Ground Truth (if n <= 6)
   - Upper Bound (UB - Minimax) : {ub_val:.4f}
   - Static Allocation (SA)     : {sa_val:.4f}
   - Relative Bound Gap (UB-LB) : {gap_pct:.2f}%
   - Iterative Tightening Gain  : LB improved by +{d_lb:.4f}, UB dropped by -{d_ub:.4f}

2. SCHEDULING DECISION ANALYSIS (Policy Comparison)
   - Initial Tasks at t=0 (AR)  : {s0_ar}
   - Initial Tasks at t=0 (LB)  : {s0_lb}  [Match AR? {match_lb}]
   - Initial Tasks at t=0 (UB)  : {s0_ub}  [Match AR? {match_ub}]
   - Initial Tasks at t=0 (SA)  : {s0_sa}  [Match AR? {match_sa}]
   - Root Cause of Any Divergence:
     [Agent explains whether divergence was due to alternative optima, 
      UB non-anticipativity violation on task X, or missing compromise candidate in LB]

3. ADVERSARIAL & TREE TELEMETRY
   - Total Theoretical Leaves   : {total_leaves}
   - Leaves Evaluated (LP calls): UB={ub_lps} ({ub_pruned_pct:.1f}% pruned), LB={lb_lps}
   - Feasible vs Infeasible LPs : {feas_count} feasible / {infeas_count} infeasible (-inf)
   - Worst-Case Budget Usage    : Tasks receiving max perturbation u_i in AR: {critical_tasks}

4. COMPUTATIONAL PERFORMANCE
   - Exact MILO Runtime         : {t_milo:.3f}s
   - UB Search Runtime          : {t_ub:.3f}s (LP solve time: {t_ub_lp:.3f}s)
   - LB Search Runtime          : {t_lb:.3f}s
   - Iterative Refinement Time  : {t_iter:.3f}s
================================================================================
```