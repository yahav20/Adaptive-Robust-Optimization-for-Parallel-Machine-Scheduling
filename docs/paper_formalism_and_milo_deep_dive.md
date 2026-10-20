# Paper Formalism, Dynamic Programming, and the MILO Model

This document provides the rigorous mathematical and theoretical foundation from the paper *"An Adaptive Robust Optimization Model for Parallel Machine Scheduling"* by Cohen, Postek, and Shtern (2021). AI coding agents must adhere to this notation and logic when analyzing or extending the exact formulations.

---

## 1. Formal Problem Statement and State Space

We schedule $n$ independent, non-preemptive tasks $[n] = \{1, \dots, n\}$ on $m$ identical parallel machines ($n \ge m$), all available at $t=0$. 

### 1.1. Uncertainty Set ($\mathcal{U}$)
The vector of task durations $d = (d_1, \dots, d_n)^\top \in \mathbb{R}_{\ge 0}^n$ lies within a convex uncertainty set $\mathcal{U}$. 
In our primary benchmarks, we use the **Budgeted Uncertainty Set** (Bertsimas and Sim, 2004):
$$\mathcal{U} = \left\{ d \in \mathbb{R}^n : d_i = d_i^0 + \bar{d}_i u_i, \quad u_i \in [0, 1] \ \forall i \in [n], \quad \sum_{i=1}^n u_i \le \Gamma \right\}$$
where:
* $d_i^0$ is the nominal duration of task $i$.
* $\bar{d}_i$ is the maximum possible perturbation of task $i$.
* $u_i$ is the normalized perturbation variable.
* $\Gamma \in [0, n]$ is the uncertainty budget parameter.

### 1.2. Dynamic System States (Continuous Time)
Decision moments occur at $t=0$ and whenever a task finishes processing and a machine becomes idle. For $m=2$ machines, a decision state at time $t$ is represented by the tuple:
$$(S, F, D, i, \bar{D}_i)$$
* $S$: Ordered list of started tasks (in order of start times).
* $F$: Ordered list of completed tasks (in order of completion times).
* $D$: Ordered list of realized durations $D_k$ for all completed tasks $k \in F$.
* $i$: The index of the task currently still in process on the other machine ($i \in S \setminus F$).
* $\bar{D}_i$: The elapsed processing time of the running task $i$ up to the current decision moment $t$.

*(Note: For general $m > 2$, $i$ and $\bar{D}_i$ are replaced by the set of running tasks $I$ and the vector of their elapsed times $\bar{D}$.)*

### 1.3. State-Dependent Uncertainty Set
As time progresses and tasks finish or remain running, the uncertainty set dynamically shrinks based on revealed information:
$$\mathcal{U}_{S, F, D, i, \bar{D}_i} = \left\{ d \in \mathcal{U} : d_k = D_k \ \forall k \in F, \quad d_i \ge \bar{D}_i \right\}$$
A state is **feasible** if and only if $\mathcal{U}_{S, F, D, i, \bar{D}_i} \neq \emptyset$.

---

## 2. Dynamic Programming (DP) Formulations

### 2.1. The Scheduler's DP
Let $T(S, F, D, i, \bar{D}_i)$ be the minimal worst-case **remaining makespan** from the current state onwards. When a machine is idle, the scheduler chooses the next unscheduled task $k \notin S$ to minimize the worst-case remaining time:

$$T(S, F, D, i, \bar{D}_i) = \min_{k \notin S} \max \begin{cases} 
\max\limits_{\substack{d_k : d \in \mathcal{U}_{[S,k], F, D, i, \bar{D}_i} \\ d_k \le d_i - \bar{D}_i}} \left\{ d_k + T([S,k], [F,k], [D, d_k], i, \bar{D}_i + d_k) \right\}, \\
\max\limits_{\substack{d_i : d \in \mathcal{U}_{[S,k], F, D, i, \bar{D}_i} \\ d_k > d_i - \bar{D}_i}} \left\{ d_i - \bar{D}_i + T([S,k], [F,i], [D, d_i], k, d_i - \bar{D}_i) \right\}
\end{cases}$$

**Boundary Condition:** When $|S| = n$ and $|F| = n - 1$ (all tasks have started, only task $i$ is still running):
$$T(S, F, D, i, \bar{D}_i) = \max_{d_i : d \in \mathcal{U}_{S, F, D, i, \bar{D}_i}} (d_i - \bar{D}_i)$$

**Why this is intractable:** Solving the Scheduler's DP directly requires optimizing over an infinite-dimensional policy space (mapping continuous realizations $D, \bar{D}_i$ to discrete decisions $k$).

### 2.2. The Adversary's DP (Dual Perspective)
To make the problem finite-dimensional, the paper flips the perspective to the **Adversary**, who makes decisions right after task $k$ is scheduled:
$$T'([S,k], F, D, i, \bar{D}_i) = \max \begin{cases} 
\max\limits_{\substack{d_k : d \in \mathcal{U}_{[S,k], F, D, i, \bar{D}_i} \\ d_k \le d_i - \bar{D}_i}} \left\{ d_k + \min\limits_{l \notin S \cup \{k\}} T'([S,k,l], [F,k], [D, d_k], i, \bar{D}_i + d_k) \right\}, \\
\max\limits_{\substack{d_i : d \in \mathcal{U}_{[S,k], F, D, i, \bar{D}_i} \\ d_k > d_i - \bar{D}_i}} \left\{ d_i - \bar{D}_i + \min\limits_{l \notin S \cup \{k\}} T'([S,k,l], [F,i], [D, d_i], k, d_i - \bar{D}_i) \right\}
\end{cases}$$

---

## 3. The Monolithic MILO Formulation (Problem $P$)

The Adversary's DP is reformulated as a Mixed-Integer Linear Optimization (MILO) problem over a discrete **Scenario Tree**.

### 3.1. Scenario Tree Nodes ($\mathcal{V} = \mathcal{D} \cup \mathcal{N}$)
A discrete state in the scenario tree is defined solely by the combinatorial sequences of starts and finishes: $\sigma = (S, F)$.
* $\mathcal{D}$: **Scheduler States** (e.g., for $m=2$, $|S|=0, |F|=0$ at the root; then $|S|=2, |F|=1$; $|S|=3, |F|=2$; up to $|S|=n-2, |F|=n-3$).
* $\mathcal{N}$: **Adversary States** (where all machines are busy, e.g., $|S|=2, |F|=0$; $|S|=3, |F|=1$; up to $|S|=n-1, |F|=n-3$).
* $\mathcal{L} \subset \mathcal{N}$: **Leaf States (End States)** where $|S| = |F| = n$.

**Tree Complexity:** For $m$ machines and $n$ tasks, the number of leaves is $|\mathcal{L}| = n! m^{n-m-1}$, which grows exponentially.

### 3.2. Leaf Sequence Matrices ($E_\sigma$ and $e_\sigma$)
At each leaf $\sigma = (S, F) \in \mathcal{L}$, the exact order in which tasks started and finished is known. Because machines are never left idle if unscheduled tasks exist, every task $k$'s completion time $C_k$ is a deterministic sum of durations of the tasks that ran on the same machine up to $k$.
* **Order Constraints ($E_\sigma d_\sigma \le 0$):** At every point along the path to leaf $\sigma$ where task $a$ finished while task $b$ was still running on another machine, we must have $C_a \le C_b$. Collecting all $n-1$ such inequalities along the root-to-leaf path forms the matrix system $E_\sigma d_\sigma \le 0$.
* **Makespan Vector ($e_\sigma$):** The last task to finish in $F$ determines the total makespan of leaf $\sigma$. Its completion time is given by the inner product $t_\sigma = e_\sigma^\top d_\sigma$, where $e_\sigma \in \{0,1\}^n$ indicates the chain of tasks executed on the machine that finished last.

### 3.3. Non-Anticipativity and Lowest Common Ancestor $\mathcal{A}(\sigma, \delta)$
The adversary chooses a separate duration vector $d_\sigma \in \mathcal{U}$ for *every* leaf $\sigma \in \mathcal{L}$. However, the adversary cannot know in advance which branch the scheduler will pick at a node $\tau \in \mathcal{D}$.
For any two leaves $\sigma, \delta \in \mathcal{L}$, let $\mathcal{A}(\sigma, \delta) = (S, F) \in \mathcal{D}$ be their **Lowest Common Ancestor** (the last scheduler decision node they share before diverging). All tasks $i \in F$ at node $\mathcal{A}(\sigma, \delta)$ have *already finished* before the scheduler makes the diverging decision. Thus, their durations must be identical across both leaves:
$$d_\sigma(i) = d_\delta(i), \quad \forall i \in F, \ (S, F) = \mathcal{A}(\sigma, \delta)$$

### 3.4. Complete MILO Equations
$$\max_{d_\sigma, t_\sigma, z_\sigma, w_{\sigma,\delta}} t_0$$
Subject to:
1. **Scheduler Minimization ($\forall \sigma \in \mathcal{D}$):**
   $$t_\sigma \le t_\delta, \quad \forall \delta \in \text{Children}(\sigma)$$
2. **Adversary Maximization ($\forall \sigma \in \mathcal{N} \setminus \mathcal{L}$):**
   $$t_\sigma \le t_\delta + M \cdot w_{\sigma,\delta}, \quad \forall \delta \in \text{Children}(\sigma)$$
   $$\sum_{\delta \in \text{Children}(\sigma)} w_{\sigma,\delta} \le |\text{Children}(\sigma)| - 1, \quad w_{\sigma,\delta} \in \{0,1\}$$
3. **Non-Anticipativity ($\forall \sigma, \delta \in \mathcal{L}$):**
   $$d_\sigma(i) = d_\delta(i), \quad \forall i \in F, \ (S, F) = \mathcal{A}(\sigma, \delta)$$
4. **Leaf Evaluation & Feasibility Penalty ($\forall \sigma \in \mathcal{L}$):**
   $$t_\sigma = e_\sigma^\top d_\sigma - z_\sigma M$$
   $$E_\sigma d_\sigma \le z_\sigma M$$
   $$d_\sigma \in \mathcal{U}, \quad z_\sigma \in \{0,1\}$$

**Role of $z_\sigma$:** Because of non-anticipativity, fixing early task durations $d_\sigma(i)$ to maximize makespan in one subtree might make the sequence constraints $E_\delta d_\delta \le 0$ mathematically impossible in another subtree $\delta$. Setting $z_\delta = 1$ relaxes the infeasible constraint $E_\delta d_\delta \le M$ while penalizing the leaf objective $t_\delta$ by $-M$ (effectively $-\infty$), so the adversary cannot exploit infeasible paths.

---

## 4. Paper Benchmarks, Theoretical Insights & The 2SSA Heuristic

### 4.1. Policies Compared in the Paper
1. **SA (Static Allocation):** Tasks are partitioned into $m$ machines upfront at $t=0$.
2. **SL (Static List):** A permutation $\pi$ is chosen at $t=0$; whenever a machine is idle, the next task in $\pi$ is scheduled.
3. **AR (Adaptive Robust):** The exact MILO policy.
4. **PH (Perfect Hindsight):** Lower bound benchmark where $d$ is known before scheduling ($PH = \max_{d \in \mathcal{U}} \min_{x} \text{makespan}(x, d)$).
5. **2SSA (Two-Stage Static Allocation):** A scalable heuristic developed by the authors for $m=2$.

### 4.2. Key Theoretical Properties
* **Box Uncertainty Equivalence (Proposition 1):** If $\mathcal{U} = \{d : d_i \in [\underline{d}_i, \bar{d}_i]\}$ (no coupling budget $\Gamma$), then $SA = SL = AR$. Adaptivity brings zero benefit because the adversary simply sets $d_i = \bar{d}_i$ for all tasks.
* **Sweet Spot for Adaptivity (Theorems 1 & 2):** Under Budgeted Uncertainty, when $\Gamma \to 0$ or $\Gamma \to n$, static policies converge to AR. Adaptivity is most valuable for **moderate budgets** ($\Gamma/n \in [0.2, 0.5]$) and moderate numbers of tasks $n$.

### 4.3. How the 2SSA Heuristic Works
Because the exact MILO scales exponentially with $n-2$ adaptation stages, **2SSA** restricts adaptivity to **only 1 adaptation stage**:
1. Schedule two initial tasks $(i, j)$ at $t=0$.
2. Observe which task finishes first ($d_i < d_j$ or $d_j \le d_i$) and its exact duration.
3. Immediately commit to a **Static Allocation (SA)** $x \in \{0,1\}^n$ for all remaining $n-2$ tasks based on that single observation.

The authors solve 2SSA for each pair $(i, j)$ using the **Column-and-Constraint Generation (C&CG)** algorithm (Zeng & Zhao, 2013), alternating between a Master Problem (adversary picking durations) and a Subproblem (scheduler finding a better second-stage static partition $x$).