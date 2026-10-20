# Algorithmic Architecture: Lower and Upper Bounds via Minimax & LP

This document explains the custom algorithmic framework developed in our codebase to approximate and bound the exact Adaptive Robust (AR) makespan $T^{AR*}$ without building the massive, intractable MILO of the paper.

Let $T^{AR*}$ be the true optimal worst-case makespan of the exact MILO (Problem $P$ in `paper_formalism_and_milo_deep_dive.md`). Our codebase implements two distinct tree-based algorithms that bracket the true value:
$$LB \le T^{AR*} \le UB$$

---

## 1. The Upper Bound ($UB$): Pure Minimax LP Relaxation

### 1.1. Algorithmic Mechanism
We traverse the discrete combinatorial game tree of states $\sigma = (S, F)$ using standard recursive Minimax (enhanced with Alpha-Beta Pruning):
* **At Scheduler Nodes ($\sigma \in \mathcal{D}$):** Return $\min_{\delta \in \text{Children}(\sigma)} \text{Eval}(\delta)$.
* **At Adversary Nodes ($\sigma \in \mathcal{N} \setminus \mathcal{L}$):** Return $\max_{\delta \in \text{Children}(\sigma)} \text{Eval}(\delta)$.
* **At Leaf Nodes ($\sigma \in \mathcal{L}$):** Solve an independent continuous Linear Program (LP):
  $$t_\sigma^{LP} = \max_{d_\sigma} \left\{ e_\sigma^\top d_\sigma \quad \text{s.t.} \quad E_\sigma d_\sigma \le 0, \quad d_\sigma \in \mathcal{U} \right\}$$
  If the LP is infeasible, return $-\infty$.

### 1.2. Mathematical Proof: Why this yields an Upper Bound ($UB \ge T^{AR*}$)
Compare this independent-leaf Minimax tree to the paper's exact MILO (Problem $P$):
1. In the exact MILO, the adversary maximizes $t_0$ subject to leaf constraints ($E_\sigma d_\sigma \le 0, d_\sigma \in \mathcal{U}$) **AND** the Non-Anticipativity constraints:
   $$d_\sigma(i) = d_\delta(i) \quad \forall i \in F, \ (S,F) = \mathcal{A}(\sigma, \delta)$$
2. When we evaluate each leaf $\sigma \in \mathcal{L}$ with an *independent* LP and simply backpropagate $\min$ and $\max$ up the tree, we are solving the exact same problem **with all Non-Anticipativity constraints completely removed (relaxed)**.
3. **Intuition:** By dropping non-anticipativity, we make the Adversary *clairvoyant* (or cheating). When the scheduler is at a decision node $\tau \in \mathcal{D}$ (say, task 1 just finished) and considers scheduling task 3 vs. task 4:
   * In the subtree where the scheduler picks task 3, the adversary solves an LP that might retroactively set $d_1 = 2.0$ to maximize the makespan of that branch.
   * In the sibling subtree where the scheduler picks task 4, the adversary solves a separate LP that might retroactively set $d_1 = 4.5$ to maximize the makespan of *that* branch.
4. Because the adversary maximizes Over a **larger feasible space** (unconstrained by shared history across branches), the adversary is strictly stronger, and the scheduler is weaker. Therefore, the resulting root value is guaranteed to be an **Upper Bound**:
   $$T^{UB} \ge T^{AR*}$$

### 1.3. Role of Alpha-Beta Pruning in $UB$
Because $T^{UB}$ is a pure zero-sum Minimax tree where leaf values depend *only* on the path from the root to that leaf (no cross-branch coupling constraints), **standard Alpha-Beta Pruning is 100% exact** for computing $T^{UB}$. It prunes subtrees without changing the $UB$ value, drastically reducing the number of Gurobi LP calls required.

---

## 2. The Lower Bound ($LB$): Candidate-Based Non-Anticipative Search

### 2.1. Algorithmic Mechanism
To restore Non-Anticipativity without solving a massive monolithic MILO, our second approach restricts the adversary at decision points to a **finite set of candidate realizations**:
1. **Candidate Collection (Probing):** We probe feasible leaves (via LPs) to collect a finite pool of valid duration vectors $\hat{\mathcal{D}} = \{ \hat{d}^{(1)}, \hat{d}^{(2)}, \dots, \hat{d}^{(K)} \} \subset \mathcal{U}$.
2. **Enforcing Shared History at Scheduler Nodes:** When evaluating a state where tasks in $F$ have already completed, the adversary is required to pick a realization from the candidate set (or fix the durations of completed tasks $i \in F$ to concrete values $D_i = \hat{d}_i$ shared across *all* child branches of the scheduler node).
3. **Scheduler Response:** Because $D_i$ ($\forall i \in F$) is now fixed and identical across all children $\delta \in \text{Children}(\sigma)$, the scheduler evaluates each decision $k \notin S$ under the exact same historical realization and picks the minimum.

### 2.2. Mathematical Proof: Why this yields a Lower Bound ($LB \le T^{AR*}$)
1. In the exact MILO, the adversary can choose *any* continuous duration vector $d \in \mathcal{U}$ in the uncountable continuous space $\mathbb{R}^n$ to serve as the shared historical durations $D_i$ at a branching point.
2. Notably, the true worst-case realization $d^*$ for the exact MILO often lies inside the polytope or on a subtle boundary where the adversary slightly compromises the makespan of Leaf A in order to keep Leaf B feasible and high-valued (balancing multiple scheduler responses simultaneously).
3. Individual leaf LPs only find **extreme points (vertices)** that are locally optimal for *one single leaf* in isolation.
4. When we force the adversary to play non-anticipatively using only durations drawn from a **finite subset of candidate vectors** $\hat{\mathcal{D}} \subset \mathcal{U}$, we are **restricting the Maximizer's strategy space** while keeping all Non-Anticipativity rules strictly intact.
5. Restricting a maximizer to a subset of its valid strategies guarantees that its achieved objective is less than or equal to the true maximum:
   $$T^{LB} \le T^{AR*}$$

---

## 3. Summary Comparison Table for AI Agents

| Feature | Exact AR (Paper MILO) | Upper Bound ($UB$) Model | Lower Bound ($LB$) Model |
| :--- | :--- | :--- | :--- |
| **Mathematical Nature** | Exact Monolithic MILO | Pure Minimax + Independent Leaf LPs | Tree Search with Fixed Candidate Durations |
| **Non-Anticipativity** | Strictly Enforced ($\forall \sigma, \delta \in \mathcal{L}$) | **Relaxed (Ignored)** | **Strictly Enforced** (over finite candidates) |
| **Adversary Power** | Exact (Continuous & Non-Anticipative) | Too Strong (Can change history per branch) | Too Weak (Restricted to finite candidate set $\hat{\mathcal{D}}$) |
| **Relation to $T^{AR*}$** | $T = T^{AR*}$ | $UB \ge T^{AR*}$ | $LB \le T^{AR*}$ |
| **Alpha-Beta Pruning** | N/A (Solved by Gurobi B&B) | **Directly Applicable & Exact** | Applicable only *given* fixed history $D_F$ |
| **Computational Bottleneck** | Exponential binary variables | Number of unpruned leaf LPs | Size of candidate set $|\hat{\mathcal{D}}|$ vs. tightness |