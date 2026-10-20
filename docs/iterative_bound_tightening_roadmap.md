# Roadmap: Iterative Bound Tightening ($LB \uparrow$ and $UB \downarrow$)

This document provides actionable mathematical and algorithmic strategies for the AI coding agent to **iteratively tighten the gap** $UB - LB \to 0$. 

---

## 1. Core Philosophy: How to Close the Gap

We have two levers to close the optimality gap $\Delta = UB - LB$:
1. **Raise the Lower Bound ($LB \uparrow$):** Generate smarter, more adversarial candidate realizations $\hat{d} \in \mathcal{U}$ at branching nodes (especially points that lie *between* the extreme vertices of individual leaves).
2. **Lower the Upper Bound ($UB \downarrow$):** Re-introduce partial Non-Anticipativity constraints by coupling sibling subtrees into small, localized multi-leaf LPs/MILPs instead of evaluating every leaf in complete isolation.

---

## 2. Strategy A: Raising $LB$ via Adversarial Candidate Generation (Benders / C&CG Style)

### 2.1. Why Current Leaf Candidates Fall Short
When a single leaf $\sigma$ solves $\max \{ e_\sigma^\top d : E_\sigma d \le 0, d \in \mathcal{U} \}$, Gurobi returns an extreme point $d_\sigma^*$ tailored *exclusively* to path $\sigma$. 
If the scheduler at an ancestor node $\tau = (S, F) \in \mathcal{D}$ simply switches to another child branch $\delta$, the realization $d_\sigma^*$ might make branch $\delta$ either very short (small makespan) or completely infeasible.

### 2.2. Iterative Joint-Branch LP (Finding "Compromise" Candidates)
To generate a stronger candidate duration vector at a scheduler node $\tau = (S, F) \in \mathcal{D}$ with children $\delta_1, \delta_2, \dots, \delta_k$ (representing scheduling task $c_1, c_2, \dots, c_k$):

1. Suppose in iteration $r$, for each child decision $\delta_j$, we have identified the adversary's best leaf path $\sigma_j^* \in \mathcal{L}$ (from the previous pass).
2. Instead of using the single-leaf vectors $d_{\sigma_1^*}^*, \dots, d_{\sigma_k^*}^*$, solve a **Joint Compromise LP** across the active leaves $\sigma_1^*, \dots, \sigma_k^*$:
   $$\max_{t, d_{\sigma_1^*}, \dots, d_{\sigma_k^*}} t$$
   Subject to:
   * $t \le e_{\sigma_j^*}^\top d_{\sigma_j^*}$ for all $j = 1, \dots, k$ (The scheduler will pick the minimum among the branches).
   * $E_{\sigma_j^*} d_{\sigma_j^*} \le 0$ and $d_{\sigma_j^*} \in \mathcal{U}$ for all $j = 1, \dots, k$ (Each active leaf must remain valid).
   * **Shared History Constraint:** $d_{\sigma_1^*}(i) = d_{\sigma_2^*}(i) = \dots = d_{\sigma_k^*}(i)$ for all $i \in F$ (Non-anticipativity at node $\tau$).

3. **Extract New Candidate:** The shared values $D_i^* = d_{\sigma_1^*}(i)$ ($\forall i \in F$) from this Joint LP represent the **optimal adversarial compromise** against the scheduler's current best responses $\sigma_1^*, \dots, \sigma_k^*$.
4. **Iterate (Counter-Response):** Inject $D_F^*$ into the candidate pool $\hat{\mathcal{D}}$ at node $\tau$ and re-evaluate the children subtrees. If the scheduler finds a new escape leaf $\sigma_j^{\text{new}}$ that beats $t$, add $\sigma_j^{\text{new}}$ to the Joint LP and repeat!
   * *Note for AI Agent:* This is a localized, tree-based version of **Column-and-Constraint Generation (C&CG)** applied at scheduler node $\tau$. Each iteration strictly increases or maintains $LB$.

---

## 3. Strategy B: Lowering $UB$ via Bottom-Up Subtree Coupling (K-Depth Merging)

### 3.1. The Concept of Partial Non-Anticipativity
Recall that $UB$ relaxes non-anticipativity across the *entire* tree, while the exact MILO enforces it across the *entire* tree. We can interpolate between $UB$ and the exact MILO by enforcing non-anticipativity **only within small subtrees of depth $h$** near the bottom of the tree:
$$T^{UB} = UB_0 \ge UB_1 \ge UB_2 \ge \dots \ge UB_{H} = T^{AR*}$$

### 3.2. Step-by-Step Implementation for Depth $h=1$ (Last Scheduler Decision)
Consider a scheduler node $\tau = (S, F) \in \mathcal{D}$ at the lowest decision level of the tree (where $|S| = n-2$ and $|F| = n-3$, meaning only 2 unscheduled tasks remain, say tasks $a$ and $b$).
* In the standard $UB$ algorithm, $\tau$ has 2 children (schedule $a$ or schedule $b$), and each child has 4 leaf grandchildren. We solve 8 tiny, independent LPs and take $\min(\max(\text{4 LPs}), \max(\text{4 LPs}))$.
* **The Upgrade ($UB_1$):** Instead of 8 independent LPs at the leaves of $\tau$, treat the entire subtree rooted at $\tau$ as a **single mini-MILO** (with only 8 leaves!):
  * Enforce $d_\sigma(i) = d_\delta(i)$ for $i \in F$ across all 8 leaves of $\tau$.
  * Solve this tiny 8-leaf MILO (takes milliseconds in Gurobi).
  * Return its objective value as the exact evaluation of node $\tau$ in the upper tree!

### 3.3. Iterative Tightening Loop ($UB_0 \to UB_1 \to UB_2$)
1. **Pass 0 ($UB_0$):** Run standard independent-leaf Minimax with Alpha-Beta pruning. Record the value $t_\sigma^{UB_0}$ at every node.
2. **Pass 1 ($UB_1$):** Traverse the tree again, using the tighter bounds from $LB$ and $UB_0$ for Alpha-Beta pruning. Whenever the search reaches a scheduler node at level $n-2$ (and it is not pruned!), solve the coupled mini-MILO for that node instead of independent leaf LPs. This yields a strictly tighter upper bound $UB_1 \le UB_0$.
3. **Selective Subtree Expansion:** You do not even need to solve the mini-MILO for *all* level $n-2$ nodes—**only for the critical nodes** that currently define the active Minimax principal variation (the winning path of the $UB$ tree)! If tightening the critical node lowers its value below a sibling's $UB_0$, the principal variation shifts, and you tighten the new critical node next.

---

## 4. Strategy C: Integrating Alpha-Beta Pruning with Iterative Bounds

When running an iterative tightening loop, the AI agent should exploit existing bounds to prune aggressively:

1. **Global Root Window:** Before starting an iteration to improve $UB$ or $LB$, initialize the root Alpha-Beta window to $[\alpha_0, \beta_0] = [LB_{\text{best}}, UB_{\text{best}}]$ instead of $[-\infty, +\infty]$.
2. **Node-Level Memoization (Transposition / Bound Caching):**
   Store at each tree node $\sigma$ its current bracket $[lb(\sigma), ub(\sigma)]$:
   * Initially, $ub(\sigma)$ comes from the independent-leaf LP minimax ($UB_0$), and $lb(\sigma)$ comes from the initial candidate search ($LB_0$).
   * At a Scheduler (Min) node $\sigma$, if a child $\delta_1$ already has $ub(\delta_1) < lb(\delta_2)$, then child $\delta_2$ **never needs to be tightened or explored further**—even after full tightening, $\delta_2$ can never beat $\delta_1$!
   * Similarly, at an Adversary (Max) node $\sigma$, if $lb(\delta_1) > ub(\delta_2)$, branch $\delta_2$ is permanently dominated and can be pruned from all future iterative refinement passes.

---

## 5. Implementation Checklist for Coding Agents

When asked to implement or debug bound-tightening routines in Python/Gurobi:
1. **Verify Monotonicity:** Always assert `LB <= UB + 1e-6` at both the root and every internal node $\sigma$. If `LB > UB` occurs, check:
   * Are the $E_\sigma$ constraints identical between the $LB$, $UB$, and MILO models?
   * Are strict vs. weak inequalities ($\le$ vs. $<$) handled consistently with tie-breaking rules?
   * Is a candidate duration vector $\hat{d}$ causing infeasibility in one branch that is being mishandled (e.g., returning $+\infty$ instead of $-\infty$ at an adversary node)?
2. **Keep Gurobi Models Lightweight:** Reuse Gurobi environments or model objects where possible, or build small parameterized LP functions that only take the active $E_\sigma$ rows and fixed historical durations $D_F$ as inputs.