# Project Context: Adaptive Robust Optimization for Parallel Machine Scheduling

## 1. Problem Overview

The core problem is **Parallel Machine Scheduling (PMS)** with identical machines, where the objective is to minimize the **makespan** (the time it takes for all tasks to finish).
In real-world scenarios, the exact duration of each task is unknown at the time of scheduling. Instead, task durations belong to an uncertainty set $\mathcal{U}$. We use a **Budgeted Uncertainty Set** (often denoted with a budget parameter $\Gamma$), where tasks have a nominal duration and can deviate up to a maximum perturbation, but the total normalized deviation across all tasks cannot exceed $\Gamma$.

## 2. Background: The Paper's Approach

Based on the paper *"An Adaptive Robust Optimization Model for Parallel Machine Scheduling"*, traditional robust scheduling uses static policies:

* **Static Allocation (SA):** Tasks are partitioned to machines in advance.
* **Static List (SL):** An ordered list of tasks is created, and the next task is assigned to the first available machine.

These approaches are limited because they do not account for the information revealed during the execution (e.g., the actual time a task finishes).
The paper introduces an **Adaptive Robust (AR)** policy. The AR policy assumes the scheduler can re-optimize decisions dynamically every time a machine becomes idle.

### 2.1. The Exact MILO Formulation

To find the optimal AR policy, the paper models the problem from the perspective of the **Adversary** (Nature). The Adversary tries to maximize the makespan by choosing the worst-case task durations, knowing that the Scheduler will make optimal, minimizing decisions at every step. 

Instead of searching a game tree node-by-node, the paper formulates the entire Adversary Dynamic Programming (DP) as a single monolithic **Mixed Integer Linear Optimization (MILO)** problem. 

**How the MILO works:**
1. **Scenario Tree Generation:** The state of the system is defined by $\sigma = (S, F)$, where $S$ is the ordered list of started tasks and $F$ is the ordered list of finished tasks. The model builds a complete scenario tree representing all possible sequences of starts and finishes. Let $\mathcal{D}$ be the set of Scheduler nodes, $\mathcal{N}$ be the Adversary nodes, and $\mathcal{L}$ be the final leaf nodes.
2. **Decision Variables:**
   * $t_\sigma$: The optimal worst-case makespan from state $\sigma$ onwards.
   * $d_\sigma \in \mathbb{R}^n$: A vector of specific task durations chosen by the adversary for a complete path ending at leaf $\sigma \in \mathcal{L}$.
   * $z_\sigma \in \{0,1\}$: A binary variable used for logical Big-M constraints at the leaves.
3. **Minimax through Constraints:**
   * For a Scheduler node $\sigma \in \mathcal{D}$, the scheduler minimizes: $t_\sigma \le t_\delta$ for all children $\delta$. (Since the adversary maximizes the root $t_0$, this acts as a `min` operation).
   * For an Adversary node $\sigma \in \mathcal{N}$, the adversary maximizes using binary variables and Big-M to pick the worst child: $t_\sigma \le t_\delta + M \cdot w_{\sigma,\delta}$.
4. **Feasibility and Big-M (Constraints at the leaves):** 
   For every leaf $\sigma \in \mathcal{L}$, the chosen durations $d_\sigma$ must respect the sequence of events (e.g., task A finished before task B). This is represented by $E_\sigma d_\sigma \le 0$. If the adversary chooses durations that make this sequence impossible under the budget $\Gamma$, the binary variable $z_\sigma$ is forced to 1. The makespan for this leaf is then calculated as $t_\sigma = e_\sigma^\top d_\sigma - z_\sigma M$. This penalizes impossible sequences with $-\infty$, ensuring they are never chosen as the maximum.
5. **Non-Anticipativity Constraints (The Crucial Part):** 
   The adversary must not use future knowledge. If two leaves $\sigma$ and $\delta$ share the same history up to a certain point (denoted by their lowest common ancestor $\mathcal{A}(\sigma, \delta)$), the adversary must assign the exact same durations to the tasks that finished during that shared history. 
   Mathematically: $d_\sigma(i) = d_\delta(i)$ for all $i \in F$ at state $(S, F) = \mathcal{A}(\sigma, \delta)$.

**The Challenge:** This MILO formulation has an exponential number of binary variables ($|\mathcal{D}| + |\mathcal{L}| - 1$) and does not scale well beyond small problems (e.g., 5 tasks).

## 3. The Proposed Algorithm: Hybrid Minimax Tree Search

To explore a different algorithmic angle, we are developing a hybrid approach that separates the **discrete decision-making** (the Minimax tree) from the **continuous uncertainty evaluation** (the Linear Programming).

Instead of building one massive MILO, we dynamically generate a Minimax game tree and evaluate the continuous uncertainty at the leaves.

### 3.1. Tree Structure (Discrete State Space)

The game tree consists of two types of alternating nodes:

1. **Scheduler Nodes (Minimizer):** A machine is idle. The scheduler decides which pending task to start next. The goal is to minimize the final makespan.
2. **Adversary Nodes (Maximizer):** All machines are busy. The adversary decides *which* of the currently running tasks will finish first. The goal is to maximize the final makespan.

Each path from the root to a leaf represents a specific deterministic sequence of events (e.g., "Task 1 & 2 start -> Task 1 finishes -> Task 3 starts -> Task 3 finishes -> Task 2 finishes").

### 3.2. Leaf Evaluation (Continuous LP)

When the tree reaches a leaf (all tasks have been scheduled and finished), we have a complete sequence of events. We translate this sequence into a set of linear constraints (the $E$ matrix). For example, if Task 1 finished before Task 2, their completion times satisfy $C_1 \le C_2$, which translates to $C_1 - C_2 \le 0$.

At each leaf, we solve a **Linear Program (LP)** using Gurobi. The recursive search propagates candidate duration realizations from feasible leaves. When multiple scheduler decisions remain after tasks finish, it evaluates those decisions under shared, fixed completed-task durations selected from that finite candidate set.

* **Variables:** Continuous task durations $d_i$, and uncertainty deviations $u_i \in [0,1]$.
* **Constraints:**
  * Uncertainty budget: $\sum u_i \le \Gamma$.
  * Sequence validity: The generated $E$ matrix constraints (ensuring the specific sequence of the leaf is mathematically possible under the durations chosen).
* **Objective:** Maximize the final makespan.

If the LP is infeasible (i.e., the adversary cannot legally make the sequence happen under the budget $\Gamma$), the leaf returns $-\infty$.

### 3.3. Backward Induction

Once the leaves are evaluated via LP, the values propagate back up the tree:

* Adversary nodes take the `max()` of their children.
* Scheduler nodes take the `min()` of their children.

The value at the root node gives the guaranteed worst-case makespan for the optimal initial scheduling decision.

## 4. The "Non-Anticipativity" Challenge (Important Context for AI)

This hybrid Minimax approach approximates the paper's exact continuous MILO using a finite candidate set.

In a true AR model (like the MILO described in section 2.1), the adversary's choice of a continuous duration for Task $i$ must be identical across different branches of the tree if those branches share the same history up to the point Task $i$ finishes ($d_{\sigma}(i) = d_{\delta}(i)$).

The Minimax search probes feasible leaves to collect candidate duration vectors. At scheduler nodes, it evaluates the available decisions with already-finished tasks fixed to shared values selected from this finite candidate set. This preserves non-anticipativity for those selected realizations, but may omit the continuous duration choices available to the exact MILO.

**Conclusion on Bounds:** Restricting the adversary to a finite subset of feasible shared-duration choices makes the Minimax value a **lower bound** on the exact AR makespan (up to solver tolerances). The MILO without non-anticipativity relaxes those shared-duration restrictions and should be at least as large as Minimax.

## 5. Agent Instructions for Future Code Modifications

When assisting with this codebase, AI agents should:

1. Understand that the separation of Gurobi (for LP at leaves) and Python recursion (for the Minimax tree) is intentional.
2. Maintain the structure of tracking Completion Times ($C$) as linear expressions (lists of coefficients for $d_i$) to easily build the $E$ matrices for the LP.
3. Be aware that any attempts to enforce Non-anticipativity across the tree using this recursive architecture will require either returning constraints up the tree (highly complex) or adopting a discretized uncertainty approach.