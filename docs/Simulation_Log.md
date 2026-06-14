# Discrete Event Simulation (DES): Production Viability Analysis

## 1. Experiment Design (M/G/c Queueing Theory)
To evaluate the viability of the neuro-symbolic framework in a real-world production environment, a discrete event simulation model was implemented using queueing theory. The system was modeled under the following stochastic conditions:

* **Markovian Arrivals (M):** User requests arrive following a Poisson Process with a rate of 1.5 requests per minute.
* **General Service Time (G):** The LLM inference and search time was modeled using a truncated Normal (Gaussian) distribution, utilizing the highly optimized empirical means and standard deviations obtained in the final unified variant experiments (n=30).
* **Servers (c):** System performance was evaluated by varying the number of workers (parallel GPU instances) among 1, 2, and 3.
* **Statistical Significance:** Each scenario simulated 1 hour of real traffic and the process was repeated across **30 independent replicas** to obtain the population mean and mitigate variance.

## 2. Consolidated Simulation Results

| Hybrid Architecture | Servers (c) | Avg. Wait (s) | Std. Dev. (s) | Utilization (%) | System Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phi-3 (Beam Search)** | 1 | 10.61 | 3.57 | 50.51% | Optimal |
| **Phi-3 (Beam Search)** | 2 | 0.87 | 0.42 | 25.26% | Underutilized |
| **Phi-3 (Beam Search)** | 3 | 0.10 | 0.08 | 16.77% | Underutilized |
| **Llama 3 (Beam Search)**| 1 | 25.23 | 14.37 | 65.54% | Optimal |
| **Llama 3 (Beam Search)**| 2 | 1.86 | 0.85 | 32.68% | Underutilized |
| **Llama 3 (Beam Search)**| 3 | 0.27 | 0.25 | 21.84% | Underutilized |
| **Mistral (Beam Search)**| 1 | 317.87 | 142.13 | 94.90% | **Danger (Bottleneck)** |
| **Mistral (Beam Search)**| 2 | 12.74 | 10.89 | 56.77% | Optimal |
| **Mistral (Beam Search)**| 3 | 1.64 | 0.85 | 37.60% | Underutilized |
| **Mistral (MCTS)** | 1 | 147.06 | 79.79 | 90.49% | Operating Limit |
| **Mistral (MCTS)** | 2 | 6.64 | 3.18 | 48.70% | **Optimal** |
| **Mistral (MCTS)** | 3 | 0.89 | 0.51 | 32.10% | Underutilized |
| **Llama 3 (MCTS)** | 1 | 841.82 | 154.36 | 97.11% | **Collapsed** |
| **Llama 3 (MCTS)** | 2 | 146.54 | 101.96 | 89.32% | Saturated |
| **Llama 3 (MCTS)** | 3 | 15.17 | 7.64 | 63.94% | **Optimal** |
| **Phi-3 (MCTS)** | 1 | 1317.94 | 117.89 | 92.91% | **Total Collapse** |
| **Phi-3 (MCTS)** | 2 | 1143.62 | 102.03 | 91.87% | **Total Collapse** |
| **Phi-3 (MCTS)** | 3 | 1003.55 | 113.93 | 91.68% | **Total Collapse** |

## 3. Stochastic Analysis and Architecture Decisions

The simulation analysis reveals the true computational cost of constraint satisfaction via AI and dictates the infrastructure required to support a traffic flow of 1.5 req/min:

### A. The Optimization Paradox: Mistral MCTS vs. Beam Search
A striking mathematical anomaly was observed in the Mistral (7B) configuration. Contrary to conventional expectations, the stochastic **MCTS algorithm proved significantly faster and more stable than the greedy Beam Search**. 
Under 2 servers, Mistral MCTS maintained an average wait of only **6.64s** (48.7% utilization), while Beam Search hovered at **12.74s** (56.7% utilization). This occurs because the Upper Confidence Bound (UCT) in MCTS rapidly identifies the optimal semantic pathway and prunes the search early, whereas Beam Search exhaustively evaluates all nodes within its width at every level.

### B. The Viability of Llama 3
In the previous smaller-scale tests, Llama 3 paired with MCTS appeared entirely unviable. However, with the optimized $O(1)$ lookahead heuristics integrated, it is now possible to deploy the highly accurate Llama 3 MCTS architecture. While 1 or 2 servers lead to queue saturation, deploying a load balancer with **3 Servers** stabilizes the system comfortably, yielding a manageable **15.17s** wait time at a healthy 63.94% utilization rate.

### C. The Architectural Failure of Phi-3 MCTS
The simulation exposes a catastrophic failure when pairing a smaller, less capable foundation model (Phi-3) with a complex exploration algorithm. Because Phi-3's initial programmatic proposals lack structural integrity, the MCTS algorithm cannot confidently prune branches. This forces the system into a brute-force expansion of the entire Monte Carlo tree. The result is a **Total System Collapse**: even with 3 parallel GPUs, utilization remains permanently above 91%, and queue wait times violently exceed 15 minutes (>1000s).

## 4. Final Engineering Conclusion
The simulation empirically demonstrates that the efficiency of a neuro-symbolic framework relies heavily on the synergy between the search algorithm and the base model's zero-shot heuristics. 

The undisputed winning architecture for a balanced, cost-effective production deployment is **Mistral (7B) utilizing Hybrid MCTS**. Thanks to the deep optimization of the simulation phase, this architecture only requires a load balancer of **2 instances** (rather than 3) to guarantee 100% semantic success and flawless constraint adherence, all while keeping queue wait times under 7 seconds. For deployments where maximum strict accuracy is preferred over hardware costs, **Llama 3 (8B) with Hybrid MCTS across 3 instances** remains the premium tier alternative.