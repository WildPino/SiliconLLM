# Phase 4.C: Learning Mechanisms

With the physical substrate defined (M4 Circular Buffer + M6 Leaky Integrator + R2 K=8 Readout), we evaluated how different learning algorithms converge on the tasks.

## 1. Integrated Throughput Check
Before learning, we measured the full pipeline overhead to ensure memory and readout didn't bottleneck the wave propagation.
- **Result**: `12,011.71 cycles/tick`
- **Analysis**: Coherent with the sum of its parts. Phase 3 (pure wave) was ~5400 cycles. Adding M4 logic, M6 traces `(state >> 1) + wave`, and R2 readout roughly doubled the cost. The system still executes ~330,000 full updates per second on a single 4GHz thread.

## 2. [L2] Perturbation + Estimation (1D Gradient Baseline)
We implemented a simulated annealing / hill climbing approach. Every 100 ticks, we mutate one rule sector randomly. If accuracy over the window improves, we keep it; if it drops, we revert.

### L2 Results
| Task | Final Accuracy | Conclusion |
| :--- | :--- | :--- |
| **XOR-2** | 50.4% | Complete failure to converge |
| **Period-7** | 85.7% | Hit optimal baseline (always predicting 0), failed to learn the pattern |
| **Echo-5** | 49.7% | Complete failure to converge |

### L2 Analysis
As anticipated, 1D numeric gradient descent is completely blind here. The search space is $4^{16} \approx 4.29$ billion combinations. Because the dynamics are chaotic, flipping a single rule rarely provides a smooth gradient towards a better solution; it just throws the system into a different attractor. The algorithm immediately gets stuck in local optima (like predicting '0' constantly for Period-7) and cannot escape. 

This establishes our baseline: **The substrate cannot be trained via blind 1D parameter wiggling.** We must use structural search (L4) or credit assignment (L6).
