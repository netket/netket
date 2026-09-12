Here is the complete, research-first execution plan formatted as a definitive markdown document. You can save this directly as `RESEARCH.md` in your project root to track your progress and keep the scientific objective strictly falsifiable.

---

# NetKet Contribution: Graph Attention Networks for Irregular Quantum Geometries

## 0. Hypotheses & Success Criteria

Before writing any code, this project evaluates the following hypotheses to determine whether a Graph Attention Network (GAT) architecture belongs upstream in the NetKet repository:

* **Hypothesis 1 (Expressivity):** Configuration-dependent graph attention improves variational expressivity on frustrated/irregular geometries relative to `netket.models.GCNN` at matched parameter budgets.
* **Hypothesis 2 (Efficiency):** The improvement, if present, remains meaningful at matched wall-clock computational budgets.
* **Hypothesis 3 (The Core Niche):** Graph attention provides its greatest advantage when the geometry lacks the regular symmetry structure (e.g., translation, point-group) explicitly exploited by `GCNN`.
* **Null Hypothesis:** GAT provides no statistically meaningful improvement after controlling for parameter count, compute time, and optimization budget.

## 1. Environment & Targeted Entry

Configure your virtual environment on your Linux machine (whether you are running Fedora or Pop!_OS on the Legion 5), ensuring JAX is compiled to leverage the discrete GPU for scaling tests later.

* **Repository Setup:** Fork and clone the `netket/netket` repository locally. Double-check your `git remote -v` to ensure your origin and upstream remotes are correctly mapped so you do not push to the wrong target later. Install with `pip install -e .[dev]`.
* **API Boundary Definition:** Trace how `netket.graph` generates `PermutationGroup` objects and how `GCNN` ingests them.
* **Graph-Adjacent PR (Optional):** If you spot a genuine documentation gap or test failure regarding irregular graphs or custom Flax modules during your reconnaissance, submit a small PR to establish project familiarity.

## 2. Falsifiable Prototyping (Standalone Package)

Build the GAT independently of the NetKet source code (`gat/layer.py`, `gat/model.py`, `gat/tests/`) to isolate JAX/Flax debugging.

### A. The Dtype Protocol

* Keep internal hidden representations real-valued with configurable precision (`float32`/`float64`).
* Compute real attention weights: $\alpha_{ij} = \operatorname{softmax}_j(e_{ij})$.
* Ensure the final readout layer produces a complex scalar, supporting NetKet's `complex_output=True` convention.

### B. Strict Symmetry Separation

Write specific unit tests targeting two distinct mathematical properties:

* **Architectural Equivariance:** Prove via tests that for a graph permutation $P$, the network satisfies $f(PX, PAP^T) = Pf(X,A)$.
* **Physical Symmetry:** Define how the model handles a physical symmetry $g$. Test whether it naturally satisfies $\psi(g\sigma) = \psi(\sigma)$ or if it requires manual symmetrization over the output.

### C. Empirical Scaling Profile

Do not assume sparse attention guarantees $\mathcal{O}(E d)$ execution. Measure scaling across system sizes ($N=16, 36, 64, 100$):

* **JIT Compilation Time:** Ensure dynamic edge counts or irregular topologies do not trigger recompilation inside the Variational Monte Carlo (VMC) loop.
* **Throughput & Memory:** Measure samples per second and peak memory during the Stochastic Reconfiguration (SR) gradient computation.

## 3. Rigorous Benchmarking

Run VMC optimizations controlling for random seeds, sample counts, SR settings, and iterations. **Crucially: All comparisons against `GCNN` must be evaluated at matched parameter budgets ($P_{\mathrm{GAT}} \approx P_{\mathrm{GCNN}}$) and matched wall-clock compute budgets.**

| Stage | Geometry | Model Comparison | Core Question |
| --- | --- | --- | --- |
| **1. Sanity** | Small 1D/2D chain | GAT vs Exact Diag. | Does the architecture optimize correctly without diverging? |
| **2. Standard** | Square Heisenberg | GAT vs `GCNN` / RBM | Does it behave competitively on conventional, non-frustrated systems? |
| **3. Frustrated** | J1-J2 / Triangular | GAT vs `GCNN` | Does learned attention actively help capture frustrated correlations? |
| **4. Scale** | Kagome | GAT vs `GCNN` | Does the advantage hold on a highly difficult frustrated lattice? |
| **5. Irregular** | Arbitrary Graph | GAT vs `GCNN` | Does GAT provide a capability on non-translationally invariant geometries that `GCNN` physically cannot? |

### Observables

Do not rely solely on total energy $\Delta E$. For frustrated and irregular systems, verify that the model correctly learns local correlation structures by measuring $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$ against exact diagonalization or trusted baselines.

## 4. The Data-Driven Decision & Upstream Pitch

Analyze the benchmark data to determine the outcome and draft the resulting proposal. Utilizing LaTeX to format the mathematical proofs of equivariance and correlation metrics will make the pitch significantly more compelling to the maintainers.

* **Outcome A (GAT wins broadly):** GAT provides better energy/correlations at matched budgets across most geometries. Submit the PR.
* **Outcome B (Irregular-geometry advantage):** GAT dominates specifically on arbitrary graphs where fixed group convolutions fail. Submit the PR, explicitly pitching it as the solution for non-crystalline/amorphous geometries.
* **Outcome C (Promising abstraction, flawed execution):** The graph abstraction proves useful, but standard attention is too heavy or slow. Pivot the research to a lighter symmetry-aware graph layer.
* **Outcome D (No advantage):** GAT provides no statistical improvement. Do not upstream. Publish the standalone repo and a brief research note to the NetKet Discussions to save future researchers time.

---