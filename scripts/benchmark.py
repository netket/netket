import netket as nk
import jax.numpy as jnp
import numpy as np

def run_vmc(graph, model_name, iters=250):
    print(f"\n--- Running {model_name} on {graph.__class__.__name__} ---")
    
    # 1. Hilbert space and Hamiltonian
    hi = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0)
    
    # 3. Model
    if model_name == "GAT":
        ma = nk.models.GAT(
            n_nodes=graph.n_nodes,
            edges=tuple(graph.edges()),
            layers=(64, 64), 
            heads=4,
            complex_output=True
        )
    elif model_name == "GCNN":
        ma = nk.models.GCNN(symmetries=graph.automorphisms(), layers=2, features=8)
    else:
        raise ValueError("Unknown model")
        
    # 4. Sampler
    sa = nk.sampler.MetropolisLocal(hilbert=hi)
    
    # 5. Optimizer
    op = nk.optimizer.Sgd(learning_rate=0.01)
    
    # 6. Variational State
    # Increased samples for higher constraint precision
    vstate = nk.vqs.MCState(sa, ma, n_samples=512)
    
    # 7. VMC
    # Using VMC_SR (minSR) for better stability in the rank-deficient regime (samples < params)
    vmc = nk.driver.VMC_SR(hamiltonian=ha, optimizer=op, diag_shift=0.1, variational_state=vstate)
    
    vmc.run(n_iter=iters, out=model_name)
    
    # Evaluate final energy
    energy = vstate.expect(ha)
    print(f"Final Energy: {energy.mean:.4f} ± {energy.error_of_mean:.4f}")
    return energy.mean

def main():
    print("Starting Benchmarking Suite (Mini-run for validation)")
    
    # Stage 1: Sanity (1D Chain)
    print("\\n=== Stage 1: Sanity (1D Chain) ===")
    g_chain = nk.graph.Chain(length=8, pbc=True)
    run_vmc(g_chain, "GAT")
    run_vmc(g_chain, "GCNN")
    
    # Stage 2: Standard (Square)
    print("\n=== Stage 2: Standard (Square) ===")
    g_square = nk.graph.Square(length=4, pbc=True)
    run_vmc(g_square, "GAT")
    run_vmc(g_square, "GCNN")
    
    # Stage 3: Frustrated (Triangular)
    print("\n=== Stage 3: Frustrated (Triangular) ===")
    g_tri = nk.graph.Triangular(extent=[3, 3], pbc=True)
    run_vmc(g_tri, "GAT")
    run_vmc(g_tri, "GCNN")
    
    # Stage 4: Scale (Kagome)
    print("\n=== Stage 4: Scale (Kagome) ===")
    g_kagome = nk.graph.Kagome(extent=[2, 2], pbc=True)
    run_vmc(g_kagome, "GAT")
    run_vmc(g_kagome, "GCNN")
    
    # Stage 5: Irregular (Arbitrary Graph - Random Regular Graph or similar)
    print("\n=== Stage 5: Irregular (Arbitrary Graph) ===")
    import networkx as nx
    nx_graph = nx.random_regular_graph(d=3, n=12, seed=42)
    edges = list(nx_graph.edges())
    g_irreg = nk.graph.Graph(edges=edges)
    run_vmc(g_irreg, "GAT")
    # GCNN will fail on arbitrary graph without full symmetries, but we can try with trivial symmetry
    try:
        run_vmc(g_irreg, "GCNN")
    except Exception as e:
        print(f"GCNN failed on irregular graph as expected: {e}")

if __name__ == "__main__":
    main()
