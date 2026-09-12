import time
import jax
import jax.numpy as jnp
from gat.model import GATModel

def test_scaling():
    print("Running scaling tests...")
    sizes = [16, 36, 64, 100]
    batch_size = 128
    
    model = GATModel(layers=[16, 16])
    rng = jax.random.PRNGKey(42)
    
    for N in sizes:
        print(f"\\n--- System size N={N} ---")
        
        # Setup data
        x = jax.random.choice(rng, jnp.array([-1.0, 1.0]), shape=(batch_size, N))
        A = jnp.ones((N, N)) - jnp.eye(N)  # fully connected for testing overhead
        
        # Init
        variables = model.init(rng, x, A)
        
        # JIT compile the apply function
        @jax.jit
        def apply_fn(vars, inputs, adj):
            return model.apply(vars, inputs, adj)
            
        # Measure JIT time
        start_jit = time.time()
        # first run triggers compilation
        out_jit = apply_fn(variables, x, A)
        # block until done
        out_jit.block_until_ready()
        jit_time = time.time() - start_jit
        print(f"JIT Compilation + 1st run time: {jit_time:.4f} s")
        
        # Measure throughput (samples / second)
        num_runs = 50
        start_run = time.time()
        for _ in range(num_runs):
            out = apply_fn(variables, x, A)
        out.block_until_ready()
        run_time = time.time() - start_run
        
        throughput = (num_runs * batch_size) / run_time
        print(f"Throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    test_scaling()
