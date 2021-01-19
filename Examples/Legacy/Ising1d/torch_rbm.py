from netket import legacy as nk
import torch
import numpy as np
import cProfile


class LogCoshSum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._ls = torch.nn.LogSigmoid()
        self._log2 = np.log(2.0)

    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.zeros((batch_size, 2), dtype=torch.float64)
        x = torch.abs(x)
        y[:, 0] = (x - self._ls.forward(2 * x) - self._log2).sum(dim=1)
        return y


# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

input_size = hi.size
alpha = 1

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, alpha * input_size),
    LogCoshSum(),
)

ma = nk.machine.Torch(model, hilbert=hi)

ma.parameters = 0.1 * (np.random.randn(ma.n_par))

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=64)

# Optimizer
op = nk.optimizer.Sgd(ma, 0.1)

gs = nk.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    n_discard=2,
    sr=nk.optimizer.SR(ma, diag_shift=0.1),
)

gs.run(out="test", n_iter=300)
