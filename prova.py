import netket as nk
import numpy as np


# some helper functions
def super_index(site=1, N=16):
    if site < N:
        return site
    else:
        return site % N


# the sign structure function
def phase_shift(Q, eta, eta_, N):
    eta_v = np.repeat([[eta, eta_]], N / 2, axis=0).flatten()
    R = np.array([[x, x] for x in range(int(N / 2))]).flatten()

    return np.exp((Q * R + eta_v) * 1j, dtype="complex")


Sx = np.array([[0, 1], [1, 0]]) * 0.5
Sy = np.array([[0, -1j], [1j, 0]]) * 0.5
Sz = np.array([[1, 0], [0, -1]]) * 0.5

diag = np.kron(Sz, Sz)
offdiag = np.kron(Sx, Sx) + np.kron(Sy, Sy)


#############################################################
#############################################################
# HILBERT

# 1D Lattice
L = 4
# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=L, total_sz=0)


edge_colors = []
color = 1
for i in range(L):
    edge_colors.append([i, (i + 1) % L, color])
    color += 1
graph = nk.graph.Graph(n_nodes=L, edges=edge_colors)


#############################################################
#############################################################
# HAMILTONIAN


bond_operator = []
bond_color = []


# setting the sign structure
Q = 1
eta = 0.5
J = 1
J_ = 1
signs = phase_shift(Q * np.pi, 0, eta * np.pi, L)
color = 1
for site in range(L):
    # diag contribution NN
    bond_operator.append((J * diag).tolist())
    bond_color.append(color)
    # OFF diag NN
    bond_operator.append(signs[site] * signs[super_index(site + 1, L)] * J * offdiag)
    bond_color.append(color)
    color += 1


ha = nk.operator.GraphOperator(
    hi, bond_ops=bond_operator, bond_ops_colors=bond_color, graph=graph, dtype=complex
)


#############################################################
#############################################################
# MACHINE

# RBM Spin Machine
ma = nk.models.RBM(
    alpha=4,
    use_visible_bias=False,
    use_hidden_bias=True,
    dtype=float,
)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.1)

# Variational State
vs = nk.vqs.MCState(sa, ma, n_samples=1000, n_discard_per_chain=100)

# Variational monte carlo driver
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)


# Run the optimization for 300 iterations
gs.run(n_iter=100, out="test")
