import json
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt


def gen_pair(row, V, PBC='False'):
    '''
    assume row is an in order array generate a cyclic pairs
    in the row array given with interaction strength V.
    For example: row = [1, 2, 3, 5]
    will gives [(1, 2, V), (2, 3, V), (3, 5, V), (5, 1, V)]
    '''
    if PBC == True:
        return [(row[i], row[(i + 1) % len(row)], V) for i in range(len(row))]
    else:
        return [(row[i], row[(i + 1) % len(row)], V) for i in range(len(row)-1)]

def gen_pair_2d_nnn(plaquette, V):
    '''
    assume the plaquette indices are given in cyclic order,
    return two pair of the cross nnn term interaction.
    For example: plaquette = [  1, 2,
                                5, 6]
    wiil gives[(1, 6, V), (2, 5, V)]
    '''
    return [(plaquette[0], plaquette[2], V), (plaquette[1], plaquette[3], V)]


def build_H(pairs, L):
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    Sy = np.array([[0., -1j],
                   [1j, 0.]])
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    # S = [Sx, Sy, Sz]
    H = scipy.sparse.csr_matrix((2 ** L, 2 ** L))
    for i, j, V in pairs:
        if i > j:
            i, j = j, i

        print("building", i, j)
        hx = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sx)
        hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (j - i - 1)))
        hx = scipy.sparse.kron(hx, Sx)
        hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (L - j)))

        hy = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sy)
        hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (j - i - 1)))
        hy = scipy.sparse.kron(hy, Sy)
        hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (L - j)))

        hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sz)
        hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (j - i - 1)))
        hz = scipy.sparse.kron(hz, Sz)
        hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - j)))

        H = H + V * (hx + hy + hz)

    H = scipy.sparse.csr_matrix(H)
    return H

def build_Sx(L):
    Sx = np.array([[0., 1.],
                   [1., 0.]])
    hx = scipy.sparse.csr_matrix(Sx)
    for i in range(1,L):
        print(i)
        hx = scipy.sparse.kron(hx, Sx)

    return hx

def spin_spin_correlation(site_i, site_j, L, vector):
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    print("correlation between", site_i, site_j)
    hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (site_i - 1)), Sz)
    hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (site_j - site_i - 1)))
    hz = scipy.sparse.kron(hz, Sz)
    hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - site_j)))
    SzSz = scipy.sparse.csr_matrix(hz)
    return vector.conjugate().dot(SzSz.dot(vector))

def sz_expectation(site_i, vector, L):
    Sz = np.array([[1., 0.],
                   [0., -1.]])

    print("spin z expectation value on site %d" % site_i)
    hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (site_i - 1)), Sz)
    hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - site_i)))
    Sz = scipy.sparse.csr_matrix(hz)
    return vector.conjugate().dot(Sz.dot(vector))

def solve_1d_J1J2(L, J1=1, J2=0.):
    lattice = np.arange(L, dtype=int) + 1
    print(lattice)
    pairs = []
    J1 = J1
    for i in range(1, L + 1):
        pairs = pairs + [(i, (i % L) + 1, J1)]

    J2 = J2
    for i in range(1, L - 1):
        pairs = pairs + [(i, i + 2, J2)]

    pairs += [(L - 1, 1, J2), (L, 2, J2)]

    print('all pairs', pairs)
    H = build_H(pairs, L)

    evals_small, evecs_small = eigsh(H, 6, which='SA')
    print(evals_small / L / 4.)
    return evals_small, evecs_small


def solve_2d_J1J2(Lx, Ly, J1=1, J2=0.):
    lattice = np.zeros((Lx, Ly), dtype=int)
    for i in range(Lx):
        for j in range(Ly):
            lattice[i, j] = int(j * Lx + (i+1))

    print(lattice)
    pairs = []
    # NN interaction : J1
    for i in range(Lx):
        print(lattice[i, :])
        pairs = pairs + gen_pair(lattice[i, :], J1)

    for j in range(Ly):
        print(lattice[:, j])
        pairs = pairs + gen_pair(lattice[:, j], J1)

    # NNN interaction : J2
    for i in range(Lx):
        for j in range(Ly):
            plaquette=[lattice[i,j]]
            plaquette.append(lattice[(i+1)%Lx, j])
            plaquette.append(lattice[(i+1)%Lx, (j+1)%Ly])
            plaquette.append(lattice[i, (j+1)%Ly])
            pairs = pairs + gen_pair_2d_nnn(plaquette, J2)


    print('all pairs', pairs)
    global H
    H = build_H(pairs, Lx*Ly)

    evals_small, evecs_small = eigsh(H, 6, which='SA')
    print('Energy : ', evals_small / Lx / Ly / 4.)
    return evals_small, evecs_small

def check_phase(vector, dim=1, site=None):
    '''
    check the phase of one site translation
    '''
    if dim==1:
        new_vector = np.zeros_like(vector)
        len_v = new_vector.size
        new_vector[:int(len_v/2)] = vector[::2]
        new_vector[int(len_v/2):] = vector[1::2]
        return new_vector.conjugate().dot(vector)
    if dim==2:
        new_vector = np.copy(vector)
        len_v = new_vector.size
        for i in range(site):
            temp_vec = np.zeros_like(new_vector)
            temp_vec[:int(len_v/2)] = new_vector[::2]
            temp_vec[int(len_v/2):] = new_vector[1::2]
            new_vector = np.copy(temp_vec)
        return new_vector.conjugate().dot(vector)

def store_eig_vec(evals_small, evecs_small, filename):
    idx_min = np.argmin(evals_small)
    print("GS energy: %f" % evals_small[idx_min])
    vec_r = np.real(evecs_small[:,idx_min])
    vec_i = np.imag(evecs_small[:,idx_min])
    vec_r = vec_r / np.linalg.norm(vec_r)
    vec_i = vec_i / np.linalg.norm(vec_i)
    if np.abs(vec_r.dot(vec_i)) - 1. < 1e-6:
        print("Eigen Vec can be casted as real")
        log_file = open(filename, 'wb')
        np.savetxt(log_file, vec_r, fmt='%.8e', delimiter=',')
        log_file.close()
    else:
        print(np.abs(vec_r.dot(vec_i)) - 1.)
        print("Complex Eigen Vec !!!")
        print("The real part <E> : %f " %  vec_r.T.dot(H.dot(vec_r)) )
        print("The imag part <E> : %f " %  vec_i.T.dot(H.dot(vec_i)) )

    return


def save_to_txt(x, L, formating=False):
    output={"samples": {}}

    output["samples"]["configs"] = []
    output["samples"]["amp"] = []
    for i in range(2**L):
        output["samples"]["configs"].append([int(num) for num in np.binary_repr(i, width=L)])
        output["samples"]["amp"].append(x)
        
    np.savetxt("1d-J1J2-samples-L-{0}.txt".format(L), output["samples"]["configs"])
    np.savetxt("1d-J1J2-targets-L-{0}.txt".format(L), output["samples"]["amp"], delimiter='\n')


if __name__ == "__main__":
    import sys

    if( len(sys.argv) < 4 ):
        print("Incorrect command-line arguments")
        print("Usage: ED.py L J1 J2")
        exit(0) 

    L, J1, J2 = sys.argv[1:]
    L, J1, J2 = int(L), float(J1), float(J2)
    evals_small, evecs_small = solve_1d_J1J2(L, J1, J2)
    save_to_txt(evecs_small[:,0], L)
