"""Sparse LU by SciPy's SuperLU, as an inverse(mat, free) for elasticity.solve_linear_system and
stokes_stress.multiplier_inverse. NGSolve 6.2.2606's UMFPACK inverse leaks memory at each
factorization (agentB-tables/goal4-stage3-leak.txt), which caps the Picard runs
(navier_stokes_stress.py: one factorization per step); SuperLU frees its factors with the object.
"""
import numpy as np
import scipy.sparse as sp
from ngsolve import BaseMatrix
from scipy.sparse.linalg import splu

PERMC = "COLAMD"        # SuperLU column ordering (agentB-tables/goal4-superlu.txt)


class SuperLUInverse(BaseMatrix):
    """The inverse of mat on the free dofs (zero on the others), as a BaseMatrix."""

    def __init__(self, mat, free=None, permc_spec=PERMC):
        super().__init__()
        self.mat = mat
        val, col, ind = mat.CSR()
        A = sp.csr_matrix((np.array(val), np.array(col), np.array(ind)), shape=(mat.height, mat.width))
        self.free = (np.arange(mat.height) if free is None
                     else np.flatnonzero(np.array([free[i] for i in range(len(free))], dtype=bool)))
        self.lu = splu(A[self.free][:, self.free].tocsc(), permc_spec=permc_spec)

    def Height(self):
        return self.mat.height

    def Width(self):
        return self.mat.width

    def CreateColVector(self):
        return self.mat.CreateColVector()

    def CreateRowVector(self):
        return self.mat.CreateRowVector()

    def Mult(self, b, x):
        y = np.zeros(self.mat.height)
        y[self.free] = self.lu.solve(b.FV().NumPy()[self.free])
        x.FV().NumPy()[:] = y


def inverse(mat, free=None):
    return SuperLUInverse(mat, free)
