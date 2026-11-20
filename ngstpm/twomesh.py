"""Interface problems on two meshes (API-PROPOSAL D9): I.Space, I.BilinearForm,
I.LinearForm, I.GridFunction.

An NGSolve space lives on one mesh, so the two sides' spaces cannot be one
compound space. `I.Space(fes1, fes2)` pairs them; its `TnT()` gives the proxies
of both, and the forms built on it take the same weak-form lines as on one mesh:
`lhs += ...` sends each integral to the side whose space its proxies belong to,
and an interface integral `... * I.ds(side=i)` becomes coupling blocks (rows of
side i, columns of both sides). `Assemble()` puts both sides' (condensed)
matrices and the blocks into one sparse matrix, solved with umfpack;
`harmonic_extension`, `harmonic_extension_trans` and `inner_solve` act on both
sides, so a condensed solve is written as on one mesh (examples/poisson.py,
solve_linear_system). Vectors are flat (side 1 first).

On one mesh these return NGSolve's own objects (fes1 * fes2, BilinearForm, ...).
"""

import numpy as np
from ngsolve import BaseMatrix, BaseVector, BilinearForm, BitArray, FESpace, GridFunction, LinearForm
from ngsolve.la import SparseMatrixd

from ._ngstpm import ComponentSpace, Interface, InterfaceCouplingBlocks, InterfaceIntegral, Leaves, _IfcSide


def _meshes(I):
    return _IfcSide(I, 1)[0], _IfcSide(I, 2)[0]


def _parts(fes, proxies):
    return list(proxies) if len(fes.components) else [proxies]


class TwoMeshSpace:
    """The spaces fes1 (on the mesh of side 1) and fes2 (side 2) of an Interface."""

    def __init__(self, I, fes1, fes2):
        for i, (fes, mesh) in enumerate(zip((fes1, fes2), _meshes(I))):
            if fes.mesh != mesh:
                raise ValueError(f"I.Space: space {i + 1} is not on the mesh of side {i + 1}")
        self.interface, self.spaces = I, (fes1, fes2)
        self.meshes = _meshes(I)
        self._tnt = (fes1.TnT(), fes2.TnT())

    @property
    def ndof(self):
        return sum(f.ndof for f in self.spaces)

    def TnT(self):
        (u1, v1), (u2, v2) = self._tnt
        f1, f2 = self.spaces
        return tuple(_parts(f1, u1) + _parts(f2, u2)), tuple(_parts(f1, v1) + _parts(f2, v2))

    def FreeDofs(self, coupling=False):
        free = BitArray(self.ndof)
        free.Clear()
        off = 0
        for fes in self.spaces:
            fd = fes.FreeDofs(coupling)
            for d in range(fes.ndof):
                free[off + d] = fd[d]
            off += fes.ndof
        return free

    def side(self, cf):
        """The side (0, 1) of the proxies of cf (None: no proxies, e.g. 0 * v simplified
        away); raises if they are on both."""
        trial, test, _, _ = Leaves(cf)
        sides = {self.meshes.index(ComponentSpace(p).mesh) for p in list(trial) + list(test)}
        if len(sides) > 1:
            raise ValueError("an integral that is not an interface integral must have the proxies of one side")
        return sides.pop() if sides else None


class _Flat(BaseMatrix):
    """A matrix of both sides on flat vectors (side 1 first): a sparse matrix, or the
    per-side operators ops (y_i = A_i x_i) plus an optional flat matrix C."""

    def __init__(self, sizes, mat=None, ops=(), C=None):
        super().__init__()
        self.sizes, self.mat, self.ops, self.C = sizes, mat, ops, C

    def Mult(self, x, y):
        if self.mat is not None:
            y.data = self.mat * x
            return
        for A, xi, yi in zip(self.ops, _blocks(x, self.sizes), _blocks(y, self.sizes)):
            yi.data = A * xi
        if self.C is not None:
            y.data += self.C * x

    def Height(self):
        return sum(self.sizes)

    def Width(self):
        return sum(self.sizes)

    def CreateColVector(self):
        return _vec(sum(self.sizes))

    def CreateRowVector(self):
        return _vec(sum(self.sizes))

    def Inverse(self, freedofs=None, inverse="umfpack"):
        return _Flat(self.sizes, self.mat.Inverse(freedofs, inverse=inverse or "umfpack"))


def _blocks(v, sizes):
    """The side blocks of the flat vector v, as ranges."""
    offs = np.cumsum([0] + list(sizes))
    return [v.Range(int(a), int(b)) for a, b in zip(offs[:-1], offs[1:])]


def _vec(n):
    v = BaseVector(n)
    v[:] = 0
    return v


def _coo(n, triplets):
    """n x n sparse matrix from a list of (rows, cols, values) arrays (duplicates are summed)."""
    i, j, v = ([np.concatenate([t[k] for t in triplets]) for k in range(3)] if triplets
               else [np.zeros(0)] * 3)
    return SparseMatrixd.CreateFromCOO(i.astype(int).tolist(), j.astype(int).tolist(), v.tolist(), n, n)


class TwoMeshBilinearForm:
    """The bilinear form of a TwoMeshSpace (see the module docstring)."""

    def __init__(self, fes, condense=False):
        self.space, self.condense = fes, condense
        self.forms = [BilinearForm(f, condense=condense) for f in fes.spaces]
        self.interface_integrals = []
        self.mat = None

    def __iadd__(self, other):
        for k in range(len(other)):
            p = other[k]
            if isinstance(p, InterfaceIntegral):
                self.interface_integrals.append(p)
            elif (side := self.space.side(p.coef)) is not None:
                self.forms[side] += p.coef * p.symbol
        return self

    def Assemble(self):
        for f in self.forms:
            f.Assemble()
        sizes = [f.ndof for f in self.space.spaces]
        offs = [0, sizes[0]]
        A, C = [], []
        for i, f in enumerate(self.forms):
            r, c, v = f.mat.COO()
            A.append((np.array(r) + offs[i], np.array(c) + offs[i], np.array(v)))
        for p in self.interface_integrals:
            s = p.side - 1
            o = 1 - s
            fes, pfes = self.space.spaces[s], self.space.spaces[o]
            bf, pbf = (self.forms[s], self.forms[o]) if self.condense else (None, None)
            B = InterfaceCouplingBlocks(p.interface, p.coef, p.side, p.bonus_intorder, fes, pfes, bf, pbf)
            for name, col, out in (("own", s, A), ("partner", o, A),
                                   ("own_correction", s, C), ("partner_correction", o, C)):
                r, c, v = B[name]
                out.append((np.asarray(r) + offs[s], np.asarray(c) + offs[col], np.asarray(v)))
        n = sum(sizes)
        self.mat = _Flat(sizes, _coo(n, A))
        if self.condense:
            self.harmonic_extension = _Flat(sizes, ops=[f.harmonic_extension for f in self.forms])
            self.harmonic_extension_trans = _Flat(sizes, ops=[f.harmonic_extension_trans for f in self.forms],
                                                  C=_coo(n, C))            # + the interface correction
            self.inner_solve = _Flat(sizes, ops=[f.inner_solve for f in self.forms])
        return self


def _symbol(p):
    """The measure of the integral p (p.symbol loses the interface of an I.ds)."""
    if isinstance(p, InterfaceIntegral):
        return p.interface.ds(side=p.side, bonus_intorder=p.bonus_intorder)
    return p.symbol


class TwoMeshLinearForm:
    """The linear form of a TwoMeshSpace: each integral goes to its side's LinearForm."""

    def __init__(self, fes):
        self.space = fes
        self.forms = [LinearForm(f) for f in fes.spaces]

    def __iadd__(self, other):
        for k in range(len(other)):
            p = other[k]
            if (side := self.space.side(p.coef)) is not None:
                self.forms[side] += p.coef * _symbol(p)
        return self

    def Assemble(self):
        for f in self.forms:
            f.Assemble()
        return self

    @property
    def vec(self):
        """Both sides' vectors as one flat vector (a copy)."""
        sizes = [f.space.ndof for f in self.forms]
        v = _vec(sum(sizes))
        for b, f in zip(_blocks(v, sizes), self.forms):
            b.data = f.vec
        return v


class TwoMeshGridFunction:
    """A GridFunction of a TwoMeshSpace: components of side 1, then of side 2.

    The coefficients live in one flat vector `vec` (side 1 first), the one the
    solver writes into; the components are each side's GridFunctions, synchronised
    with it: reading `components` copies vec into them, and the next access to
    `vec` copies them back (e.g. after components[i].Set(...)). Re-read
    `components` after changing `vec`; references held from before are stale."""

    def __init__(self, fes):
        self.space = fes
        self.gfs = [GridFunction(f) for f in fes.spaces]
        self.sizes = [f.ndof for f in fes.spaces]
        self._vec = _vec(sum(self.sizes))
        self._handed_out = False

    @property
    def vec(self):
        if self._handed_out:
            for b, g in zip(_blocks(self._vec, self.sizes), self.gfs):
                b.data = g.vec
            self._handed_out = False
        return self._vec

    @property
    def components(self):
        if not self._handed_out:
            for b, g in zip(_blocks(self._vec, self.sizes), self.gfs):
                g.vec.data = b
            self._handed_out = True
        return tuple(c for g in self.gfs for c in (g.components if len(g.components) else (g,)))


def _one_mesh(I):
    m1, m2 = _meshes(I)
    return m1 == m2


def _space(self, fes1, fes2):
    """The space of both sides: fes1 * fes2 on one mesh, a TwoMeshSpace on two."""
    return fes1 * fes2 if _one_mesh(self) else TwoMeshSpace(self, fes1, fes2)


def _bilinear(self, fes, condense=False):
    return TwoMeshBilinearForm(fes, condense) if isinstance(fes, TwoMeshSpace) else BilinearForm(fes, condense=condense)


def _linear(self, fes):
    return TwoMeshLinearForm(fes) if isinstance(fes, TwoMeshSpace) else LinearForm(fes)


def _gridfunction(self, fes):
    return TwoMeshGridFunction(fes) if isinstance(fes, TwoMeshSpace) else GridFunction(fes)


_fes_mul = FESpace.__mul__


def _mul(self, other):
    if isinstance(other, FESpace) and other.mesh != self.mesh:
        raise ValueError("a compound space of two meshes is not possible: use I.Space(fes1, fes2)")
    return _fes_mul(self, other)


FESpace.__mul__ = _mul
Interface.Space = _space
Interface.BilinearForm = _bilinear
Interface.LinearForm = _linear
Interface.GridFunction = _gridfunction
