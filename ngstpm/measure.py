"""The interface measure I.ds(side=i) (API-PROPOSAL D3, D7, D12, D15-D17, section 6).

`I.ds(side=i, bonus_intorder=b)` is a DifferentialSymbol: `cf * I.ds(...)` is an ordinary
SumOfIntegrals, mixable with dx terms. It integrates over the facets of Gamma_h^i seen from
D_h^i: one-sided, specialcf.normal is the outward normal of the element K, and a trial function
that is not extrapolated is K's. Not NGSolve's ds (BND elements).

- BilinearForm: `+=` splits the SumOfIntegrals; the interface integrals are assembled on the
  segments of the common subdivision (per-piece Gauss rules of order 2k + bonus), the rest by
  NGSolve unchanged. `q.Extrapolate(...)` of a space without dofs on K is evaluated on the
  partner K' (D7); with condense=True the coupling is condensed with the rest (D16).
- LinearForm and Integrate: local integrands, on the same per-piece rules (rule order
  2k + bonus in a LinearForm, as in the BilinearForm; 5 + bonus in Integrate).
- Raises: a test function with element-local dofs (D16); a trial function of the other side
  that is not extrapolated (D7).
"""

from functools import reduce
from operator import add

from ngsolve import BilinearForm
from ngsolve.comp import SumOfIntegrals

from ._ngstpm import Interface, InterfaceDifferentialSymbol, InterfaceIntegral, _AddInterfaceIntegral, _IfcSide


def _ds(self, side, bonus_intorder=0):
    """The interface measure on Gamma_h^side (side: 1, 2, or a TransferPath of this Interface).

    One-sided: it integrates over the facets of Gamma_h^side seen from the
    elements K of D_h^side, so specialcf.normal is the outward normal of K and
    a trial function that is not extrapolated is K's. The facets are integrated
    as element boundaries of K (as dx(element_boundary=True)), also when they
    are interior facets of the mesh; this is not NGSolve's ds, which integrates
    over BND elements. One Gauss rule per piece of the common subdivision, of
    order 2k + bonus_intorder in forms (5 + bonus_intorder in Integrate)."""
    if not isinstance(side, int):
        same = lambda a, b: (a & ~b).NumSet() == 0 and (b & ~a).NumSet() == 0
        side = next((i for i in (1, 2) if _IfcSide(self, i)[0] == side.mesh
                     and same(_IfcSide(self, i)[1], side.elements)), None)
        if side is None:
            raise ValueError("I.ds: the path is not a side of this Interface")
    return InterfaceDifferentialSymbol(self, side, bonus_intorder)


_previous_iadd = BilinearForm.__iadd__


def _iadd(self, other):
    if isinstance(other, SumOfIntegrals):
        parts = [other[i] for i in range(len(other))]
        ifc = [p for p in parts if isinstance(p, InterfaceIntegral)]
        if ifc:
            rest = [p.coef * p.symbol for p in parts if not isinstance(p, InterfaceIntegral)]
            if rest:
                _previous_iadd(self, reduce(add, rest))
            for p in ifc:
                _AddInterfaceIntegral(self, p)
            return self
    return _previous_iadd(self, other)


Interface.ds = _ds
BilinearForm.__iadd__ = _iadd
