"""Stokes flow on snapped meshes (the meshes of darcy_snapped.py: the background mesh with
the vertices of the cut elements moved onto Gamma, D_h the elements that were entirely in
Omega, gap O(h^2)), standalone. Cases and variants as stokes_interpolating.py:
disk, flower (u = g on Gamma, stokes_tpm.py) and traction (sigma nu = gN nu on the circle
r = 0.6, u = g on the box, stokes_stress_tpm_neumann.py), each -tpm or -closest.
"""

from ngsolve import *

from convergence import main
from darcy_interpolating import CURVES
from darcy_snapped import snapped_mesh
from stokes_interpolating import GEOMETRY, run
from ngstpm import ParametricCurve, boundary_ds, boundary_facets

CASES = {f"snapped-{g}-{v}": {k: (0.2, 0.1, 0.05, 0.025) for k in (1, 2, 3)}
         for g in GEOMETRY for v in ("tpm", "closest")}


def row(case, k, maxh, solution=False, **kw):
    name = case[len("snapped-"):]
    geometry = GEOMETRY[name.rsplit("-", 1)[0]]
    mesh, D, checks = snapped_mesh(geometry, maxh)
    facets = boundary_facets(mesh, D)[0]
    dB = boundary_ds(mesh, D, bonus_intorder=max(6, 2 * k))
    r = run(name, k, mesh, D, dB, facets, ".*", ParametricCurve(CURVES[geometry]), **kw)
    return dict(r, **checks)


if __name__ == "__main__":
    import sys
    if any("traction" in a for a in sys.argv):
        main(row, CASES, ("|s-sh|", "|u-uh|", "|p-ph|"), {"mean(ph-p)": ".2e"}, hfmt="6.4f")
    else:
        main(row, CASES, ("|L-Lh|", "|u-uh|", "|p-ph|", "unshifted"), {"|rho|": ".1e", "alpha": ".1e"}, hfmt="6.4f")
