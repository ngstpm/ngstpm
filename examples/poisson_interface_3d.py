"""The interface problem of poisson_interface.py in 3D on one background mesh,
carved: D_h^1 the tetrahedra of a Netgen mesh of the box [-1.5, 1.5]^3 inside
the unit ball, D_h^2 those outside it, so Gamma_h^1 and Gamma_h^2 are two
staircases of faces of the background mesh with a gap of width O(h) between
them. The weak form is the one of poisson_interface.py (solve is imported from
there); the data are those of poisson_interface_two_meshes_3d.py.

Carved staircases fold in 3D (faces of Gamma_h facing back into D_h, against
their paths) or are not surfaces (four faces at an edge), and pair raises on
them; unfitted_domain(..., max_normal_deviation=85) prunes the elements owning
such faces (and those along the paths) and the sides pair.
Smoke-sized: three coarse levels, under ~2 GB.
"""

from math import log

from netgen.csg import CSGeometry, OrthoBrick, Pnt
from ngsolve import *

from ngstpm import unfitted_domain
from poisson_interface import solve
from poisson_interface_two_meshes_3d import f1, f2, gD, gN, kappa1, kappa2, phi, q1, q2, u1, u2

MAX_DEVIATION = 85
LEVELS = (0.4, 0.3, 0.2)


def domains(maxh, L=1.5, max_normal_deviation=MAX_DEVIATION):
    geo = CSGeometry()
    geo.Add(OrthoBrick(Pnt(-L, -L, -L), Pnt(L, L, L)).bc("outer"))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    D1, D2 = (unfitted_domain(mesh, phi, side, max_normal_deviation=max_normal_deviation)[0] for side in (1, 2))
    return mesh, D1, D2


def errors(mesh, sol, D1, D2):
    eu = eq = 0
    for D, uh, qh, ue, qe in ((D1, sol[1], sol[0], u1, q1), (D2, sol[4], sol[3], u2, q2)):
        dV = dx(definedonelements=D, bonus_intorder=6)
        eu += Integrate((uh - ue(x, y, z)) ** 2 * dV, mesh)
        eq += Integrate((qh - qe(x, y, z)) ** 2 * dV, mesh)
    return sqrt(eu), sqrt(eq)


def run(k, maxh):
    mesh, D1, D2 = domains(maxh)
    sol = solve(mesh, k, phi, D1, D2, kappa1, kappa2, f1, f2, gD, gN, u2(x, y, z))
    return (sum(D1) + sum(D2), *errors(mesh, sol, D1, D2))


if __name__ == "__main__":
    # rates against N^(-1/3), N the number of active tetrahedra (the meshes are not nested)
    for k in (1, 2):
        print(f"k = {k}")
        prev = None
        for maxh in LEVELS:
            N, eu, eq = run(k, maxh)
            if prev:
                r = log(N / prev[0]) / 3
                print(f"  maxh={maxh:5.3f}  N={N:6d}  |u-uh|={eu:.3e} ({log(prev[1] / eu) / r:.2f})"
                      f"  |q-qh|={eq:.3e} ({log(prev[2] / eq) / r:.2f})", flush=True)
            else:
                print(f"  maxh={maxh:5.3f}  N={N:6d}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}", flush=True)
            prev = (N, eu, eq)
