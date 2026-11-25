"""The interface problem of poisson_interface_two_meshes.py in 3D: Omega_1 (the
unit ball) and Omega_2 (a box minus the ball), each meshed on its own with
straight tetrahedra, so Gamma_h^1 and Gamma_h^2 are two non-matching
triangulated surfaces near the unit sphere Gamma. The weak form is the one of
poisson_interface_two_meshes.py (solve is imported from there). Smoke-sized:
the box [-1.2, 1.2]^3 and three coarse levels keep the run under ~1.5 GB.
"""

from math import log

from netgen.csg import CSGeometry, OrthoBrick, Pnt, Sphere
from ngsolve import *

from poisson_interface_two_meshes import solve


def meshes(maxh1, maxh2, L=1.2):
    g1 = CSGeometry()
    g1.Add(Sphere(Pnt(0, 0, 0), 1).bc("interface"))
    g2 = CSGeometry()
    g2.Add(OrthoBrick(Pnt(-L, -L, -L), Pnt(L, L, L)).bc("outer") - Sphere(Pnt(0, 0, 0), 1).bc("interface"))
    return Mesh(g1.GenerateMesh(maxh=maxh1)), Mesh(g2.GenerateMesh(maxh=maxh2))


phi = lambda X, Y, Z: sqrt(X*X + Y*Y + Z*Z) - 1
kappa1, kappa2 = (lambda X, Y, Z: CF(1.0)), (lambda X, Y, Z: CF(10.0))
u1 = lambda X, Y, Z: sin(X) * cos(Y) * exp(0.5 * Z)
u2 = lambda X, Y, Z: exp(0.5 * X) * cos(0.5 * Y) + X * Y * Z
q1 = lambda X, Y, Z: -kappa1(X, Y, Z) * CF((cos(X) * cos(Y) * exp(0.5 * Z), -sin(X) * sin(Y) * exp(0.5 * Z),
                                             0.5 * sin(X) * cos(Y) * exp(0.5 * Z)))
q2 = lambda X, Y, Z: -kappa2(X, Y, Z) * CF((0.5 * exp(0.5 * X) * cos(0.5 * Y) + Y * Z,
                                             -0.5 * exp(0.5 * X) * sin(0.5 * Y) + X * Z, X * Y))
f1 = sum(q1(x, y, z)[i].Diff(c) for i, c in enumerate((x, y, z)))
f2 = sum(q2(x, y, z)[i].Diff(c) for i, c in enumerate((x, y, z)))
gD = lambda X, Y, Z: u1(X, Y, Z) - u2(X, Y, Z)
gN = lambda X, Y, Z: q1(X, Y, Z) - q2(X, Y, Z)
LEVELS = (0.5, 0.35, 0.25)


def run(k, maxh):
    mesh1, mesh2 = meshes(maxh, 1.23 * maxh)                          # non-matching on Gamma
    sol = solve(mesh1, mesh2, k, phi, kappa1, kappa2, f1, f2, gD, gN, u2(x, y, z))
    e = lambda a, b, m: Integrate((a - b)**2 * dx(bonus_intorder=6), m)
    eu = sqrt(e(sol[1], u1(x, y, z), mesh1) + e(sol[4], u2(x, y, z), mesh2))
    eq = sqrt(e(sol[0], q1(x, y, z), mesh1) + e(sol[3], q2(x, y, z), mesh2))
    return mesh1.ne + mesh2.ne, eu, eq


if __name__ == "__main__":
    # the meshes are not nested and the layer between the sphere and the box limits the
    # size of the elements of mesh 2: rates against N^(-1/3), N the number of tetrahedra
    for k in (1, 2):
        print(f"k = {k}")
        prev = None
        for maxh in LEVELS:
            N, eu, eq = run(k, maxh)
            if prev:
                r = log(N / prev[0]) / 3
                print(f"  maxh={maxh:5.3f}  N={N:5d}  |u-uh|={eu:.3e} ({log(prev[1] / eu) / r:.2f})"
                      f"  |q-qh|={eq:.3e} ({log(prev[2] / eq) / r:.2f})")
            else:
                print(f"  maxh={maxh:5.3f}  N={N:5d}  |u-uh|={eu:.3e}         |q-qh|={eq:.3e}")
            prev = (N, eu, eq)
