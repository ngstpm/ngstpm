"""Draw the polygonal free-slip paradox of slip_paradox.py: S (slip on the N-gon) against
TPM (slip transferred to the circle), rotating cylinder, exact solution u = Omega x^perp.

- slip_paradox_fields.png: on one mesh (N, k below), the angular velocity
  u_theta / r = x^perp . u_h / r^2 (exact: Omega) and the vorticity of u_h (exact:
  2 Omega; Hodge limit: 0); rows S, TPM. TPM also fills the band between Gamma_h and
  Gamma by polynomial extrapolation (ngstpm.plot, extend=True); Gamma is drawn on top.
- slip_paradox_vs_N.png: the torque |M_s| on Gamma_h and the mean vorticity on Gamma_h
  against N (k = 1, 2), and u_theta(r) along the ray through an edge midpoint of the
  N-gon, with the exact, Hodge and no-slip references.

Needs matplotlib. Prints the diagnostics, writes the two figures and opens them.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from ngsolve import *

from ngstpm import plot, transfer_paths
import slip_paradox as sp

N, k = 16, 2                                       # the mesh of the fields and the profile
NS, KS = (8, 16, 32, 64, 128), (1, 2)


def run(N, k, transfer):
    mesh = sp.polygon_mesh(N)
    sol, _, _ = sp.solve(mesh, k, sp.MU, CF((0, 0)), sp.OMEGA * sp.XPERP, transfer=transfer)
    return mesh, sol


def omega(u):                                      # u_theta / r
    return InnerProduct(sp.XPERP, u) / (x * x + y * y)


def vorticity(u):
    G = grad(u)                                    # G[i, j] = d_j u_i
    return G[1, 0] - G[0, 1]


# ---------------------------------------------------------------- fields on one mesh
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
profiles = {}
for row, form in enumerate(("S", "TPM")):
    mesh, sol = run(N, k, form == "TPM")
    u = sol[1]
    p = transfer_paths(mesh, sp.PHI, facets=mesh.Boundaries("interface"), order=k + 4)
    extend = form == "TPM"
    for col, (name, cf, lim) in enumerate((("u_theta / r", omega(u), (0, 1.2 * sp.OMEGA)),
                                           ("vorticity", vorticity(u), (-0.5, 2.5 * sp.OMEGA)))):
        ax = plot(cf, p, extend=extend, ax=axes[row, col], cmap="viridis", vmin=lim[0], vmax=lim[1],
                  edges=False)
        ax.set_title(f"{form}: {name}" + (" (band extrapolated)" if extend else ""))
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    th = math.pi / N                               # ray through an edge midpoint
    r = np.linspace(sp.R1 * 1.001, sp.R2 * math.cos(th) * 0.999, 200)
    profiles[form] = r, np.array([u(mesh(ri * math.cos(th), ri * math.sin(th)))
                                  @ np.array([-math.sin(th), math.cos(th)]) for ri in r])
fig.suptitle(f"rotating cylinder, free slip on r = {sp.R2}, N = {N}, k = {k}")
fig.savefig("slip_paradox_fields.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- against N
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8))
print(f"{'form':>4} {'k':>2} {'N':>4} {'M_s':>12} {'vort_iface':>12} {'err_exact':>12}")
for form, ls in (("S", "-o"), ("TPM", "--s")):
    for kk in KS:
        d = []
        for n in NS:
            mesh, sol = run(n, kk, form == "TPM")
            d.append(sp.diagnostics(mesh, sol, kk))
            print(f"{form:>4} {kk:2d} {n:4d} {d[-1]['M_s']:12.4e} {d[-1]['vort_iface']:12.6f} "
                  f"{d[-1]['err_exact']:12.4e}")
        a1.loglog(NS, [max(abs(q["M_s"]), 1e-16) for q in d], ls, label=f"{form}, k = {kk}")
        a2.semilogx(NS, [q["vort_iface"] for q in d], ls, label=f"{form}, k = {kk}")
a1.loglog(NS, [sp.REF_TORQUE["hodge"] * 8 / n for n in NS], "k:", label="1/N")
a1.axhline(sp.REF_TORQUE["hodge"], color="gray", lw=0.8)
a1.text(NS[0], sp.REF_TORQUE["hodge"] * 1.2, "Hodge limit pi", color="gray")
a1.set(xlabel="N", ylabel="|M_s|", title="torque on Gamma_h (exact 0)")
for val, lab in ((2 * sp.OMEGA, "exact"), (0.0, "Hodge")):
    a2.axhline(val, color="gray", lw=0.8)
    a2.text(NS[0], val + 0.05, lab, color="gray")
a2.set(xlabel="N", ylabel="mean vorticity on Gamma_h", title="vorticity on Gamma_h (exact 2)")
r = np.linspace(sp.R1, sp.R2, 200)
for name, ref, st in (("exact", sp.OMEGA * r, "k-"),
                      ("Hodge", sp.OMEGA * sp.R1**2 / r, "k:"),
                      ("no-slip", (sp._A * r + sp._B / r), "k-.")):
    a3.plot(r, ref, st, lw=1, label=name)
for form, (rr, ut) in profiles.items():
    a3.plot(rr, ut, "-" if form == "S" else "--", lw=2, label=f"{form}, N = {N}, k = {k}")
a3.set(xlabel="r", ylabel="u_theta", title="u_theta along the ray through an edge midpoint")
for a in (a1, a2, a3):
    a.legend(fontsize=8)
fig.savefig("slip_paradox_vs_N.png", dpi=200, bbox_inches="tight")
plt.show()
