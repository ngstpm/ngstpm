# ngstpm

The Transfer Path Method (Cockburn–Solano) for HDG on unfitted level-set
domains, as an NGSolve add-on.

## Quick install

```
git clone https://github.com/ngstpm/ngstpm.git
cd ngstpm
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test,examples]"
pytest
```

## Install

`ngstpm` has a small C++ part that is compiled on your machine against the
NGSolve that pip installs. You need:

- Python 3.10 to 3.14;
- a C++ compiler: the Xcode command line tools on macOS
  (`xcode-select --install`), `g++` on Linux (`sudo apt install build-essential`),
  Visual Studio Build Tools (C++ workload) on Windows.

You do not need to install NGSolve or CMake yourself: pip fetches both.
NGSolve is pinned to version 6.2.2606, so install into a fresh virtual
environment and not next to another NGSolve (conda, system or older pip
install).

From the cloned `ngstpm` folder:

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test,examples]"
```

The last line downloads NGSolve, compiles the C++ module (about half a
minute) and installs `ngstpm` together with pytest and matplotlib (for the
drawing example). Check the installation with the tests (about a minute;
expect `186 passed, 10 skipped`):

```
pytest
```

Then run an example:

```
cd examples
python poisson_tpm.py
```

Activate the environment (`source .venv/bin/activate`) in every new
terminal before running anything. After updating the sources, run
`pip install -e .` again to recompile.

If something fails:

- `No module named 'ngstpm._ngstpm'`: the C++ module was not compiled. Use
  the `-e` form above and read the pip output for the compiler error.
- A compiler error mentioning NGSolve headers, or a crash on `import ngstpm`:
  another NGSolve is being picked up. `python -c "import ngsolve; print(ngsolve.__version__, ngsolve.__file__)"`
  must print 6.2.2606 and a path inside `.venv`.

`transfer_paths(mesh, geo, D, kind=...)` gives the `TransferPath` of the
element set `D` (one side): the straight paths `x -> p.target = x + ell sigma`
from `Gamma_h` of `D` to `Gamma`, with `p.ell`, `p.sigma`, `p.target` and the
normal of `Gamma` at the target, `p.target_normal`. Data at the end of the
path is `g(*p.target)`. The `kind` says how `sigma` is chosen:

| | |
|---|---|
| `"quasi-normal"` | `grad phi / \|grad phi\|` at `x` (level sets; the default) |
| `"closest"` | the path ends at the closest point of `Gamma` (explicit geometries) |
| `"facet-normal"` | the facet normal (level sets; legacy only: on unfitted meshes its paths can miss `Gamma`) |

`import ngstpm` adds three methods to NGSolve trial functions (test
functions are never extrapolated):

| | |
|---|---|
| `q.Extrapolate(target)` | `q` of the current element evaluated at `target` (default: `x`), also outside the element (polynomial extrapolation) |
| `G.PathIntegral(p)` | `∫ G · dl` along the path from `x` to `p.target`; `G` any CF linear in trial functions, its coefficients evaluated along the path |
| `u.Transfer(p, gradient=G)` | `u.Extrapolate() + G.PathIntegral(p)`: `u` moved to `p.target`, with `G ≈ grad u` |

`shifted_eval(u, forth=d)` is the ngsxfem-compatible spelling of
`u.Extrapolate(x + d)`. All are ordinary proxies, so they go straight into
the bilinear form and are statically condensed with the rest.

`unfitted_domain(mesh, phi)` gives the interior subdomain `D_h` of
`{phi < 0}`. Two measures integrate over its facets, seen from `D`
(one-sided element-boundary integrals, `specialcf.normal` the outward normal
of the element; mesh-boundary faces are left to NGSolve's `ds`):

| | |
|---|---|
| `interior_ds(mesh, D)` | the faces with both neighbours in `D` |
| `boundary_ds(mesh, D)` | `Gamma_h`: the faces with one neighbour in `D` |

```python
p = transfer_paths(mesh, phi, D)
dB = boundary_ds(mesh, D, bonus_intorder=b)
lhs += qhat_n * mu * interior_ds(mesh, D)                 # HDG conservativity
lhs += uhat.Transfer(p, gradient=-q) * mu * dB            # u = g on Gamma (q = -grad u)
rhs += g(*p.target) * mu * dB
```

Everything works in 2D and 3D.

### Explicit geometries

Instead of a level set, `Gamma` can be given explicitly; the object goes
wherever `phi` does (`unfitted_domain`, `snapped_domains`, `transfer_paths`),
and `-geometry` swaps the two sides as `-phi` does:

```python
geo = ParametricCurve(lambda t: (np.cos(t), np.sin(t)))    # closed curve, t in [0, 2 pi)
geo = BSplineCurve(ctrl, degree=3)                         # closed uniform B-spline, ctrl: (n, 2)
geo = PointCloud(points)                                   # 2D, points ordered along Gamma
geo = PointCloud(points, normals)                          # 2D or 3D, any order
D = unfitted_domain(mesh, geo)[0]
p = transfer_paths(mesh, geo, D)                           # kind="closest"
```

The path of `x` ends at the closest point of `Gamma` (for a cloud: on the
tangent plane of the nearest cloud point), so `sigma` is also the normal of
`Gamma` at the end of the path, and `h` must be below the reach of `Gamma`.
A KD-tree is used when scipy is installed.

### Interface problems

Each side `i` has its own element set `D_i` and its own paths
`p_i = transfer_paths(mesh, phi, D_i)` (`D_2` lies in `{phi > 0}`; its paths
run the other way). `I = pair(p1, p2)` is the `Interface`: it subdivides
`Gamma` into segments, each reached from one facet of `Gamma_h^1` and one of
`Gamma_h^2`, so every point `x` of `Gamma_h^i` has a partner `x'` on the
facing `Gamma_h^j` whose own path ends at the same `xbar` (the two paths may
meet at `xbar` with a kink). It raises if a piece of `Gamma_h^i` has no
partner (h too large for the geometry, or inconsistent paths). In 3D
(tetrahedra) a segment is a pair of faces and its pieces are polygons, one
quadrature rule per triangle of a piece.

| | |
|---|---|
| `I.ds(side=i, bonus_intorder=b)` | integral over `Gamma_h^i`, seen from `D_i`; one quadrature rule (order `2k + b`) per segment piece |
| `I.points(side, intorder)` | `x`, `xbar`, `x'`, `sigma(x')`, weights, own and partner element/facet at the quadrature points (numpy) |
| `I.segments()` | the common subdivision: per segment, the pieces of the two facets and their ends on `Gamma` (numpy) |

Under `I.ds(side=i)` a trial function whose space lives on side `i` is
evaluated on the element `K` of `x`, one of the other side on the partner
element `K'`: `u2.Extrapolate()` is `u2(x')`, `q2.Extrapolate(xbar)` is the
polynomial of `K'` at `xbar`, and `uhat2.Transfer(p2, gradient=-q2/kappa2(x, y))`
follows side 2's own path from `x'` to `xbar`. The same measure serves the
bilinear and the linear form and `Integrate`, and the coupling is statically
condensed with the rest (`BilinearForm(fes, condense=True)`). When one space
covers both sides, say which element is meant: `u.Extrapolate(target, side=p)`
with a `TransferPath`, or `side=1` / `side=2` under `I.ds`.

```python
p1, p2 = transfer_paths(mesh, phi, D1), transfer_paths(mesh, phi, D2)
I = pair(p1, p2)
dI = [I.ds(side=i, bonus_intorder=b) for i in (1, 2)]

xbar, nubar = p1.target, p1.target_normal                  # flux, on Gamma_h^1
qn1 = q1.Extrapolate(xbar)*nubar + tau*(u1 - uhat1)
qn2 = q2.Extrapolate(xbar)*nubar - tau*(u2.Extrapolate() - uhat2.Extrapolate())
lhs += (qn1 - qn2) * mu1 * dI[0]
rhs += gN(*xbar) * nubar * mu1 * dI[0]

xbar = p2.target                                           # trace, on Gamma_h^2
ub1 = uhat1.Transfer(p1, gradient=-q1/kappa1(x, y))
ub2 = uhat2.Transfer(p2, gradient=-q2/kappa2(x, y))
lhs += (ub1 - ub2) * mu2 * dI[1]
rhs += gD(*xbar) * mu2 * dI[1]
```

The gap between `D_1` and `D_2` may be the band of cut elements of a
background mesh (width O(h)), empty (a fitted mesh, `D1` and `D2` sharing
facets), or the slivers left by `snapped_domains(mesh, phi)`, which moves the
cut elements onto `Gamma` (width O(h^2)).

The two sides may also live on two independent meshes:
`p1 = transfer_paths(mesh1, phi, facets=mesh1.Boundaries("interface"))` takes
`Gamma_h` as a boundary region, `D` defaults to the whole mesh
(`p1.elements`), and the product space and the forms come from the
Interface: `I.Space(fes1, fes2)`, `I.BilinearForm`, `I.LinearForm`,
`I.GridFunction`. The weak form is unchanged.

### Plotting

`plot(u, p)` draws a scalar field (`uh`, `qh[0]`, `Norm(qh)`, `uh - uex`, ...)
on `D_h = p.elements` with matplotlib (2D; `pip install matplotlib`) and
returns the Axes, with the mesh and `Gamma` (the ends of the paths) on top.
`extend=True` also fills the band between `Gamma_h` and `Gamma` along the
paths: a point `y` of the path of `x`, with `K` the element of `x`, gets
`u_h|K(x) + ∫_x^y G · dl` with `gradient=G` (`G ≈ grad u`), or the polynomial
extrapolation `u_h|K(y)` without it.

```python
ax = plot(uh, p, extend=True, gradient=-qh, cmap="jet")   # q = -grad u
```

## Examples and tests

```
python examples/poisson.py              # fitted HDG, unit square
python examples/poisson_tpm.py          # Dirichlet: disk, flower, annulus, square with a hole
python examples/poisson_tpm_draw.py     # draws u_h on each of them with plot() (needs matplotlib)
python examples/poisson_tpm_RT.py       # the same with mixed Raviart-Thomas instead of HDG
python examples/poisson_tpm_quads.py    # HDG, HDG-Q[k] (enriched flux) and RT on quadrilaterals
python examples/poisson_tpm_3d.py       # Dirichlet: ball, torus (a few minutes)
python examples/poisson_tpm_neumann.py  # Neumann condition on the unfitted boundary
python examples/poisson_interface.py    # interface problem: fitted, snapped, unfitted
python examples/poisson_interface_two_meshes.py  # the same on two non-matching meshes
pytest
```
