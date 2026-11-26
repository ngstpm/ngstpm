"""The suggestion of examples/elastodynamics_with_damping.py (transfer="grad", damping): not the
formulation of the preprints; kept for later study."""
import sys
from pathlib import Path

import pytest
from ngsolve import CF, GridFunction, x, y

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import elastodynamics as ed  # noqa: E402
import elastodynamics_with_damping as edd  # noqa: E402
from elastodynamics_with_damping import t  # noqa: E402
from elasticity_tpm import EXAMPLES, domain  # noqa: E402

TPM = "Example 2 (unfitted disk)"
POLY = {1: CF((1 + 2 * x - y, -3 + x + 4 * y)),
        2: CF((x * x - 2 * x * y + y, 3 * y * y + x * y - x)),
        3: CF((x ** 3 - 2 * x * y * y + y, 3 * y ** 3 + x * x * y - x))}
MAT = edd.material(rho=2.0, b0=1.5)
STABLE = {"displacement": (edd.Elastodynamics, edd.run, dict(transfer="grad", damping=1.0)),
          "velocity": (edd.VelocityStress, edd.run_velocity, dict(transfer="grad"))}


def unfitted(name, h, prune=True):
    phi = EXAMPLES[name]
    mesh, D, dirichlet, _ = domain(name, h, phi, prune)
    return mesh, dict(phi=phi, D=D, dirichlet=dirichlet)


@pytest.mark.parametrize("form", STABLE)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_stable_settings_exact(form, k):
    """Polynomial solutions stay exact with transfer="grad" (and damping) on the unfitted disk."""
    eta = POLY[k] * (1 + t + t * t / 2) + CF((y, -x)) * t * t
    mesh, kw = unfitted(TPM, 0.3, prune=False)
    _, runner, extra = STABLE[form]
    worst = []
    runner(mesh, k, eta, MAT, 0.1, 4, monitor=lambda r: worst.append(max(r.errors(eta))), **kw, **extra)
    assert max(worst) < 1e-9


def growth(cls, mesh, k, kw, dt=0.01, T=1.0):
    """Growth rate per unit time of the zero-data march from a random displacement (velocity)."""
    import math
    import numpy as np
    r = cls(mesh, k, edd.material(), CF((0, 0)), CF((0, 0)) * t, dt, **kw)
    gf = GridFunction(r.fes)
    gf.vec[:] = 0
    v = gf.components[3].vec.FV().NumPy()
    v[:] = np.random.default_rng(0).standard_normal(len(v))
    r.U.data = gf.vec
    norms = []
    for _ in range(round(T / dt)):
        r.step()
        norms.append(np.linalg.norm(r.U.FV().NumPy()))
    n = len(norms)
    return math.log(norms[-1] / norms[n // 2]) / ((n - 1 - n // 2) * dt)


# Kidney h = 0.1, k = 2. Observed rates: formulation of the preprints (elastodynamics.py, no
# prune) 1.62 (displacement), 119.7 (velocity); with the former 2k-bubble stress enrichment
# 270, 80. Stable settings -0.46, -0.00. Smooth data to T = 10 with the stable settings: sigma
# error bounded (disk and kidney, h 0.1, 0.05, k = 1..3).
@pytest.mark.parametrize("form", STABLE)
def test_stable_settings_no_growth(form):
    cls, _, extra = STABLE[form]
    mesh, kw = unfitted("Example 3 (kidney)", 0.1)
    assert growth(cls, mesh, 2, dict(kw, **extra)) < 0.05
    plain_cls = {"displacement": ed.Elastodynamics, "velocity": ed.VelocityStress}[form]
    plain, pkw = unfitted("Example 3 (kidney)", 0.1, prune=False)
    rate = growth(plain_cls, plain, 2, pkw, dt=0.005, T=0.4)
    assert rate < 5 if form == "displacement" else rate > 10
