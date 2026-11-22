import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ngsolve import CF, GridFunction, InnerProduct, Integrate, Mesh, cos, dx, pi, sin, sqrt, x, y
from netgen.geom2d import unit_square

EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import elasticity  # noqa: E402
import stokes_stress  # noqa: E402
from study_io import Record, save  # noqa: E402


def test_cli_saves_and_reloads(tmp_path):
    """elasticity.py through its command line: the saved dof vector, loaded into the space
    rebuilt on the saved mesh, reproduces the saved L2 error of u."""
    out = tmp_path / "el"
    subprocess.run([sys.executable, str(EXAMPLES / "elasticity.py"), "--case", "traction",
                    "--k", "2", "--h", "0.4", "0.3", "--out", str(out)], check=True, capture_output=True)
    d = np.load(f"{out}.npz")
    params = json.loads(str(d["params"]))
    assert params["case"] == "traction" and params["k"] == 2 and list(d["h"]) == [0.4, 0.3]
    assert d["errors"].shape == (2, len(elasticity.COLUMNS))
    u = CF((sin(pi * x) * cos(pi * y), cos(pi * x) * sin(pi * y)))     # the driver's displacement
    for i, name in enumerate(d["meshes"]):
        mesh = Mesh(str(tmp_path / str(name)))
        fes, _ = elasticity.spaces(mesh, 2, "left|bottom")
        gfu = GridFunction(fes)
        assert gfu.vec.size == d["ndof"][i]
        gfu.vec.FV().NumPy()[:] = d[f"dofs_{i}"]
        u_h = gfu.components[3]
        e = sqrt(Integrate(InnerProduct(u_h - u, u_h - u) * dx(bonus_intorder=4), mesh))
        assert abs(e / d["errors"][i, 1] - 1) < 1e-12


def test_saved_table_is_the_returned_one(tmp_path):
    """stokes_stress.study with a Record: the saved errors, EOCs and m_h are the returned ones."""
    record = Record(stokes_stress.COLUMNS)
    meshes = [(h, Mesh(unit_square.GenerateMesh(maxh=h))) for h in (0.4, 0.3)]
    rows = stokes_stress.study(meshes, 1, 1.0, record=record)
    save(str(tmp_path / "st.npz"), record, dict(driver="stokes_stress", k=1))
    d = np.load(tmp_path / "st.npz")
    assert np.array_equal(d["errors"], np.array([e for e, _ in rows]))
    assert np.array_equal(d["m_h"], np.array([m for _, m in rows]))
    eoc = np.log(d["errors"][0] / d["errors"][1]) / np.log(0.4 / 0.3)
    assert np.allclose(d["eoc"][1], eoc, rtol=1e-14) and np.isnan(d["eoc"][0]).all()
    assert all((tmp_path / str(name)).exists() for name in d["meshes"])
