"""Command line and output of the convergence drivers: one case (setting, k, mesh sizes)
per invocation, and with --out PATH.npz the table and the solutions in double precision.

Saved (numpy.savez): h, columns, errors (levels x columns), eoc, ndof, the extra columns
of the study (R, #D_h, m_h, c, ... ; NaN where absent), params (JSON: case, k, materials,
tau, flags, git commit), dofs_<i> (the dof vector of the solution GridFunction of level i)
and meshes (file names): each mesh is saved as PATH_h<h>.vol.gz next to the .npz.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


class Record:
    """The levels of one study."""

    def __init__(self, columns):
        self.columns, self.levels = list(columns), []

    def add(self, h, mesh, errors, gfu, **extra):
        self.levels.append(dict(h=float(h), mesh=mesh, errors=[float(e) for e in errors],
                                dofs=np.array(gfu.vec.FV().NumPy(), dtype=np.float64),
                                extra={k: float(v) for k, v in extra.items() if v is not None}))


def parser(doc, cases):
    """--case (one of cases, default the first), --k, --h, --out; drivers add their own."""
    p = argparse.ArgumentParser(description=doc.strip().splitlines()[0],
                                epilog="Without --k and --h: the default tables, one process per case.")
    p.add_argument("--case", choices=cases, default=cases[0], metavar="CASE",
                   help="one of: " + "; ".join(repr(c) for c in cases) + f" (default {cases[0]!r})")
    p.add_argument("--k", type=int, help="polynomial degree")
    p.add_argument("--h", type=float, nargs="+", help="mesh sizes (maxh), coarse to fine")
    p.add_argument("--out", help="save to OUT.npz (and the meshes to OUT_h<h>.vol.gz)")
    return p


def parse(p):
    a = p.parse_args()
    if (a.k is None) != (a.h is None):
        p.error("--k and --h go together")
    return a


def run_defaults(script, cases):
    """Run each default case (a list of arguments) in its own process."""
    for argv in cases:
        subprocess.run([sys.executable, script, *[str(v) for v in argv]], check=True)


def commit():
    """The short commit of the worktree, with "-dirty" if examples/ differs from it."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "diff", "--quiet", "HEAD", "--", here]).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save(path, record, params):
    stem = path[:-4] if path.endswith(".npz") else path
    levels = record.levels
    h = np.array([lv["h"] for lv in levels])
    E = np.array([lv["errors"] for lv in levels])
    eoc = np.full_like(E, np.nan)
    if len(levels) > 1:
        eoc[1:] = np.log(E[:-1] / E[1:]) / np.log(h[:-1] / h[1:])[:, None]
    data = dict(h=h, columns=np.array(record.columns), errors=E, eoc=eoc,
                ndof=np.array([lv["dofs"].size for lv in levels]),
                params=json.dumps({**params, "commit": commit()}))
    for key in sorted({k for lv in levels for k in lv["extra"]}):
        data[key] = np.array([lv["extra"].get(key, np.nan) for lv in levels])
    meshes = []
    for i, lv in enumerate(levels):
        data[f"dofs_{i}"] = lv["dofs"]
        name = f"{stem}_h{lv['h']:g}.vol.gz"
        lv["mesh"].ngmesh.Save(name)
        meshes.append(os.path.basename(name))
    data["meshes"] = np.array(meshes)
    np.savez(stem + ".npz", **data)
