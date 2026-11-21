"""The command line of the convergence drivers (stokes.py, darcy.py, ...).

Each driver has a module-level row(case, k, h, solution) (one mesh, one solve)
returning a dict of floats: the error columns (printed with the EOC between
consecutive rows, log(e0/e1) / log(size(h0)/size(h1))), the printed extras
(N, |rho|, alpha, ...), any other values to store (mesh flags) and, with
solution=True, "solution": the coefficient vector (float64, the components in
the order solve returns them). Every row runs in a new process: NGSolve's
umfpack inverse does not give its memory back.

    python driver.py                                   all default cases and k
    python driver.py --case NAME... --k K...           a subset
    python driver.py --case NAME --k K --h H... --out FILE.npz [--solution]
                                                       one case on given meshes (COSMOS)

--out writes one .npz per (case, k) (with several, "_<case>_k<k>" is inserted
before ".npz"): float64 arrays h and one per returned value, case, k, and with
--solution solution_<i> per mesh and ndof.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from math import log

import numpy as np


def main(row, cases, columns, extras=None, size=lambda h: h, hname="h", hfmt="5.3f", htype=float, argv=None):
    """cases: case name -> {k: default mesh list}; columns: the error keys of row's dict;
    extras: printed key -> format. Returns {(case, k): [row dicts]}."""
    extras = extras or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs="+", choices=list(cases))
    parser.add_argument("--k", nargs="+", type=int)
    parser.add_argument("--h", nargs="+", type=htype, help="mesh list; needs exactly one --case and one --k")
    parser.add_argument("--out", help="FILE.npz")
    parser.add_argument("--solution", action="store_true", help="also store the coefficient vectors")
    args = parser.parse_args(argv)
    names = args.case or list(cases)
    if args.h is not None and (len(names) != 1 or args.k is None or len(args.k) != 1):
        parser.error("--h needs exactly one --case and one --k")
    several = len(names) * len(args.k or [0, 0]) > 1
    results = {}
    with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
        for case in names:
            print(case)
            for k in args.k or cases[case]:
                print(f"  k = {k}")
                rows = []
                for h in args.h or cases[case][k]:
                    r = pool.submit(row, case, k, h, args.solution).result()
                    prev = rows[-1] if rows else None
                    eoc = [f"({log(prev[c] / r[c]) / log(size(prev['h']) / size(h)):.2f})" if prev else " " * 6
                           for c in columns]
                    print(f"    {hname}={h:{hfmt}}" + "".join(f"  {c}={r[c]:.3e} {e}" for c, e in zip(columns, eoc))
                          + "".join(f"  {c}={r[c]:{f}}" for c, f in extras.items()), flush=True)
                    rows.append(dict(r, h=h))
                if args.out:
                    out = args.out[:-4] + f"_{case}_k{k}.npz" if several else args.out
                    save(out, case, k, rows)
                results[(case, k)] = rows
    return results


def save(path, case, k, rows):
    values = {key: np.array([r[key] for r in rows], dtype=np.float64) for key in rows[0] if key != "solution"}
    vectors = {}
    if "solution" in rows[0]:
        vectors = {f"solution_{i}": np.asarray(r["solution"], dtype=np.float64) for i, r in enumerate(rows)}
        vectors["ndof"] = np.array([len(r["solution"]) for r in rows], dtype=np.int64)
    np.savez(path, case=case, k=k, **values, **vectors)


def coefficients(components):
    """The coefficient vectors of GridFunctions, concatenated (float64)."""
    return np.concatenate([np.array(c.vec, dtype=np.float64) for c in components])
