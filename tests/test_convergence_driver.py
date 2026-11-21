import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from convergence import main  # noqa: E402

CASES = {"toy": {1: (0.4, 0.2, 0.1), 2: (0.4, 0.2)}, "other": {1: (0.5, 0.25)}}


def toy_row(case, k, h, solution=False):
    """e = h^(k+1) exactly: the EOC is k+1."""
    r = {"e": h ** (k + 1), "N": 1 / h ** 2, "flag": 1.0}
    if solution:
        r["solution"] = np.arange(3.0) * h
    return r


def test_eoc_and_table(capsys):
    results = main(toy_row, CASES, ("e",), {"N": "6.0f"}, argv=["--case", "toy", "--k", "1", "2"])
    out = capsys.readouterr().out
    assert out.count("(2.00)") == 2 and out.count("(3.00)") == 1
    assert [r["h"] for r in results[("toy", 1)]] == [0.4, 0.2, 0.1]


def test_npz(tmp_path):
    out = tmp_path / "toy.npz"
    main(toy_row, CASES, ("e",), argv=["--case", "toy", "--k", "1", "--h", "0.3", "0.15", "--out", str(out), "--solution"])
    z = np.load(out)
    assert {"h", "e", "N", "flag", "case", "k", "solution_0", "solution_1", "ndof"} <= set(z.files)
    assert z["e"].dtype == np.float64 and z["h"].dtype == np.float64
    assert np.array_equal(z["e"], np.array([0.3 ** 2, 0.15 ** 2])) and str(z["case"]) == "toy" and int(z["k"]) == 1
    assert list(z["ndof"]) == [3, 3] and np.array_equal(z["solution_1"], np.arange(3.0) * 0.15)


def test_several_files(tmp_path):
    main(toy_row, CASES, ("e",), argv=["--k", "1", "--out", str(tmp_path / "t.npz")])
    assert {p.name for p in tmp_path.iterdir()} == {"t_toy_k1.npz", "t_other_k1.npz"}


@pytest.mark.parametrize("argv", [["--h", "0.1"], ["--case", "toy", "--h", "0.1"],
                                  ["--case", "toy", "other", "--k", "1", "--h", "0.1"]])
def test_h_needs_one_case_and_one_k(argv):
    with pytest.raises(SystemExit):
        main(toy_row, CASES, ("e",), argv=argv)


def test_darcy_driver(tmp_path):
    """A real driver through the same command line: two coarse meshes, k = 1."""
    import darcy
    out = tmp_path / "darcy.npz"
    results = main(darcy.row, darcy.CASES, ("|u-uh|", "|p-ph|"),
                   argv=["--case", "neumann-identity", "--k", "1", "--h", "0.1", "0.05", "--out", str(out)])
    z = np.load(out)
    rows = results[("neumann-identity", 1)]
    assert np.array_equal(z["|u-uh|"], [r["|u-uh|"] for r in rows])
    assert np.isclose(z["|u-uh|"][0], 8.238e-03, rtol=1e-3)               # the committed table
