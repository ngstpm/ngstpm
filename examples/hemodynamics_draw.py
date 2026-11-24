"""Draw the three profiles of the hemodynamics benchmark (hemodynamics.py; Meddahi 2025,
Sec. 4.2, Example 2) at the snapshots t = 4, 8, 12 ms: rows flow (2/3) uhat_x on y = 0,
pressure on the bottom elements, d_y on Sigma; one column per snapshot; one curve per gap
delta (total gap 2 delta, the coupling transferred across it); one figure per placement of
the gap (MATLAB hemodynamics.m gap_split):

  symmetric  fluid up to y = 0.5 - delta, wall from 0.5 + delta (the wall loses delta of its
             thickness 0.1) -> hemodynamics_symmetric.png;
  fluid      fluid up to y = 0.5 - 2 delta, wall conforming to Sigma at full thickness
             (solid paths of length 0) -> hemodynamics_fluid.png.

The axes keep the range of the no-gap curves. At h = 0.1, k = 2 the symmetric gap is
unstable from delta = 0.02 on (it leaves the axes by 12 ms; 0.05 diverges within 4 ms;
MATLAB's gap runs grow too, unfitted-hdg JOURNAL 2026-08-05), while the fluid gap stays bounded
to 24 ms up to delta = 0.05 (fluid paths of length 0.1 = h).

Runs the benchmark for each split and gap below (about 10 s each, 0.6 GB), prints the table
of hemodynamics.py for each, writes the two figures and opens them. Needs matplotlib.
"""
import matplotlib.pyplot as plt

import hemodynamics as hm

k, maxh, dt = 2, 0.1, 1e-4
GAPS = {"symmetric": (0.0, 0.005, 0.01, 0.015, 0.02), "fluid": (0.0, 0.005, 0.01, 0.02, 0.05)}
nsteps = hm.snapshot_steps(dt)[-1]                                  # up to 12 ms
labels = {"flow": "flow (2/3) û_x, y = 0", "pressure": "pressure, bottom elements",
          "displacement": "d_y on Σ"}

nogap = None
for split, gaps in GAPS.items():
    results = {}
    for gap in gaps:
        if gap == 0 and nogap is not None:                          # the same run for both splits
            results[gap] = nogap
            continue
        _, snaps, _, _ = hm.benchmark(maxh, k, dt, nsteps, gap=gap, split=split)
        print(f"gap = {gap:g} ({split})")
        hm.print_table(snaps, dt)
        results[gap] = snaps
    nogap = results[0.0]

    colors = plt.cm.viridis_r([0.15 + 0.7 * i / (len(gaps) - 2) for i in range(len(gaps) - 1)])
    style = {gap: dict(color="k", lw=1.8) if gap == 0 else dict(color=c, lw=1.1, ls="--")
             for gap, c in zip(gaps, ["k", *colors])}
    steps = sorted(nogap)
    fig, ax = plt.subplots(3, len(steps), figsize=(13, 9), sharex=True, squeeze=False,
                           constrained_layout=True)
    for i, name in enumerate(hm.PROFILES):
        for j, n in enumerate(steps):
            a = ax[i, j]
            for gap in gaps:
                a.plot(*results[gap][n][name], **style[gap], label="no gap" if gap == 0 else f"δ = {gap:g}")
            lo, hi = nogap[n][name][1].min(), nogap[n][name][1].max()
            a.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))  # the no-gap range
            a.grid(color="#dddddd", lw=0.5)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            if i == 0:
                a.set_title(f"t = {1e3 * n * dt:g} ms")
            if j == 0:
                a.set_ylabel(labels[name])
            if i == 2:
                a.set_xlabel("x")
    ax[0, -1].legend(fontsize=8)
    where = ("fluid to 0.5 - δ, wall from 0.5 + δ" if split == "symmetric"
             else "fluid to 0.5 - 2δ, wall conforming to Σ")
    fig.suptitle(f"Meddahi 2025 Ex. 2, k = {k}, h = {maxh}, dt = {dt:g}, μ_F = {hm.MU_F:g}, S_S = 1; "
                 f"{split} gap: {where}")
    fig.savefig(f"hemodynamics_{split}.png", dpi=150)
plt.show()
