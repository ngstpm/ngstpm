"""Draw the three profiles of the hemodynamics benchmark (hemodynamics.py; Meddahi 2025,
Sec. 4.2, Example 2) at the snapshots t = 4, 8, 12 ms: rows flow (2/3) uhat_x on y = 0,
pressure on the bottom elements, d_y on Sigma; one column per snapshot.

Runs the benchmark with the defaults below (about 10 s, 0.6 GB), prints the table of
hemodynamics.py, writes hemodynamics.png and opens it. Needs matplotlib.
"""
import matplotlib.pyplot as plt

import hemodynamics as hm

k, maxh, dt = 2, 0.1, 1e-4
nsteps = hm.snapshot_steps(dt)[-1]                                  # up to 12 ms

run, snaps, _, _ = hm.benchmark(maxh, k, dt, nsteps)
hm.print_table(snaps, dt)

labels = {"flow": "flow (2/3) û_x, y = 0", "pressure": "pressure, bottom elements",
          "displacement": "d_y on Σ"}
steps = sorted(snaps)
fig, ax = plt.subplots(3, len(steps), figsize=(13, 9), sharex=True, squeeze=False,
                       constrained_layout=True)
for i, name in enumerate(hm.PROFILES):
    for j, n in enumerate(steps):
        a = ax[i, j]
        a.plot(*snaps[n][name], color="#2a78d6", lw=1.5)
        a.grid(color="#dddddd", lw=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        if i == 0:
            a.set_title(f"t = {1e3 * n * dt:g} ms")
        if j == 0:
            a.set_ylabel(labels[name])
        if i == 2:
            a.set_xlabel("x")
fig.suptitle(f"Meddahi 2025 Ex. 2, k = {k}, h = {maxh}, dt = {dt:g}, μ_F = {hm.MU_F:g}, S_S = 1")
fig.savefig("hemodynamics.png", dpi=150)
plt.show()
