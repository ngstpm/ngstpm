"""Gamma given explicitly instead of by a level set: a parametrized curve, a
B-spline or a cloud of points. Each has

    closest(X) -> (P, N)

for an (n, d) numpy array X: the closest points P of Gamma and the unit
normals N there, pointing out of Omega_1 (the inside); `-geometry` is the
same Gamma with the two sides swapped, as `-phi` for a level set. They are
accepted wherever a level set phi is: unfitted_domain, snapped_domains,
transfer_paths."""

import copy

import numpy as np


def _nearest(S):
    """Function X -> index of the nearest row of S (KD-tree when scipy is
    installed, otherwise brute force)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        def query(X, chunk=max(1, 2_000_000 // len(S))):
            return np.concatenate([
                np.argmin(((X[i:i + chunk, None, :] - S[None]) ** 2).sum(axis=2), axis=1)
                for i in range(0, len(X), chunk)])
        return query
    tree = cKDTree(S)
    return lambda X: tree.query(X)[1]


def _signed_area(P):
    return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])


class _Geometry:
    _sign = 1.0
    kinds = ("closest",)                    # transfer path kinds supported (transfer_paths)

    def __neg__(self):
        other = copy.copy(self)
        other._sign = -self._sign
        return other

    def signed_distance(self, X):
        """(X - P) . N: negative in Omega_1."""
        X = np.asarray(X, dtype=float)
        P, N = self.closest(X)
        return np.sum((X - P) * N, axis=1)


class ParametricCurve(_Geometry):
    """Closed plane curve t -> gamma(t), t in [t0, t1), of either orientation;
    Omega_1 is the region it encloses.

    gamma, dgamma, d2gamma: numpy-vectorised callables t -> (x(t), y(t)).
    The derivatives default to central differences; they only steer the
    Newton iteration and the normal, the closest point itself is gamma(t)
    and lies on the curve.

    closest(X) minimises |gamma(t) - X| by Newton on (gamma - X) . gamma' = 0
    from the nearest of nsample samples."""

    dim = 2

    def __init__(self, gamma, t0=0.0, t1=2 * np.pi, dgamma=None, d2gamma=None,
                 nsample=2000, newton_steps=30):
        self.t0, self.T = t0, t1 - t0
        self._gamma = gamma
        e = 1e-6 * self.T
        self._dgamma = dgamma or (lambda t: tuple(
            (a - b) / (2 * e) for a, b in zip(gamma(t + e), gamma(t - e))))
        e2 = 1e-5 * self.T
        self._d2gamma = d2gamma or (lambda t: tuple(
            (a - b) / (2 * e2) for a, b in zip(self._dgamma(t + e2), self._dgamma(t - e2))))
        self.newton_steps = newton_steps
        self._ts = t0 + self.T * np.arange(nsample) / nsample
        S = self.point(self._ts)
        self._query = _nearest(S)
        self._orient = 1.0 if _signed_area(S) > 0 else -1.0

    def _eval(self, f, t):
        return np.stack(f(self.t0 + np.mod(t - self.t0, self.T)), axis=-1)

    def point(self, t):
        return self._eval(self._gamma, np.asarray(t, dtype=float))

    def normal(self, t):
        dg = self._eval(self._dgamma, np.asarray(t, dtype=float))
        N = self._sign * self._orient * np.stack((dg[:, 1], -dg[:, 0]), axis=-1)
        return N / np.linalg.norm(N, axis=1, keepdims=True)

    def closest_parameter(self, X):
        X = np.asarray(X, dtype=float)
        t = self._ts[self._query(X)]
        dt = self.T / len(self._ts)
        for _ in range(self.newton_steps):
            r = self._eval(self._gamma, t) - X
            dg = self._eval(self._dgamma, t)
            f = np.sum(r * dg, axis=1)
            gn = np.sum(dg * dg, axis=1)
            df = gn + np.sum(r * self._eval(self._d2gamma, t), axis=1)
            df = np.where(df > 0.1 * gn, df, gn)        # Gauss-Newton fallback
            step = np.clip(f / df, -dt, dt)
            t = t - step
            if np.max(np.abs(step)) < 1e-15 * self.T:
                break
        return t

    def closest(self, X):
        t = self.closest_parameter(X)
        return self.point(t), self.normal(t)


def _cardinal(p, s, deriv=0):
    """Cardinal B-spline of degree p (support [0, p+1]) or a derivative."""
    if deriv:
        return _cardinal(p - 1, s, deriv - 1) - _cardinal(p - 1, s - 1, deriv - 1)
    if p == 0:
        return ((s >= 0) & (s < 1)).astype(float)
    return (s * _cardinal(p - 1, s) + (p + 1 - s) * _cardinal(p - 1, s - 1)) / p


class BSplineCurve(ParametricCurve):
    """Closed uniform B-spline curve of the given degree with control points
    ctrl ((n, 2) array, not repeated at the end); t in [0, n). Exact
    derivatives."""

    def __init__(self, ctrl, degree=3, **kw):
        self.ctrl, self.degree = np.asarray(ctrl, dtype=float), degree
        n = len(self.ctrl)
        if n <= degree:
            raise ValueError("need more control points than the degree")
        kw.setdefault("nsample", max(2000, 20 * n))
        super().__init__(lambda t: self._spline(t, 0), 0.0, float(n),
                         dgamma=lambda t: self._spline(t, 1),
                         d2gamma=lambda t: self._spline(t, 2), **kw)

    def _spline(self, t, deriv):
        i = np.floor(t).astype(int)
        out = sum(_cardinal(self.degree, t - i + m, deriv)[:, None]
                  * self.ctrl[(i - m) % len(self.ctrl)] for m in range(self.degree + 1))
        return out.T


class PointCloud(_Geometry):
    """Gamma known through a cloud of points ((n, d) array) only, 2D or 3D.

    closest(X) returns the nearest cloud point and the normal attached to
    it. normals: (n, d) normals pointing out of Omega_1; if omitted (2D
    only) the points are taken as ordered along the closed curve and the
    normals come from central differences of the polygon."""

    def __init__(self, points, normals=None):
        self.points = np.asarray(points, dtype=float)
        self.dim = self.points.shape[1]
        if normals is None:
            if self.dim != 2:
                raise ValueError("an unordered or 3D point cloud needs normals")
            tau = np.roll(self.points, -1, axis=0) - np.roll(self.points, 1, axis=0)
            orient = 1.0 if _signed_area(self.points) > 0 else -1.0
            normals = orient * np.stack((tau[:, 1], -tau[:, 0]), axis=-1)
        normals = np.asarray(normals, dtype=float)
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self._query = _nearest(self.points)

    def closest(self, X):
        idx = self._query(np.asarray(X, dtype=float))
        return self.points[idx], self._sign * self.normals[idx]
