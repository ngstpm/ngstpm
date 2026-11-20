"""pair(p1, p2): the Interface between two sides' transfer paths (API-PROPOSAL D6)."""

from typing import NamedTuple

import numpy as np
from ngsolve import CF

from ._ngstpm import Interface

# Relative to the mean facet length of Gamma_h^1 (see the Interface docstring):
MERGE_TOL = 1e-10
GAP_TOL = 5e-2


def pair(p1, p2):
    """The Interface of the paths p1 (side 1) and p2 (side 2), each with the
    attributes mesh, elements (D_h^i), facets (Gamma_h^i), target (xbar_i) and
    sigma. Builds the common subdivision of Gamma (2D): segments, each reached
    from one facet of Gamma_h^1 and one of Gamma_h^2. Raises on a piece of
    Gamma_h^i with no partner, on D_h^1, D_h^2 sharing an element of one mesh,
    and in 3D.

    D14 absorption, up to GAP_TOL * h (h: mean facet length of Gamma_h^1): gaps
    between the images of a vertex (element-wise paths), and quadrature points
    whose xbar no path of the partner piece reaches exactly (discontinuous paths,
    e.g. a coarse point cloud; the partner is then the point whose path line
    misses xbar by the least). Anything larger raises as an unpaired piece."""
    return Interface(p1.mesh, p1.elements, p1.facets, CF(p1.target), CF(p1.sigma),
                     p2.mesh, p2.elements, p2.facets, CF(p2.target), CF(p2.sigma),
                     merge_tol=MERGE_TOL, gap_tol=GAP_TOL)


class InterfacePoints(NamedTuple):
    """I.points(side, intorder): one row per quadrature point x of Gamma_h^side
    (n points, d = mesh dimension). Coordinates of x are in the mesh of side,
    those of xp in the mesh of the other side."""
    x: np.ndarray        # (n, d) the point of Gamma_h^side
    weight: np.ndarray   # (n,)   quadrature weight (sum over side = length of Gamma_h^side)
    xbar: np.ndarray     # (n, d) end of the path of x, on Gamma
    xp: np.ndarray       # (n, d) partner x' on Gamma_h^other, whose own path ends at xbar
    sigmap: np.ndarray   # (n, d) unit path direction sigma(x') of the partner
    el: np.ndarray       # (n,)   element K of x (in D_h^side)
    loc: np.ndarray      # (n,)   local facet number of x's facet in K
    facet: np.ndarray    # (n,)   facet number of x's facet F
    pel: np.ndarray      # (n,)   partner element K' (in D_h^other)
    ploc: np.ndarray     # (n,)   local facet number of F' in K'
    pfacet: np.ndarray   # (n,)   partner facet number F'
    seg: np.ndarray      # (n,)   segment of I.segments() containing x


class InterfaceSegments(NamedTuple):
    """I.segments(): the common subdivision of Gamma (D6), n segments; axis 1 is
    the side (0: side 1, 1: side 2), axis 2 the two ends of the piece. Segment s
    is the arc of Gamma reached from the piece t[s, i, 0]..t[s, i, 1] of the
    facet F = facet[s, i] (x = p0 + t (p1 - p0), p0, p1 the facet's vertices in
    mesh order) on both sides."""
    facet: np.ndarray    # (n, 2)          facet F of Gamma_h^i
    el: np.ndarray       # (n, 2)          element of D_h^i owning F
    loc: np.ndarray      # (n, 2)          local facet number of F in el
    t: np.ndarray        # (n, 2, 2)       piece parameters on F
    x: np.ndarray        # (n, 2, 2, d)    piece ends on F
    xbar: np.ndarray     # (n, 2, 2, d)    their path ends on Gamma


def _points(self, side, intorder=4):
    """Diagnostics at the points of the order-intorder rule on the pieces of
    Gamma_h^side (side = 1, 2): an InterfacePoints of numpy arrays."""
    R = self._rule(side, intorder)
    S = self._segments()
    i = side - 1
    return InterfacePoints(x=R["x"], weight=R["weight"], xbar=R["xbar"], xp=R["xp"],
                           sigmap=R["sigmap"], el=R["el"], loc=R["loc"],
                           facet=S["facet"][R["seg"], i], pel=R["pel"], ploc=R["ploc"],
                           pfacet=S["facet"][R["seg"], 1 - i], seg=R["seg"])


def _segments(self):
    """The common subdivision of Gamma (D6): an InterfaceSegments of numpy arrays."""
    return InterfaceSegments(**self._segments())


Interface.points = _points
Interface.segments = _segments
