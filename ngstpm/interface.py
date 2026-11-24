"""pair(p1, p2): the Interface between two sides' transfer paths (API-PROPOSAL D6)."""

from typing import NamedTuple

import numpy as np
from ngsolve import CF

from ._ngstpm import Interface

# Relative to the mean facet length of Gamma_h^1 (3D: the larger of the two sides' mean
# edge lengths of Gamma_h; see the Interface docstring):
MERGE_TOL = 1e-10
GAP_TOL = 5e-2


def pair(p1, p2):
    """The Interface of the paths p1 (side 1) and p2 (side 2), each with the
    attributes mesh, elements (D_h^i), facets (Gamma_h^i), target (xbar_i) and
    sigma. Builds the common subdivision of Gamma (2D): segments, each reached
    from one facet of Gamma_h^1 and one of Gamma_h^2. Raises on a piece of
    Gamma_h^i with no partner, and on D_h^1, D_h^2 sharing an element of one mesh.

    D14 absorption, up to GAP_TOL * h (h: mean facet length of Gamma_h^1): gaps
    between the images of a vertex (element-wise paths), and quadrature points
    whose xbar no path of the partner piece reaches exactly (discontinuous paths,
    e.g. a coarse point cloud; the partner is then the point whose path line
    misses xbar by the least). Anything larger raises as an unpaired piece.

    And in 3D (tetrahedra, straight triangular faces): a segment is a pair of
    faces (F, F'); each side's pieces are polygons on its faces, clipped in the
    chart of the face (the plane normal to the mean path direction at its
    vertices) against the other side's image triangles T_F' (on the path ends
    of the vertices of F'; the sides lie on either side of Gamma: only those
    oriented opposite to the face's), and they tile Gamma_h^i face by face
    (else raises: unpaired piece, or overlap); the path of a partner x' must
    end within GAP_TOL * h of xbar. h is the larger of the two sides' mean
    edge lengths of Gamma_h (2D: of Gamma_h^1 only); pieces below
    MERGE_TOL * h^2 are dropped; GAP_TOL * h bounds how far x' lies outside
    F' and is the depth of the strips that pair the rims of open surfaces.

    The conditions in 3D, for each side: (1) Gamma_h^i is a surface (at most
    two faces at an edge), closed or with rims whose path ends match the
    other side's within GAP_TOL * h; (2) the path map x -> xbar_i is
    injective on Gamma_h^i and does not fold: no face whose outward normal
    is on the wrong side of sigma_i, and every T_F oriented as F along the
    paths, not parallel to them; (3) on each face the path map is affine up
    to GAP_TOL * h, measured normal to the paths: |(I - sigma sigma^T)
    (xbar_i(x) - T_F(x))| <= GAP_TOL * h, T_F(x) the affine interpolant of the
    vertices' path ends (the pieces are computed from the T_F). Sides fitted
    to Gamma or interpolating it (paths of length O(h^2)) meet (3) with
    1e-9 h. Carved sides of one background mesh (paths of length O(h)) do
    not: on the unit sphere 0.04 - 0.2 h at h = 0.19 - 0.48, and their
    Gamma_h may fold or not be a surface; pair raises on them (a point
    absorbed beyond GAP_TOL * h, an overlap near a fold, or a degenerate
    chart)."""
    return Interface(p1.mesh, p1.elements, p1.facets, CF(p1.target), CF(p1.sigma),
                     p2.mesh, p2.elements, p2.facets, CF(p2.target), CF(p2.sigma),
                     merge_tol=MERGE_TOL, gap_tol=GAP_TOL)


class InterfacePoints(NamedTuple):
    """I.points(side, intorder): one row per quadrature point x of Gamma_h^side
    (n points, d = mesh dimension). Coordinates of x are in the mesh of side,
    those of xp in the mesh of the other side."""
    x: np.ndarray        # (n, d) the point of Gamma_h^side
    weight: np.ndarray   # (n,)   quadrature weight (sum over side = length (3D: area) of Gamma_h^side)
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


class InterfacePieces(NamedTuple):
    """I.segments() in 3D: n segments, each a pair of faces (F of Gamma_h^1, F' of
    Gamma_h^2), axis 1 the side; the pieces are unions of triangles (m in all, on
    F of side tri_side of segment tri_seg), in the face's reference coordinates
    (xi, eta): x = p0 + xi (p1 - p0) + eta (p2 - p0), p0, p1, p2 the face's
    vertices in mesh order. A piece may be empty on one side (area 0)."""
    facet: np.ndarray    # (n, 2)          face F of Gamma_h^i
    el: np.ndarray       # (n, 2)          element of D_h^i owning F
    loc: np.ndarray      # (n, 2)          local facet number of F in el
    area: np.ndarray     # (n, 2)          physical area of the piece on F (0: empty)
    tri_seg: np.ndarray  # (m,)            segment of the triangle
    tri_side: np.ndarray # (m,)            side of the triangle (0: side 1, 1: side 2)
    tri_ref: np.ndarray  # (m, 3, 2)       its vertices in F's reference coordinates
    tri_x: np.ndarray    # (m, 3, 3)       its vertices in space


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
    """The common subdivision of Gamma (D6): an InterfaceSegments of numpy arrays
    (3D: an InterfacePieces)."""
    S = self._segments()
    return InterfacePieces(**S) if "tri_seg" in S else InterfaceSegments(**S)


Interface.points = _points
Interface.segments = _segments
