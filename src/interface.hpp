#ifndef NGSTPM_INTERFACE_HPP
#define NGSTPM_INTERFACE_HPP

// Interface: the pairing of two sides' transfer paths with the common
// subdivision of Gamma (API-PROPOSAL D6). Each side has its own mesh, element
// set D_h^i, facets of Gamma_h^i, path end xbar_i and direction sigma_i;
// nothing is shared. A segment is an arc of Gamma reached from exactly one
// facet F of Gamma_h^1 and one facet F' of Gamma_h^2; its pieces are the
// parts of F and F' whose paths end on that arc. In 3D (D10) a segment is a
// pair (F, F') of faces whose paths' ends overlap on Gamma; its pieces are
// polygons, computed per side in local charts from flat images of the path
// map on m_F^2 subtriangles of each face F (src/interface3d.cpp).

#include <comp.hpp>
#include <python_ngstd.hpp>

struct IfcSide            // one side, as given by its TransferPath
{
  std::shared_ptr<ngcomp::MeshAccess> ma;
  std::shared_ptr<ngcore::BitArray> elements, facets;          // D_h^i (elements), Gamma_h^i (facets)
  std::shared_ptr<ngfem::CoefficientFunction> target, sigma;   // xbar_i, sigma_i
};

struct IfcFacet { int facet, el, loc; ngbla::Vec<3> p[3]; double len; };   // facet nr, owner K in D_h^i, local nr; vertices
// (2D: p[2] = 0; 3D: p[0..2] the face's vertices in mesh order, len its area |F|)

// Piece of the facet F = Facet(side, f): the points p[0] + t (p[1] - p[0]), t in [t[0], t[1]].
// In 3D a piece is a union of triangles tri in F's reference coordinates (xi, eta),
// x = p[0] + xi (p[1] - p[0]) + eta (p[2] - p[0]), of physical area `area`; it may be empty.
struct IfcPiece { int f; double t[2]; std::vector<std::array<ngbla::Vec<2>,3>> tri; double area = 0; };

struct IfcSegment { IfcPiece piece[2]; ngbla::Vec<3> xbar[2][2]; };   // per side; xbar[side][end] at t[end]

struct IfcPoint           // a quadrature point of the piece of side i of segment seg
{
  ngfem::IntegrationPoint ip, pip;   // reference points on K and on the partner K' (facet nr set)
  double weight;                     // physical: Gauss weight times the length (3D: area) of the piece
  int el, loc, pel, ploc, seg;       // K and its local facet; K' and the local facet of x'
  ngbla::Vec<3> x, xbar, xp, sigmap; // x on F, xbar = target_i(x), partner x' on F', unit sigma_j(x')
};

class Interface
{
  IfcSide sides[2];
  ngcore::Array<IfcFacet> facets[2];
  ngcore::Array<IfcSegment> segments;
  ngcore::Array<std::array<ngbla::Vec<3>,3>> images[2];   // 3D: the image triangle T_F of each face
  ngcore::Array<int> nsub[2];                              // 3D: each face split into nsub^2 subtriangles
  double merge_tol, gap_tol, hmean;   // hmean: mean facet length of Gamma_h^1 (3D: see Build3D)
  int dim = 2;
  void Build ();
  void Build3D ();
  void Rule3D (int side, size_t seg, int intorder, ngcore::Array<IfcPoint> & pts, ngcore::LocalHeap & lh) const;
public:
  // tolerances relative to the mean facet length of Gamma_h^1: breakpoints of the two
  // sides closer than merge_tol are merged; a vertex image of side 2 falling up to gap_tol
  // outside the images of Gamma_h^1 (the gaps of element-wise paths) is moved onto them.
  // D14 absorption: a rule point whose xbar no path of its partner piece reaches exactly
  // gets the partner whose path line misses xbar by the least, if by at most gap_tol;
  // beyond that (and for any other unpaired piece) the construction raises.
  // 3D (Build3D): relative to h = the larger of the two sides' mean edge lengths of Gamma_h;
  // pieces of area below merge_tol h^2 are dropped (also image triangles that small raise);
  // gap_tol h is the depth of the strips beyond the rims of open surfaces, how far a
  // partner x' may lie outside its face F' (the D14 absorption as in 2D) and how far the
  // path of x' may end from xbar; the path map is resolved on subtriangles of the faces to
  // 0.2 gap_tol h across the paths (at most 16^2 per face, else raises).
  Interface (IfcSide s1, IfcSide s2, double amerge, double agap);
  int Dim () const { return dim; }
  double MergeTol () const { return merge_tol; }
  double GapTol () const { return gap_tol; }
  const std::shared_ptr<ngcomp::MeshAccess> & Mesh (int side) const { return sides[side].ma; }   // side 0, 1
  const IfcSide & Side (int side) const { return sides[side]; }
  const IfcFacet & Facet (int side, int f) const { return facets[side][f]; }
  int NSub (int side, int f) const { return nsub[side][f]; }                // 3D
  size_t NSegments () const { return segments.Size(); }
  const IfcSegment & Segment (size_t s) const { return segments[s]; }
  // 3D: a segment's piece of one side may be empty (a sliver kept on the other side only)
  bool Empty (int side, size_t s) const { return dim == 3 && segments[s].piece[side].tri.empty(); }
  // Gauss rule of order intorder on the piece of side `side` of segment seg, with the partners
  void Rule (int side, size_t seg, int intorder, ngcore::Array<IfcPoint> & pts, ngcore::LocalHeap & lh) const;
};

namespace ngstpm
{
  // reference coordinates of the point p in the element of trafo (Newton; one step if affine)
  ngfem::IntegrationPoint Locate (const ngfem::ElementTransformation & trafo, const ngbla::Vec<3> & p, int D);
}

// I._segments() in 3D (src/interface3d.cpp)
pybind11::dict InterfacePieces3D (const Interface & I);

void ExportInterface (pybind11::module & m);

#endif
