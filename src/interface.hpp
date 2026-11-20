#ifndef NGSTPM_INTERFACE_HPP
#define NGSTPM_INTERFACE_HPP

// Interface: the pairing of two sides' transfer paths with the common
// subdivision of Gamma (API-PROPOSAL D6). Each side has its own mesh, element
// set D_h^i, facets of Gamma_h^i, path end xbar_i and direction sigma_i;
// nothing is shared. A segment is an arc of Gamma reached from exactly one
// facet F of Gamma_h^1 and one facet F' of Gamma_h^2; its pieces are the
// parts of F and F' whose paths end on that arc. 2D only (D10).

#include <comp.hpp>
#include <python_ngstd.hpp>

struct IfcSide            // one side, as given by its TransferPath
{
  std::shared_ptr<ngcomp::MeshAccess> ma;
  std::shared_ptr<ngcore::BitArray> elements, facets;          // D_h^i (elements), Gamma_h^i (facets)
  std::shared_ptr<ngfem::CoefficientFunction> target, sigma;   // xbar_i, sigma_i
};

struct IfcFacet { int facet, el, loc; ngbla::Vec<3> p[2]; double len; };   // facet nr, owner K in D_h^i, local nr; vertices

// Piece of the facet F = Facet(side, f): the points p[0] + t (p[1] - p[0]), t in [t[0], t[1]].
// In 3D a piece is a polygon in F's reference coordinates (clipping in a chart of Gamma).
struct IfcPiece { int f; double t[2]; };

struct IfcSegment { IfcPiece piece[2]; ngbla::Vec<3> xbar[2][2]; };   // per side; xbar[side][end] at t[end]

struct IfcPoint           // a quadrature point of the piece of side i of segment seg
{
  ngfem::IntegrationPoint ip, pip;   // reference points on K and on the partner K' (facet nr set)
  double weight;                     // physical: Gauss weight times the length of the piece
  int el, loc, pel, ploc, seg;       // K and its local facet; K' and the local facet of x'
  ngbla::Vec<3> x, xbar, xp, sigmap; // x on F, xbar = target_i(x), partner x' on F', unit sigma_j(x')
};

class Interface
{
  IfcSide sides[2];
  ngcore::Array<IfcFacet> facets[2];
  ngcore::Array<IfcSegment> segments;
  double merge_tol, gap_tol, hmean;   // hmean: mean facet length of Gamma_h^1
  void Build ();
public:
  // tolerances relative to the mean facet length of Gamma_h^1: breakpoints of the two
  // sides closer than merge_tol are merged; a vertex image of side 2 falling up to gap_tol
  // outside the images of Gamma_h^1 (the gaps of element-wise paths) is moved onto them.
  // D14 absorption: a rule point whose xbar no path of its partner piece reaches exactly
  // gets the partner whose path line misses xbar by the least, if by at most gap_tol;
  // beyond that (and for any other unpaired piece) the construction raises.
  Interface (IfcSide s1, IfcSide s2, double amerge, double agap);
  int Dim () const { return 2; }
  double MergeTol () const { return merge_tol; }
  double GapTol () const { return gap_tol; }
  const std::shared_ptr<ngcomp::MeshAccess> & Mesh (int side) const { return sides[side].ma; }   // side 0, 1
  const IfcSide & Side (int side) const { return sides[side]; }
  const IfcFacet & Facet (int side, int f) const { return facets[side][f]; }
  size_t NSegments () const { return segments.Size(); }
  const IfcSegment & Segment (size_t s) const { return segments[s]; }
  // Gauss rule of order intorder on the piece of side `side` of segment seg, with the partners
  void Rule (int side, size_t seg, int intorder, ngcore::Array<IfcPoint> & pts, ngcore::LocalHeap & lh) const;
};

namespace ngstpm
{
  // reference coordinates of the point p in the element of trafo (Newton; one step if affine)
  ngfem::IntegrationPoint Locate (const ngfem::ElementTransformation & trafo, const ngbla::Vec<3> & p, int D);
}

void ExportInterface (pybind11::module & m);

#endif
