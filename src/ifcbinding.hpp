#ifndef NGSTPM_IFCBINDING_HPP
#define NGSTPM_IFCBINDING_HPP

// The interface binding of one piece rule: what ExtrapolateDiffOp and SideEvalCF (binding.hpp)
// read while an interface integral is assembled on a segment of the common subdivision.
//
// Public API (for the assemblers of src/ifcmeasure.cpp and src/ifclinear.cpp):
//
//   ngstpm::PieceRule r;
//   ngstpm::MakePieceRule (I, side, seg, intorder, r, lh);
//       side 0 or 1 (Gamma_h^1, Gamma_h^2), seg < I.NSegments(). Fills
//       r.el, r.loc   K (in D_h^side, on I.Mesh(side)) and the local facet of its piece;
//       r.pel, r.ploc K' (in D_h^other, on I.Mesh(1-side)) and the local facet of x';
//       r.pts         the IfcPoints of Interface::Rule (r.pts[q].weight: physical weight);
//       r.mir, r.pmir the points mapped on K and on K' (allocated on lh), point numbers
//                     q = 0..n-1 (ip.Nr()), facet numbers set, outward normals of K / K'
//                     computed (specialcf.normal).
//   {
//     ngstpm::PieceBinding b (I, side, r);       // an InterfaceBinding (binding.hpp)
//     ngstpm::BindingGuard guard (b);            // interface_binding = &b until end of scope
//     ... cf->Evaluate (*r.mir, values) ...      // SideEval nodes, Extrapolate: partner on K'
//   }
//
// PieceBinding (binding.hpp's InterfaceBinding): IntegrationMesh() = I.Mesh(side), PartnerMesh() =
// I.Mesh(1-side), MeasureSide() = side + 1, PathEnd(mip) = r.pts[nr].xbar (the integration point's
// xbar; partner paths end there), Partner(mip) = (*r.pmir)[nr], nr = mip.IP().Nr(); mip must be on
// K or K' with nr in range (a point of r.mir / r.pmir or a copy), else it throws. Not thread-safe across
// threads by sharing: the binding pointer is thread_local, each thread binds its own rule.

#include <comp.hpp>
#include "binding.hpp"
#include "interface.hpp"

namespace ngstpm
{
  struct PieceRule
  {
    ngcomp::ElementId el, pel;
    int loc = -1, ploc = -1;
    ngcore::Array<IfcPoint> pts;
    ngfem::BaseMappedIntegrationRule * mir = nullptr, * pmir = nullptr;
  };

  inline void MakePieceRule (const Interface & I, int side, size_t seg, int intorder, PieceRule & r,
                             ngcore::LocalHeap & lh)
  {
    using namespace ngcomp;
    I.Rule (side, seg, intorder, r.pts, lh);
    size_t n = r.pts.Size();
    if (n == 0) throw Exception ("MakePieceRule: empty rule");
    r.el = ElementId(VOL, r.pts[0].el); r.loc = r.pts[0].loc;
    r.pel = ElementId(VOL, r.pts[0].pel); r.ploc = r.pts[0].ploc;
    auto ir = new (lh) IntegrationRule (n, lh);
    auto pir = new (lh) IntegrationRule (n, lh);
    for (size_t q = 0; q < n; q++)
      {
        (*ir)[q] = r.pts[q].ip; (*ir)[q].SetNr (q); (*ir)[q].SetWeight (r.pts[q].weight);
        (*pir)[q] = r.pts[q].pip; (*pir)[q].SetNr (q); (*pir)[q].SetWeight (r.pts[q].weight);
      }
    auto & ma = *I.Mesh(side), & pma = *I.Mesh(1 - side);
    const ElementTransformation & trafo = ma.GetTrafo (r.el, lh);
    const ElementTransformation & ptrafo = pma.GetTrafo (r.pel, lh);
    r.mir = &trafo (*ir, lh);
    r.mir->ComputeNormalsAndMeasure (trafo.GetElementType(), r.loc);
    r.pmir = &ptrafo (*pir, lh);
    r.pmir->ComputeNormalsAndMeasure (ptrafo.GetElementType(), r.ploc);
  }

  class PieceBinding : public InterfaceBinding
  {
    const ngcomp::MeshAccess & ma, & pma;
    const PieceRule & r;
    int side;
  public:
    PieceBinding (const Interface & I, int aside, const PieceRule & ar)
      : ma(*I.Mesh(aside)), pma(*I.Mesh(1 - aside)), r(ar), side(aside) { }
    const ngcomp::MeshAccess & IntegrationMesh () const override { return ma; }
    const ngcomp::MeshAccess & PartnerMesh () const override { return pma; }
    int MeasureSide () const override { return side + 1; }
    void PathEnd (const ngfem::BaseMappedIntegrationPoint & mip, ngbla::FlatVector<double> xbar) const override
    {
      int nr = mip.IP().Nr();
      if (nr < 0 || nr >= int(r.pts.Size())) throw ngcore::Exception ("interface binding: PathEnd() of a point not on the rule");
      for (size_t i = 0; i < xbar.Size(); i++) xbar(i) = r.pts[nr].xbar(i);
    }
    const ngfem::BaseMappedIntegrationPoint & Partner (const ngfem::BaseMappedIntegrationPoint & mip,
                                                       ngcore::LocalHeap & lh) const override
    {
      auto & trafo = mip.GetTransformation();
      int nr = mip.IP().Nr(), el = trafo.GetElementNr();
      if (!(el == int(r.el.Nr()) || el == int(r.pel.Nr())) || nr < 0 || nr >= int(r.pmir->Size()))
        throw ngcore::Exception ("interface binding: Partner() of a point that is not on the bound piece rule");
      return (*r.pmir)[nr];
    }
  };

  class BindingGuard
  {
    InterfaceBinding * prev;
  public:
    BindingGuard (InterfaceBinding & b) : prev(interface_binding) { interface_binding = &b; }
    ~BindingGuard () { interface_binding = prev; }
    BindingGuard (const BindingGuard &) = delete;
    BindingGuard & operator= (const BindingGuard &) = delete;
  };
}

#endif
