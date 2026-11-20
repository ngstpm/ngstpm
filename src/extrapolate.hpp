#ifndef NGSTPM_EXTRAPOLATE_HPP
#define NGSTPM_EXTRAPOLATE_HPP

#include <comp.hpp>
#include <python_ngstd.hpp>
#include "binding.hpp"

namespace ngstpm
{
  // A trial function evaluated at target(x), by polynomial extrapolation outside the element.
  // The element is the own (integration) element K if K is in side (given) or the proxy's
  // component space has all its dofs on K (inferred); otherwise the partner K' of the
  // interface binding. Without a binding, a K that is not own is an error (no partner).
  class ExtrapolateDiffOp : public ngfem::DifferentialOperator
  {
    std::shared_ptr<ngfem::DifferentialOperator> base;
    std::shared_ptr<ngfem::CoefficientFunction> target;   // nullptr: the start of the path (x on K, x' on K')
    std::shared_ptr<ngcomp::MeshAccess> ma;                // mesh of the proxy's space
    std::shared_ptr<ngcore::BitArray> own;                 // elements with all dofs of the component space
    std::shared_ptr<ngcore::BitArray> side;                // nullptr: infer from own
    int side_nr = 0;                                       // 1, 2: the side of an interface measure (I.ds)
    template <int D>
    void T_Calc (const ngfem::FiniteElement & fel, const ngfem::BaseMappedIntegrationPoint & mip,
                 ngbla::BareSliceMatrix<double,ngbla::ColMajor> mat, ngcore::LocalHeap & lh) const;
  public:
    ExtrapolateDiffOp (std::shared_ptr<ngfem::DifferentialOperator> abase, const ngfem::ProxyFunction & proxy,
                       std::shared_ptr<ngfem::CoefficientFunction> atarget,
                       std::shared_ptr<ngcore::BitArray> aside, int aside_nr = 0);

    const std::shared_ptr<ngfem::DifferentialOperator> & Base () const { return base; }
    const std::shared_ptr<ngfem::CoefficientFunction> & Target () const { return target; }
    const std::shared_ptr<ngcore::BitArray> & Side () const { return side; }     // nullptr: inferred
    int SideNr () const { return side_nr; }                                       // 0: not given as 1/2
    // true: evaluate on the element of mip (pass its fel); false: on the partner K' (pass K''s fel)
    bool OnOwnElement (const ngfem::BaseMappedIntegrationPoint & mip) const;

    std::string Name () const override { return "extrapolate " + base->Name(); }
    bool SupportsVB (ngfem::VorB vb) const override { return base->SupportsVB(vb); }
    ngcore::IntRange UsedDofs (const ngfem::FiniteElement & fel) const override { return base->UsedDofs(fel); }
    void CalcMatrix (const ngfem::FiniteElement & fel, const ngfem::BaseMappedIntegrationPoint & mip,
                     ngbla::BareSliceMatrix<double,ngbla::ColMajor> mat, ngcore::LocalHeap & lh) const override;
    void CalcMatrix (const ngfem::FiniteElement & fel, const ngfem::SIMD_BaseMappedIntegrationRule & mir,
                     ngbla::BareSliceMatrix<ngcore::SIMD<double>> mat) const override;
  };

  // cf evaluated on an element of one side (mesh, elements) at target(x) (nullptr: at the start
  // of the path). Own element K unless an interface binding is set and K is not on this side;
  // then the partner K'. Opaque to tree traversal: Replace does not descend into cf or target.
  class SideEvalCF : public ngfem::CoefficientFunction
  {
    std::shared_ptr<ngfem::CoefficientFunction> cf, target;
    std::shared_ptr<ngcomp::MeshAccess> ma;
    std::shared_ptr<ngcore::BitArray> elements;
    bool path_end = false;     // cf is the end of this side's paths: on the partner, the binding's PathEnd
  public:
    SideEvalCF (std::shared_ptr<ngfem::CoefficientFunction> acf, std::shared_ptr<ngfem::CoefficientFunction> atarget,
                std::shared_ptr<ngcomp::MeshAccess> ama, std::shared_ptr<ngcore::BitArray> aelements,
                bool apath_end = false);
    bool OnOwnElement (const ngfem::BaseMappedIntegrationPoint & mip) const;
    using ngfem::CoefficientFunction::Evaluate;
    double Evaluate (const ngfem::BaseMappedIntegrationPoint & mip) const override;
    void Evaluate (const ngfem::BaseMappedIntegrationPoint & mip, ngbla::FlatVector<> result) const override;
  };

  // the component space of a trial/test function of a (nested) compound space
  std::shared_ptr<ngcomp::FESpace> ComponentSpace (const ngfem::ProxyFunction & proxy);

  void ExportExtrapolate (pybind11::module & m);
}

#endif
