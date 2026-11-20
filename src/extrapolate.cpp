#include "extrapolate.hpp"
using namespace ngcomp;

namespace ngstpm
{
  shared_ptr<FESpace> ComponentSpace (const ProxyFunction & proxy)
  {
    auto fes = proxy.GetFESpace();
    auto diffop = proxy.Evaluator();
    while (auto comp = dynamic_pointer_cast<CompoundDifferentialOperator> (diffop))
      {
        auto compound = dynamic_pointer_cast<CompoundFESpace> (fes);
        if (!compound) break;
        fes = (*compound)[comp->Component()];
        diffop = comp->BaseDiffOp();
      }
    return fes;
  }

  // the elements on which fes has all its dofs (Compress marks the others -1)
  static shared_ptr<BitArray> OwnElements (const FESpace & fes)
  {
    auto ma = fes.GetMeshAccess();
    auto own = make_shared<BitArray> (ma->GetNE(VOL));
    own->Clear();
    Array<DofId> dnums;
    for (size_t i = 0; i < ma->GetNE(VOL); i++)
      {
        fes.GetDofNrs (ElementId(VOL, i), dnums);
        bool all = dnums.Size() > 0;
        for (auto d : dnums) all = all && d >= 0;
        if (all) own->SetBit (i);
      }
    return own;
  }

  // reference coordinates of the point p in the element of mip (outside it too), by Newton
  // from mip; keeps the facet number of mip (facet spaces). Affine elements: one step.
  template <int D>
  static IntegrationPoint MoveTo (const BaseMappedIntegrationPoint & bmip, const Vec<D> & p)
  {
    const ElementTransformation & trafo = bmip.GetTransformation();
    IntegrationPoint ip(bmip.IP());
    MappedIntegrationPoint<D,D> mip(ip, trafo);
    Vec<D> d = p - mip.GetPoint();
    Vec<D> dxi = mip.GetJacobianInverse() * d;
    for (int i = 0; i < D; i++) ip(i) += dxi(i);
    for (int it = 0; it < 20; it++)
      {
        MappedIntegrationPoint<D,D> mip2(ip, trafo);
        Vec<D> res = p - mip2.GetPoint();
        if (L2Norm(res) < 1e-13 * (1 + L2Norm(d))) break;
        dxi = mip2.GetJacobianInverse() * res;
        for (int i = 0; i < D; i++) ip(i) += dxi(i);
      }
    return ip;
  }

  // -------------------------------------------------------------------------

  ExtrapolateDiffOp::ExtrapolateDiffOp (shared_ptr<DifferentialOperator> abase, const ProxyFunction & proxy,
                                        shared_ptr<CoefficientFunction> atarget, shared_ptr<BitArray> aside,
                                        int aside_nr)
    : DifferentialOperator(abase->Dim(), abase->BlockDim(), abase->VB(), abase->DiffOrder()),
      base(abase), target(atarget), ma(proxy.GetFESpace()->GetMeshAccess()),
      own(OwnElements (*ComponentSpace (proxy))), side(aside), side_nr(aside_nr)
  {
    SetDimensions (abase->Dimensions());
  }

  bool ExtrapolateDiffOp::OnOwnElement (const BaseMappedIntegrationPoint & mip) const
  {
    if (side_nr)               // side=1/2: the measure's side decides
      {
        if (!interface_binding)
          throw Exception ("Extrapolate(side=1|2): only under an interface measure I.ds; "
                           "elsewhere pass the TransferPath of the side");
        return interface_binding->MeasureSide() == side_nr;
      }
    if (interface_binding && &interface_binding->IntegrationMesh() != ma.get()) return false;
    int el = mip.GetTransformation().GetElementNr();
    return side ? side->Test(el) : own->Test(el);
  }

  template <int D>
  void ExtrapolateDiffOp::T_Calc (const FiniteElement & fel, const BaseMappedIntegrationPoint & bmip,
                                  BareSliceMatrix<double,ColMajor> mat, LocalHeap & lh) const
  {
    const BaseMappedIntegrationPoint * from = &bmip;      // the start of the path
    auto binding = interface_binding;
    if (OnOwnElement (bmip))
      {
        if (binding && !side && !side_nr)
          {
            auto & partner = binding->Partner (bmip, lh);
            if (&binding->PartnerMesh() == ma.get()
                && own->Test (partner.GetTransformation().GetElementNr()))
              throw Exception ("Extrapolate: ambiguous side, the space of this trial function has dofs on "
                               "both the element and its partner; pass side=");
          }
      }
    else
      {
        if (!binding)
          throw Exception ("Extrapolate: the space of this trial function has no dofs on element "
                           + ToString (bmip.GetTransformation().GetElementNr())
                           + "; partner evaluation needs an interface measure");
        from = &binding->Partner (bmip, lh);
      }
    if (!target)
      {
        base->CalcMatrix (fel, *from, mat, lh);
        return;
      }
    Vec<D> p;
    target->Evaluate (bmip, FlatVector<> (D, &p(0)));
    IntegrationPoint ip = MoveTo<D> (*from, p);
    MappedIntegrationPoint<D,D> mip(ip, from->GetTransformation());
    base->CalcMatrix (fel, mip, mat, lh);
  }

  void ExtrapolateDiffOp::CalcMatrix (const FiniteElement & fel, const BaseMappedIntegrationPoint & bmip,
                                      BareSliceMatrix<double,ColMajor> mat, LocalHeap & lh) const
  {
    if (bmip.DimSpace() != bmip.DimElement())
      throw Exception ("Extrapolate: volume elements only (integrate with dx(element_boundary=True))");
    if (bmip.DimSpace() == 2) T_Calc<2> (fel, bmip, mat, lh);
    else if (bmip.DimSpace() == 3) T_Calc<3> (fel, bmip, mat, lh);
    else throw Exception ("Extrapolate: 2D/3D only");
  }

  void ExtrapolateDiffOp::CalcMatrix (const FiniteElement & fel, const SIMD_BaseMappedIntegrationRule & mir,
                                      BareSliceMatrix<SIMD<double>> mat) const
  {
    throw ExceptionNOSIMD ("ExtrapolateDiffOp");
  }

  // -------------------------------------------------------------------------

  SideEvalCF::SideEvalCF (shared_ptr<CoefficientFunction> acf, shared_ptr<CoefficientFunction> atarget,
                          shared_ptr<MeshAccess> ama, shared_ptr<BitArray> aelements, bool apath_end)
    : CoefficientFunction(acf->Dimension()), cf(acf), target(atarget), ma(ama), elements(aelements),
      path_end(apath_end)
  {
    if (path_end && (target || acf->Dimension() != ama->GetDimension()))
      throw Exception ("SideEval(path_end=True): the path end, a vector of the mesh dimension, without target");
    SetDimensions (acf->Dimensions());
  }

  bool SideEvalCF::OnOwnElement (const BaseMappedIntegrationPoint & mip) const
  {
    if (!interface_binding) return true;
    return &interface_binding->IntegrationMesh() == ma.get()
      && elements->Test (mip.GetTransformation().GetElementNr());
  }

  double SideEvalCF::Evaluate (const BaseMappedIntegrationPoint & mip) const
  {
    double val;
    Evaluate (mip, FlatVector<> (1, &val));
    return val;
  }

  void SideEvalCF::Evaluate (const BaseMappedIntegrationPoint & mip, FlatVector<> result) const
  {
    static thread_local LocalHeap lh(1000000, "side-eval");
    HeapReset hr(lh);
    int D = mip.DimSpace();
    Vec<3> p = 0.0;
    if (target) target->Evaluate (mip, FlatVector<> (D, &p(0)));      // on K
    const BaseMappedIntegrationPoint * from = &mip;
    if (!OnOwnElement (mip))
      {
        if (path_end) { interface_binding->PathEnd (mip, result); return; }
        from = &interface_binding->Partner (mip, lh);
      }
    auto & trafo = from->GetTransformation();
    auto ud = trafo.PushUserData();       // cf sees a plain point, no integrator caches
    if (!target)
      cf->Evaluate (*from, result);
    else if (D == 2)
      {
        MappedIntegrationPoint<2,2> mip2(MoveTo<2> (*from, Vec<2>(p(0), p(1))), trafo);
        cf->Evaluate (mip2, result);
      }
    else
      {
        MappedIntegrationPoint<3,3> mip2(MoveTo<3> (*from, Vec<3>(p)), trafo);
        cf->Evaluate (mip2, result);
      }
  }

  // -------------------------------------------------------------------------

  void ExportExtrapolate (py::module & m)
  {
    m.def ("Extrapolate", [] (shared_ptr<CoefficientFunction> cf, shared_ptr<CoefficientFunction> target,
                              shared_ptr<BitArray> side, int side_nr) -> shared_ptr<CoefficientFunction>
    {
      auto proxy = dynamic_pointer_cast<ProxyFunction> (cf);
      if (!proxy) throw Exception ("Extrapolate: trial function expected");
      if (proxy->IsTestFunction()) throw Exception ("Extrapolate: test functions are never extrapolated");
      if (proxy->IsOther()) throw Exception ("Extrapolate: .Other() is not supported");
      if (dynamic_pointer_cast<ExtrapolateDiffOp> (proxy->Evaluator()))
        throw Exception ("Extrapolate: the trial function is already extrapolated");
      int D = proxy->GetFESpace()->GetMeshAccess()->GetDimension();
      if (target && target->Dimension() != D)
        throw Exception ("Extrapolate: the target must be a vector of the mesh dimension");
      return make_shared<ProxyFunction> (proxy->GetFESpace(), false, proxy->IsComplex(),
                                         make_shared<ExtrapolateDiffOp> (proxy->Evaluator(), *proxy, target, side, side_nr),
                                         nullptr, nullptr, nullptr, nullptr, nullptr);
    }, py::arg("proxy"), py::arg("target") = nullptr, py::arg("side") = nullptr, py::arg("side_nr") = 0,
    "Trial function evaluated at target(x) (default: the start of the path), on the own element if\n"
    "it is in side (BitArray; default: if the space has all its dofs there), else on the partner\n"
    "element of an interface measure.");

    m.def ("SideEval", [] (shared_ptr<CoefficientFunction> cf, shared_ptr<CoefficientFunction> target,
                           shared_ptr<MeshAccess> mesh, shared_ptr<BitArray> elements, bool path_end)
           -> shared_ptr<CoefficientFunction>
    {
      if (target && target->Dimension() != mesh->GetDimension())
        throw Exception ("SideEval: the target must be a vector of the mesh dimension");
      return make_shared<SideEvalCF> (cf, target, mesh, elements, path_end);
    }, py::arg("cf"), py::arg("target"), py::arg("mesh"), py::arg("elements"), py::arg("path_end") = false,
    "cf on the element of one side (mesh, elements) at target(x) (None: the start of the path).");

    m.def ("Leaves", [] (shared_ptr<CoefficientFunction> cf)
    {
      py::list trial, test, gfs;
      bool coefficients = false;
      Array<CoefficientFunction*> seen;
      cf->TraverseTree ([&] (CoefficientFunction & node)
      {
        if (seen.Contains (&node)) return;
        seen.Append (&node);
        auto sp = node.shared_from_this();
        if (auto proxy = dynamic_cast<ProxyFunction*> (&node))
          (proxy->IsTestFunction() ? test : trial).append (sp);
        else if (dynamic_cast<GridFunctionCoefficientFunction*> (&node))
          { gfs.append (sp); coefficients = true; }
        else if (node.InputCoefficientFunctions().Size() == 0
                 && !dynamic_cast<ConstantCoefficientFunction*> (&node) && !node.IsZeroCF())
          coefficients = true;
      });
      return py::make_tuple (trial, test, gfs, coefficients);
    }, py::arg("cf"),
    "(trial proxies, test proxies, GridFunctions, has_coefficients) of cf; has_coefficients: cf has\n"
    "a leaf other than proxies and constants (coordinates, GridFunctions, other CFs).");

    m.def ("ComponentSpace", [] (shared_ptr<CoefficientFunction> cf)
    {
      auto proxy = dynamic_pointer_cast<ProxyFunction> (cf);
      if (!proxy) throw Exception ("ComponentSpace: trial or test function expected");
      return ComponentSpace (*proxy);
    }, py::arg("proxy"), "the space of a trial/test function (its component of a compound space)");

    m.def ("OwnElements", [] (shared_ptr<CoefficientFunction> cf)
    {
      auto proxy = dynamic_pointer_cast<ProxyFunction> (cf);
      if (!proxy) throw Exception ("OwnElements: trial or test function expected");
      return OwnElements (*ComponentSpace (*proxy));
    }, py::arg("proxy"), "BitArray: the elements on which the space of the trial/test function has all its dofs");
  }
}
