#include <comp.hpp>
#include <python_ngstd.hpp>
#include "ifcbinding.hpp"
#include "ifclinear.hpp"
using namespace ngcomp;

namespace ngstpm
{
  namespace
  {
    class InterfaceLFI : public LinearFormIntegrator
    {
      shared_ptr<Interface> I;
      int side, bonus;
      shared_ptr<CoefficientFunction> cf;
      Array<ProxyFunction*> test;
      std::map<int, Array<size_t>> pieces;     // element K -> the segments whose piece K owns

    public:
      InterfaceLFI (shared_ptr<Interface> aI, int aside, shared_ptr<CoefficientFunction> acf, int abonus)
        : I(aI), side(aside), bonus(abonus), cf(acf)
      {
        if (side != 0 && side != 1) throw Exception ("interface linear form: side must be 0 or 1");
        if (cf->Dimension() != 1) throw Exception ("interface linear form: scalar integrand expected");
        cf->TraverseTree ([&] (CoefficientFunction & node)
        {
          auto proxy = dynamic_cast<ProxyFunction*> (&node);
          if (!proxy) return;
          if (!proxy->IsTestFunction()) throw Exception ("interface linear form: trial function in the integrand");
          if (!test.Contains(proxy)) test.Append (proxy);
        });
        if (test.Size() == 0) throw Exception ("interface linear form: no test function in the integrand");
        for (size_t s = 0; s < I->NSegments(); s++)
          pieces[I->Facet(side, I->Segment(s).piece[side].f).el].Append (s);
      }

      VorB VB () const override { return VOL; }
      bool BoundaryForm () const override { return false; }
      string Name () const override { return "InterfaceLFI"; }

      void CalcElementVector (const FiniteElement & fel, const ElementTransformation & eltrans,
                              FlatVector<double> elvec, LocalHeap & lh) const override
      {
        elvec = 0.0;
        auto it = pieces.find (eltrans.GetElementNr());
        if (it == pieces.end()) return;
        int intorder = 2 * fel.Order() + bonus;
        for (size_t s : it->second)
          {
            HeapReset hr(lh);
            PieceRule r;
            MakePieceRule (*I, side, s, intorder, r, lh);
            BaseMappedIntegrationRule & mir = *r.mir;
            size_t np = mir.Size();
            auto & trafo = const_cast<ElementTransformation&> (mir.GetTransformation());
            ProxyUserData ud;
            ud.fel = &fel;
            ud.lh = &lh;
            trafo.userdata = &ud;
            {
              PieceBinding b(*I, side, r);
              BindingGuard guard(b);
              for (ProxyFunction * t : test)
                {
                  HeapReset hr2(lh);
                  size_t dt = t->Dimension();
                  FlatMatrix<double> val(np, 1, lh), coef(np, dt, lh);
                  for (size_t l = 0; l < dt; l++)
                    {
                      ud.testfunction = t; ud.test_comp = l;
                      cf->Evaluate (mir, val);
                      coef.Col(l) = val.Col(0);
                    }
                  FlatMatrix<double,ColMajor> bt(dt, elvec.Size(), lh);
                  for (size_t q = 0; q < np; q++)
                    {
                      bt = 0.0;
                      t->Evaluator()->CalcMatrix (fel, mir[q], bt, lh);
                      elvec += r.pts[q].weight * (Trans(bt) * coef.Row(q));
                    }
                }
            }
            trafo.userdata = nullptr;
          }
      }
    };
  }

  shared_ptr<LinearFormIntegrator>
  MakeInterfaceLFI (shared_ptr<Interface> I, int side, shared_ptr<CoefficientFunction> cf, int bonus_intorder)
  {
    return make_shared<InterfaceLFI> (I, side, cf, bonus_intorder);
  }

  double IntegrateInterface (const Interface & I, int side, CoefficientFunction & cf, int bonus_intorder,
                             LocalHeap & lh)
  {
    if (side != 0 && side != 1) throw Exception ("interface Integrate: side must be 0 or 1");
    if (cf.Dimension() != 1) throw Exception ("interface Integrate: scalar integrand expected");
    double sum = 0;
    for (size_t s = 0; s < I.NSegments(); s++)
      {
        HeapReset hr(lh);
        PieceRule r;
        MakePieceRule (I, side, s, 5 + bonus_intorder, r, lh);
        FlatMatrix<double> val(r.mir->Size(), 1, lh);
        {
          PieceBinding b(I, side, r);
          BindingGuard guard(b);
          cf.Evaluate (*r.mir, val);
        }
        for (size_t q = 0; q < r.pts.Size(); q++) sum += r.pts[q].weight * val(q, 0);
      }
    return sum;
  }
}

void ExportIfcLinear (py::module & m)
{
  m.def ("_AddInterfaceLF", [] (shared_ptr<LinearForm> lf, shared_ptr<Interface> I, int side,
                               shared_ptr<CoefficientFunction> cf, int bonus_intorder)
  {
    if (lf->GetFESpace()->GetMeshAccess() != I->Mesh(side - 1))
      throw Exception ("interface linear form: the space is not on the mesh of side " + ToString(side));
    lf->AddIntegrator (ngstpm::MakeInterfaceLFI (I, side - 1, cf, bonus_intorder));
  }, py::arg("lf"), py::arg("interface"), py::arg("side"), py::arg("cf"), py::arg("bonus_intorder") = 0,
  "(testing) lf += cf * I.ds(side) for the linear part: side = 1, 2");

  m.def ("_IntegrateInterface", [] (shared_ptr<Interface> I, int side, shared_ptr<CoefficientFunction> cf,
                                   int bonus_intorder)
  {
    LocalHeap lh(10000000, "interface-integrate");
    return ngstpm::IntegrateInterface (*I, side - 1, *cf, bonus_intorder, lh);
  }, py::arg("interface"), py::arg("side"), py::arg("cf"), py::arg("bonus_intorder") = 0,
  "(testing) Integrate(cf * I.ds(side)): side = 1, 2");
}
