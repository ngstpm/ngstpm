#include <comp.hpp>
#include <integratorcf.hpp>
#include <python_ngstd.hpp>
#include "extrapolate.hpp"
#include "ifcbinding.hpp"
#include "ifclinear.hpp"
#include "ifcmeasure.hpp"
using namespace ngcomp;
using namespace ngstpm;

// ---------------------------------------------------------------------------
// The interface measure I.ds(side=i): the facets of Gamma_h^i seen from D_h^i.
// Subclasses of NGSolve's DifferentialSymbol/Integral (the hook ngsxfem's
// dFacetPatch uses), so cf * I.ds(...) is an ordinary SumOfIntegrals: + - and
// scaling keep the type (CreateSameIntegralType).
//  - LinearForm, Integrate: local integrands, on the same per-piece rules
//    (src/ifclinear.cpp, MakeInterfaceLFI / IntegrateInterface).
//  - BilinearForm: split off in Python (ngstpm/measure.py) and assembled by a
//    SegmentGroup: elements = the segments of the common subdivision, per-piece
//    Gauss rules, the partner K' bound for ExtrapolateDiffOp/SideEvalCF through
//    the interface binding (ifcbinding.hpp).
// ---------------------------------------------------------------------------

namespace {

struct IfcData { shared_ptr<Interface> I; int side; };   // side: 1, 2

class InterfaceIntegral : public Integral
{
public:
  IfcData ifc;
  InterfaceIntegral (shared_ptr<CoefficientFunction> acf, const DifferentialSymbol & adx, IfcData aifc)
    : Integral(acf, adx), ifc(aifc) { }

  // element_wise is not filled
  double Integrate (const MeshAccess & ma, FlatVector<double> ew) override
  {
    if (&ma != ifc.I->Mesh(ifc.side - 1).get())
      throw Exception ("Integrate(cf * I.ds(side=" + ToString(ifc.side) + ")): mesh is not the mesh of that side");
    LocalHeap lh(10000000, "interface-integrate");
    return IntegrateInterface (*ifc.I, ifc.side - 1, *cf, dx.bonus_intorder, lh);
  }
  Complex Integrate (const MeshAccess & ma, FlatVector<Complex> ew) override
  { throw Exception ("I.ds: complex Integrate is not implemented"); }
  shared_ptr<LinearFormIntegrator> MakeLinearFormIntegrator () const override
  {
    // the integrator cannot see the form's space: check its test functions' mesh here
    cf->TraverseTree ([&] (CoefficientFunction & node)
    {
      auto p = dynamic_cast<ProxyFunction*> (&node);
      if (!p) return;
      if (!p->IsTestFunction())
        throw Exception ("I.ds in a LinearForm: trial function in the integrand");
      if (p->GetFESpace()->GetMeshAccess() != ifc.I->Mesh(ifc.side - 1))
        throw Exception ("I.ds(side=" + ToString(ifc.side) + ") in a LinearForm: the test function's "
                         "space is not on the mesh of that side");
    });
    return MakeInterfaceLFI (ifc.I, ifc.side - 1, cf, dx.bonus_intorder);
  }
  shared_ptr<BilinearFormIntegrator> MakeBilinearFormIntegrator () const override
  {
    throw Exception ("interface integral (I.ds): add it to a BilinearForm with += "
                     "after `import ngstpm`, not through a BilinearFormIntegrator");
  }
  shared_ptr<Integral> CreateSameIntegralType (shared_ptr<CoefficientFunction> acf) override
  { return make_shared<InterfaceIntegral> (acf, dx, ifc); }
};

class InterfaceDifferentialSymbol : public DifferentialSymbol
{
public:
  IfcData ifc;
  InterfaceDifferentialSymbol (IfcData aifc, shared_ptr<BitArray> D, int bonus)
    : DifferentialSymbol(VOL, BND, false, bonus), ifc(aifc)
  { definedonelements = D; }
  shared_ptr<Integral> MakeIntegral (shared_ptr<CoefficientFunction> cf) const override
  { return make_shared<InterfaceIntegral> (cf, *this, ifc); }
};

// ---------------------------------------------------------------------------
// Assembly of one interface integrand on the segments. Rows: the test functions
// on K (side i, space fes); columns: the trial functions on K (fes) and, for
// Extrapolate proxies evaluated on the partner, on K' (their own space: fes on
// one mesh, the other side's space pfes on two meshes).
// ---------------------------------------------------------------------------

struct TrialInfo { ProxyFunction * p; const ExtrapolateDiffOp * ex; };

class SegmentAssembler
{
public:
  shared_ptr<Interface> I;
  int side;                                   // 0, 1
  shared_ptr<CoefficientFunction> cf;
  int bonus;
  shared_ptr<FESpace> fes, pfes;              // one mesh: pfes == fes
  Array<TrialInfo> trial;
  Array<ProxyFunction*> test;

  SegmentAssembler (shared_ptr<Interface> aI, int aside, shared_ptr<CoefficientFunction> acf, int abonus,
                    shared_ptr<FESpace> afes, shared_ptr<FESpace> apfes)
    : I(aI), side(aside - 1), cf(acf), bonus(abonus), fes(afes), pfes(apfes)
  {
    if (aside != 1 && aside != 2) throw Exception ("I.ds: side is 1 or 2");
    if (cf->Dimension() != 1) throw Exception ("I.ds: scalar integrand expected");
    if (fes->GetMeshAccess() != I->Mesh(side))
      throw Exception ("I.ds(side=" + ToString(aside) + "): the form's space is not on the mesh of that side");
    if (pfes->GetMeshAccess() != I->Mesh(1 - side))
      throw Exception ("I.ds: the partner space is not on the mesh of the other side");
    cf->TraverseTree ([&] (CoefficientFunction & node)
    {
      auto p = dynamic_cast<ProxyFunction*> (&node);
      if (!p) return;
      if (p->IsTestFunction()) { if (!test.Contains(p)) test.Append (p); return; }
      for (auto & t : trial) if (t.p == p) return;
      trial.Append (TrialInfo { p, dynamic_cast<const ExtrapolateDiffOp*> (p->Evaluator().get()) });
    });
    // the elements K of this side's pieces
    Array<int> els;
    for (size_t s = 0; s < I->NSegments(); s++)
      {
        if (I->Empty (side, s)) continue;
        int el = I->Facet (side, I->Segment(s).piece[side].f).el;
        if (!els.Contains(el)) els.Append (el);
      }
    Array<DofId> dn;
    for (auto t : test)                                     // D16
      {
        if (t->GetFESpace() != fes) throw Exception ("I.ds: test function of another space than the form's");
        auto space = ComponentSpace (*t);
        for (int el : els)
          {
            space->GetDofNrs (ElementId(VOL, el), dn);
            for (auto d : dn)
              if (IsRegularDof(d) && space->GetDofCouplingType(d) == LOCAL_DOF)
                throw Exception ("I.ds: test function '" + t->Evaluator()->Name() + "' has element-local dofs; "
                                 "interface integrands are tested only with facet test functions (D16)");
          }
      }
    for (auto & t : trial)
      {
        string name = t.p->Evaluator()->Name();
        if (t.ex) continue;
        // a plain trial function is evaluated on K: its space must live there
        if (t.p->GetFESpace() != fes)
          throw Exception ("I.ds: trial function '" + name + "' of the other side's space; use .Extrapolate()");
        auto space = ComponentSpace (*t.p);
        for (int el : els)
          {
            space->GetDofNrs (ElementId(VOL, el), dn);
            for (auto d : dn)
              if (!IsRegularDof(d))
                throw Exception ("I.ds: trial function '" + name + "' has no dofs on the integration element; "
                                 "a partner value is written .Extrapolate() (D7)");
          }
      }
  }

  int IntOrder (const FiniteElement & fel) const { return 2 * fel.Order() + bonus; }

  // emit(rows, cols_fes, mat_fes, cols_pfes, mat_pfes, ei, lh); on one mesh the pfes block is empty
  // and the K' columns are part of the fes block. Columns are unique.
  template <typename EMIT>
  void Assemble (size_t s, LocalHeap & lh, EMIT && emit) const
  {
    if (I->Empty (side, s)) return;                 // 3D: no piece of this side
    HeapReset hr(lh);
    const IfcSegment & seg = I->Segment(s);
    ElementId ei(VOL, I->Facet (side, seg.piece[side].f).el);
    ElementId pei(VOL, I->Facet (1 - side, seg.piece[1 - side].f).el);
    bool one = pfes == fes;
    const FiniteElement & fel = fes->GetFE (ei, lh);
    PieceRule r;
    MakePieceRule (*I, side, s, IntOrder (fel), r, lh);
    if (r.el != ei || r.pel != pei) throw Exception ("I.ds: piece rule of another element");
    const BaseMappedIntegrationRule & mir = *r.mir;
    size_t np = mir.Size();

    Array<DofId> rows, dK, dKp_own, dKp;
    fes->GetDofNrs (ei, rows);
    dK = rows;
    if (one) fes->GetDofNrs (pei, dKp_own); else pfes->GetDofNrs (pei, dKp);
    // fes block: K's dofs then K''s (one mesh), unique
    Array<DofId> cols; Array<int> posK(dK.Size()), posKp(one ? dKp_own.Size() : 0);
    for (size_t j = 0; j < dK.Size(); j++) { posK[j] = cols.Size(); cols.Append (dK[j]); }
    for (size_t j = 0; j < dKp_own.Size(); j++)
      {
        int pos = IsRegularDof(dKp_own[j]) ? cols.Pos (dKp_own[j]) : -1;
        if (pos < 0) { pos = cols.Size(); cols.Append (dKp_own[j]); }
        posKp[j] = pos;
      }
    FlatMatrix<double> M(rows.Size(), cols.Size(), lh), P(rows.Size(), dKp.Size(), lh);
    M = 0.0; P = 0.0;

    PieceBinding b (*I, side, r);
    BindingGuard guard (b);
    const ElementTransformation & trafo = mir[0].GetTransformation();
    ProxyUserData ud;
    const_cast<ElementTransformation&>(trafo).userdata = &ud;
    ud.fel = &fel;
    ud.lh = &lh;
    const FiniteElement & pfel = (one ? fes : pfes)->GetFE (pei, lh);     // K''s element of its own space

    for (auto & tk : trial)
      for (auto t : test)
        {
          HeapReset hr2(lh);
          ProxyFunction * p = tk.p;
          size_t dp = p->Dimension(), dt = t->Dimension();
          FlatMatrix<double> val(np, 1, lh), coef(np, dt * dp, lh);
          for (size_t l = 0; l < dt; l++)
            for (size_t k = 0; k < dp; k++)
              {
                ud.trialfunction = p; ud.trial_comp = k;
                ud.testfunction = t; ud.test_comp = l;
                cf->Evaluate (mir, val);
                coef.Col(l * dp + k) = val.Col(0);
              }
          bool pspace = p->GetFESpace() != fes;               // two meshes: the other side's space
          for (size_t i = 0; i < np; i++)
            {
              if (L2Norm (coef.Row(i)) == 0.0) continue;
              HeapReset hr3(lh);
              const BaseMappedIntegrationPoint & mip = mir[i];
              double w = r.pts[i].weight;
              FlatMatrix<double> dmat(dt, dp, lh);
              for (size_t l = 0; l < dt; l++)
                for (size_t k = 0; k < dp; k++) dmat(l, k) = coef(i, l * dp + k);
              FlatMatrix<double,ColMajor> bt(dt, fel.GetNDof(), lh);
              bt = 0.0;
              t->Evaluator()->CalcMatrix (fel, mip, bt, lh);
              bool own = tk.ex ? tk.ex->OnOwnElement (mip) : true;
              const FiniteElement & efel = own ? fel : pfel;
              if (own && pspace) throw Exception ("I.ds: a trial function of the other mesh on K");
              if (!own && !one && !pspace) throw Exception ("I.ds: a trial function of this mesh on the partner K'");
              FlatMatrix<double,ColMajor> bp(dp, efel.GetNDof(), lh);
              bp = 0.0;
              p->Evaluator()->CalcMatrix (efel, mip, bp, lh);
              FlatMatrix<double> c(rows.Size(), efel.GetNDof(), lh);
              c = w * Trans(bt) * (dmat * bp);
              for (size_t j = 0; j < efel.GetNDof(); j++)
                {
                  if (own) M.Col(posK[j]) += c.Col(j);
                  else if (!pspace) M.Col(posKp[j]) += c.Col(j);
                  else P.Col(j) += c.Col(j);
                }
            }
        }
    const_cast<ElementTransformation&>(trafo).userdata = nullptr;
    emit (FlatArray<DofId>(rows), FlatArray<DofId>(cols), M, FlatArray<DofId>(dKp), P, ei, lh);
  }

  void Dofs (size_t s, Array<DofId> & dn) const      // one mesh: K and K'
  {
    const IfcSegment & seg = I->Segment(s);
    Array<DofId> d2;
    fes->GetDofNrs (ElementId(VOL, I->Facet (side, seg.piece[side].f).el), dn);
    fes->GetDofNrs (ElementId(VOL, I->Facet (1 - side, seg.piece[1 - side].f).el), d2);
    for (auto d : d2) if (!dn.Contains(d)) dn.Append (d);
  }
};

class SegmentGroup : public SpecialElementGroup
{
public:
  SegmentAssembler A;
  SegmentGroup (shared_ptr<Interface> I, int side, shared_ptr<CoefficientFunction> cf, int bonus,
                shared_ptr<FESpace> fes)
    : A(I, side, cf, bonus, fes, fes) { }
  int GetNElements () override { return A.I->NSegments(); }
  void GetDofNrs (std::function<void(int,FlatArray<DofId>)> eldofs) override
  {
    Array<DofId> dn;
    for (size_t s = 0; s < A.I->NSegments(); s++) { A.Dofs (s, dn); eldofs (s, dn); }
  }
  void Assemble (std::function<void(FlatArray<DofId>,FlatArray<DofId>,FlatMatrix<double>,ElementId,LocalHeap&)> addelmat,
                 LocalHeap & lh) override
  {
    for (size_t s = 0; s < A.I->NSegments(); s++)
      A.Assemble (s, lh, [&] (FlatArray<DofId> rows, FlatArray<DofId> cols, FlatMatrix<double> M,
                              FlatArray<DofId>, FlatMatrix<double>, ElementId ei, LocalHeap & lh)
                  { addelmat (rows, cols, M, ei, lh); });
  }
};

// ---------------------------------------------------------------------------
// Static condensation of the interface coupling (D16). NGSolve condenses each
// element in isolation: x_l = H x_g + S f_l (H = harmonic_extension, S =
// inner_solve, element by element). The interface rows C = [C_g | C_l] are
// tested with global functions only (D16), so the local equations are unchanged
// and the Schur complement gains
//   matrix:      C_g + C_l H          (element block by block, below)
//   right side:  -C_l S f_l           (added to harmonic_extension_trans, which
//                                      every condensed NGSolve solve applies to f)
// ---------------------------------------------------------------------------

// THE ONE PLACE where ngstpm reads NGSolve's protected members (no fork: a
// pointer-to-member formed in a derived class, legal C++, tied to the pinned
// NGSolve headers). If NGSolve renames them, this stops compiling; if the getter
// stops returning the member, CondensedGroup::Assemble raises (see InstallCorrection).
struct NGSolveInternals : S_BilinearForm<double>
{
  static Array<shared_ptr<SpecialElementGroup>> & SpecialGroups (BilinearForm & bf)
  { return bf.*(&NGSolveInternals::se_groups); }
  static shared_ptr<BaseMatrix> & HarmonicExtensionTrans (S_BilinearForm<double> & bf)
  { return bf.*(&NGSolveInternals::harmonicexttrans); }
};

bool install_correction = true;          // false only to test the loud failure

struct LocalSolvers                      // H and S of a condensed, assembled BilinearForm, per dof
{
  shared_ptr<FESpace> fes;
  shared_ptr<ElementByElementMatrix<double>> H, S;
  Array<int> hel, hpos, sel, spos;

  LocalSolvers (const BilinearForm & bf) : fes(bf.GetTrialSpace())
  {
    if (!bf.UsesEliminateInternal()) throw Exception ("condensed interface coupling: the form has condense=False");
    H = dynamic_pointer_cast<ElementByElementMatrix<double>> (bf.GetHarmonicExtension());
    S = dynamic_pointer_cast<ElementByElementMatrix<double>> (bf.GetInnerSolve());
    if (!H || !S) throw Exception ("condensed interface coupling: no element-by-element harmonic extension / "
                                   "inner solve (form not assembled, or special elements assembled first?)");
    size_t ndof = fes->GetNDof();
    hel.SetSize(ndof); hpos.SetSize(ndof); sel.SetSize(ndof); spos.SetSize(ndof);
    hel = -1; sel = -1;
    for (size_t e = 0; e < H->GetNumElMats(); e++)
      {
        auto rd = H->GetElementRowDNums(e);
        for (size_t i = 0; i < rd.Size(); i++) if (IsRegularDof(rd[i])) { hel[rd[i]] = e; hpos[rd[i]] = i; }
      }
    for (size_t e = 0; e < S->GetNumElMats(); e++)
      {
        auto rd = S->GetElementRowDNums(e);
        for (size_t i = 0; i < rd.Size(); i++) if (IsRegularDof(rd[i])) { sel[rd[i]] = e; spos[rd[i]] = i; }
      }
  }

  bool Local (DofId d) const { return IsRegularDof(d) && fes->GetDofCouplingType(d) == LOCAL_DOF; }

  // columns cols of elmat (rows: global row numbers grows, NO_DOF_NR skipped): returns the global
  // columns out and C_g + C_l H in m (on lh); appends -C_l S as triplets (grows, local dofs of fes)
  FlatMatrix<double> Condense (FlatArray<DofId> grows, FlatArray<DofId> cols, FlatMatrix<double> elmat,
                               Array<DofId> & out, Array<int> & ri, Array<int> & rj, Array<double> & rv,
                               LocalHeap & lh) const
  {
    out.SetSize0();
    for (auto d : cols)
      {
        if (!IsRegularDof(d)) continue;
        if (!Local(d)) { if (!out.Contains(d)) out.Append (d); continue; }
        if (hel[d] < 0 || sel[d] < 0) throw Exception ("condensed interface coupling: local dof without local solver");
        for (auto c : H->GetElementColumnDNums(hel[d])) if (IsRegularDof(c) && !out.Contains(c)) out.Append (c);
      }
    FlatMatrix<double> m(grows.Size(), out.Size(), lh);
    m = 0.0;
    for (size_t j = 0; j < cols.Size(); j++)
      {
        DofId d = cols[j];
        if (!IsRegularDof(d)) continue;
        if (!Local(d)) { m.Col(out.Pos(d)) += elmat.Col(j); continue; }
        auto Hb = H->GetElementMatrix(hel[d]);
        auto hc = H->GetElementColumnDNums(hel[d]);
        for (size_t k = 0; k < hc.Size(); k++)
          if (IsRegularDof(hc[k])) m.Col(out.Pos(hc[k])) += Hb(hpos[d], k) * elmat.Col(j);
        auto Sb = S->GetElementMatrix(sel[d]);
        auto sc = S->GetElementColumnDNums(sel[d]);
        for (size_t i = 0; i < grows.Size(); i++)
          if (IsRegularDof(grows[i]) && elmat(i, j) != 0.0)
            for (size_t k = 0; k < sc.Size(); k++)
              if (IsRegularDof(sc[k]) && Sb(spos[d], k) != 0.0)
                { ri.Append (grows[i]); rj.Append (sc[k]); rv.Append (-elmat(i, j) * Sb(spos[d], k)); }
      }
    return m;
  }
};

// rows of element-local dofs must be empty (D16); they become NO_DOF_NR
FlatArray<DofId> GlobalRows (const FESpace & fes, FlatArray<DofId> rows, FlatMatrix<double> elmat, LocalHeap & lh)
{
  FlatArray<DofId> g(rows.Size(), lh);
  for (size_t i = 0; i < rows.Size(); i++)
    {
      g[i] = rows[i];
      if (!(IsRegularDof(rows[i]) && fes.GetDofCouplingType(rows[i]) == LOCAL_DOF)) continue;
      if (L2Norm (elmat.Row(i)) != 0.0)
        throw Exception ("condensed interface coupling: interface row of an element-local dof (D16)");
      g[i] = NO_DOF_NR;
    }
  return g;
}

class CondensedGroup : public SpecialElementGroup
{
  BilinearForm * bf;                                  // owns this group
  Array<shared_ptr<SegmentGroup>> inner;
  shared_ptr<BaseMatrix> base_het, installed;
public:
  shared_ptr<BaseMatrix> correction;                  // -C_l S
  CondensedGroup (BilinearForm * abf) : bf(abf) { }
  void Append (shared_ptr<SegmentGroup> g) { inner += g; }

  int GetNElements () override
  {
    int n = 0;
    for (auto & g : inner) n += g->GetNElements();
    return n;
  }

  void GetDofNrs (std::function<void(int,FlatArray<DofId>)> eldofs) override
  {
    auto fes = bf->GetTrialSpace();
    int off = 0;
    Array<DofId> glob;
    for (auto & g : inner)
      {
        g->GetDofNrs ([&] (int i, FlatArray<DofId> dn)
        {
          glob.SetSize0();
          for (auto d : dn)
            if (IsRegularDof(d) && fes->GetDofCouplingType(d) != LOCAL_DOF) glob.Append (d);
          eldofs (off + i, glob);
        });
        off += g->GetNElements();
      }
  }

  void Assemble (std::function<void(FlatArray<DofId>,FlatArray<DofId>,FlatMatrix<double>,ElementId,LocalHeap&)> addelmat,
                 LocalHeap & lh) override
  {
    auto fes = bf->GetTrialSpace();
    if (bf->GetTestSpace() != fes) throw Exception ("I.ds with condense=True: one space for trial and test");
    LocalSolvers ls (*bf);
    Array<int> ri, rj;
    Array<double> rv;
    Array<DofId> out;
    for (auto & g : inner)
      g->Assemble ([&] (FlatArray<DofId> rows, FlatArray<DofId> cols, FlatMatrix<double> elmat,
                        ElementId ei, LocalHeap & lh)
      {
        HeapReset hr(lh);
        auto grows = GlobalRows (*fes, rows, elmat, lh);
        auto m = ls.Condense (grows, cols, elmat, out, ri, rj, rv, lh);
        addelmat (grows, out, m, ei, lh);
      }, lh);
    size_t ndof = fes->GetNDof();
    correction = SparseMatrix<double>::CreateFromCOO (ri, rj, rv, ndof, ndof);
    InstallCorrection ();
  }

  void InstallCorrection ()
  {
    auto sbf = dynamic_cast<S_BilinearForm<double>*> (bf);
    if (!sbf) throw Exception ("I.ds with condense=True: real BilinearForm expected");
    if (install_correction)
      {
        auto & het = NGSolveInternals::HarmonicExtensionTrans (*sbf);
        if (installed && het == installed) het = base_het;     // re-assembly into the same matrices
        base_het = het;
        installed = make_shared<SumMatrix> (base_het, correction);
        het = installed;
      }
    if (!installed || bf->GetHarmonicExtensionTrans() != installed)
      throw Exception ("I.ds with condense=True: the right-hand-side correction -C_l S is not in "
                       "harmonic_extension_trans; a condensed solve would be wrong");
  }

  bool Installed () const { return installed && bf->GetHarmonicExtensionTrans() == installed; }
};

shared_ptr<CondensedGroup> FindCondensed (BilinearForm & bf)
{
  for (auto & g : NGSolveInternals::SpecialGroups (bf))
    if (auto c = dynamic_pointer_cast<CondensedGroup> (g)) return c;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Two meshes: the coupling blocks of one interface integral as COO.
// ---------------------------------------------------------------------------

py::array_t<int> ToNumpy (const Array<int> & a) { return py::array_t<int> (a.Size(), a.Data()); }
py::array_t<double> ToNumpy (const Array<double> & a) { return py::array_t<double> (a.Size(), a.Data()); }

py::dict CouplingBlocks (shared_ptr<Interface> I, shared_ptr<CoefficientFunction> cf, int side, int bonus,
                         shared_ptr<FESpace> fes, shared_ptr<FESpace> pfes,
                         shared_ptr<BilinearForm> bf, shared_ptr<BilinearForm> pbf)
{
  SegmentAssembler A(I, side, cf, bonus, fes, pfes);
  unique_ptr<LocalSolvers> ls, pls;
  if (bf) { if (bf->GetTrialSpace() != fes) throw Exception ("CouplingBlocks: bf is not on fes"); ls = make_unique<LocalSolvers> (*bf); }
  if (pbf) { if (pbf->GetTrialSpace() != pfes) throw Exception ("CouplingBlocks: pbf is not on pfes"); pls = make_unique<LocalSolvers> (*pbf); }
  bool one = pfes == fes;
  if (one && (bool(bf) != bool(pbf) || bf != pbf)) throw Exception ("CouplingBlocks: one mesh needs bf == pbf");
  Array<int> oi, oj, pi, pj, roi, roj, rpi, rpj;
  Array<double> ov, pv, rov, rpv;
  Array<DofId> out;
  LocalHeap lh(50000000, "coupling-blocks");
  auto add = [] (FlatArray<DofId> rows, FlatArray<DofId> cols, FlatMatrix<double> m,
                 Array<int> & I, Array<int> & J, Array<double> & V)
  {
    for (size_t i = 0; i < rows.Size(); i++)
      if (IsRegularDof(rows[i]))
        for (size_t j = 0; j < cols.Size(); j++)
          if (IsRegularDof(cols[j]) && m(i, j) != 0.0) { I.Append (rows[i]); J.Append (cols[j]); V.Append (m(i, j)); }
  };
  for (size_t s = 0; s < I->NSegments(); s++)
    A.Assemble (s, lh, [&] (FlatArray<DofId> rows, FlatArray<DofId> cols, FlatMatrix<double> M,
                            FlatArray<DofId> pcols, FlatMatrix<double> P, ElementId ei, LocalHeap & lh)
    {
      HeapReset hr(lh);
      auto grows = ls ? GlobalRows (*fes, rows, M, lh) : rows;
      if (ls) { auto m = ls->Condense (grows, cols, M, out, roi, roj, rov, lh); add (grows, out, m, oi, oj, ov); }
      else add (rows, cols, M, oi, oj, ov);
      if (one) return;
      if (pls) { auto m = pls->Condense (grows, pcols, P, out, rpi, rpj, rpv, lh); add (grows, out, m, pi, pj, pv); }
      else add (grows, pcols, P, pi, pj, pv);
    });
  py::dict d;
  d["own"] = py::make_tuple (ToNumpy(oi), ToNumpy(oj), ToNumpy(ov));
  d["partner"] = py::make_tuple (ToNumpy(pi), ToNumpy(pj), ToNumpy(pv));
  d["own_correction"] = py::make_tuple (ToNumpy(roi), ToNumpy(roj), ToNumpy(rov));
  d["partner_correction"] = py::make_tuple (ToNumpy(rpi), ToNumpy(rpj), ToNumpy(rpv));
  return d;
}

} // namespace

void ExportIfcMeasure (py::module & m)
{
  py::class_<InterfaceDifferentialSymbol, DifferentialSymbol> (m, "InterfaceDifferentialSymbol",
      "I.ds(side=i): the facets of Gamma_h^i seen from D_h^i (one-sided: specialcf.normal is the\n"
      "outward normal of K, an un-extrapolated proxy is K's). Not NGSolve's ds (BND elements).")
    .def (py::init ([] (shared_ptr<Interface> I, int side, int bonus_intorder)
                    {
                      if (side != 1 && side != 2) throw Exception ("I.ds: side is 1 or 2");
                      return new InterfaceDifferentialSymbol (IfcData { I, side }, I->Side(side - 1).elements,
                                                              bonus_intorder);
                    }),
          py::arg("interface"), py::arg("side"), py::arg("bonus_intorder") = 0);

  py::class_<InterfaceIntegral, Integral, shared_ptr<InterfaceIntegral>> (m, "InterfaceIntegral")
    .def_property_readonly ("interface", [] (InterfaceIntegral & self) { return self.ifc.I; })
    .def_property_readonly ("side", [] (InterfaceIntegral & self) { return self.ifc.side; })
    .def_property_readonly ("bonus_intorder", [] (InterfaceIntegral & self) { return self.dx.bonus_intorder; });

  m.def ("_IfcSide", [] (shared_ptr<Interface> I, int side)
  {
    if (side != 1 && side != 2) throw Exception ("side is 1 or 2");
    auto & S = I->Side (side - 1);
    return py::make_tuple (S.ma, S.elements, S.facets);
  }, py::arg("interface"), py::arg("side"), "(mesh, elements D_h^side, facets of Gamma_h^side)");

  m.def ("_AddInterfaceIntegral", [] (shared_ptr<BilinearForm> bf, shared_ptr<InterfaceIntegral> igl)
  {
    auto fes = bf->GetTrialSpace();
    if (bf->GetTestSpace() != fes) throw Exception ("I.ds: one space for trial and test functions");
    auto g = make_shared<SegmentGroup> (igl->ifc.I, igl->ifc.side, igl->cf, igl->dx.bonus_intorder, fes);
    if (!bf->UsesEliminateInternal()) { bf->Add (g); return; }
    auto cg = FindCondensed (*bf);
    if (!cg) { cg = make_shared<CondensedGroup> (bf.get()); bf->Add (cg); }
    cg->Append (g);
  }, py::arg("bf"), py::arg("integral"));

  m.def ("_CondensedCorrection", [] (shared_ptr<BilinearForm> bf) -> shared_ptr<BaseMatrix>
  {
    auto c = FindCondensed (*bf);
    return c ? c->correction : nullptr;
  }, "the right-hand-side correction -C_l S of a condensed form with interface integrals (None: none)");

  m.def ("_CondensationInstalled", [] (shared_ptr<BilinearForm> bf)
  {
    auto c = FindCondensed (*bf);
    return c && c->Installed();
  }, "True if harmonic_extension_trans of bf carries the interface correction");

  m.def ("_SetCondensationHookForTesting", [] (bool on) { install_correction = on; },
         "testing only: False skips installing the correction (assembly must then raise)");

  m.def ("InterfaceCouplingBlocks", &CouplingBlocks,
         py::arg("interface"), py::arg("cf"), py::arg("side"), py::arg("bonus_intorder"),
         py::arg("fes"), py::arg("pfes"), py::arg("bf") = nullptr, py::arg("pbf") = nullptr,
         "Coupling blocks of the interface integral cf * I.ds(side) on the segments, as COO numpy\n"
         "triplets (rows, cols, values) in a dict:\n"
         "  own:     rows fes (side), columns fes (K)\n"
         "  partner: rows fes, columns pfes (K', the other side's space; empty on one mesh,\n"
         "           where pfes is fes and the K' columns are in own)\n"
         "  own_correction, partner_correction: -C_l S (rows fes, columns fes / pfes)\n"
         "bf / pbf: the condensed, assembled BilinearForm of fes / pfes; if given, the local\n"
         "columns of that block are condensed (C_g + C_l H) and its correction is filled;\n"
         "else the block is uncondensed and its correction empty.");
}
