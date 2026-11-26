#include <comp.hpp>
#include <python_ngstd.hpp>
#include <map>
#include "binding.hpp"
#include "interface.hpp"
using namespace ngcomp;

// ---------------------------------------------------------------------------
// Interface (2D): the common subdivision of Gamma by the two sides' paths.
//  1. every vertex v of Gamma_h^i is mapped along its own paths to Gamma; the
//     paths are element-wise, so v has one image per facet of Gamma_h^i at v,
//     and these images are averaged (they differ by the projection error);
//  2. Gamma is parametrised by arc length s along the chains of Gamma_h^1 (the
//     point X of Gamma has the parameter of the x of Gamma_h^1 with X - x
//     parallel to sigma_1(x)); the side-2 vertex images are pulled back to s
//     (accepted up to gap_tol outside a facet: the gaps between images) and
//     merged with the side-1 vertices when closer than merge_tol (a vertex of
//     both sides on one mesh gives one breakpoint, by its number);
//  3. between consecutive breakpoints lies a segment, covered by exactly one
//     facet of each side (else: unpaired piece or overlap, raised);
//  4. a Gauss rule on each piece, with the partner x' of each point by the
//     same pull-back on the partner facet.
// ---------------------------------------------------------------------------

namespace ngstpm
{
  template <int DIM>
  static IntegrationPoint T_Locate (const ElementTransformation & trafo, const Vec<3> & p)
  {
    IntegrationPoint ip(1.0 / (DIM + 1), 1.0 / (DIM + 1), DIM == 3 ? 0.25 : 0.0, 0.0);
    for (int it = 0; it < 20; it++)      // Newton; affine elements stop after one step
      {
        MappedIntegrationPoint<DIM,DIM> mip(ip, trafo);
        Vec<DIM> res;
        for (int i = 0; i < DIM; i++) res(i) = p(i) - mip.GetPoint()(i);
        if (L2Norm(res) < 1e-13 * (1 + L2Norm(p))) break;
        Vec<DIM> dxi = mip.GetJacobianInverse() * res;
        for (int i = 0; i < DIM; i++) ip(i) += dxi(i);
      }
    return ip;
  }

  IntegrationPoint Locate (const ElementTransformation & trafo, const Vec<3> & p, int D)
  {
    return D == 2 ? T_Locate<2> (trafo, p) : T_Locate<3> (trafo, p);
  }
}
using ngstpm::Locate;

namespace
{
  // outward unit normal of the facet F of its owner element
  Vec<3> OutwardNormal (const IfcSide & S, const IfcFacet & F)
  {
    Vec<3> e = F.p[1] - F.p[0], n = 0.0, c = 0.0;
    n(0) = e(1); n(1) = -e(0);
    n /= L2Norm (n);
    auto vs = S.ma->GetElVertices (ElementId(VOL, F.el));
    for (auto v : vs) c += (1.0 / vs.Size()) * S.ma->GetPoint<3> (v);
    if (InnerProduct (n, F.p[0] - c) < 0) n *= -1.0;
    return n;
  }

  // the vector CF cf at the point p of the line of F, evaluated on F's owner
  // element (extrapolated if p is outside it), with facet nr and normal set
  Vec<3> EvalAt (const IfcSide & S, const IfcFacet & F, const Vec<3> & p,
                 const CoefficientFunction & cf, LocalHeap & lh)
  {
    HeapReset hr(lh);
    // the paths' SideEval nodes on their own element, also when called during an
    // interface assembly (which would redirect them to the partner)
    struct Unbind { ngstpm::InterfaceBinding * b = ngstpm::interface_binding;
                    Unbind () { ngstpm::interface_binding = nullptr; }
                    ~Unbind () { ngstpm::interface_binding = b; } } unbind;
    const ElementTransformation & trafo = S.ma->GetTrafo (ElementId(VOL, F.el), lh);
    IntegrationPoint ip = Locate (trafo, p, 2);
    ip.SetFacetNr (F.loc, BND);
    MappedIntegrationPoint<2,2> mip(ip, trafo);
    Vec<3> n = OutwardNormal (S, F);
    mip.SetNV (Vec<2> (n(0), n(1)));
    Vec<2> v;
    cf.Evaluate (mip, FlatVector<> (2, &v(0)));
    return Vec<3> (v(0), v(1), 0.0);
  }

  Vec<3> UnitSigma (const IfcSide & S, const IfcFacet & F, const Vec<3> & p, LocalHeap & lh)
  {
    Vec<3> s = EvalAt (S, F, p, *S.sigma, lh);
    return (1.0 / L2Norm (s)) * s;
  }

  Vec<3> PointAt (const IfcFacet & F, double t) { return F.p[0] + t * (F.p[1] - F.p[0]); }

  // The point x = p0 + t (p1 - p0) of F, up to the distance gap outside it,
  // with X - x parallel to sigma(x) whose own path ends nearest X: the roots
  // of the scalar cross product are bracketed on 16 subintervals of
  // [-1/2, 3/2] (there can be several: the line of a far point may cross X as
  // well; sigma outside F, a meaningless extrapolation, only serves to
  // bracket), refined by Newton safeguarded by bisection, and kept if inside
  // F up to gap and consistent (own path end near X). Returns the score
  // |t outside [0, 1]| |F| + |xbar(x) - X| (1e300 if there is no root), and t.
  double PullBack (const IfcSide & S, const IfcFacet & F, const Vec<3> & X, double gap, double & t, LocalHeap & lh,
                   double from = -0.5, double to = 1.5)
  {
    auto r = [&] (double tt)
    {
      Vec<3> p = PointAt (F, tt), w = X - p, sg = UnitSigma (S, F, p, lh);
      return w(0) * sg(1) - w(1) * sg(0);
    };
    const int n = 16;
    double best = 1e300, scale = F.len + L2Norm (X - F.p[0]);
    double d = gap / F.len, a = from, ra = r(a);
    for (int k = 1; k <= n; k++)
      {
        double b = from + (to - from) * k / n, rb = r(b);
        if (ra * rb <= 0 && !(ra == 0 && k > 1))
          {
            double lo = a, hi = b, rlo = ra, tt = fabs (ra) < fabs (rb) ? a : b, rt = fabs (ra) < fabs (rb) ? ra : rb;
            for (int it = 0; it < 60 && fabs (rt) > 1e-15 * scale && hi - lo > 1e-15; it++)
              {
                double d = (r(tt + 1e-7) - rt) / 1e-7;
                double tn = d != 0 ? tt - rt / d : lo;
                if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);
                tt = tn; rt = r(tt);
                if (rt * rlo <= 0) hi = tt; else { lo = tt; rlo = rt; }
              }
            if (tt < -d || tt > 1 + d) { a = b; ra = rb; continue; }
            double score = max2 (0.0, max2 (-tt, tt - 1)) * F.len
              + L2Norm (EvalAt (S, F, PointAt (F, tt), *S.target, lh) - X);
            if (score < best) { best = score; t = tt; }
          }
        a = b; ra = rb;
      }
    return best;
  }

  string Pt (const Vec<3> & p) { return "(" + ToString (p(0)) + ", " + ToString (p(1)) + ")"; }
}

Interface::Interface (IfcSide s1, IfcSide s2, double amerge, double agap) : merge_tol(amerge), gap_tol(agap)
{
  sides[0] = s1; sides[1] = s2;
  Build();
}

void Interface::Build ()
{
  dim = sides[0].ma->GetDimension();
  if ((dim != 2 && dim != 3) || sides[1].ma->GetDimension() != dim)
    throw Exception ("pair: the meshes of both sides must be 2D or both 3D");
  for (int i = 0; i < 2; i++)
    {
      auto & S = sides[i];
      if (S.target->Dimension() != dim || S.sigma->Dimension() != dim)
        throw Exception ("pair: target and sigma must be vectors of the mesh dimension");
      if (S.elements->Size() != S.ma->GetNE() || S.facets->Size() != S.ma->GetNFacets())
        throw Exception ("pair: elements/facets BitArrays do not match the mesh");
    }
  if (sides[0].ma == sides[1].ma)
    for (size_t i = 0; i < sides[0].elements->Size(); i++)
      if (sides[0].elements->Test(i) && sides[1].elements->Test(i))
        throw Exception ("pair: the two sides share element " + ToString(i)
                         + " of their common mesh (D_h^1 and D_h^2 must be disjoint)");
  if (dim == 3) { Build3D(); return; }      // src/interface3d.cpp

  LocalHeap lh(10000000, "interface");
  // facets of Gamma_h^i with their owners and vertex numbers
  Array<std::array<int,2>> pv[2];
  for (int i = 0; i < 2; i++)
    {
      auto & S = sides[i];
      Array<int> elnums, pnums;
      for (size_t fnr = 0; fnr < S.facets->Size(); fnr++)
        {
          if (!S.facets->Test(fnr)) continue;
          S.ma->GetFacetElements (fnr, elnums);
          IfcFacet F;
          F.facet = fnr; F.el = -1;
          for (int el : elnums)
            if (S.elements->Test(el))
              {
                if (F.el >= 0) throw Exception ("pair: facet " + ToString(fnr) + " of side " + ToString(i+1)
                                                + " has both neighbours in the element set");
                F.el = el;
              }
          if (F.el < 0) throw Exception ("pair: facet " + ToString(fnr) + " of side " + ToString(i+1)
                                         + " has no neighbour in the element set");
          F.loc = S.ma->GetElFacets(ElementId(VOL, F.el)).Pos(fnr);
          S.ma->GetFacetPNums (fnr, pnums);
          for (int a = 0; a < 2; a++) F.p[a] = S.ma->GetPoint<3> (pnums[a]);
          F.p[2] = 0.0;                                       // unused in 2D
          F.len = L2Norm (F.p[1] - F.p[0]);
          facets[i].Append (F);
          pv[i].Append ({pnums[0], pnums[1]});
        }
      if (facets[i].Size() == 0) throw Exception ("pair: side " + ToString(i+1) + " has no facets");
    }

  // vertex images on Gamma: the mean over the facets of the side at the vertex
  std::map<int,Vec<3>> image[2];
  std::map<int,Array<int>> vfac[2];
  for (int i = 0; i < 2; i++)
    {
      for (int f = 0; f < facets[i].Size(); f++)
        for (int a = 0; a < 2; a++)
          {
            vfac[i][pv[i][f][a]].Append (f);
            Vec<3> xb = EvalAt (sides[i], facets[i][f], facets[i][f].p[a], *sides[i].target, lh);
            if (!image[i].count(pv[i][f][a])) image[i][pv[i][f][a]] = 0.0;
            image[i][pv[i][f][a]] += xb;
          }
      for (auto & [v, fs] : vfac[i])
        {
          if (fs.Size() > 2)
            throw Exception ("pair: Gamma_h^" + ToString(i+1) + " is not a curve at "
                             + Pt (sides[i].ma->GetPoint<3>(v)) + " (" + ToString(fs.Size()) + " facets meet)");
          image[i][v] *= 1.0 / fs.Size();
        }
    }

  // chains of Gamma_h^1, oriented; s = arc length along the chain
  struct Chain { Array<int> fac, dir; Array<double> cum; double L; bool closed; };
  Array<Chain> chains;
  Array<int> chainof(facets[0].Size()), posof(facets[0].Size());
  chainof = -1;
  auto walk = [&] (int f, int v)
  {
    Chain c; c.closed = false;
    int start = v;
    double s = 0;
    while (true)
      {
        int d = pv[0][f][0] == v ? 1 : -1;
        chainof[f] = chains.Size(); posof[f] = c.fac.Size();
        c.fac.Append (f); c.dir.Append (d); c.cum.Append (s);
        s += facets[0][f].len;
        v = pv[0][f][d == 1 ? 1 : 0];
        int next = -1;
        for (int g : vfac[0][v]) if (g != f) next = g;
        if (next < 0) break;
        if (chainof[next] >= 0) { c.closed = v == start; break; }
        f = next;
      }
    c.L = s;
    chains.Append (std::move(c));
  };
  for (auto & [v, fs] : vfac[0])
    if (fs.Size() == 1 && chainof[fs[0]] < 0) walk (fs[0], v);
  for (int f = 0; f < facets[0].Size(); f++)
    if (chainof[f] < 0) walk (f, pv[0][f][0]);

  hmean = 0;
  for (auto & F : facets[0]) hmean += F.len / facets[0].Size();
  double mtol = merge_tol * hmean, gtol = gap_tol * hmean;

  // breakpoints per chain: side-1 vertices (exact s) and pulled-back side-2 vertex images
  struct Brk { double s; int v[2]; Vec<3> X; };
  Array<Array<Brk>> brks(chains.Size());
  std::map<int,std::pair<int,int>> vbrk;   // side-1 vertex -> (chain, breakpoint)
  for (int c = 0; c < chains.Size(); c++)
    {
      auto & C = chains[c];
      int n = C.fac.Size();
      for (int k = 0; k <= n; k++)
        {
          if (k == n && C.closed) break;
          int f = C.fac[k < n ? k : n - 1], d = C.dir[k < n ? k : n - 1];
          int v = k < n ? pv[0][f][d == 1 ? 0 : 1] : pv[0][f][d == 1 ? 1 : 0];
          if (!vbrk.count(v)) vbrk[v] = { c, int(brks[c].Size()) };
          brks[c].Append (Brk { k < n ? C.cum[k] : C.L, {v, -1}, image[0][v] });
        }
    }
  std::map<int,int> vchain;           // side-2 vertex -> chain
  Array<int> unpaired;                // side-2 vertices whose image no side-1 path reaches
  double miss = 0;
  bool onemesh = sides[0].ma == sides[1].ma;
  for (auto & [v, X] : image[1])
    {
      if (onemesh && vbrk.count(v))     // a vertex of both Gamma_h^1 and Gamma_h^2: one breakpoint
        {
          auto [c, j] = vbrk[v];
          brks[c][j].v[1] = v;
          brks[c][j].X = 0.5 * (brks[c][j].X + X);
          vchain[v] = c;
          continue;
        }
      // the facets of Gamma_h^1 by a lower bound of |X - x| (x on the facet)
      int nf = facets[0].Size();
      Array<double> lb(nf); Array<int> order(nf);
      for (int f = 0; f < nf; f++)
        {
          auto & F = facets[0][f];
          lb[f] = L2Norm (X - 0.5 * (F.p[0] + F.p[1])) - 0.5 * F.len - gtol;
          order[f] = f;
        }
      QuickSortI (lb, order);
      // the facet with the least score (see PullBack); accepted up to gtol
      double best = 1e300, bdist = 1e300, bt = 0;
      int bf = -1;
      for (int f : order)
        {
          if (lb[f] > 2 * bdist + hmean) break;
          double t, score = PullBack (sides[0], facets[0][f], X, gtol, t, lh);
          if (score < best) { best = score; bt = t; bf = f; bdist = L2Norm (X - PointAt (facets[0][f], t)); }
        }
      if (bf < 0 || best > gtol) { unpaired.Append (v); miss = max2 (miss, best); continue; }
      bt = min2 (1.0, max2 (0.0, bt));
      auto & C = chains[chainof[bf]];
      int k = posof[bf];
      double s = C.cum[k] + (C.dir[k] == 1 ? bt : 1 - bt) * facets[0][bf].len;
      if (C.closed && s > C.L - mtol) s -= C.L;
      vchain[v] = chainof[bf];
      brks[chainof[bf]].Append (Brk { s, {-1, v}, X });
    }
  if (unpaired.Size())
    {
      double len = 0;                 // the facets of Gamma_h^2 at these vertices
      for (int f = 0; f < facets[1].Size(); f++)
        if (unpaired.Contains (pv[1][f][0]) || unpaired.Contains (pv[1][f][1])) len += facets[1][f].len;
      throw Exception ("pair: unpaired piece of Gamma_h^2 (side 2) near " + Pt (sides[1].ma->GetPoint<3>(unpaired[0]))
                       + ", at most " + ToString (len) + " long: the paths of " + ToString (unpaired.Size())
                       + " of its vertices end where no path of side 1 arrives (missed by up to "
                       + (miss < 1e300 ? ToString (miss) : string("inf")) + ")");
    }

  // merge breakpoints closer than mtol, then the segments between them
  for (int c = 0; c < chains.Size(); c++)
    {
      auto & C = chains[c];
      auto & B = brks[c];
      if (B.Size()) std::sort (&B[0], &B[0] + B.Size(), [] (const Brk & a, const Brk & b) { return a.s < b.s; });
      Array<Brk> M;
      for (auto & b : B)
        {
          if (M.Size() && b.s - M.Last().s <= mtol)
            {
              auto & m = M.Last();
              for (int i = 0; i < 2; i++)
                if (b.v[i] >= 0)
                  {
                    if (m.v[i] >= 0)
                      throw Exception ("pair: two vertices of Gamma_h^" + ToString(i+1) + " reach Gamma closer than "
                                       "the merge tolerance, near " + Pt (b.X));
                    m.v[i] = b.v[i];
                  }
              if (b.v[0] >= 0) { m.s = b.s; m.X = b.X; }   // a side-1 vertex keeps its own s
              continue;
            }
          M.Append (b);
        }
      if (C.closed && M.Size() > 1 && M.Last().s >= C.L - mtol)
        throw Exception ("pair: internal error, unmerged breakpoint at the end of a closed chain");
      std::map<int,int> where;                   // side-2 vertex -> breakpoint index
      for (int j = 0; j < M.Size(); j++) if (M[j].v[1] >= 0) where[M[j].v[1]] = j;
      int nseg = C.closed ? M.Size() : M.Size() - 1;

      // facets of Gamma_h^2 covering each segment
      Array<Array<int>> cover(nseg);
      for (int f = 0; f < facets[1].Size(); f++)
        {
          int a = pv[1][f][0], b = pv[1][f][1];
          bool ina = vchain.count(a) && vchain[a] == c, inb = vchain.count(b) && vchain[b] == c;
          if (ina != inb)
            throw Exception ("pair: unpaired piece of Gamma_h^2 (side 2): the paths of facet "
                             + Pt (facets[1][f].p[0]) + " - " + Pt (facets[1][f].p[1])
                             + " end on two different components of Gamma_h^1");
          if (!ina) continue;
          int ja = where[a], jb = where[b];
          if (ja == jb)
            throw Exception ("pair: the paths of the facet " + Pt (facets[1][f].p[0]) + " - "
                             + Pt (facets[1][f].p[1]) + " of Gamma_h^2 end closer than the merge tolerance");
          int lo = min2 (ja, jb), hi = max2 (ja, jb);
          if (C.closed && 2 * (M[hi].s - M[lo].s) > C.L)        // the short way round
            { for (int j = hi; j < nseg; j++) cover[j].Append (f); for (int j = 0; j < lo; j++) cover[j].Append (f); }
          else
            for (int j = lo; j < hi; j++) cover[j].Append (f);
        }

      std::vector<IfcSegment> segs;
      int k = 0;                                // facet of the chain containing the segment
      for (int j = 0; j < nseg; j++)
        {
          const Brk & A = M[j], & Bk = M[(j + 1) % M.Size()];
          double sA = A.s, sB = j + 1 < M.Size() ? Bk.s : C.L;
          while (k + 1 < C.fac.Size() && C.cum[k + 1] <= sA + 1e-14 * C.L) k++;
          int f0 = C.fac[k];
          auto & F0 = facets[0][f0];
          if (cover[j].Size() != 1)
            {
              if (cover[j].Size() > 1)
                throw Exception ("pair: the paths of " + ToString (cover[j].Size()) + " facets of Gamma_h^2 "
                                 "overlap on Gamma near " + Pt (A.X));
              double len = sB - sA;                  // the run of unpaired segments
              Vec<3> xa = PointAt (F0, (C.dir[k] == 1 ? 0.0 : 1.0) + C.dir[k] * (sA - C.cum[k]) / F0.len);
              for (int jj = j + 1; jj < nseg && cover[jj].Size() == 0; jj++)
                len += (jj + 1 < M.Size() ? M[jj + 1].s : C.L) - M[jj].s;
              throw Exception ("pair: unpaired piece of Gamma_h^1 (side 1) starting at " + Pt (xa)
                               + ", of length " + ToString (len) + ": no path of side 2 ends on its arc of Gamma");
            }
          int f1 = cover[j][0];
          auto & F1 = facets[1][f1];
          IfcSegment seg;
          seg.piece[0].f = f0; seg.piece[1].f = f1;
          const Brk * ends[2] = { &A, &Bk };
          double ss[2] = { sA, sB };
          for (int e = 0; e < 2; e++)
            {
              const Brk & b = *ends[e];
              // side 1: the vertex, or the arc length on the facet
              double t0;
              if (b.v[0] == pv[0][f0][0]) t0 = 0;
              else if (b.v[0] == pv[0][f0][1]) t0 = 1;
              else
                {
                  double u = (ss[e] - C.cum[k]) / F0.len;
                  t0 = C.dir[k] == 1 ? u : 1 - u;
                }
              // side 2: the vertex, or the pull-back of the breakpoint
              double t1;
              if (b.v[1] == pv[1][f1][0]) t1 = 0;
              else if (b.v[1] == pv[1][f1][1]) t1 = 1;
              else if (PullBack (sides[1], F1, b.X, gtol, t1, lh) > gtol)
                throw Exception ("pair: unpaired piece of Gamma_h^2 (side 2) on the facet " + Pt (F1.p[0]) + " - "
                                 + Pt (F1.p[1]) + ": no path of it reaches the point " + Pt (b.X)
                                 + " of Gamma within the gap tolerance " + ToString (gtol));
              seg.piece[0].t[e] = min2 (1.0, max2 (0.0, t0));
              seg.piece[1].t[e] = min2 (1.0, max2 (0.0, t1));
            }
          for (int i = 0; i < 2; i++)
            for (int e = 0; e < 2; e++)
              {
                auto & F = facets[i][seg.piece[i].f];
                seg.xbar[i][e] = EvalAt (sides[i], F, PointAt (F, seg.piece[i].t[e]), *sides[i].target, lh);
              }
          segs.push_back (seg);
        }

      // A breakpoint of one side inside the gap between the vertex images of
      // the other side leaves an empty piece on that side (both ends clamped to
      // the vertex): give the other piece to the neighbour with the same facet.
      for (size_t j = 0; j < segs.size(); )
        {
          bool empty[2];
          for (int i = 0; i < 2; i++) empty[i] = fabs (segs[j].piece[i].t[1] - segs[j].piece[i].t[0]) < 1e-12;
          if (!empty[0] && !empty[1]) { j++; continue; }
          int o = empty[0] ? 1 : 0;
          size_t n = segs.size(), jp = (j + n - 1) % n, jn = (j + 1) % n;
          bool hasp = (C.closed || j > 0) && jp != j, hasn = (C.closed || j + 1 < n) && jn != j;
          auto & P = segs[j].piece[o];
          if (empty[0] && empty[1]) ;
          else if (hasp && segs[jp].piece[o].f == P.f && fabs (segs[jp].piece[o].t[1] - P.t[0]) < 1e-12)
            { segs[jp].piece[o].t[1] = P.t[1]; segs[jp].xbar[o][1] = segs[j].xbar[o][1]; }
          else if (hasn && segs[jn].piece[o].f == P.f && fabs (segs[jn].piece[o].t[0] - P.t[1]) < 1e-12)
            { segs[jn].piece[o].t[0] = P.t[0]; segs[jn].xbar[o][0] = segs[j].xbar[o][0]; }
          else
            throw Exception ("pair: a piece of Gamma_h^" + ToString (2 - o) + " near "
                             + Pt (segs[j].xbar[o][0]) + " has no partner (empty piece on the other side)");
          segs.erase (segs.begin() + j);
        }
      for (auto & sg : segs) segments.Append (sg);
    }

  // the rules raise where the absorption above would exceed gap_tol: check it here, so
  // that pair raises (at the order-12 rule; a higher order may still raise in Rule)
  Array<IfcPoint> pts;
  for (int i = 0; i < 2; i++)
    for (size_t s = 0; s < segments.Size(); s++)
      {
        HeapReset hr(lh);
        Rule (i, s, 12, pts, lh);
      }
}

void Interface::Rule (int side, size_t s, int intorder, Array<IfcPoint> & pts, LocalHeap & lh) const
{
  if (dim == 3) { Rule3D (side, s, intorder, pts, lh); return; }
  const IfcSegment & seg = segments[s];
  const IfcSide & S = sides[side], & P = sides[1 - side];
  const IfcFacet & F = facets[side][seg.piece[side].f], & G = facets[1 - side][seg.piece[1 - side].f];
  double t0 = seg.piece[side].t[0], t1 = seg.piece[side].t[1];
  const IntegrationRule & ir = SelectIntegrationRule (ET_SEGM, intorder);
  pts.SetSize (ir.Size());
  for (size_t q = 0; q < ir.Size(); q++)
    {
      HeapReset hr(lh);
      IfcPoint & pt = pts[q];
      double t = t0 + ir[q](0) * (t1 - t0);
      pt.seg = s; pt.el = F.el; pt.loc = F.loc; pt.pel = G.el; pt.ploc = G.loc;
      pt.weight = ir[q].Weight() * fabs (t1 - t0) * F.len;
      pt.x = PointAt (F, t);
      pt.xbar = EvalAt (S, F, pt.x, *S.target, lh);
      // the partner on the segment's piece of F' (up to the gap tolerance)
      double tp, gap = gap_tol * hmean, d = gap / G.len;
      double lo = min2 (seg.piece[1-side].t[0], seg.piece[1-side].t[1]) - d;
      double hi = max2 (seg.piece[1-side].t[0], seg.piece[1-side].t[1]) + d;
      if (PullBack (P, G, pt.xbar, gap, tp, lh, lo, hi) == 1e300)
        {
          // D14 absorption: no exact partner on the piece (paths that are not continuous,
          // e.g. a point cloud, near a merged breakpoint): the point of the piece whose
          // path line passes closest to xbar, if it misses xbar by at most the gap tolerance
          double rmin = 1e300;
          for (int k = 0; k <= 64; k++)
            {
              double tt = lo + (hi - lo) * k / 64;
              Vec<3> q = PointAt (G, tt), w = pt.xbar - q, sg = UnitSigma (P, G, q, lh);
              double rr = fabs (w(0) * sg(1) - w(1) * sg(0));
              if (rr < rmin) { rmin = rr; tp = tt; }
            }
          if (rmin > gap)
            throw Exception ("pair: unpaired piece of Gamma_h^" + ToString (side + 1) + " (side "
                             + ToString (side + 1) + ") at " + Pt (pt.x) + ": its path ends at " + Pt (pt.xbar)
                             + ", which the paths of the facet " + Pt (G.p[0]) + " - " + Pt (G.p[1])
                             + " of the other side miss by " + ToString (rmin) + " > the gap tolerance "
                             + ToString (gap));
        }
      pt.xp = PointAt (G, tp);
      pt.sigmap = UnitSigma (P, G, pt.xp, lh);
      pt.ip = Locate (S.ma->GetTrafo (ElementId(VOL, F.el), lh), pt.x, 2);
      pt.ip.SetFacetNr (F.loc, BND);
      pt.pip = Locate (P.ma->GetTrafo (ElementId(VOL, G.el), lh), pt.xp, 2);
      pt.pip.SetFacetNr (G.loc, BND);
    }
}

void ExportInterface (py::module & m)
{
  py::class_<Interface, shared_ptr<Interface>> (m, "Interface",
      "Pairing of two sides' transfer paths with the common subdivision of Gamma (2D and 3D).\n"
      "Built by ngstpm.pair(p1, p2). Segments: arcs of Gamma reached from exactly one facet\n"
      "F of Gamma_h^1 and one facet F' of Gamma_h^2; their pieces of F and F'. In 3D a segment\n"
      "is a pair of faces (F, F'), its pieces are polygons (triangulated) on F and on F'; it is\n"
      "supported under the conditions in the docstring of ngstpm.pair (Gamma_h^i a surface, the\n"
      "two sides on opposite sides of Gamma, the path maps injective without folds and resolved\n"
      "by at most 16^2 subtriangles per face).")
    .def (py::init ([] (shared_ptr<MeshAccess> m1, shared_ptr<BitArray> D1, shared_ptr<BitArray> F1,
                        shared_ptr<CoefficientFunction> xb1, shared_ptr<CoefficientFunction> sg1,
                        shared_ptr<MeshAccess> m2, shared_ptr<BitArray> D2, shared_ptr<BitArray> F2,
                        shared_ptr<CoefficientFunction> xb2, shared_ptr<CoefficientFunction> sg2,
                        double merge_tol, double gap_tol)
                    { return make_shared<Interface> (IfcSide { m1, D1, F1, xb1, sg1 },
                                                     IfcSide { m2, D2, F2, xb2, sg2 }, merge_tol, gap_tol); }),
          py::arg("mesh1"), py::arg("elements1"), py::arg("facets1"), py::arg("target1"), py::arg("sigma1"),
          py::arg("mesh2"), py::arg("elements2"), py::arg("facets2"), py::arg("target2"), py::arg("sigma2"),
          py::arg("merge_tol"), py::arg("gap_tol"))
    .def_property_readonly ("nsegments", &Interface::NSegments)
    .def_property_readonly ("merge_tol", [] (shared_ptr<Interface> self) { return self->MergeTol(); })
    .def_property_readonly ("gap_tol", [] (shared_ptr<Interface> self) { return self->GapTol(); })
    .def ("_segments", [] (shared_ptr<Interface> self)
          {
            // per segment s and side i (axis 1: 0 = side 1, 1 = side 2):
            // facet, el, loc of the piece's facet; t[s,i,end]; x and xbar[s,i,end,:]
            if (self->Dim() == 3) return InterfacePieces3D (*self);
            size_t n = self->NSegments();
            py::array_t<int> facet({n, size_t(2)}), el({n, size_t(2)}), loc({n, size_t(2)});
            py::array_t<double> t({n, size_t(2), size_t(2)}), X({n, size_t(2), size_t(2), size_t(2)}),
              XB({n, size_t(2), size_t(2), size_t(2)});
            auto fa = facet.mutable_unchecked<2>(), ea = el.mutable_unchecked<2>(), la = loc.mutable_unchecked<2>();
            auto ta = t.mutable_unchecked<3>();
            auto xa = X.mutable_unchecked<4>(), ba = XB.mutable_unchecked<4>();
            for (size_t s = 0; s < n; s++)
              for (int i = 0; i < 2; i++)
                {
                  auto & P = self->Segment(s).piece[i];
                  auto & F = self->Facet (i, P.f);
                  fa(s, i) = F.facet; ea(s, i) = F.el; la(s, i) = F.loc;
                  for (int e = 0; e < 2; e++)
                    {
                      ta(s, i, e) = P.t[e];
                      Vec<3> p = PointAt (F, P.t[e]);
                      for (int j = 0; j < 2; j++)
                        { xa(s, i, e, j) = p(j); ba(s, i, e, j) = self->Segment(s).xbar[i][e](j); }
                    }
                }
            py::dict d;
            d["facet"] = facet; d["el"] = el; d["loc"] = loc; d["t"] = t; d["x"] = X; d["xbar"] = XB;
            return d;
          },
          "dict of numpy arrays over segments s and sides i: facet, el, loc [s,i]; t [s,i,end]\n"
          "(the piece: x = p0 + t (p1 - p0), p0, p1 the facet's vertices); x, xbar [s,i,end,:].\n"
          "3D: facet, el, loc, area, m [s,i] (area 0: empty piece; m: the face split into m^2\n"
          "subtriangles) and the pieces' triangles k:\n"
          "tri_seg, tri_side [k], tri_ref [k,vertex,:] (reference coordinates on the face:\n"
          "x = p0 + xi (p1 - p0) + eta (p2 - p0)), tri_x [k,vertex,:]")
    .def ("_rule", [] (shared_ptr<Interface> self, int side, int intorder)
          {
            if (side != 1 && side != 2) throw Exception ("side must be 1 or 2");
            LocalHeap lh(10000000, "interface-rule");
            Array<IfcPoint> pts;
            std::vector<double> X, XB, XP, SG, W;
            std::vector<int> EL, LOC, PEL, PLOC, SEG;
            for (size_t s = 0; s < self->NSegments(); s++)
              {
                self->Rule (side - 1, s, intorder, pts, lh);
                for (auto & p : pts)
                  {
                    for (int j = 0; j < self->Dim(); j++)
                      { X.push_back (p.x(j)); XB.push_back (p.xbar(j)); XP.push_back (p.xp(j)); SG.push_back (p.sigmap(j)); }
                    W.push_back (p.weight);
                    EL.push_back (p.el); LOC.push_back (p.loc); PEL.push_back (p.pel); PLOC.push_back (p.ploc);
                    SEG.push_back (p.seg);
                  }
              }
            size_t n = W.size(), D = self->Dim();
            auto vec = [&] (std::vector<double> & v) { return py::array_t<double> (std::vector<size_t>{n, D}, v.data()); };
            auto ints = [&] (std::vector<int> & v) { return py::array_t<int> (n, v.data()); };
            py::dict d;
            d["x"] = vec(X); d["weight"] = py::array_t<double> (n, W.data()); d["xbar"] = vec(XB);
            d["xp"] = vec(XP); d["sigmap"] = vec(SG); d["el"] = ints(EL); d["loc"] = ints(LOC);
            d["pel"] = ints(PEL); d["ploc"] = ints(PLOC); d["seg"] = ints(SEG);
            return d;
          }, py::arg("side"), py::arg("intorder") = 4,
          "dict of numpy arrays at the points of the order-intorder Gauss rule on the pieces of\n"
          "Gamma_h^side: x, weight (physical), xbar, partner xp, unit sigma(xp), el, loc (K and\n"
          "its facet), pel, ploc (K' and the facet of xp), seg");
}
