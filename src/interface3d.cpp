#include <comp.hpp>
#include <python_ngstd.hpp>
#include <map>
#include "binding.hpp"
#include "interface.hpp"
using namespace ngcomp;
using ngstpm::Locate;

// ---------------------------------------------------------------------------
// Interface (3D; tetrahedra, triangular straight faces): pieces by clipping
// in local charts (no global parametrisation of Gamma).
//  1. as in 2D, every vertex v of Gamma_h^i has one image per face of the side
//     at v (element-wise paths), averaged; the flat triangles T_F on the images
//     of the vertices of the faces F form a conforming surface S_i near Gamma;
//  2. the chart of F is the plane through X0 (the image of F's first vertex)
//     normal to n0, the mean unit path direction at F's vertices (axes e1 along
//     X1 - X0 projected onto it, e2 = n0 x e1): projected to it along (nearly)
//     the paths, S_j stays a conforming triangulation also where T_F is a sliver
//     far off the tangent plane of Gamma (carved meshes); T_F, projected to the
//     chart, only gives the affine map to F's reference coordinates;
//  3. near F, S_j projected to this chart is a planar conforming triangulation,
//     so the chart triangle of T_F clipped against the projected T_F' (F' of the
//     other side; the two sides lie on either side of Gamma, so only the T_F'
//     whose normal, oriented by the outward normal of F', points to the other
//     side of the chart than T_F's) gives polygons that tile the chart triangle
//     of T_F up to round-off; mapped affinely
//     to F (T_F -> F vertex by vertex) they are the pieces of side i of the
//     segments (F, F'). Side j's pieces are computed the same way in side j's
//     charts: the same polygons pulled back to F' would not tile F' (different
//     faces, different charts). A segment is a pair (F, F') found by either side;
//     its piece on one side may be empty (a sliver dropped there);
//  4. open surfaces: beyond each rim edge of S_j (an edge in one face of S_j), a
//     strip of depth gap belongs to that edge's face, so that the O(h^2) by
//     which T_F sticks out of S_j near the rim is paired;
//  5. coverage per face: the pieces' areas sum to |F| (else: unpaired piece, or
//     the paths of side j overlap); then the rules (partner x' on F' by
//     Gauss-Newton on the path line through xbar; its own path must end within
//     the gap of xbar) are checked at order 12.
// ---------------------------------------------------------------------------

namespace
{
  // outward unit normal of the face F of its owner element
  Vec<3> OutwardNormal3 (const IfcSide & S, const IfcFacet & F)
  {
    Vec<3> n = Cross (Vec<3> (F.p[1] - F.p[0]), Vec<3> (F.p[2] - F.p[0])), c = 0.0;
    n /= L2Norm (n);
    auto vs = S.ma->GetElVertices (ElementId(VOL, F.el));
    for (auto v : vs) c += (1.0 / vs.Size()) * S.ma->GetPoint<3> (v);
    if (InnerProduct (n, F.p[0] - c) < 0) n *= -1.0;
    return n;
  }

  // the vector CF cf at the point p of the plane of F, evaluated on F's owner
  // element (extrapolated if p is outside it), with facet nr and normal set
  Vec<3> EvalAt3 (const IfcSide & S, const IfcFacet & F, const Vec<3> & p,
                  const CoefficientFunction & cf, LocalHeap & lh)
  {
    HeapReset hr(lh);
    // the paths' SideEval nodes on their own element (see EvalAt in interface.cpp)
    struct Unbind { ngstpm::InterfaceBinding * b = ngstpm::interface_binding;
                    Unbind () { ngstpm::interface_binding = nullptr; }
                    ~Unbind () { ngstpm::interface_binding = b; } } unbind;
    const ElementTransformation & trafo = S.ma->GetTrafo (ElementId(VOL, F.el), lh);
    IntegrationPoint ip = Locate (trafo, p, 3);
    ip.SetFacetNr (F.loc, BND);
    MappedIntegrationPoint<3,3> mip(ip, trafo);
    mip.SetNV (OutwardNormal3 (S, F));
    Vec<3> v;
    cf.Evaluate (mip, FlatVector<> (3, &v(0)));
    return v;
  }

  Vec<3> UnitSigma3 (const IfcSide & S, const IfcFacet & F, const Vec<3> & p, LocalHeap & lh)
  {
    Vec<3> s = EvalAt3 (S, F, p, *S.sigma, lh);
    return (1.0 / L2Norm (s)) * s;
  }

  Vec<3> PointAt3 (const IfcFacet & F, const Vec<2> & r)
  { return F.p[0] + r(0) * (F.p[1] - F.p[0]) + r(1) * (F.p[2] - F.p[0]); }

  double Cross2 (const Vec<2> & a, const Vec<2> & b) { return a(0) * b(1) - a(1) * b(0); }

  string Pt3 (const Vec<3> & p) { return "(" + ToString (p(0)) + ", " + ToString (p(1)) + ", " + ToString (p(2)) + ")"; }

  using Tri = std::array<Vec<3>,3>;

  Vec<3> UnitNormal (const Tri & X)
  {
    Vec<3> n = Cross (Vec<3> (X[1] - X[0]), Vec<3> (X[2] - X[0]));
    return (1.0 / L2Norm (n)) * n;
  }

  // the plane through X0 = X[0] with unit normal n: P(X) = (e1.(X - X0), e2.(X - X0)),
  // e1 along X[1] - X[0] projected onto it (n not parallel to the plane of X)
  struct Chart
  {
    Vec<3> X0, e1, e2;
    Chart (const Tri & X, const Vec<3> & n) : X0(X[0])
    {
      e1 = X[1] - X[0]; e1 -= InnerProduct (e1, n) * n; e1 /= L2Norm (e1);
      e2 = Cross (n, e1);
    }
    Vec<2> Lin (const Vec<3> & v) const { return Vec<2> (InnerProduct (e1, v), InnerProduct (e2, v)); }
    Vec<2> operator() (const Vec<3> & X) const { return Lin (Vec<3> (X - X0)); }
  };

  // A line of the chart through O and E, g(q) = (E - O) x (q - O). An edge of S_j has O, E
  // the projected images of its vertices ordered by vertex number (va < vb), so that g is
  // evaluated identically from both faces at the edge (one of them flips the sign); a ray
  // from the rim vertex va has vb = -1; tf >= 0: the edge tf of T_F itself (O, E: its ends
  // ordered by vertex number).
  struct Line { Vec<2> O, E; int va = -1, vb = -1, tf = -1; };

  double G (const Line & L, const Vec<2> & q) { return Cross2 (Vec<2> (L.E - L.O), Vec<2> (q - L.O)); }

  // the point where the polygon edge a -> b, on the line A, crosses B: on an edge of T_F
  // from that edge's ends; at a vertex of S_j common to A and B, that vertex; else on a -> b
  // (canonical operand order). So the pieces on both sides of a line share their vertices.
  Vec<2> Meet (const Line & A, const Line & B, const Vec<2> & a, const Vec<2> & b)
  {
    if (A.tf >= 0)
      {
        double gu = G(B, A.O), gw = G(B, A.E);
        double t = gu != gw ? min2 (1.0, max2 (0.0, gu / (gu - gw))) : 0.5;
        return A.O + t * (A.E - A.O);
      }
    for (int v : { A.va, A.vb })
      if (v >= 0 && (v == B.va || v == B.vb)) return v == A.va ? A.O : A.E;
    bool sw = a(0) > b(0) || (a(0) == b(0) && a(1) > b(1));
    const Vec<2> & u = sw ? b : a, & w = sw ? a : b;
    double gu = G(B, u), gw = G(B, w);
    double t = gu != gw ? min2 (1.0, max2 (0.0, gu / (gu - gw))) : 0.5;
    return u + t * (w - u);
  }

  // Sutherland-Hodgman: the part of the convex polygon pts (edge k from pts[k] to pts[k+1]
  // on the line lines[lab[k]]) where s g_L >= 0
  void Clip (Array<Vec<2>> & pts, Array<int> & lab, const Array<Line> & lines, int L, double s)
  {
    Array<Vec<2>> np; Array<int> nl;
    size_t n = pts.Size();
    for (size_t k = 0; k < n; k++)
      {
        const Vec<2> & a = pts[k], & b = pts[(k + 1) % n];
        bool ina = s * G(lines[L], a) >= 0, inb = s * G(lines[L], b) >= 0;
        if (ina) { np.Append (a); nl.Append (lab[k]); }
        if (ina != inb)
          {
            np.Append (Meet (lines[lab[k]], lines[L], a, b));
            nl.Append (ina ? L : lab[k]);
          }
      }
    pts = std::move (np); lab = std::move (nl);
  }
}

void Interface::Build3D ()
{
  LocalHeap lh(10000000, "interface3d");
  // faces of Gamma_h^i with their owners and vertex numbers
  Array<std::array<int,3>> pv[2];
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
          if (pnums.Size() != 3)
            throw Exception ("pair: face " + ToString(fnr) + " of side " + ToString(i+1) + " is not a triangle "
                             "(3D interfaces need tetrahedra with straight triangular faces)");
          for (int a = 0; a < 3; a++) F.p[a] = S.ma->GetPoint<3> (pnums[a]);
          F.len = 0.5 * L2Norm (Cross (Vec<3> (F.p[1] - F.p[0]), Vec<3> (F.p[2] - F.p[0])));
          facets[i].Append (F);
          pv[i].Append ({pnums[0], pnums[1], pnums[2]});
        }
      if (facets[i].Size() == 0) throw Exception ("pair: side " + ToString(i+1) + " has no facets");
    }

  // h: the larger of the two sides' mean edge lengths (the partner's overshoot is set by
  // the coarser side's path map)
  hmean = 0;
  for (int i = 0; i < 2; i++)
    {
      double m = 0;
      for (auto & F : facets[i])
        for (int a = 0; a < 3; a++) m += L2Norm (F.p[(a + 1) % 3] - F.p[a]) / (3 * facets[i].Size());
      hmean = max2 (hmean, m);
    }
  double marea = merge_tol * hmean * hmean, gap = gap_tol * hmean;

  // vertex images (mean over the faces of the side at the vertex), image triangles, edges
  std::map<int,Vec<3>> image[2];
  std::map<std::pair<int,int>,Array<int>> efac[2];      // edge (va < vb) -> faces of the side
  for (int i = 0; i < 2; i++)
    {
      std::map<int,int> count;
      for (int f = 0; f < facets[i].Size(); f++)
        for (int a = 0; a < 3; a++)
          {
            int v = pv[i][f][a], w = pv[i][f][(a + 1) % 3];
            efac[i][{min2 (v, w), max2 (v, w)}].Append (f);
            Vec<3> xb = EvalAt3 (sides[i], facets[i][f], facets[i][f].p[a], *sides[i].target, lh);
            if (!image[i].count(v)) image[i][v] = 0.0;
            image[i][v] += xb;
            count[v]++;
          }
      for (auto & [v, X] : image[i]) X *= 1.0 / count[v];
      for (auto & [e, fs] : efac[i])
        if (fs.Size() > 2)
          throw Exception ("pair: Gamma_h^" + ToString(i+1) + " is not a surface at the edge "
                           + Pt3 (sides[i].ma->GetPoint<3>(e.first)) + " - " + Pt3 (sides[i].ma->GetPoint<3>(e.second))
                           + " (" + ToString(fs.Size()) + " faces meet)");
      images[i].SetSize (facets[i].Size());
      for (int f = 0; f < facets[i].Size(); f++)
        {
          for (int a = 0; a < 3; a++) images[i][f][a] = image[i][pv[i][f][a]];
          auto & X = images[i][f];
          if (0.5 * L2Norm (Cross (Vec<3> (X[1] - X[0]), Vec<3> (X[2] - X[0]))) < marea)
            throw Exception ("pair: the paths of the face " + Pt3 (facets[i][f].p[0]) + " - " + Pt3 (facets[i][f].p[1])
                             + " - " + Pt3 (facets[i][f].p[2]) + " of Gamma_h^" + ToString(i+1)
                             + " end closer than the merge tolerance");
        }
    }

  // rims of S_i: the edges in one face; at a rim vertex, the direction of the rays bounding
  // the strips: the normalised sum of the outward conormals (in T_F) of its rim edges
  struct Rim { int f, a, b; };                          // face, the edge's vertices (a < b)
  Array<Rim> rims[2];
  std::map<int,Vec<3>> rimdir[2];
  for (int i = 0; i < 2; i++)
    {
      for (auto & [e, fs] : efac[i])
        {
          if (fs.Size() != 1) continue;
          int f = fs[0];
          rims[i].Append (Rim { f, e.first, e.second });
          int t = 0;
          while (pv[i][f][t] == e.first || pv[i][f][t] == e.second) t++;
          Vec<3> Xa = image[i][e.first], u = image[i][e.second] - Xa, w = Xa - images[i][f][t];
          u /= L2Norm (u);
          Vec<3> nu = w - InnerProduct (w, u) * u;
          nu /= L2Norm (nu);
          for (int v : { e.first, e.second })
            {
              if (!rimdir[i].count(v)) rimdir[i][v] = 0.0;
              rimdir[i][v] += nu;
            }
        }
      for (auto & [v, d] : rimdir[i]) d /= L2Norm (d);
    }
  Array<Array<int>> frims[2];                           // face -> its rim edges
  for (int i = 0; i < 2; i++)
    {
      frims[i].SetSize (facets[i].Size());
      for (int r = 0; r < rims[i].Size(); r++) frims[i][rims[i][r].f].Append (r);
    }

  // bounding spheres of the image triangles; their unit normals, oriented as the outward
  // normal of the owner element on F (the vertex order of F carried over to T_F)
  Array<Vec<3>> ctr[2], nimg[2]; Array<double> rad[2];
  for (int i = 0; i < 2; i++)
    for (int f = 0; f < images[i].Size(); f++)
      {
        const Tri & X = images[i][f];
        const IfcFacet & F = facets[i][f];
        Vec<3> c = (1.0 / 3) * (X[0] + X[1] + X[2]);
        ctr[i].Append (c);
        rad[i].Append (max2 (L2Norm (X[0] - c), max2 (L2Norm (X[1] - c), L2Norm (X[2] - c))));
        Vec<3> nF = Cross (Vec<3> (F.p[1] - F.p[0]), Vec<3> (F.p[2] - F.p[0]));
        nimg[i].Append (InnerProduct (nF, OutwardNormal3 (sides[i], F)) > 0 ? UnitNormal (X) : Vec<3> (-1.0 * UnitNormal (X)));
      }

  // pieces: for each face F of side i, clip in the chart of F against the faces of side j
  std::map<std::pair<int,int>,int> segof;              // (face of side 1, face of side 2) -> segment
  auto piece = [&] (int i, int f, int g) -> IfcPiece &
  {
    std::pair<int,int> key = i == 0 ? std::make_pair (f, g) : std::make_pair (g, f);
    auto it = segof.find (key);
    if (it == segof.end())
      {
        IfcSegment seg;
        seg.piece[0].f = key.first; seg.piece[1].f = key.second;
        for (int k = 0; k < 2; k++) { seg.piece[k].t[0] = seg.piece[k].t[1] = 0; }
        for (int k = 0; k < 2; k++) for (int e = 0; e < 2; e++) seg.xbar[k][e] = 0.0;
        it = segof.emplace (key, segments.Size()).first;
        segments.Append (seg);
      }
    return segments[it->second].piece[i];
  };
  // candidates: the image triangles of each side binned in a uniform grid of cells of width
  // 2 rmax + 2 gap (rmax: the largest bounding radius of both sides), so that the bounding-sphere
  // test below can only pass for triangles in the 27 cells around the centre's cell
  double rmax = 0;
  for (int i = 0; i < 2; i++) for (double r : rad[i]) rmax = max2 (rmax, r);
  double cell = 2 * rmax + 2 * gap;
  auto cellof = [&] (const Vec<3> & c)
  { return std::array<int,3> { int (floor (c(0) / cell)), int (floor (c(1) / cell)), int (floor (c(2) / cell)) }; };
  std::map<std::array<int,3>,Array<int>> grid[2];
  for (int i = 0; i < 2; i++)
    for (int f = 0; f < ctr[i].Size(); f++) grid[i][cellof (ctr[i][f])].Append (f);
  for (int i = 0; i < 2; i++)
    {
      int j = 1 - i;
      for (int f = 0; f < facets[i].Size(); f++)
        {
          const IfcFacet & F = facets[i][f];
          const Tri & XF = images[i][f];
          // the chart: normal to the mean path direction at the vertices of F (projecting S_j
          // along the paths), oriented as T_F's normal; T_F (possibly a sliver far off the
          // tangent plane) only gives the affine map to F
          Vec<3> n0 = 0.0, s0 = UnitSigma3 (sides[i], F, F.p[0], lh);
          for (int a = 0; a < 3; a++)
            {
              Vec<3> sa = UnitSigma3 (sides[i], F, F.p[a], lh);
              n0 += InnerProduct (sa, s0) < 0 ? Vec<3> (-1.0 * sa) : sa;
            }
          n0 /= L2Norm (n0);
          if (InnerProduct (n0, nimg[i][f]) < 0) n0 *= -1.0;
          if (InnerProduct (n0, nimg[i][f]) < 1e-3)
            throw Exception ("pair: the path ends of the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - "
                             + Pt3 (F.p[2]) + " of Gamma_h^" + ToString(i+1) + " span a triangle (nearly) parallel "
                             "to its paths (cosine " + ToString (InnerProduct (n0, nimg[i][f])) + "): no chart");
          Chart P(XF, n0);
          Vec<2> c[3];
          for (int a = 0; a < 3; a++) c[a] = P(XF[a]);
          // chart -> F's reference coordinates: c0 + xi (c1 - c0) + eta (c2 - c0)
          Vec<2> d1 = c[1] - c[0], d2 = c[2] - c[0];
          double det = Cross2 (d1, d2);
          auto toref = [&] (const Vec<2> & q)
          {
            Vec<2> w = q - c[0];
            return Vec<2> (Cross2 (w, d2) / det, Cross2 (d1, w) / det);
          };
          Array<Line> lines;
          for (int a = 0; a < 3; a++)                          // the edges of T_F
            {
              int u = a, w = (a + 1) % 3;
              if (pv[i][f][u] > pv[i][f][w]) std::swap (u, w);
              Line L; L.O = c[u]; L.E = c[w]; L.tf = a;
              lines.Append (L);
            }
          // a line of S_j through the images of its vertices v, w, canonically ordered
          auto sjline = [&] (int v, int w)
          {
            Line L;
            if (v > w) std::swap (v, w);
            L.O = P(image[j][v]); L.E = P(image[j][w]); L.va = v; L.vb = w;
            return L;
          };
          int ncand = 0;
          double covered = 0;
          auto add = [&] (int g, Array<Vec<2>> & pts)
          {
            if (pts.Size() < 3) return;
            Array<Vec<2>> r;
            for (auto & q : pts) r.Append (toref (q));
            double a = 0;
            for (size_t k = 1; k + 1 < r.Size(); k++) a += 0.5 * Cross2 (r[k] - r[0], r[k + 1] - r[0]);
            a = fabs (a) * 2 * F.len;                          // physical area of the polygon
            if (a < marea) return;                             // a sliver below the merge tolerance
            IfcPiece & pc = piece (i, f, g);
            for (size_t k = 1; k + 1 < r.Size(); k++) pc.tri.push_back ({ r[0], r[k], r[k + 1] });
            pc.area += a;
            covered += a;
          };
          // the faces of side j in the cells around T_F's, in increasing order (as a plain search)
          Array<int> cand;
          auto c0 = cellof (ctr[i][f]);
          for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) for (int d = -1; d <= 1; d++)
            {
              auto it = grid[j].find ({ c0[0] + a, c0[1] + b, c0[2] + d });
              if (it != grid[j].end()) for (int g : it->second) cand.Append (g);
            }
          std::sort (cand.begin(), cand.end());
          for (int g : cand)
            {
              if (L2Norm (ctr[i][f] - ctr[j][g]) > rad[i][f] + rad[j][g] + 2 * gap) continue;
              const Tri & XG = images[j][g];
              // the two sides lie on either side of Gamma: paired faces have (oriented) image
              // normals on opposite sides of the chart; this tells apart sheets of Gamma closer
              // than h (no margin: a sliver T_G may be nearly parallel to the paths)
              if (InnerProduct (nimg[j][g], n0) >= 0) continue;
              ncand++;
              Vec<2> Q[3];
              for (int a = 0; a < 3; a++) Q[a] = P(XG[a]);
              int ord[3] = { 0, 1, 2 };                         // counter-clockwise in the chart
              if (Cross2 (Q[1] - Q[0], Q[2] - Q[0]) < 0) std::swap (ord[1], ord[2]);
              lines.SetSize (3);
              Array<Vec<2>> pts; Array<int> lab;
              for (int a = 0; a < 3; a++) { pts.Append (c[a]); lab.Append (a); }
              // inside: left of the counter-clockwise edge A -> B; with canonical (O, E), the sign s
              for (int a = 0; a < 3 && pts.Size(); a++)
                {
                  int va = pv[j][g][ord[a]], vb = pv[j][g][ord[(a + 1) % 3]];
                  lines.Append (sjline (va, vb));
                  Clip (pts, lab, lines, lines.Size() - 1, va < vb ? 1.0 : -1.0);
                }
              add (g, pts);
              // the strips beyond the rim edges of G
              for (int r : frims[j][g])
                {
                  ncand++;
                  int va = rims[j][r].a, vb = rims[j][r].b;
                  int ka = 0; while (pv[j][g][ord[ka]] != va) ka++;
                  bool ccw = pv[j][g][ord[(ka + 1) % 3]] == vb;  // A -> B = va -> vb counter-clockwise
                  lines.SetSize (3);
                  pts.SetSize0 (); lab.SetSize0 ();
                  for (int a = 0; a < 3; a++) { pts.Append (c[a]); lab.Append (a); }
                  Line Lb = sjline (va, vb);                    // O = va, E = vb
                  lines.Append (Lb);
                  Clip (pts, lab, lines, 3, ccw ? -1.0 : 1.0);  // outside G
                  Vec<2> t = Lb.E - Lb.O, nout = ccw ? Vec<2> (t(1), -t(0)) : Vec<2> (-t(1), t(0));
                  nout /= L2Norm (nout);
                  Line Lo; Lo.O = Lb.O + gap * nout; Lo.E = Lb.E + gap * nout;
                  lines.Append (Lo);                            // depth gap: on the side of Lb
                  Clip (pts, lab, lines, 4, G(Lo, Lb.O) >= 0 ? 1.0 : -1.0);
                  for (int v : { va, vb })                      // the rays at the rim vertices
                    {
                      Line R; R.O = P(image[j][v]); R.E = R.O + P.Lin (rimdir[j][v]); R.va = v;
                      lines.Append (R);
                      Vec<2> other = v == va ? Lb.E : Lb.O;
                      if (pts.Size()) Clip (pts, lab, lines, lines.Size() - 1, G(R, other) >= 0 ? 1.0 : -1.0);
                    }
                  add (g, pts);
                }
            }
          // coverage: the pieces tile F, up to the slivers dropped (each below marea)
          double tol = (ncand + 1) * marea + 1e-12 * F.len;
          if (covered < F.len - tol)
            throw Exception ("pair: unpaired piece of Gamma_h^" + ToString(i+1) + " (side " + ToString(i+1)
                             + ") on the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - " + Pt3 (F.p[2])
                             + ", uncovered area " + ToString (F.len - covered) + " of " + ToString (F.len)
                             + ": no path of side " + ToString(j+1) + " ends on its part of Gamma, or the paths of "
                             "a side fold there (see pair: the conditions in 3D)");
          if (covered > F.len + tol)
            throw Exception ("pair: the pieces of the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - " + Pt3 (F.p[2])
                             + " of Gamma_h^" + ToString(i+1) + " (side " + ToString(i+1) + ") cover "
                             + ToString (covered) + " of its area " + ToString (F.len) + ": near " + Pt3 (ctr[i][f])
                             + " the paths of side " + ToString(j+1) + " overlap on Gamma, or the paths of a side "
                             "fold there (see pair: the conditions in 3D)");
        }
    }

  // the rules raise where the absorption would exceed gap_tol: check it here, so that
  // pair raises (at the order-12 rule; a higher order may still raise in Rule)
  Array<IfcPoint> pts;
  for (int i = 0; i < 2; i++)
    for (size_t s = 0; s < segments.Size(); s++)
      {
        HeapReset hr(lh);
        Rule3D (i, s, 12, pts, lh);
      }
}

void Interface::Rule3D (int side, size_t s, int intorder, Array<IfcPoint> & pts, LocalHeap & lh) const
{
  const IfcSegment & seg = segments[s];
  const IfcSide & S = sides[side], & P = sides[1 - side];
  int gf = seg.piece[1 - side].f;
  const IfcFacet & F = facets[side][seg.piece[side].f], & G = facets[1 - side][gf];
  const auto & tri = seg.piece[side].tri;
  const IntegrationRule & ir = SelectIntegrationRule (ET_TRIG, intorder);
  pts.SetSize (tri.size() * ir.Size());
  double gap = gap_tol * hmean, scale = sqrt (G.len) + L2Norm (G.p[0]);
  // G: chart of its image triangle (for the first guess), heights (for the overshoot)
  const Tri & XG = images[1 - side][gf];
  Chart C(XG, UnitNormal (XG));
  Vec<2> c0 = C(XG[0]), d1 = C(XG[1]) - c0, d2 = C(XG[2]) - c0;
  double det = Cross2 (d1, d2), hgt[3];
  for (int a = 0; a < 3; a++) hgt[a] = 2 * G.len / L2Norm (G.p[(a + 2) % 3] - G.p[(a + 1) % 3]);
  auto overshoot = [&] (const Vec<2> & u)        // distance of x'(u) outside G
  {
    double lam[3] = { 1 - u(0) - u(1), u(0), u(1) }, o = 0;
    for (int a = 0; a < 3; a++) o = max2 (o, -lam[a] * hgt[a]);
    return o;
  };
  // the miss of the path line of x'(u): (I - sigma sigma^T)(xbar - x'(u))
  auto miss = [&] (const Vec<3> & xbar, const Vec<2> & u)
  {
    Vec<3> xq = PointAt3 (G, u), sg = UnitSigma3 (P, G, xq, lh), w = xbar - xq;
    return Vec<3> (w - InnerProduct (w, sg) * sg);
  };
  size_t q = 0;
  for (auto & T : tri)
    {
      double jac = fabs (Cross2 (Vec<2> (T[1] - T[0]), Vec<2> (T[2] - T[0])));
      for (size_t k = 0; k < ir.Size(); k++, q++)
        {
          HeapReset hr(lh);
          IfcPoint & pt = pts[q];
          Vec<2> r = T[0] + ir[k](0) * (T[1] - T[0]) + ir[k](1) * (T[2] - T[0]);
          pt.seg = s; pt.el = F.el; pt.loc = F.loc; pt.pel = G.el; pt.ploc = G.loc;
          pt.weight = ir[k].Weight() * jac * 2 * F.len;
          pt.x = PointAt3 (F, r);
          pt.xbar = EvalAt3 (S, F, pt.x, *S.target, lh);
          // the partner on the plane of F': its own path line passes through xbar (Gauss-Newton
          // on the 2 reference coordinates, Jacobian by differences), from the affine guess
          Vec<2> w = C(pt.xbar) - c0, u (Cross2 (w, d2) / det, Cross2 (d1, w) / det);
          Vec<3> res = miss (pt.xbar, u);
          for (int it = 0; it < 30 && L2Norm (res) > 1e-14 * scale; it++)
            {
              Mat<3,2> J;
              for (int l = 0; l < 2; l++)
                {
                  Vec<2> ul = u; ul(l) += 1e-7;
                  Vec<3> dr = miss (pt.xbar, ul) - res;
                  for (int a = 0; a < 3; a++) J(a, l) = 1e7 * dr(a);
                }
              Mat<2,2> JtJ = Trans (J) * J;
              Vec<2> du = -1.0 * (Inv (JtJ) * (Trans (J) * res));
              u += du;
              res = miss (pt.xbar, u);
              if (L2Norm (du) < 1e-15) break;
            }
          if (!(L2Norm (res) <= 1e-11 * scale && overshoot (u) <= gap))
            {
              // D14 absorption (as in 2D): the point of F', extended by the gap (overshoot <= gap),
              // whose path line passes closest to xbar, on a barycentric lattice refined once
              // around its best point; accepted up to the gap
              double ext = gap / min2 (hgt[0], min2 (hgt[1], hgt[2])), rmin = 1e300;
              const int n = 24, m = 6;
              double step = (1 + 3 * ext) / n;
              auto test = [&] (const Vec<2> & ul)
              {
                if (overshoot (ul) > gap) return;
                double rr = L2Norm (miss (pt.xbar, ul));
                if (rr < rmin) { rmin = rr; u = ul; }
              };
              for (int a = 0; a <= n; a++)
                for (int b = 0; a + b <= n; b++)
                  test (Vec<2> (-ext + step * a, -ext + step * b));
              Vec<2> u0 = u;
              for (int a = -m; a <= m; a++)
                for (int b = -m; b <= m; b++)
                  test (Vec<2> (u0(0) + step * a / m, u0(1) + step * b / m));
              if (rmin > gap)
                throw Exception ("pair: unpaired piece of Gamma_h^" + ToString (side + 1) + " (side "
                                 + ToString (side + 1) + ") at " + Pt3 (pt.x) + ": its path ends at " + Pt3 (pt.xbar)
                                 + ", which the paths of the face " + Pt3 (G.p[0]) + " - " + Pt3 (G.p[1]) + " - "
                                 + Pt3 (G.p[2]) + " of the other side miss by " + ToString (rmin)
                                 + " > the gap tolerance " + ToString (gap));
            }
          pt.xp = PointAt3 (G, u);
          // the path line of x' may cross Gamma twice (sheets closer than h): x''s own path
          // must end at xbar, up to the gap
          Vec<3> xbarp = EvalAt3 (P, G, pt.xp, *P.target, lh);
          if (L2Norm (xbarp - pt.xbar) > gap)
            throw Exception ("pair: unpaired piece of Gamma_h^" + ToString (side + 1) + " (side "
                             + ToString (side + 1) + ") at " + Pt3 (pt.x) + ": its path ends at " + Pt3 (pt.xbar)
                             + ", the path of its partner on the face " + Pt3 (G.p[0]) + " - " + Pt3 (G.p[1]) + " - "
                             + Pt3 (G.p[2]) + " of the other side at " + Pt3 (xbarp) + ", "
                             + ToString (L2Norm (xbarp - pt.xbar)) + " away > the gap tolerance " + ToString (gap));
          pt.sigmap = UnitSigma3 (P, G, pt.xp, lh);
          pt.ip = Locate (S.ma->GetTrafo (ElementId(VOL, F.el), lh), pt.x, 3);
          pt.ip.SetFacetNr (F.loc, BND);
          pt.pip = Locate (P.ma->GetTrafo (ElementId(VOL, G.el), lh), pt.xp, 3);
          pt.pip.SetFacetNr (G.loc, BND);
        }
    }
}

py::dict InterfacePieces3D (const Interface & I)
{
  // per segment s and side i: facet, el, loc, area; per triangle k of the pieces: its
  // segment, side, reference and physical vertices
  size_t n = I.NSegments(), m = 0;
  for (size_t s = 0; s < n; s++)
    for (int i = 0; i < 2; i++) m += I.Segment(s).piece[i].tri.size();
  py::array_t<int> facet({n, size_t(2)}), el({n, size_t(2)}), loc({n, size_t(2)}), tseg(m), tside(m);
  py::array_t<double> area({n, size_t(2)}), tref({m, size_t(3), size_t(2)}), tx({m, size_t(3), size_t(3)});
  auto fa = facet.mutable_unchecked<2>(), ea = el.mutable_unchecked<2>(), la = loc.mutable_unchecked<2>();
  auto aa = area.mutable_unchecked<2>();
  auto sa = tseg.mutable_unchecked<1>(), da = tside.mutable_unchecked<1>();
  auto ra = tref.mutable_unchecked<3>(), xa = tx.mutable_unchecked<3>();
  size_t k = 0;
  for (size_t s = 0; s < n; s++)
    for (int i = 0; i < 2; i++)
      {
        auto & P = I.Segment(s).piece[i];
        auto & F = I.Facet (i, P.f);
        fa(s, i) = F.facet; ea(s, i) = F.el; la(s, i) = F.loc; aa(s, i) = P.area;
        for (auto & T : P.tri)
          {
            sa(k) = s; da(k) = i;
            for (int v = 0; v < 3; v++)
              {
                Vec<3> x = PointAt3 (F, T[v]);
                for (int c = 0; c < 2; c++) ra(k, v, c) = T[v](c);
                for (int c = 0; c < 3; c++) xa(k, v, c) = x(c);
              }
            k++;
          }
      }
  py::dict d;
  d["facet"] = facet; d["el"] = el; d["loc"] = loc; d["area"] = area;
  d["tri_seg"] = tseg; d["tri_side"] = tside; d["tri_ref"] = tref; d["tri_x"] = tx;
  return d;
}
