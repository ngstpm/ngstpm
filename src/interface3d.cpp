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

  // subdivision of the faces (Build3D): m_F in 1, 2, 4, .., MMAX; the path map resolved to
  // RESOLVE gap; points of F at (A, B) / U in its reference coordinates, U a multiple of 2 MMAX
  // and 3 MMAX (the nodes, edge midpoints and centroids of the subtriangles are integer)
  const int MMAX = 16, U = 96;
  const double RESOLVE = 0.2;
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
        for (int a = 0; a < 3; a++) images[i][f][a] = image[i][pv[i][f][a]];
    }

  // Subdivision: the pieces are computed from flat images. Where the path map is not affine on a
  // face up to the gap (carved sides: paths of length O(h)), the face f is split into m_f^2
  // congruent subtriangles, m_f in 1, 2, 4, .., MMAX, uniformly in its reference coordinates; its
  // points are (A, B) / U. The image of a node: at F's vertices the mean above; on an edge e of
  // Gamma_h^i, at the multiples of 1/m_e (m_e the smallest m_f of the faces at e), the mean of
  // the faces' path ends, and linear in between (the hanging nodes of a finer face); inside F,
  // F's path end. So the flat images of the subtriangles form a conforming surface S_i (for
  // m = 1: the T_F). m_f: the smallest m for which, on each subtriangle, the path map of F
  // deviates from the affine interpolant of F's own path ends at the nodes by at most RESOLVE gap
  // across the paths (normal to sigma), at the centroid and the edge midpoints. Then on each edge
  // with a coarser face, the linear images of the finer face's hanging nodes (and of the
  // midpoints between them) must be within RESOLVE gap of the mean path end there, else the
  // coarser face's m is doubled (to a fixed point). The means at vertices and edges differ from a
  // face's own path ends by the spread of element-wise paths, which no m reduces: that is left to
  // the absorption (D14), as for m = 1.
  Array<std::map<std::pair<int,int>,Vec<3>>> xbc[2], sgc[2];   // per face: path end, unit sigma at (A, B)
  for (int i = 0; i < 2; i++) { xbc[i].SetSize (facets[i].Size()); sgc[i].SetSize (facets[i].Size()); }
  auto point = [&] (int i, int f, int A, int B) -> Vec<3>     // (A, B) of F; its vertices exactly
  {
    const IfcFacet & F = facets[i][f];
    if (A == 0 && B == 0) return F.p[0];
    if (A == U) return F.p[1];
    if (B == U) return F.p[2];
    return PointAt3 (F, Vec<2> (double (A) / U, double (B) / U));
  };
  auto xbar = [&] (int i, int f, int A, int B) -> Vec<3>
  {
    auto [it, nw] = xbc[i][f].try_emplace (std::make_pair (A, B));
    if (nw) it->second = EvalAt3 (sides[i], facets[i][f], point (i, f, A, B), *sides[i].target, lh);
    return it->second;
  };
  auto sig = [&] (int i, int f, int A, int B) -> Vec<3>
  {
    auto [it, nw] = sgc[i][f].try_emplace (std::make_pair (A, B));
    if (nw) it->second = UnitSigma3 (sides[i], facets[i][f], point (i, f, A, B), lh);
    return it->second;
  };
  // the point k / U of the way from va to vb (va < vb, an edge of the face g) in g's coordinates
  auto onedge = [&] (int i, int g, int va, int vb, int k) -> std::array<int,2>
  {
    int l[3];
    for (int a = 0; a < 3; a++) l[a] = pv[i][g][a] == va ? U - k : pv[i][g][a] == vb ? k : 0;
    return { l[1], l[2] };
  };
  // whether (A, B) of F is on an edge (not a vertex): the edge (va < vb), k / U of the way from va
  auto edgeof = [&] (int i, int f, int A, int B, int & va, int & vb, int & k)
  {
    int l[3] = { U - A - B, A, B }, z = -1, nz = 0;
    for (int a = 0; a < 3; a++) if (l[a] == 0) { z = a; nz++; }
    if (nz != 1) return false;
    int a1 = (z + 1) % 3, a2 = (z + 2) % 3;
    if (pv[i][f][a1] > pv[i][f][a2]) std::swap (a1, a2);
    va = pv[i][f][a1]; vb = pv[i][f][a2]; k = l[a2];
    return true;
  };
  // the image of the node k / U of the way from va to vb, k a multiple of U / m_e: the mean of
  // the path ends of the faces at the edge
  auto edgenode = [&] (int i, int va, int vb, int k) -> Vec<3>
  {
    if (k == 0) return image[i][va];
    if (k == U) return image[i][vb];
    const Array<int> & fs = efac[i][{va, vb}];
    Vec<3> X = 0.0;
    for (int g : fs) { auto c = onedge (i, g, va, vb, k); X += xbar (i, g, c[0], c[1]); }
    return (1.0 / fs.Size()) * X;
  };
  // the image of the node (A, B) of F on S_i
  auto nodeimg = [&] (int i, int f, int A, int B) -> Vec<3>
  {
    if (A == 0 && B == 0) return image[i][pv[i][f][0]];
    if (A == U) return image[i][pv[i][f][1]];
    if (B == U) return image[i][pv[i][f][2]];
    int va, vb, k;
    if (!edgeof (i, f, A, B, va, vb, k)) return xbar (i, f, A, B);
    int me = MMAX;
    for (int g : efac[i][{va, vb}]) me = min2 (me, nsub[i][g]);
    int st = U / me, k0 = k - k % st;
    if (k0 == k) return edgenode (i, va, vb, k);
    Vec<3> X0 = edgenode (i, va, vb, k0), X1 = edgenode (i, va, vb, k0 + st);
    return X0 + (double (k - k0) / st) * (X1 - X0);
  };
  // the subtriangles of F at the level m, by their nodes (A, B), oriented as F
  auto subtri = [] (int m)
  {
    Array<std::array<std::array<int,2>,3>> T;
    int s = U / m;
    for (int A = 0; A < U; A += s)
      for (int B = 0; A + B < U; B += s)
        {
          T.Append ({{ {A, B}, {A + s, B}, {A, B + s} }});
          if (A + B + 2 * s <= U) T.Append ({{ {A + s, B}, {A + s, B + s}, {A, B + s} }});
        }
    return T;
  };
  auto across = [] (const Vec<3> & d, const Vec<3> & s) { return L2Norm (Vec<3> (d - InnerProduct (d, s) * s)); };
  auto deviation = [&] (int i, int f, int m)
  {
    double dev = 0;
    for (auto & T : subtri (m))
      {
        Vec<3> X[3];
        for (int a = 0; a < 3; a++) X[a] = xbar (i, f, T[a][0], T[a][1]);
        for (int c = 0; c < 4; c++)                     // the edge midpoints (c < 3) and the centroid
          {
            double w[3] = { 0, 0, 0 };
            if (c < 3) w[c] = w[(c + 1) % 3] = 0.5; else w[0] = w[1] = w[2] = 1.0 / 3;
            int A = 0, B = 0;
            for (int a = 0; a < 3; a++)
              if (w[a] > 0) { A += T[a][0]; B += T[a][1]; }
            A /= c < 3 ? 2 : 3; B /= c < 3 ? 2 : 3;
            Vec<3> d = xbar (i, f, A, B) - (w[0] * X[0] + w[1] * X[1] + w[2] * X[2]);
            dev = max2 (dev, across (d, sig (i, f, A, B)));
          }
      }
    return dev;
  };
  double resolved = RESOLVE * gap;
  for (int i = 0; i < 2; i++)
    {
      nsub[i].SetSize (facets[i].Size());
      for (int f = 0; f < facets[i].Size(); f++)
        {
          int m = 1;
          for (double dev; (dev = deviation (i, f, m)) > resolved; m *= 2)
            if (m == MMAX)
              {
                const IfcFacet & F = facets[i][f];
                throw Exception ("pair: the path map of the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - "
                                 + Pt3 (F.p[2]) + " of Gamma_h^" + ToString(i+1) + " is not resolved by "
                                 + ToString (MMAX) + "^2 subtriangles (deviation " + ToString (dev) + " > "
                                 + ToString (resolved) + " across the paths; see pair: the conditions in 3D)");
              }
          nsub[i][f] = m;
        }
      // the hanging nodes of the finer face on an edge, and the midpoints between them
      for (bool changed = true; changed; )
        {
          changed = false;
          for (auto & [e, fs] : efac[i])
            {
              if (fs.Size() != 2 || nsub[i][fs[0]] == nsub[i][fs[1]]) continue;
              int fc = nsub[i][fs[0]] < nsub[i][fs[1]] ? fs[0] : fs[1], ff = fs[0] + fs[1] - fc;
              int st = U / nsub[i][fc];
              double dev = 0;
              for (int k = 0; k < U; k += U / (2 * nsub[i][ff]))
                {
                  if (k % st == 0) continue;
                  int k0 = k - k % st;
                  Vec<3> X0 = edgenode (i, e.first, e.second, k0), X1 = edgenode (i, e.first, e.second, k0 + st);
                  auto c = onedge (i, ff, e.first, e.second, k);
                  dev = max2 (dev, across (Vec<3> (edgenode (i, e.first, e.second, k) - X0
                                                   - (double (k - k0) / st) * (X1 - X0)), sig (i, ff, c[0], c[1])));
                }
              if (dev > resolved) { nsub[i][fc] *= 2; changed = true; }
            }
        }
    }

  // the subtriangles; their nodes: the vertices of Gamma_h^i (numbered as in the mesh), the nodes
  // on its edges and inside the faces (numbered from nv on), with their images in image[i]
  struct Sub { int f; int v[3]; std::array<int,2> c[3]; Vec<2> r[3]; Tri X; };   // face, nodes, (A, B), ref., images
  Array<Sub> subs[2];
  Array<int> sfirst[2];                                 // face -> its first subtriangle (and the end)
  std::map<std::array<int,4>,int> nodeid[2];
  auto nid = [&] (int i, int f, int A, int B)
  {
    if (A == 0 && B == 0) return pv[i][f][0];
    if (A == U) return pv[i][f][1];
    if (B == U) return pv[i][f][2];
    int va, vb, k;
    std::array<int,4> key = edgeof (i, f, A, B, va, vb, k) ? std::array<int,4> { 0, va, vb, k }
                                                           : std::array<int,4> { 1, f, A, B };
    return nodeid[i].try_emplace (key, int (sides[i].ma->GetNV() + nodeid[i].size())).first->second;
  };
  for (int i = 0; i < 2; i++)
    for (int f = 0; f <= facets[i].Size(); f++)
      {
        sfirst[i].Append (subs[i].Size());
        if (f == facets[i].Size()) break;
        for (auto & T : subtri (nsub[i][f]))
          {
            Sub t;
            t.f = f;
            for (int a = 0; a < 3; a++)
              {
                auto [A, B] = T[a];
                t.v[a] = nid (i, f, A, B);
                t.c[a] = T[a];
                t.r[a] = Vec<2> (double (A) / U, double (B) / U);
                auto it = image[i].find (t.v[a]);
                if (it == image[i].end()) it = image[i].emplace (t.v[a], nodeimg (i, f, A, B)).first;
                t.X[a] = it->second;
              }
            auto & X = t.X;
            if (0.5 * L2Norm (Cross (Vec<3> (X[1] - X[0]), Vec<3> (X[2] - X[0]))) < marea)
              throw Exception ("pair: the paths of the face " + Pt3 (facets[i][f].p[0]) + " - " + Pt3 (facets[i][f].p[1])
                               + " - " + Pt3 (facets[i][f].p[2]) + " of Gamma_h^" + ToString(i+1)
                               + " end closer than the merge tolerance");
            subs[i].Append (t);
          }
      }

  // rims of S_i: the subtriangle edges on the edges of Gamma_h^i in one face; at a rim node, the
  // direction of the rays bounding the strips: the normalised sum of the outward conormals (in
  // the image subtriangle) of its rim edges
  struct Rim { int t, a, b; };                          // subtriangle, the edge's nodes (a < b)
  Array<Rim> rims[2];
  std::map<int,Vec<3>> rimdir[2];
  for (int i = 0; i < 2; i++)
    {
      for (auto & [e, fs] : efac[i])
        {
          if (fs.Size() != 1) continue;
          int f = fs[0], st = U / nsub[i][f];
          for (int k = 0; k < U; k += st)
            {
              auto c0 = onedge (i, f, e.first, e.second, k), c1 = onedge (i, f, e.first, e.second, k + st);
              int na = nid (i, f, c0[0], c0[1]), nb = nid (i, f, c1[0], c1[1]);
              auto has = [] (const Sub & s, int v) { return s.v[0] == v || s.v[1] == v || s.v[2] == v; };
              int t = sfirst[i][f];
              while (!(has (subs[i][t], na) && has (subs[i][t], nb))) t++;
              Rim R { t, min2 (na, nb), max2 (na, nb) };
              rims[i].Append (R);
              int q = 0;
              while (subs[i][t].v[q] == R.a || subs[i][t].v[q] == R.b) q++;
              Vec<3> Xa = image[i][R.a], u = image[i][R.b] - Xa, w = Xa - subs[i][t].X[q];
              u /= L2Norm (u);
              Vec<3> nu = w - InnerProduct (w, u) * u;
              nu /= L2Norm (nu);
              for (int v : { R.a, R.b })
                {
                  if (!rimdir[i].count(v)) rimdir[i][v] = 0.0;
                  rimdir[i][v] += nu;
                }
            }
        }
      for (auto & [v, d] : rimdir[i]) d /= L2Norm (d);
    }
  Array<Array<int>> frims[2];                           // subtriangle -> its rim edges
  for (int i = 0; i < 2; i++)
    {
      frims[i].SetSize (subs[i].Size());
      for (int r = 0; r < rims[i].Size(); r++) frims[i][rims[i][r].t].Append (r);
    }

  // bounding spheres of the image subtriangles; their unit normals, oriented as the outward
  // normal of the owner element on F (the vertex order of F carried over); the charts' normals
  Array<Vec<3>> ctr[2], nimg[2], nrm[2]; Array<double> rad[2];
  for (int i = 0; i < 2; i++)
    {
      Array<Vec<3>> nout;
      for (auto & F : facets[i]) nout.Append (OutwardNormal3 (sides[i], F));
      for (int t = 0; t < subs[i].Size(); t++)
        {
          const Tri & X = subs[i][t].X;
          int f = subs[i][t].f;
          const IfcFacet & F = facets[i][f];
          Vec<3> c = (1.0 / 3) * (X[0] + X[1] + X[2]);
          ctr[i].Append (c);
          rad[i].Append (max2 (L2Norm (X[0] - c), max2 (L2Norm (X[1] - c), L2Norm (X[2] - c))));
          Vec<3> nF = Cross (Vec<3> (F.p[1] - F.p[0]), Vec<3> (F.p[2] - F.p[0]));
          nimg[i].Append (InnerProduct (nF, OutwardNormal3 (sides[i], F)) > 0 ? UnitNormal (X) : Vec<3> (-1.0 * UnitNormal (X)));
          // the chart: normal to the mean path direction at the vertices of t (projecting S_j along
          // the paths), oriented as t's image normal; the image (possibly a sliver far off the
          // tangent plane) only gives the affine map to F
          auto & cs = subs[i][t].c;
          Vec<3> n0 = 0.0, s0 = sig (i, f, cs[0][0], cs[0][1]);
          for (int a = 0; a < 3; a++)
            {
              Vec<3> sa = sig (i, f, cs[a][0], cs[a][1]);
              n0 += InnerProduct (sa, s0) < 0 ? Vec<3> (-1.0 * sa) : sa;
            }
          n0 /= L2Norm (n0);
          // a fold: the image of t is oriented opposite to F along the paths
          double cf = InnerProduct (n0, nout[f]), ct = InnerProduct (n0, nimg[i][t]);
          if (cf * ct < 0)
            throw Exception ("pair: the paths of Gamma_h^" + ToString(i+1) + " fold at the face " + Pt3 (F.p[0])
                             + " - " + Pt3 (F.p[1]) + " - " + Pt3 (F.p[2]) + ": the image of its subtriangle at "
                             + Pt3 (PointAt3 (F, (1.0 / 3) * (subs[i][t].r[0] + subs[i][t].r[1] + subs[i][t].r[2])))
                             + " (of " + ToString (nsub[i][f] * nsub[i][f]) + ") is oriented opposite to the face "
                             "along the paths (cosines with the mean path direction: face " + ToString (cf)
                             + ", image " + ToString (ct) + "; see pair: the conditions in 3D)");
          if (InnerProduct (n0, nimg[i][t]) < 0) n0 *= -1.0;
          if (InnerProduct (n0, nimg[i][t]) < 1e-3)
            throw Exception ("pair: the path ends of the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - "
                             + Pt3 (F.p[2]) + " of Gamma_h^" + ToString(i+1) + " span a triangle (nearly) parallel "
                             "to its paths (cosine " + ToString (InnerProduct (n0, nimg[i][t])) + "): no chart");
          nrm[i].Append (n0);
        }
    }

  // pieces: for each subtriangle t of a face F of side i, clip in the chart of t against the
  // subtriangles of side j; a polygon in the subtriangle t' of F' belongs to the segment (F, F')
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
  // candidates: the image subtriangles of each side binned in a uniform grid of cells of width
  // 2 rmax + 2 gap (rmax: the largest bounding radius of both sides), so that the bounding-sphere
  // test below can only pass for subtriangles in the 27 cells around the centre's cell
  double rmax = 0;
  for (int i = 0; i < 2; i++) for (double r : rad[i]) rmax = max2 (rmax, r);
  double cell = 2 * rmax + 2 * gap;
  auto cellof = [&] (const Vec<3> & c)
  { return std::array<int,3> { int (floor (c(0) / cell)), int (floor (c(1) / cell)), int (floor (c(2) / cell)) }; };
  std::map<std::array<int,3>,Array<int>> grid[2];
  for (int i = 0; i < 2; i++)
    for (int t = 0; t < ctr[i].Size(); t++) grid[i][cellof (ctr[i][t])].Append (t);
  for (int i = 0; i < 2; i++)
    {
      int j = 1 - i;
      for (int f = 0; f < facets[i].Size(); f++)
        {
          const IfcFacet & F = facets[i][f];
          int ncand = 0, nnear = 0;
          double covered = 0;
          for (int t = sfirst[i][f]; t < sfirst[i][f + 1]; t++)
            {
              const Sub & T = subs[i][t];
              const Tri & XF = T.X;
              const Vec<3> & n0 = nrm[i][t];
              Chart P(XF, n0);
              Vec<2> c[3];
              for (int a = 0; a < 3; a++) c[a] = P(XF[a]);
              // chart -> t's reference coordinates: c0 + xi (c1 - c0) + eta (c2 - c0) -> F's
              Vec<2> d1 = c[1] - c[0], d2 = c[2] - c[0];
              double det = Cross2 (d1, d2);
              auto toref = [&] (const Vec<2> & q)
              {
                Vec<2> w = q - c[0];
                double xi = Cross2 (w, d2) / det, eta = Cross2 (d1, w) / det;
                return Vec<2> (T.r[0] + xi * (T.r[1] - T.r[0]) + eta * (T.r[2] - T.r[0]));
              };
              Array<Line> lines;
              for (int a = 0; a < 3; a++)                          // the edges of t's image
                {
                  int u = a, w = (a + 1) % 3;
                  if (T.v[u] > T.v[w]) std::swap (u, w);
                  Line L; L.O = c[u]; L.E = c[w]; L.tf = a;
                  lines.Append (L);
                }
              // a line of S_j through the images of its nodes v, w, canonically ordered
              auto sjline = [&] (int v, int w)
              {
                Line L;
                if (v > w) std::swap (v, w);
                L.O = P(image[j][v]); L.E = P(image[j][w]); L.va = v; L.vb = w;
                return L;
              };
              auto add = [&] (int g, Array<Vec<2>> & pts)
              {
                if (pts.Size() < 3) return;
                Array<Vec<2>> r;
                for (auto & q : pts) r.Append (toref (q));
                double a = 0;
                for (size_t k = 1; k + 1 < r.Size(); k++) a += 0.5 * Cross2 (r[k] - r[0], r[k + 1] - r[0]);
                a = fabs (a) * 2 * F.len;                          // physical area of the polygon
                if (a < marea) return;                             // a sliver below the merge tolerance
                IfcPiece & pc = piece (i, f, subs[j][g].f);
                for (size_t k = 1; k + 1 < r.Size(); k++) pc.tri.push_back ({ r[0], r[k], r[k + 1] });
                pc.area += a;
                covered += a;
              };
              // the subtriangles of side j in the cells around t's image, in increasing order
              Array<int> cand;
              auto c0 = cellof (ctr[i][t]);
              for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++) for (int d = -1; d <= 1; d++)
                {
                  auto it = grid[j].find ({ c0[0] + a, c0[1] + b, c0[2] + d });
                  if (it != grid[j].end()) for (int g : it->second) cand.Append (g);
                }
              std::sort (cand.begin(), cand.end());
              for (int g : cand)
                {
                  if (L2Norm (ctr[i][t] - ctr[j][g]) > rad[i][t] + rad[j][g] + 2 * gap) continue;
                  nnear++;
                  const Tri & XG = subs[j][g].X;
                  const int * vg = subs[j][g].v;
                  // the two sides lie on either side of Gamma: paired faces have (oriented) image
                  // normals on opposite sides of the chart; this tells apart sheets of Gamma closer
                  // than h (no margin: a sliver image may be nearly parallel to the paths)
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
                      int va = vg[ord[a]], vb = vg[ord[(a + 1) % 3]];
                      lines.Append (sjline (va, vb));
                      Clip (pts, lab, lines, lines.Size() - 1, va < vb ? 1.0 : -1.0);
                    }
                  add (g, pts);
                  // the strips beyond the rim edges of t'
                  for (int r : frims[j][g])
                    {
                      ncand++;
                      int va = rims[j][r].a, vb = rims[j][r].b;
                      int ka = 0; while (vg[ord[ka]] != va) ka++;
                      bool ccw = vg[ord[(ka + 1) % 3]] == vb;       // A -> B = va -> vb counter-clockwise
                      lines.SetSize (3);
                      pts.SetSize0 (); lab.SetSize0 ();
                      for (int a = 0; a < 3; a++) { pts.Append (c[a]); lab.Append (a); }
                      Line Lb = sjline (va, vb);                    // O = va, E = vb
                      lines.Append (Lb);
                      Clip (pts, lab, lines, 3, ccw ? -1.0 : 1.0);  // outside t'
                      Vec<2> tv = Lb.E - Lb.O, nout = ccw ? Vec<2> (tv(1), -tv(0)) : Vec<2> (-tv(1), tv(0));
                      nout /= L2Norm (nout);
                      Line Lo; Lo.O = Lb.O + gap * nout; Lo.E = Lb.E + gap * nout;
                      lines.Append (Lo);                            // depth gap: on the side of Lb
                      Clip (pts, lab, lines, 4, G(Lo, Lb.O) >= 0 ? 1.0 : -1.0);
                      for (int v : { va, vb })                      // the rays at the rim nodes
                        {
                          Line R; R.O = P(image[j][v]); R.E = R.O + P.Lin (rimdir[j][v]); R.va = v;
                          lines.Append (R);
                          Vec<2> other = v == va ? Lb.E : Lb.O;
                          if (pts.Size()) Clip (pts, lab, lines, lines.Size() - 1, G(R, other) >= 0 ? 1.0 : -1.0);
                        }
                      add (g, pts);
                    }
                }
            }
          // coverage: the pieces tile F, up to the slivers dropped (each below marea)
          double tol = (ncand + 1) * marea + 1e-12 * F.len;
          if (covered < F.len - tol)
            throw Exception ("pair: unpaired piece of Gamma_h^" + ToString(i+1) + " (side " + ToString(i+1)
                             + ") on the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - " + Pt3 (F.p[2])
                             + ", uncovered area " + ToString (F.len - covered) + " of " + ToString (F.len)
                             + (ncand == 0 && nnear > 0
                                ? ": the faces of side " + ToString(j+1) + " near its path ends are all oriented as "
                                  "it, not on the other side of Gamma (see pair: the conditions in 3D)"
                                : ": no path of side " + ToString(j+1) + " ends on its part of Gamma, or the paths "
                                  "of a side fold there (see pair: the conditions in 3D)"));
          if (covered > F.len + tol)
            {
              const Tri & XF = images[i][f];
              throw Exception ("pair: the pieces of the face " + Pt3 (F.p[0]) + " - " + Pt3 (F.p[1]) + " - " + Pt3 (F.p[2])
                               + " of Gamma_h^" + ToString(i+1) + " (side " + ToString(i+1) + ") cover "
                               + ToString (covered) + " of its area " + ToString (F.len) + ": near "
                               + Pt3 (Vec<3> ((1.0 / 3) * (XF[0] + XF[1] + XF[2])))
                               + " the paths of side " + ToString(j+1) + " overlap on Gamma, or the paths of a side "
                               "fold there (see pair: the conditions in 3D)");
            }
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
  // per segment s and side i: facet, el, loc, area, m (the face's subdivision); per triangle
  // k of the pieces: its segment, side, reference and physical vertices
  size_t n = I.NSegments(), m = 0;
  for (size_t s = 0; s < n; s++)
    for (int i = 0; i < 2; i++) m += I.Segment(s).piece[i].tri.size();
  py::array_t<int> facet({n, size_t(2)}), el({n, size_t(2)}), loc({n, size_t(2)}), msub({n, size_t(2)}), tseg(m), tside(m);
  py::array_t<double> area({n, size_t(2)}), tref({m, size_t(3), size_t(2)}), tx({m, size_t(3), size_t(3)});
  auto fa = facet.mutable_unchecked<2>(), ea = el.mutable_unchecked<2>(), la = loc.mutable_unchecked<2>();
  auto aa = area.mutable_unchecked<2>(); auto ma = msub.mutable_unchecked<2>();
  auto sa = tseg.mutable_unchecked<1>(), da = tside.mutable_unchecked<1>();
  auto ra = tref.mutable_unchecked<3>(), xa = tx.mutable_unchecked<3>();
  size_t k = 0;
  for (size_t s = 0; s < n; s++)
    for (int i = 0; i < 2; i++)
      {
        auto & P = I.Segment(s).piece[i];
        auto & F = I.Facet (i, P.f);
        fa(s, i) = F.facet; ea(s, i) = F.el; la(s, i) = F.loc; aa(s, i) = P.area; ma(s, i) = I.NSub (i, P.f);
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
  d["facet"] = facet; d["el"] = el; d["loc"] = loc; d["area"] = area; d["m"] = msub;
  d["tri_seg"] = tseg; d["tri_side"] = tside; d["tri_ref"] = tref; d["tri_x"] = tx;
  return d;
}
