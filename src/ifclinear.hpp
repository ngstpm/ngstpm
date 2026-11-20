#ifndef NGSTPM_IFCLINEAR_HPP
#define NGSTPM_IFCLINEAR_HPP

// The local parts of an interface measure I.ds(side): linear forms and Integrate,
// on the same points and weights as the bilinear segment group (the piece rules
// of the common subdivision, ifcbinding.hpp), with the binding set.

#include <comp.hpp>
#include <python_ngstd.hpp>
#include "interface.hpp"

namespace ngstpm
{
  // LinearFormIntegrator of cf (linear in test proxies, no trial proxies) over the
  // pieces of Gamma_h^side (side 0, 1). Element K of the linear form's mesh (which
  // must be I.Mesh(side)) gets the pieces owned by K, each with the order
  // 2 * fel.Order() + bonus_intorder rule (fel: the test space's element of K).
  std::shared_ptr<ngfem::LinearFormIntegrator>
  MakeInterfaceLFI (std::shared_ptr<Interface> I, int side, std::shared_ptr<ngfem::CoefficientFunction> cf,
                    int bonus_intorder);

  // int cf over the pieces of Gamma_h^side (side 0, 1); cf scalar, no proxies.
  // Rule of order 5 + bonus_intorder (5: NGSolve's default order of Integrate).
  double IntegrateInterface (const Interface & I, int side, ngfem::CoefficientFunction & cf,
                             int bonus_intorder, ngcore::LocalHeap & lh);
}

void ExportIfcLinear (pybind11::module & m);

#endif
