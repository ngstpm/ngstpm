#ifndef NGSTPM_BINDING_HPP
#define NGSTPM_BINDING_HPP

#include <comp.hpp>

namespace ngstpm
{
  // Set (non-null) only while an interface integral is assembled; null in ordinary NGSolve assembly.
  class InterfaceBinding
  {
  public:
    virtual ~InterfaceBinding () = default;
    // mesh of the integration element K (the side being integrated)
    virtual const ngcomp::MeshAccess & IntegrationMesh () const = 0;
    // mesh of the partner element K' (the other side; the same mesh when both sides share one)
    virtual const ngcomp::MeshAccess & PartnerMesh () const = 0;
    // the side of the integration element: 1 or 2 (I.ds(side=i)); resolves side=1/2 of Extrapolate
    virtual int MeasureSide () const = 0;
    // xbar of mip: the end on Gamma of the integration point's own path. The partner's path is
    // taken to end there too (straight from x' to xbar), so both transfers meet at one point.
    virtual void PathEnd (const ngfem::BaseMappedIntegrationPoint & mip, ngbla::FlatVector<double> xbar) const = 0;
    // the partner point x' of mip, mapped on the partner element K' of the other side (on that side's mesh)
    virtual const ngfem::BaseMappedIntegrationPoint & Partner (const ngfem::BaseMappedIntegrationPoint & mip,
                                                               ngcore::LocalHeap & lh) const = 0;
  };

  inline thread_local InterfaceBinding * interface_binding = nullptr;
}

#endif
