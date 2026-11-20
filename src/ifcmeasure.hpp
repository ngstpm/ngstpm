#ifndef NGSTPM_IFCMEASURE_HPP
#define NGSTPM_IFCMEASURE_HPP

#include <python_ngstd.hpp>

// The interface measure I.ds(side=i) (DifferentialSymbol/Integral subclasses)
// and the static condensation of the interface coupling.
void ExportIfcMeasure (pybind11::module & m);

#endif
