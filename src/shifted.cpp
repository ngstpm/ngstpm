#include <comp.hpp>
#include <python_ngstd.hpp>
#include "extrapolate.hpp"
#include "interface.hpp"
#include "ifcmeasure.hpp"
#include "ifclinear.hpp"
using namespace ngcomp;

PYBIND11_MODULE(_ngstpm, m)
{
  ngstpm::ExportExtrapolate (m);
  ExportInterface (m);
  ExportIfcMeasure (m);
  ExportIfcLinear (m);
}
