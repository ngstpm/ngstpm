cmake_minimum_required(VERSION 3.16)
project(ngstpm LANGUAGES CXX)

# Locate the CMake configs shipped inside the pip wheels of netgen and ngsolve.
find_package(Python 3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import netgen, os; print(os.path.join(os.path.dirname(netgen.__file__), 'cmake'))"
                OUTPUT_VARIABLE Netgen_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import ngsolve.config as c; print(c.get_cmake_dir())"
                OUTPUT_VARIABLE NGSolve_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(NGSolve CONFIG REQUIRED)

add_ngsolve_python_module(_ngstpm src/shifted.cpp src/extrapolate.cpp src/interface.cpp src/ifcmeasure.cpp src/ifclinear.cpp)
# Find the NGSolve libraries of the installed ngsolve wheel next to us in
# site-packages, not in the temporary build environment.
if(APPLE)
  set_target_properties(_ngstpm PROPERTIES INSTALL_RPATH "@loader_path/../netgen")
elseif(UNIX)
  set_target_properties(_ngstpm PROPERTIES INSTALL_RPATH "$ORIGIN/../netgen_mesher.libs;$ORIGIN/../netgen")
endif()
install(TARGETS _ngstpm DESTINATION ngstpm)
