cmake_minimum_required(VERSION 3.22)
project(pio_bench LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "PIC-like MPI-IO/HDF5/ADIOS2 checkpoint benchmark (modern C++)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

# Probe the MPI_Type_create_subarray index width: MPI-3 says MPI_Aint,
# older Intel MPI / MPICH / OMPI declare const int[]. Wrong width passed
# across is silent little-endian corruption, so detect, never assume.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
#include <mpi.h>
int main() {
  MPI_Aint s[1] = {4}, ss[1] = {2}, st[1] = {0};
  MPI_Datatype t;
  return MPI_Type_create_subarray(1, s, ss, st, MPI_ORDER_C, MPI_DOUBLE, &t) == 0;
}" PIO_MPI_SUBARRAY_AINT)
if(PIO_MPI_SUBARRAY_AINT)
  message(STATUS "pio_bench: MPI subarray uses MPI_Aint extents (MPI-3+)")
else()
  message(STATUS "pio_bench: MPI subarray uses int extents (pre-MPI-3 signature)")
endif()

add_executable(pio_bench src/pio_bench.cpp)
target_include_directories(pio_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pio_bench PRIVATE MPI::MPI_CXX)
target_compile_options(pio_bench PRIVATE -Wall -Wextra)

# ---- optional backends (compile-time availability, cf. pio/backends.hpp) --
option(PIO_WITH_HDF5  "Build the parallel HDF5 backend"  ON)
option(PIO_WITH_ADIOS2 "Build the ADIOS2 (BP5) backend"  ON)

if(PIO_WITH_HDF5)
  # Manual override for exotic/Debian-flavoured installs (e.g.
  # /usr/lib/<triplet>/libhdf5_openmpi.so + .../hdf5/openmpi/include) that
  # FindHDF5 cannot discover or classify -- the libraries MUST be parallel.
  set(PIO_HDF5_LIBRARIES "" CACHE STRING
      "Explicit parallel-HDF5 libraries (bypasses find_package)")
  set(PIO_HDF5_INCLUDE_DIRS "" CACHE STRING
      "Explicit HDF5 include dirs when using PIO_HDF5_LIBRARIES")

  if(PIO_HDF5_LIBRARIES)
    target_compile_definitions(pio_bench PRIVATE PIO_HAVE_HDF5=1)
    target_include_directories(pio_bench PRIVATE ${PIO_HDF5_INCLUDE_DIRS})
    target_link_libraries(pio_bench PRIVATE ${PIO_HDF5_LIBRARIES})
    message(STATUS "pio_bench: HDF5 backend ON (manual libs, assumed parallel; "
                   "H5_HAVE_PARALLEL compile guard still protects the guess)")
  else()
    # FindHDF5 sets HDF5_IS_PARALLEL; a serial HDF5 is USELESS here
    # (no H5Pset_fapl_mpio) -- gate on it explicitly so the failure mode
    # is "backend compiled out", not "mysterious link error".
    find_package(HDF5 COMPONENTS C)
    if(HDF5_FOUND AND HDF5_IS_PARALLEL)
      target_compile_definitions(pio_bench PRIVATE PIO_HAVE_HDF5=1)
      target_include_directories(pio_bench PRIVATE ${HDF5_C_INCLUDE_DIRS})
      target_link_libraries(pio_bench PRIVATE ${HDF5_C_LIBRARIES})
      message(STATUS "pio_bench: HDF5 backend ON")
    elseif(HDF5_FOUND)
      message(STATUS "pio_bench: HDF5 found but SERIAL -- backend compiled OUT "
                     "(install a parallel HDF5, point HDF5_ROOT to one, or set "
                     "PIO_HDF5_LIBRARIES)")
    else()
      message(STATUS "pio_bench: HDF5 not found -- backend compiled OUT")
    endif()
  endif()
endif()

if(PIO_WITH_ADIOS2)
  find_package(ADIOS2 CONFIG QUIET)
  if(ADIOS2_FOUND)
    target_compile_definitions(pio_bench PRIVATE PIO_HAVE_ADIOS2=1)
    target_link_libraries(pio_bench PRIVATE adios2::cxx11)
    message(STATUS "pio_bench: ADIOS2 backend ON")
  else()
    message(STATUS "pio_bench: ADIOS2 not found -- backend compiled OUT")
  endif()
endif()

install(TARGETS pio_bench RUNTIME DESTINATION bin)
