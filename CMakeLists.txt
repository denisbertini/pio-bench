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

# Detection, not a pass/fail test: MPI-3 declares the subarray extents as
# MPI_Aint, but Open MPI (all 4.x/5.x) still ships the `const int[]`
# signature. Passing the wrong width is silent little-endian corruption,
# so we probe the actual headers and configure pio-bench accordingly --
# and hand the result to the compiler via PIO_MPI_SUBARRAY_AINT below.
# (Probe is phrased against the int signature so Open MPI reports "Passed".)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
check_cxx_source_compiles("
#include <mpi.h>
int main() {
  int s[1] = {4}, ss[1] = {2}, st[1] = {0};
  MPI_Datatype t;
  return MPI_Type_create_subarray(1, s, ss, st, MPI_ORDER_C, MPI_DOUBLE, &t) == 0;
}" PIO_MPI_SUBARRAY_INT)
if(PIO_MPI_SUBARRAY_INT)
  message(STATUS "pio_bench: subarray extents = int (Open MPI convention) -- detected, handled")
else()
  message(STATUS "pio_bench: subarray extents = MPI_Aint (MPI-3 conformant) -- detected, handled")
endif()

add_executable(pio_bench src/pio_bench.cpp)
target_include_directories(pio_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pio_bench PRIVATE MPI::MPI_CXX)
target_compile_options(pio_bench PRIVATE -Wall -Wextra)
if(NOT PIO_MPI_SUBARRAY_INT)
  # (applied here, not at probe time: the target does not exist yet there)
  target_compile_definitions(pio_bench PRIVATE PIO_MPI_SUBARRAY_AINT=1)
endif()

# ---- optional backends (compile-time availability, cf. pio/backends.hpp) --
option(PIO_WITH_HDF5  "Build the parallel HDF5 backend"  ON)
option(PIO_WITH_ADIOS2 "Build the ADIOS2 (BP5) backend"  ON)
option(PIO_WITH_OPENPMD "Build the openPMD-api backends (pmd_hdf5/pmd_adios2)" ON)

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
    # ADIOS2 >= 2.12 split the MPI API into its OWN target (adios2::cxx_mpi):
    # only it carries the ADIOS2_USE_MPI interface macro that #if-guards
    # ADIOS(MPI_Comm) and Open(..., comm). The plain cxx target therefore
    # LOOKS serial even on a perfectly MPI-enabled install -- probe order
    # matters, and the compile test below is still the final judge.
    if(TARGET adios2::cxx_mpi)
      set(PIO_ADIOS2_TARGET adios2::cxx_mpi)
    elseif(TARGET adios2::cxx11)
      set(PIO_ADIOS2_TARGET adios2::cxx11)
    elseif(TARGET adios2::cxx)
      set(PIO_ADIOS2_TARGET adios2::cxx)
    else()
      set(PIO_ADIOS2_TARGET "")
    endif()
    if(NOT PIO_ADIOS2_TARGET)
      message(STATUS "pio_bench: ADIOS2 package found but no C++ target -- backend compiled OUT")
    else()
      # ADIOS2's MPI bindings are a compile-time option (ADIOS2_USE_MPI).
      # A serial-built ADIOS2 has NO communicator anywhere in its C++ API
      # and cannot merge multi-rank output -- probe it, don't let the
      # compiler discover it via header errors mid-build.
      include(CheckCXXSourceCompiles)
      set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX ${PIO_ADIOS2_TARGET})
      check_cxx_source_compiles("
#include <adios2.h>
#include <mpi.h>
int main() {
  adios2::ADIOS a(MPI_COMM_WORLD);
  adios2::IO io = a.DeclareIO(\"probe\");
  (void)io;
  return 0;
}" PIO_ADIOS2_HAS_MPI)
      set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)  # restore
      if(PIO_ADIOS2_HAS_MPI)
        target_compile_definitions(pio_bench PRIVATE PIO_HAVE_ADIOS2=1)
        target_link_libraries(pio_bench PRIVATE ${PIO_ADIOS2_TARGET})
        message(STATUS "pio_bench: ADIOS2 backend ON (MPI-enabled build detected, "
                       "target ${PIO_ADIOS2_TARGET})")
      else()
        message(STATUS "pio_bench: ADIOS2 found (${PIO_ADIOS2_TARGET}) but WITHOUT usable "
                       "MPI bindings -- backend compiled OUT (this ADIOS2 cannot "
                       "merge multi-rank output; rebuild it with -DADIOS2_USE_MPI=ON)")
      endif()
    endif()
  else()
    message(STATUS "pio_bench: ADIOS2 not found -- backend compiled OUT")
  endif()
endif()

if(PIO_WITH_OPENPMD)
  find_package(openPMD CONFIG QUIET)
  if(openPMD_FOUND AND TARGET openPMD::openPMD)
    # API archaeology: current openPMD (>=0.15, and the image's git build)
    # constructs Series as (path, access, comm, options-JSON) and selects
    # the backend via {"backend": "..."}; the positional engine-string ctor
    # only exists in old releases. Probe BOTH forms -- and make the probe
    # exercise the exact call surface the backend uses (scalar record vs
    # component storeChunk, void* store, typed shared_ptr load), so a
    # "compiled OUT" verdict means the build is truly incompatible, not
    # that one guessed signature missed.
    include(CheckCXXSourceCompiles)
    set(PIO_PMD_PROBE "
#include <openPMD/openPMD.hpp>
#include <mpi.h>
int main() {
  openPMD::Series s(\"probe_only_unused\", openPMD::Access::Create,
                    MPI_COMM_WORLD, std::string(\"{\\\"backend\\\":\\\"hdf5\\\"}\"));
  auto it = s.iterations[0];
  double buf[2] = {1.0, 2.0};
  auto comp = it.meshes[\"m\"][\"c\"];
  comp.resetDataset(openPMD::Dataset(openPMD::determineType<double>(),
                                     openPMD::Shape{2}));
  comp.setChunkSize(openPMD::Shape{1});
  comp.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
  auto scalar = it.particles[\"sp\"][\"mass\"];      // scalar record
  scalar.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
  auto px = it.particles[\"sp\"][\"position\"][\"x\"]; // component leaf
  px.storeChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
  (void)px.loadChunk<double>(openPMD::Offset{0}, openPMD::Extent{2});
  (void)comp.loadChunk(buf, openPMD::Offset{0}, openPMD::Extent{2});
  s.flush();
  return 0;
}")
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX openPMD::openPMD)
    check_cxx_source_compiles("${PIO_PMD_PROBE}" PIO_PMD_CTOR_CURRENT)
    if(NOT PIO_PMD_CTOR_CURRENT)
      # legacy form: Series(path, access, engine, comm)
      string(REPLACE "MPI_COMM_WORLD, std::string(\"{\\\"backend\\\":\\\"hdf5\\\"}\")"
                     "std::string(\"HDF5\"), MPI_COMM_WORLD"
                     PIO_PMD_PROBE_LEGACY "${PIO_PMD_PROBE}")
      check_cxx_source_compiles("${PIO_PMD_PROBE_LEGACY}" PIO_PMD_CTOR_LEGACY)
    endif()
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)  # restore
    if(PIO_PMD_CTOR_CURRENT)
      target_compile_definitions(pio_bench PRIVATE PIO_HAVE_OPENPMD=1)
      target_link_libraries(pio_bench PRIVATE openPMD::openPMD)
      message(STATUS "pio_bench: openPMD backend ON (pmd_hdf5 + pmd_adios2, "
                     "current (path,access,comm,options) API)")
    elseif(PIO_PMD_CTOR_LEGACY)
      target_compile_definitions(pio_bench PRIVATE PIO_HAVE_OPENPMD=1
                                 PIO_PMD_LEGACY_CTOR=1)
      target_link_libraries(pio_bench PRIVATE openPMD::openPMD)
      message(STATUS "pio_bench: openPMD backend ON (legacy engine-string ctor)")
    else()
      message(STATUS "pio_bench: openPMD found but NEITHER Series ctor form "
                     "compiles against its headers -- pmd_* backends compiled "
                     "OUT. Check the probe log:")
      message(STATUS "  ${CMAKE_BINARY_DIR}/CMakeFiles/CMakeError.log "
                     "(or rerun cmake with --debug-trycompile)")
    endif()
  else()
    message(STATUS "pio_bench: openPMD not found -- pmd_* backends compiled OUT")
  endif()
endif()

install(TARGETS pio_bench RUNTIME DESTINATION bin)
