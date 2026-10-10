// Minimal openPMD stub: reproduces ONLY the API surface + signatures that
// pio/backend_openpmd.hpp uses (shaped after the 0.17.1 tag sources). It is
// NOT functional -- it exists so `mpicxx -fsyntax-only` can catch C++ misuse
// of that surface locally, without the real library. When the cluster
// compiler accepts the real headers, that remains the final gate; this stub
// only stops us from shipping plain C++ errors.
//
// 0.17.1 facts mirrored here:
//   * BaseRecord<T> inherits BOTH Container<T> AND T: a scalar record is
//     used directly as a RecordComponent (resetDataset/storeChunk on the
//     Record object -- example 8a: currSpecies["id"]).
//   * storeChunk owning overloads: shared_ptr<T> and shared_ptr<T[]>
//     (canonical createData() pattern), plus the non-owning container one.
//   * makeConstant() on RecordComponent (positionOffset pattern, 8a/3b).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <mpi.h>

namespace openPMD {

using Offset = std::vector<std::uint64_t>;
using Extent = std::vector<std::uint64_t>;

enum class Access { READ_ONLY, CREATE, APPEND, READ_WRITE };

// 0.17.1: lowerCamelCase enumerators (IterationEncoding.hpp).
enum class IterationEncoding { fileBased, groupBased, variableBased };

template <typename T>
struct DatatypeTag {};
template <typename T>
int determineDatatype() { return 0; }

struct Dataset {
    Dataset(int dtype, Extent extent, std::string options = "{}") {}
};

struct RecordComponent {
    RecordComponent() {}
    void resetDataset(Dataset) {}
    template <typename T>
    void storeChunk(std::shared_ptr<T>, Offset, Extent) {}
    template <typename T>
    void storeChunk(std::shared_ptr<T[]>, Offset, Extent) {}
    template <typename Container>
    void storeChunk(Container&, Offset, Extent) {}
    template <typename T>
    void makeConstant(T) {}
    template <typename T>
    void loadChunkRaw(T*, Offset, Extent) {}
};

struct MeshRecordComponent : RecordComponent {
    using RecordComponent::RecordComponent;
};

template <typename ComponentT>
struct ContainerLike {
    ComponentT operator[](const std::string&) { return ComponentT{}; }
    ComponentT operator[](int) { return ComponentT{}; }  // iterations: int-key
};

// The real 0.17.1 BaseRecord<T> is `: public Container<T>, public T`.
template <typename ComponentT>
struct BaseRecord : ContainerLike<ComponentT>, ComponentT {};

struct Mesh : BaseRecord<MeshRecordComponent> {
    enum class Geometry { cartesian, thetaMode, cylindrical, spherical, other };
    void setGeometry(Geometry) {}
};

struct Record : BaseRecord<RecordComponent> {};

struct Species : ContainerLike<Record> {};

struct Iteration {
    ContainerLike<Mesh> meshes;
    ContainerLike<Species> particles;
    void close() {}
};

struct Series {
    ContainerLike<Iteration> iterations;
    Series(const std::string&, Access, MPI_Comm, const std::string& = "{}") {}
    Series(const std::string&, Access, const std::string& = "{}") {}
    Series& setIterationEncoding(IterationEncoding) { return *this; }
    void flush(std::string = "{}") {}
};

} // namespace openPMD
