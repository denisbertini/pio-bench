// Minimal openPMD stub: reproduces ONLY the API surface + signatures that
// pio/backend_openpmd.hpp uses (shaped after the 0.17.1 tag sources). It is
// NOT functional -- it exists so `mpicxx -fsyntax-only` can catch C++ misuse
// of that surface locally, without the real library. When the cluster
// compiler accepts the real headers, that remains the final gate; this stub
// only stops us from shipping plain C++ errors.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <mpi.h>

namespace openPMD {

using Offset = std::vector<std::uint64_t>;
using Extent = std::vector<std::uint64_t>;

enum class Access { READ_ONLY, CREATE, APPEND, READ_WRITE };

template <typename T>
struct DatatypeTag {};
template <typename T>
int determineDatatype() { return 0; }

struct Dataset {
    Dataset(int dtype, Extent extent, std::string options = "{}") {}
};

struct RecordComponent {
    struct FromRecord {};
    template <typename BaseRecordT>
    RecordComponent(BaseRecordT const&) {}
    RecordComponent() {}
    void resetDataset(Dataset) {}
    template <typename Container>
    void storeChunk(Container&, Offset, Extent) {}
    template <typename T>
    void loadChunkRaw(T*, Offset, Extent) {}
};

struct MeshRecordComponent : RecordComponent {
    using RecordComponent::RecordComponent;
};

template <typename ComponentT>
struct BaseRecord {
    ComponentT operator[](const std::string&) { return ComponentT{}; }
};

struct Mesh : BaseRecord<MeshRecordComponent> {
    enum class Geometry { cartesian, thetaMode, cylindrical, spherical, other };
    void setGeometry(Geometry) {}
};

struct Record : BaseRecord<RecordComponent> {};

template <typename RecordT>
struct Container {
    RecordT operator[](const std::string&) { return RecordT{}; }
    RecordT operator[](int) { return RecordT{}; }  // iterations: int-indexed
};

struct Species : Container<Record> {};

struct Iteration {
    Container<Mesh> meshes;
    Container<Species> particles;
    void close() {}
};

struct Series {
    Container<Iteration> iterations;
    Series(const std::string&, Access, MPI_Comm, const std::string& = "{}") {}
    Series(const std::string&, Access, const std::string& = "{}") {}
    void flush(std::string = "{}") {}
};

} // namespace openPMD
