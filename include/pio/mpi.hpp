// pio/mpi.hpp -- thin RAII layer over the MPI C API.
//
// Design notes (vs. the old ci_ompi CHECK_ERR macro):
//  * CHECK_ERR printed and CONTINUED after a failed MPI call -- broken
//    error handling that turns partial I/O into silent wrong results.
//    Here every failure throws pio::MpiError with the MPI error string
//    and the source location; main() catches and aborts the job.
//  * MPI_Info / MPI_Datatype / MPI_File are wrapped in move-only RAII
//    handles: no leak paths on early return / exception.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include <mpi.h>

namespace pio {

class MpiError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {
[[noreturn]] inline void mpi_fail(int rc, const char* what,
                                  std::source_location loc) {
    char buf[MPI_MAX_ERROR_STRING];
    int len = 0;
    if (MPI_Error_string(rc, buf, &len) == MPI_SUCCESS)
        throw MpiError(std::string(what) + ": " + std::string(buf, len) + " @ " +
                       loc.file_name() + ":" + std::to_string(loc.line()));
    throw MpiError(std::string(what) + ": <unknown error " +
                   std::to_string(rc) + "> @ " + loc.file_name() + ":" +
                   std::to_string(loc.line()));
}
} // namespace detail

inline void mpi_check(int rc, const char* what,
                      std::source_location loc = std::source_location::current()) {
    if (rc != MPI_SUCCESS)
        detail::mpi_fail(rc, what, loc);
}

#define PIO_MPI(expr) ::pio::mpi_check((expr), #expr)

inline int rank_of(MPI_Comm comm) {
    int r = 0;
    PIO_MPI(MPI_Comm_rank(comm, &r));
    return r;
}

inline int size_of(MPI_Comm comm) {
    int s = 0;
    PIO_MPI(MPI_Comm_size(comm, &s));
    return s;
}

// ---------------------------------------------------------------- Info ----
/// RAII MPI_Info builder (starts empty, never MPI_INFO_NULL).
class Info {
public:
    Info() { PIO_MPI(MPI_Info_create(&info_)); }
    Info(const Info&) = delete;
    Info& operator=(const Info&) = delete;
    Info(Info&& o) noexcept : info_(std::exchange(o.info_, MPI_INFO_NULL)) {}
    Info& operator=(Info&& o) noexcept {
        if (this != &o) {
            free();
            info_ = std::exchange(o.info_, MPI_INFO_NULL);
        }
        return *this;
    }
    ~Info() { free(); }

    void set(std::string_view key, std::string_view value) {
        PIO_MPI(MPI_Info_set(info_, std::string(key).c_str(), std::string(value).c_str()));
    }
    /// set() only when `cond` -- keeps hint tables declarative.
    void set_if(bool cond, std::string_view key, std::string_view value) {
        if (cond)
            set(key, value);
    }
    MPI_Info get() const { return info_; }

private:
    void free() {
        if (info_ != MPI_INFO_NULL)
            MPI_Info_free(&info_);
    }
    MPI_Info info_{MPI_INFO_NULL};
};

// ----------------------------------------------------------- Datatype ----
/// RAII committed MPI_Datatype.
class Datatype {
public:
    Datatype() = default;
    Datatype(const Datatype&) = delete;
    Datatype& operator=(const Datatype&) = delete;
    Datatype(Datatype&& o) noexcept : type_(std::exchange(o.type_, MPI_DATATYPE_NULL)) {}
    Datatype& operator=(Datatype&& o) noexcept {
        if (this != &o) {
            free();
            type_ = std::exchange(o.type_, MPI_DATATYPE_NULL);
        }
        return *this;
    }
    ~Datatype() { free(); }

    MPI_Datatype get() const { return type_; }
    operator MPI_Datatype() const { return type_; }

    /// MPI-3 declares the subarray extents as MPI_Aint, but several
    /// widely-deployed implementations (Intel MPI / MPICH < 4, OMPI < 4.1)
    /// still declare `const int[]`. Passing the wrong width is SILENT
    /// corruption on little-endian (4-byte misreads of 8-byte arrays), so
    /// the signature is probed at configure time (PIO_MPI_SUBARRAY_AINT)
    /// and values are converted -- with range checks -- here.
#ifdef PIO_MPI_SUBARRAY_AINT
    using SubarrayIndex = MPI_Aint;
#else
    using SubarrayIndex = int;
#endif

    /// 1-D subarray over arbitrary etype (used for particle record blocks).
    static Datatype subarray_1d(MPI_Aint global_size, MPI_Aint local_size,
                                MPI_Aint start, MPI_Datatype etype) {
        return subarray(1, &global_size, &local_size, &start, etype);
    }

    static Datatype subarray(int ndims, const MPI_Aint sizes[],
                             const MPI_Aint subsizes[], const MPI_Aint starts[],
                             MPI_Datatype etype) {
        if (ndims < 1 || ndims > 3)
            throw MpiError("subarray ndims out of range");
        auto narrow = [](const MPI_Aint in[], SubarrayIndex out[]) {
            for (int i = 0; i < 3; ++i) {
                if (in[i] < 0)
                    throw MpiError("negative subarray extent");
                if (in[i] > static_cast<MPI_Aint>(
                                 std::numeric_limits<SubarrayIndex>::max()))
                    throw MpiError(
                        "subarray extent exceeds this MPI implementation's "
                        "int-based subarray signature (MPI-3 requires MPI_Aint; "
                        "upgrade the MPI stack for larger arrays)");
                out[i] = static_cast<SubarrayIndex>(in[i]);
            }
        };
        SubarrayIndex s[3], ss[3], st[3];
        narrow(sizes, s);
        narrow(subsizes, ss);
        narrow(starts, st);
        MPI_Datatype t = MPI_DATATYPE_NULL;
        PIO_MPI(MPI_Type_create_subarray(ndims, s, ss, st, MPI_ORDER_C, etype, &t));
        PIO_MPI(MPI_Type_commit(&t));
        return Datatype{t};
    }

    static Datatype contiguous(MPI_Aint count, MPI_Datatype base) {
        MPI_Datatype t = MPI_DATATYPE_NULL;
        PIO_MPI(MPI_Type_contiguous(static_cast<int>(count), base, &t));
        PIO_MPI(MPI_Type_commit(&t));
        return Datatype{t};
    }

private:
    explicit Datatype(MPI_Datatype t) : type_(t) {}
    void free() {
        if (type_ != MPI_DATATYPE_NULL)
            MPI_Type_free(&type_);
    }
    MPI_Datatype type_{MPI_DATATYPE_NULL};
};

// --------------------------------------------------------------- File ----
/// RAII MPI_File (collective open; close in dtor unless released).
class File {
public:
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&& o) noexcept : fh_(std::exchange(o.fh_, MPI_FILE_NULL)) {}
    File& operator=(File&& o) noexcept {
        if (this != &o) {
            close();
            fh_ = std::exchange(o.fh_, MPI_FILE_NULL);
        }
        return *this;
    }
    ~File() { close(); }

    static File open(MPI_Comm comm, const std::string& path, int amode,
                     MPI_Info info) {
        File f;
        PIO_MPI(MPI_File_open(comm, path.c_str(), amode, info, &f.fh_));
        return f;
    }

    MPI_File get() const { return fh_; }
    explicit operator bool() const { return fh_ != MPI_FILE_NULL; }

    void set_view(MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype,
                  MPI_Info info) {
        PIO_MPI(MPI_File_set_view(fh_, disp, etype, filetype, "native", info));
    }

    void close() {
        if (fh_ != MPI_FILE_NULL) {
            MPI_File_close(&fh_);
            fh_ = MPI_FILE_NULL;
        }
    }

private:
    MPI_File fh_{MPI_FILE_NULL};
};

} // namespace pio
