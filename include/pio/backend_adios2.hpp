// pio/backend_adios2.hpp -- ADIOS2 (BP5 engine) backend.
//
// One .bp dataset per checkpoint:
//   "fields/data"     3-D double, per-rank global block (ConstantDims)
//   "particles/{x,y,z,px,py,pz,mass,id}"  1-D vars with LocalValueDim:
//                     rank blocks are merged by the engine into the
//                     global arrays -- the ADIOS-native "merge via
//                     back end" pattern.
//
// ADIOS2 Put() requires a CONTIGUOUS local block; with ghost > 0 the
// padded interior is not contiguous, so the field is gathered into a
// temporary (documented O(N) cost; MPI-IO/HDF5 backends avoid it with
// native strided views). Particle members are gathered likewise (AoS ->
// per-member arrays).
//
// Timing split: field Put() -> field_seconds; particle Puts + EndStep +
// Close (the flush) -> particle_seconds. Approximate but consistent.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "pio/backends.hpp"

#ifdef PIO_HAVE_ADIOS2
#include <adios2.h>
#endif

namespace pio {

#ifdef PIO_HAVE_ADIOS2

class Adios2Backend {
public:
    static constexpr std::string_view name = "adios2";
    static constexpr std::string_view extension = ".bp";
    static constexpr bool available = true;

    explicit Adios2Backend(MPI_Comm comm, const IoSettings& s)
        : comm_(comm), settings_(s) {}

    WriteSplit write(const std::filesystem::path& path, const Domain& d,
                     const Field3d<double>& field, const ParticleSet& parts) const {
        WriteSplit split;

        adios2::ADIOS adios(comm_);
        adios2::IO io = adios.DeclareIO("pio_bench");
        io.SetEngine("BP5");
        if (settings_.buffer_bytes)
            io.SetParameter("BufferVSize", std::to_string(*settings_.buffer_bytes));
        if (settings_.aggregators && *settings_.aggregators > 0)
            io.SetParameter("NumAggregators", std::to_string(*settings_.aggregators));

        // ---- field variable: this rank's global block ----
        const adios2::Dims shape{size_t(d.global[0]), size_t(d.global[1]), size_t(d.global[2])};
        const adios2::Dims fstart{size_t(d.start[0]), size_t(d.start[1]), size_t(d.start[2])};
        const adios2::Dims fcount{size_t(d.local[0]), size_t(d.local[1]), size_t(d.local[2])};
        auto var = io.DefineVariable<double>("fields/data", shape, fstart, fcount,
                                             adios2::ConstantDims);

        // ---- particle member variables (rank blocks merged by engine) ----
        const adios2::Dims lvd{adios2::LocalValueDim};
        auto vx = io.DefineVariable<double>("particles/x", lvd);
        auto vy = io.DefineVariable<double>("particles/y", lvd);
        auto vz = io.DefineVariable<double>("particles/z", lvd);
        auto vpx = io.DefineVariable<double>("particles/px", lvd);
        auto vpy = io.DefineVariable<double>("particles/py", lvd);
        auto vpz = io.DefineVariable<double>("particles/pz", lvd);
        auto vm = io.DefineVariable<double>("particles/mass", lvd);
        auto vi = io.DefineVariable<std::uint64_t>("particles/id", lvd);

        adios2::Engine engine = io.Open(path.string(), adios2::Mode::Write);
        engine.BeginStep();

        Stopwatch w;
        {
            // contiguous interior block (skipped when ghost == 0)
            if (d.ghost == 0) {
                engine.Put(var, field.data());
            } else {
                std::vector<double> tmp(d.interior_count());
                std::size_t n = 0;
                for (int i = d.ghost; i < d.ghost + d.local[0]; ++i)
                    for (int j = d.ghost; j < d.ghost + d.local[1]; ++j)
                        for (int k = d.ghost; k < d.ghost + d.local[2]; ++k)
                            tmp[n++] = field.at(i, j, k);
                engine.Put(var, tmp);
            }
        }
        split.field_seconds = w.elapsed();

        w.restart();
        {
            const std::size_t n = parts.local_count();
            std::vector<double> cx(n), cy(n), cz(n), cpx(n), cpy(n), cpz(n), cm(n);
            std::vector<std::uint64_t> ci(n);
            const Particle* p = parts.data();
            for (std::size_t i = 0; i < n; ++i) {
                cx[i] = p[i].x;
                cy[i] = p[i].y;
                cz[i] = p[i].z;
                cpx[i] = p[i].px;
                cpy[i] = p[i].py;
                cpz[i] = p[i].pz;
                cm[i] = p[i].mass;
                ci[i] = p[i].id;
            }
            engine.Put(vx, cx);
            engine.Put(vy, cy);
            engine.Put(vz, cz);
            engine.Put(vpx, cpx);
            engine.Put(vpy, cpy);
            engine.Put(vpz, cpz);
            engine.Put(vm, cm);
            engine.Put(vi, ci);
            engine.EndStep();
            engine.Close(); // collective flush point
        }
        split.particle_seconds = w.elapsed();
        return split;
    }

    void read_field(const std::filesystem::path& path, const Domain& d,
                    Field3d<double>& dst) const {
        adios2::ADIOS adios(comm_);
        adios2::IO io = adios.DeclareIO("pio_bench_read");
        io.SetEngine("BP5");
        auto var = io.InquireVariable<double>("fields/data");
        if (!var)
            throw std::runtime_error("adios2: fields/data not found in " + path.string());

        const adios2::Dims fstart{size_t(d.start[0]), size_t(d.start[1]), size_t(d.start[2])};
        const adios2::Dims fcount{size_t(d.local[0]), size_t(d.local[1]), size_t(d.local[2])};
        var.SetSelection(adios2::Box<adios2::Dims>(fstart, fcount));

        adios2::Engine engine = io.Open(path.string(), adios2::Mode::Read);
        engine.BeginStep();
        if (d.ghost == 0) {
            engine.Get(var, dst.data(), adios2::Mode::Sync);
        } else {
            std::vector<double> tmp(d.interior_count());
            engine.Get(var, tmp, adios2::Mode::Sync);
            std::size_t n = 0;
            for (int i = d.ghost; i < d.ghost + d.local[0]; ++i)
                for (int j = d.ghost; j < d.ghost + d.local[1]; ++j)
                    for (int k = d.ghost; k < d.ghost + d.local[2]; ++k)
                        dst.at(i, j, k) = tmp[n++];
        }
        engine.EndStep();
        engine.Close();
    }

private:
    MPI_Comm comm_;
    IoSettings settings_;
};

#else // !PIO_HAVE_ADIOS2 -- compile-time stub keeps the concept satisfied

class Adios2Backend {
public:
    static constexpr std::string_view name = "adios2";
    static constexpr std::string_view extension = ".bp";
    static constexpr bool available = false;

    explicit Adios2Backend(MPI_Comm, const IoSettings&) {}
    WriteSplit write(const std::filesystem::path&, const Domain&,
                     const Field3d<double>&, const ParticleSet&) const {
        throw std::logic_error("pio_bench built without ADIOS2 support");
    }
    void read_field(const std::filesystem::path&, const Domain&, Field3d<double>&) const {
        throw std::logic_error("pio_bench built without ADIOS2 support");
    }
};

#endif

} // namespace pio
