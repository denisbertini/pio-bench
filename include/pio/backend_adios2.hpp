// pio/backend_adios2.hpp -- ADIOS2 (BP5 engine) backend.
//
// One .bp dataset per checkpoint:
//   "fields/data"     3-D double, per-rank global block (ConstantDims)
//   "particles/{x,y,z,px,py,pz,mass,id}"  1-D double/uint64, explicit
//                     global shape with per-rank blocks at rank_offset --
//                     merged by the engine into the global arrays.
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

        // ADIOS2 3.x removed MPI_Comm from the ADIOS constructor (it is
        // passed per-Open now, like 2.x's optional overload); using the
        // pointer-based Put/Get everywhere keeps this source valid across
        // ADIOS 2.6+ and 3.x.
        adios2::ADIOS adios;
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

        // ---- particle member variables: explicit global arrays -------------
        // shape = merged total, this rank's block at rank_offset -- exactly
        // the field pattern, which works across BP engines. (LocalValueDim
        // was the original choice, but ADIOS 2.12 has no Put(var, ptr, Dims)
        // overload to pass the local size, and Put(var, ptr) on a
        // LocalValueDim variable silently writes NOTHING -- particle
        // read-back verify caught it as a read-side segfault.)
        const adios2::Dims pshape{size_t(parts.global_count())};
        const adios2::Dims pstart{size_t(parts.rank_offset())};
        const adios2::Dims pcount{size_t(parts.local_count())};
        auto vx = io.DefineVariable<double>("particles/x", pshape, pstart, pcount,
                                            adios2::ConstantDims);
        auto vy = io.DefineVariable<double>("particles/y", pshape, pstart, pcount,
                                            adios2::ConstantDims);
        auto vz = io.DefineVariable<double>("particles/z", pshape, pstart, pcount,
                                            adios2::ConstantDims);
        auto vpx = io.DefineVariable<double>("particles/px", pshape, pstart, pcount,
                                             adios2::ConstantDims);
        auto vpy = io.DefineVariable<double>("particles/py", pshape, pstart, pcount,
                                             adios2::ConstantDims);
        auto vpz = io.DefineVariable<double>("particles/pz", pshape, pstart, pcount,
                                             adios2::ConstantDims);
        auto vm = io.DefineVariable<double>("particles/mass", pshape, pstart,
                                            pcount, adios2::ConstantDims);
        auto vi = io.DefineVariable<std::uint64_t>("particles/id", pshape, pstart,
                                                   pcount, adios2::ConstantDims);

        adios2::Engine engine = io.Open(path.string(), adios2::Mode::Write, comm_);
        engine.BeginStep();

        // ADIOS2 Put() MEMORY CONTRACT: the copy may be deferred until
        // EndStep -- BP5 keeps the USER POINTER for blocks exceeding the
        // staging pool (BufferVSize, default 128 MiB) instead of copying.
        // Every buffer handed to Put must therefore stay alive until after
        // EndStep. Field staging is hoisted to this scope for exactly that
        // reason: a tmp destroyed before EndStep made BP5 flush freed
        // memory (verify caught it at local=256, ALL field cells wrong;
        // invisible at local=64 where the 2 MiB block fit the pool and
        // was copied eagerly).
        std::vector<double> tmp;

        Stopwatch w;
        {
            // contiguous interior block (skipped when ghost == 0)
            if (d.ghost == 0) {
                engine.Put(var, field.data());
            } else {
                tmp.resize(d.interior_count());
                std::size_t n = 0;
                for (int i = d.ghost; i < d.ghost + d.local[0]; ++i)
                    for (int j = d.ghost; j < d.ghost + d.local[1]; ++j)
                        for (int k = d.ghost; k < d.ghost + d.local[2]; ++k)
                            tmp[n++] = field.at(i, j, k);
                engine.Put(var, tmp.data());
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
            // count comes from the variable definition (explicit global
            // block), so the two-argument Put is the complete form here.
            engine.Put(vx, cx.data());
            engine.Put(vy, cy.data());
            engine.Put(vz, cz.data());
            engine.Put(vpx, cpx.data());
            engine.Put(vpy, cpy.data());
            engine.Put(vpz, cpz.data());
            engine.Put(vm, cm.data());
            engine.Put(vi, ci.data());
            engine.EndStep();
            engine.Close(); // collective flush point
        }
        split.particle_seconds = w.elapsed();
        return split;
    }

    void read_field(const std::filesystem::path& path, const Domain& d,
                    Field3d<double>& dst) const {
        adios2::ADIOS adios;
        adios2::IO io = adios.DeclareIO("pio_bench_read");
        io.SetEngine("BP5");

        // ADIOS2 read ordering rule: the variables of a file only appear
        // in the IO AFTER Open(Read) + BeginStep() have loaded its
        // metadata -- inquiring before returns nothing (smoke test caught
        // exactly this).
        adios2::Engine engine = io.Open(path.string(), adios2::Mode::Read, comm_);
        engine.BeginStep();
        auto var = io.InquireVariable<double>("fields/data");
        if (!var)
            throw std::runtime_error("adios2: fields/data not found in " + path.string());

        const adios2::Dims fstart{size_t(d.start[0]), size_t(d.start[1]), size_t(d.start[2])};
        const adios2::Dims fcount{size_t(d.local[0]), size_t(d.local[1]), size_t(d.local[2])};
        var.SetSelection(adios2::Box<adios2::Dims>(fstart, fcount));

        if (d.ghost == 0) {
            engine.Get(var, dst.data(), adios2::Mode::Sync);
        } else {
            std::vector<double> tmp(d.interior_count());
            engine.Get(var, tmp.data(), adios2::Mode::Sync);
            std::size_t n = 0;
            for (int i = d.ghost; i < d.ghost + d.local[0]; ++i)
                for (int j = d.ghost; j < d.ghost + d.local[1]; ++j)
                    for (int k = d.ghost; k < d.ghost + d.local[2]; ++k)
                        dst.at(i, j, k) = tmp[n++];
        }
        engine.EndStep();
        engine.Close();
    }

    /// Read this rank's own block back out of the merged global arrays:
    /// select [rank_offset, local_count) of each member and reassemble
    /// the AoS (same selection style as the field read).
    void read_particles(const std::filesystem::path& path, const Domain&,
                        const ParticleSet& src, std::vector<Particle>& out) const {
        const std::size_t n = src.local_count();
        out.assign(n, Particle{});
        if (n == 0)
            return;

        adios2::ADIOS adios;
        adios2::IO io = adios.DeclareIO("pio_bench_particle_read");
        io.SetEngine("BP5");
        adios2::Engine engine = io.Open(path.string(), adios2::Mode::Read, comm_);
        engine.BeginStep(); // variables exist only from here on (see read_field)

        const adios2::Dims sstart{size_t(src.rank_offset())};
        const adios2::Dims scount{n};
        auto get_member = [&](const char* name, double Particle::*mem) {
            auto var = io.InquireVariable<double>(name);
            if (!var)
                throw std::runtime_error(std::string("adios2: ") + name +
                                         " not found in " + path.string());
            var.SetSelection(adios2::Box<adios2::Dims>(sstart, scount));
            std::vector<double> tmp(n);
            engine.Get(var, tmp.data(), adios2::Mode::Sync);
            for (std::size_t i = 0; i < n; ++i)
                out[i].*mem = tmp[i];
        };
        get_member("particles/x", &Particle::x);
        get_member("particles/y", &Particle::y);
        get_member("particles/z", &Particle::z);
        get_member("particles/px", &Particle::px);
        get_member("particles/py", &Particle::py);
        get_member("particles/pz", &Particle::pz);
        get_member("particles/mass", &Particle::mass);

        auto var_id = io.InquireVariable<std::uint64_t>("particles/id");
        if (!var_id)
            throw std::runtime_error("adios2: particles/id not found in " +
                                     path.string());
        var_id.SetSelection(adios2::Box<adios2::Dims>(sstart, scount));
        std::vector<std::uint64_t> tmp_id(n);
        engine.Get(var_id, tmp_id.data(), adios2::Mode::Sync);
        for (std::size_t i = 0; i < n; ++i)
            out[i].id = tmp_id[i];

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
    void read_particles(const std::filesystem::path&, const Domain&,
                        const ParticleSet&, std::vector<Particle>&) const {
        throw std::logic_error("pio_bench built without ADIOS2 support");
    }
};

#endif

} // namespace pio
