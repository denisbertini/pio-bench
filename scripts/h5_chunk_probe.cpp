// h5_chunk_probe.cpp -- standalone HDF5-parallel chunked-write probe.
//
// Isolates the pmd_hdf5 MPI_ERR_IO wall WITHOUT openPMD and WITHOUT
// pio-bench: creates one file, one 1-D dataset (chunked or contiguous),
// and every rank collective-writes its contiguous slice of doubles --
// the exact transfer shape of openPMD's particle component writes.
//
// Build (cluster, inside container; HDF5 from /usr/local):
//   mpicxx -O2 scripts/h5_chunk_probe.cpp -o /tmp/h5probe -lhdf5
// Run (32 ranks, crash-geometry numbers):
//   mpirun -n 32 /tmp/h5probe p1.h5 128000000 4000000          # chunk 32 MiB, default libver
//   mpirun -n 32 /tmp/h5probe p2.h5 128000000 4000000 v114     # libver bounds v114
//   mpirun -n 32 /tmp/h5probe p3.h5 128000000 1000000          # chunk 8 MiB
//   mpirun -n 32 /tmp/h5probe p4.h5 128000000 0                # contiguous control
//   (trailing flags, any order: "acorder" = attribute-creation-order
//    on the file-creation list + one attribute, mimicking openPMD's
//    file-customization habits; "indep" = INDEPENDENT H5Dwrite, which
//    is what openPMD's HDF5 backend actually uses for chunk writes)
//
// Verdict table:
//   p1 fails, p4 passes  -> HDF5 2.2 parallel CHUNK layer is the wall.
//       p2 passing would mean libver bounds dodge it (cheap fix via
//       openPMD file options); otherwise pin container HDF5 to 1.14.x.
//   p1 passes             -> HDF5 alone is innocent; openPMD's own
//       file/dataset customization triggers it -> probe stage 2
//       (try acorder; then fill values / multiple datasets).
#include <hdf5.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void h5die(const char* what, MPI_Comm comm) {
    fprintf(stderr, "[probe] %s FAILED\n", what);
    MPI_Abort(comm, 1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (!rank)
            fprintf(stderr,
                    "usage: %s <file.h5> <global_elems> <chunk_elems|0=contig>"
                    " [default|earliest|v16|v18|v110|v112|v114] [acorder]"
                    " [indep]\n",
                    argv[0]);
        MPI_Finalize();
        return 2;
    }
    const hsize_t global = strtoull(argv[2], nullptr, 10);
    const hsize_t chunk = strtoull(argv[3], nullptr, 10);
    // argv[4..] free-form: one libver name + optional "acorder"/"indep" flags
    std::string libver = "default";
    bool acorder = false, indep = false;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "acorder") acorder = true;
        else if (a == "indep") indep = true;
        else libver = a;
    }

    if (global % (hsize_t)size) {
        if (!rank) fprintf(stderr, "[probe] global must divide by ranks\n");
        MPI_Finalize();
        return 2;
    }
    const hsize_t per = global / (hsize_t)size;

    hid_t fcpl = H5Pcreate(H5P_FILE_CREATE);
    // HDF5 2.x removes old libver enum values (V16 confirmed gone in 2.2.0);
    // guard each so the probe compiles on both 1.14-era and 2.x headers.
    bool libver_set = libver == "default";
#define PIO_LIBVER(NAME, LOW, HIGH)                         \
    if (libver == NAME) {                                   \
        H5Pset_libver_bounds(fcpl, LOW, HIGH);              \
        libver_set = true;                                  \
    }
#ifdef H5F_LIBVER_EARLIEST
    PIO_LIBVER("earliest", H5F_LIBVER_EARLIEST, H5F_LIBVER_EARLIEST)
#endif
#ifdef H5F_LIBVER_V16
    PIO_LIBVER("v16", H5F_LIBVER_EARLIEST, H5F_LIBVER_V16)
#endif
#ifdef H5F_LIBVER_V18
    PIO_LIBVER("v18", H5F_LIBVER_EARLIEST, H5F_LIBVER_V18)
#endif
#ifdef H5F_LIBVER_V110
    PIO_LIBVER("v110", H5F_LIBVER_EARLIEST, H5F_LIBVER_V110)
#endif
#ifdef H5F_LIBVER_V112
    PIO_LIBVER("v112", H5F_LIBVER_EARLIEST, H5F_LIBVER_V112)
#endif
#ifdef H5F_LIBVER_V114
    PIO_LIBVER("v114", H5F_LIBVER_EARLIEST, H5F_LIBVER_V114)
#endif
#undef PIO_LIBVER
    if (!libver_set) {
        if (!rank)
            fprintf(stderr,
                    "[probe] libver '%s' not in this HDF5's enum set\n",
                    libver.c_str());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (acorder)
        H5Pset_attr_creation_order(
            fcpl, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED);

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file = H5Fcreate(argv[1], H5F_ACC_TRUNC, fcpl, fapl);
    if (file < 0) h5die("H5Fcreate", MPI_COMM_WORLD);

    if (acorder) {  // one root attribute, like openPMD stamps everywhere
        hid_t attr = H5Acreate2(file, "probe_attr", H5T_NATIVE_INT,
                                H5Screate(H5S_SCALAR), H5P_DEFAULT,
                                H5P_DEFAULT);
        if (attr < 0) h5die("H5Acreate2", MPI_COMM_WORLD);
        int v = 42;
        if (H5Awrite(attr, H5T_NATIVE_INT, &v) < 0)
            h5die("H5Awrite", MPI_COMM_WORLD);
        H5Aclose(attr);
    }

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (chunk > 0) H5Pset_chunk(dcpl, 1, &chunk);
    hid_t space = H5Screate_simple(1, &global, nullptr);
    hid_t dset = H5Dcreate2(file, "probe", H5T_NATIVE_DOUBLE, space,
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) h5die("H5Dcreate2", MPI_COMM_WORLD);

    std::vector<double> buf(per, 1.0);
    hid_t mem = H5Screate_simple(1, &per, nullptr);
    hid_t fspace = H5Dget_space(dset);
    hsize_t start = (hsize_t)rank * per;
    if (H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &start, nullptr, &per,
                            nullptr) < 0)
        h5die("H5Sselect_hyperslab", MPI_COMM_WORLD);
    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, indep ? H5FD_MPIO_INDEPENDENT
                                 : H5FD_MPIO_COLLECTIVE);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    herr_t rc = H5Dwrite(dset, H5T_NATIVE_DOUBLE, mem, fspace, xfer,
                         buf.data());
    double dt = MPI_Wtime() - t0;

    if (!rank)
        printf("[probe] file=%s global=%llu chunk=%llu libver=%s acorder=%d "
               "xfer=%s write/rank=%.1f MiB  rc=%d  %.2f s"
               " (%.0f MiB/s aggregate)\n",
               argv[1], (unsigned long long)global, (unsigned long long)chunk,
               libver.c_str(), (int)acorder, indep ? "INDEPENDENT" : "collective",
               per * 8.0 / 1048576.0, (int)rc,
               dt, global * 8.0 / 1048576.0 / (dt > 0 ? dt : 1e-9));

    H5Dclose(dset);
    H5Fclose(file);
    MPI_Finalize();
    return rc < 0 ? 1 : 0;
}
