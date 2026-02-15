#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Timers {
  double pivot = 0.0;
  double swap  = 0.0;
  double bcast = 0.0;
  double update = 0.0;
  double solve = 0.0;
};

static inline double now_sec() { return MPI_Wtime(); }

// -------------------------
// Utilities: block row distribution
// -------------------------
struct Dist1D {
  int rank = 0, size = 1;
  int64_t N = 0;

  // counts[r] = number of rows owned by rank r
  std::vector<int> counts;
  // displs[r] = starting global row index for rank r
  std::vector<int> displs;

  void init(int64_t n, int r, int s) {
    N = n; rank = r; size = s;
    counts.assign(size, 0);
    displs.assign(size, 0);

    int64_t base = N / size;
    int64_t rem  = N % size;
    for (int p = 0; p < size; ++p) {
      int64_t c = base + (p < rem ? 1 : 0);
      counts[p] = (int)c;
    }
    displs[0] = 0;
    for (int p = 1; p < size; ++p) displs[p] = displs[p-1] + counts[p-1];
  }

  inline int owner(int64_t global_i) const {
    // Find rank such that displs[r] <= i < displs[r] + counts[r]
    for (int r = 0; r < size; ++r) {
      int64_t start = displs[r];
      int64_t end   = start + counts[r];
      if (global_i >= start && global_i < end) return r;
    }
    return -1;
  }

  inline bool owns(int64_t global_i) const { return owner(global_i) == rank; }

  inline int64_t local_index(int64_t global_i) const {
    return global_i - displs[rank];
  }

  inline int64_t local_rows() const { return counts[rank]; }
};

// Row-major local storage
static inline int64_t lidx(int64_t li, int64_t j, int64_t N) { return li * N + j; }

// Generate A and b on root, then scatter by row
static void generate_on_root_and_scatter(int64_t N, uint64_t seed,
                                        const Dist1D& dist,
                                        std::vector<double>& A_local,
                                        std::vector<double>& b_local,
                                        MPI_Comm comm) {
  int rank = dist.rank;
  int size = dist.size;

  int64_t local_rows = dist.local_rows();
  A_local.assign((size_t)local_rows * (size_t)N, 0.0);
  b_local.assign((size_t)local_rows, 0.0);

  std::vector<double> A_full;
  std::vector<double> b_full;
  if (rank == 0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> distu(-1.0, 1.0);
    A_full.assign((size_t)N * (size_t)N, 0.0);
    b_full.assign((size_t)N, 0.0);

    // Make A diagonally dominant to reduce singularity risk
    for (int64_t i = 0; i < N; ++i) {
      double rowsum = 0.0;
      for (int64_t j = 0; j < N; ++j) {
        double v = distu(rng);
        A_full[(size_t)i * (size_t)N + (size_t)j] = v;
        rowsum += std::abs(v);
      }
      A_full[(size_t)i * (size_t)N + (size_t)i] += rowsum + 1.0;
      b_full[(size_t)i] = distu(rng);
    }
  }

  // Scatter rows of A
  std::vector<int> sendcountsA(size, 0), displsA(size, 0);
  std::vector<int> sendcountsB(size, 0), displsB(size, 0);
  for (int r = 0; r < size; ++r) {
    sendcountsA[r] = dist.counts[r] * (int)N;
    displsA[r]     = dist.displs[r] * (int)N;
    sendcountsB[r] = dist.counts[r];
    displsB[r]     = dist.displs[r];
  }

  MPI_Scatterv(rank == 0 ? A_full.data() : nullptr, sendcountsA.data(), displsA.data(), MPI_DOUBLE,
               A_local.data(), (int)(local_rows * N), MPI_DOUBLE, 0, comm);

  MPI_Scatterv(rank == 0 ? b_full.data() : nullptr, sendcountsB.data(), displsB.data(), MPI_DOUBLE,
               b_local.data(), (int)local_rows, MPI_DOUBLE, 0, comm);
}

// Swap two global rows across ranks, distribution stays by global index.
static void swap_global_rows(int64_t N, const Dist1D& dist,
                             std::vector<double>& A_local,
                             std::vector<double>& b_local,
                             int64_t ra, int64_t rb,
                             MPI_Comm comm) {
  if (ra == rb) return;

  int owner_a = dist.owner(ra);
  int owner_b = dist.owner(rb);

  int rank = dist.rank;

  // If both rows owned by the same rank then local swap
  if (owner_a == owner_b) {
    if (rank == owner_a) {
      int64_t la = dist.local_index(ra);
      int64_t lb = dist.local_index(rb);
      for (int64_t j = 0; j < N; ++j) std::swap(A_local[lidx(la, j, N)], A_local[lidx(lb, j, N)]);
      std::swap(b_local[(size_t)la], b_local[(size_t)lb]);
    }
    return;
  }

  // Otherwise, exchange row buffers between owner_a and owner_b
  std::vector<double> rowbuf_a, rowbuf_b;
  double bval_a = 0.0, bval_b = 0.0;

  if (rank == owner_a) {
    int64_t la = dist.local_index(ra);
    rowbuf_a.resize((size_t)N);
    for (int64_t j = 0; j < N; ++j) rowbuf_a[(size_t)j] = A_local[lidx(la, j, N)];
    bval_a = b_local[(size_t)la];

    // Receive row b into rowbuf_b
    rowbuf_b.resize((size_t)N);
    MPI_Sendrecv(rowbuf_a.data(), (int)N, MPI_DOUBLE, owner_b, 100,
                 rowbuf_b.data(), (int)N, MPI_DOUBLE, owner_b, 101,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&bval_a, 1, MPI_DOUBLE, owner_b, 200,
                 &bval_b, 1, MPI_DOUBLE, owner_b, 201,
                 comm, MPI_STATUS_IGNORE);

    // Write received rowbuf_b into row a position
    for (int64_t j = 0; j < N; ++j) A_local[lidx(la, j, N)] = rowbuf_b[(size_t)j];
    b_local[(size_t)la] = bval_b;
  }

  if (rank == owner_b) {
    int64_t lb = dist.local_index(rb);
    rowbuf_b.resize((size_t)N);
    for (int64_t j = 0; j < N; ++j) rowbuf_b[(size_t)j] = A_local[lidx(lb, j, N)];
    bval_b = b_local[(size_t)lb];

    // Receive row a into rowbuf_a
    rowbuf_a.resize((size_t)N);
    MPI_Sendrecv(rowbuf_b.data(), (int)N, MPI_DOUBLE, owner_a, 101,
                 rowbuf_a.data(), (int)N, MPI_DOUBLE, owner_a, 100,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&bval_b, 1, MPI_DOUBLE, owner_a, 201,
                 &bval_a, 1, MPI_DOUBLE, owner_a, 200,
                 comm, MPI_STATUS_IGNORE);

    // Write received rowbuf_a into row b position
    for (int64_t j = 0; j < N; ++j) A_local[lidx(lb, j, N)] = rowbuf_a[(size_t)j];
    b_local[(size_t)lb] = bval_a;
  }
}

// Broadcast the full pivot row and pivot b value.
static void bcast_pivot_row(int64_t N, const Dist1D& dist,
                            const std::vector<double>& A_local,
                            const std::vector<double>& b_local,
                            int64_t k,
                            std::vector<double>& pivot_row,
                            double& pivot_b,
                            MPI_Comm comm) {
  pivot_row.assign((size_t)N, 0.0);
  pivot_b = 0.0;

  int owner_k = dist.owner(k);
  if (dist.rank == owner_k) {
    int64_t lk = dist.local_index(k);
    for (int64_t j = 0; j < N; ++j) pivot_row[(size_t)j] = A_local[lidx(lk, j, N)];
    pivot_b = b_local[(size_t)lk];
  }

  MPI_Bcast(pivot_row.data(), (int)N, MPI_DOUBLE, owner_k, comm);
  MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner_k, comm);
}

// Naive MPI Gaussian elimination with partial pivoting.
static bool mpi_lu_elim_partial_pivot(int64_t N, const Dist1D& dist,
                                     std::vector<double>& A_local,
                                     std::vector<double>& b_local,
				     Timers& T,
                                     MPI_Comm comm) {
  int rank = dist.rank;

  std::vector<double> pivot_row;
  double pivot_b = 0.0;

  for (int64_t k = 0; k < N; ++k) {
    // Find global pivot in column k among rows i>=k
    // Local search on owned rows
    double t0 = now_sec();
    double local_max = -1.0;
    int local_idx = -1; // global row index of local best
    int64_t start = dist.displs[rank];
    int64_t end   = start + dist.counts[rank];

    for (int64_t i = std::max<int64_t>(k, start); i < end; ++i) {
      int64_t li = dist.local_index(i);
      double v = std::abs(A_local[lidx(li, k, N)]);
      if (v > local_max) { local_max = v; local_idx = (int)i; }
    }

    struct { double val; int idx; } in, out;
    in.val = local_max;
    in.idx = local_idx;

    // MAXLOC gives the rank's idx associated with max val
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

    T.pivot += (now_sec()-t0);

    int64_t piv = (int64_t)out.idx;
    if (out.idx < 0 || !(out.val > 0.0) || !std::isfinite(out.val)) {
      if (rank == 0) std::cerr << "Pivot failed at k=" << k << "\n";
      return false;
    }

    // Swap row k with pivot row globally
    t0 = now_sec();
    swap_global_rows(N, dist, A_local, b_local, k, piv, comm);
    T.swap += (now_sec()-t0);

    // Broadcast pivot row k to all ranks
    t0 = now_sec();
    bcast_pivot_row(N, dist, A_local, b_local, k, pivot_row, pivot_b, comm);
    T.bcast += (now_sec()-t0);

    double Akk = pivot_row[(size_t)k];
    if (Akk == 0.0 || !std::isfinite(Akk)) {
      if (rank == 0) std::cerr << "Singular/NaN pivot at k=" << k << "\n";
      return false;
    }

    // Update local rows i>k
    t0 = now_sec();
    for (int64_t i = std::max<int64_t>(k + 1, start); i < end; ++i) {
      int64_t li = dist.local_index(i);

      double Lik = A_local[lidx(li, k, N)] / Akk;
      A_local[lidx(li, k, N)] = Lik;  // store L

      // A[i, j] -= Lik * A[k, j]
      for (int64_t j = k + 1; j < N; ++j) {
        A_local[lidx(li, j, N)] -= Lik * pivot_row[(size_t)j];
      }
      // b[i] -= Lik * b[k]
      b_local[(size_t)li] -= Lik * pivot_b;
    }
    T.update += (now_sec()-t0);
  }
  return true;
}

// Gather U and b to root,then do back substitution on root, then broadcast x.
static void gather_and_backsolve(int64_t N, const Dist1D& dist,
                                 const std::vector<double>& A_local,
                                 const std::vector<double>& b_local,
                                 std::vector<double>& x,
                                 MPI_Comm comm) {
  int rank = dist.rank;
  int size = dist.size;

  // Gather full A and b on root
  std::vector<int> recvcountsA(size, 0), displsA(size, 0);
  std::vector<int> recvcountsB(size, 0), displsB(size, 0);
  for (int r = 0; r < size; ++r) {
    recvcountsA[r] = dist.counts[r] * (int)N;
    displsA[r]     = dist.displs[r] * (int)N;
    recvcountsB[r] = dist.counts[r];
    displsB[r]     = dist.displs[r];
  }

  std::vector<double> A_full, b_full;
  if (rank == 0) {
    A_full.assign((size_t)N * (size_t)N, 0.0);
    b_full.assign((size_t)N, 0.0);
  }

  MPI_Gatherv(A_local.data(), (int)(dist.local_rows() * N), MPI_DOUBLE,
              rank == 0 ? A_full.data() : nullptr, recvcountsA.data(), displsA.data(),
              MPI_DOUBLE, 0, comm);

  MPI_Gatherv(b_local.data(), (int)dist.local_rows(), MPI_DOUBLE,
              rank == 0 ? b_full.data() : nullptr, recvcountsB.data(), displsB.data(),
              MPI_DOUBLE, 0, comm);

  if (rank == 0) {
    x.assign((size_t)N, 0.0);
    // Back substitution solve Ux = b_full
    for (int64_t ii = 0; ii < N; ++ii) {
      int64_t i = N - 1 - ii;
      double sum = b_full[(size_t)i];
      for (int64_t j = i + 1; j < N; ++j) {
        sum -= A_full[(size_t)i * (size_t)N + (size_t)j] * x[(size_t)j];
      }
      double Uii = A_full[(size_t)i * (size_t)N + (size_t)i];
      x[(size_t)i] = sum / Uii;
    }
  } else {
    x.assign((size_t)N, 0.0);
  }

  MPI_Bcast(x.data(), (int)N, MPI_DOUBLE, 0, comm);
}

// Compute relative residual
// To keep MPI code simple, recompute A and b deterministically on root and broadcast x already.
// Here do the residual distributed by regenerating A and b on root and scattering.
static double residual_check_distributed(int64_t N, uint64_t seed,
                                         const Dist1D& dist,
                                         const std::vector<double>& x,
                                         MPI_Comm comm) {
  // Recreate and scatter A0 and b0
  std::vector<double> A0_local, b0_local;
  generate_on_root_and_scatter(N, seed, dist, A0_local, b0_local, comm);

  double local_norm_r2 = 0.0, local_normA_F2 = 0.0, local_normx2 = 0.0, local_normb2 = 0.0;
  for (int64_t i_local = 0; i_local < dist.local_rows(); ++i_local) {
    int64_t i_global = dist.displs[dist.rank] + i_local;

    double ax = 0.0;
    for (int64_t j = 0; j < N; ++j) {
      double aij = A0_local[lidx(i_local, j, N)];
      ax += aij * x[(size_t)j];
      local_normA_F2 += aij * aij;
    }
    double ri = ax - b0_local[(size_t)i_local];
    local_norm_r2 += ri * ri;

    local_normb2 += b0_local[(size_t)i_local] * b0_local[(size_t)i_local];
  }
  for (int64_t j = 0; j < N; ++j) local_normx2 += x[(size_t)j] * x[(size_t)j];

  double norm_r2 = 0.0, normA_F2 = 0.0, normx2 = 0.0, normb2 = 0.0;
  MPI_Allreduce(&local_norm_r2, &norm_r2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_normA_F2, &normA_F2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_normx2, &normx2, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&local_normb2, &normb2, 1, MPI_DOUBLE, MPI_SUM, comm);

  double norm_r = std::sqrt(norm_r2);
  double normA_F = std::sqrt(normA_F2);
  double normx = std::sqrt(normx2);
  double normb = std::sqrt(normb2);
  double denom = normA_F * normx + normb;
  if (denom == 0.0) return norm_r;
  return norm_r / denom;
}

static void usage(const char* prog) {
  if (prog) {
    std::cerr << "Usage: " << prog << " N [seed] [--no-check]\n"
              << "  N          : matrix size\n"
              << "  seed       : optional RNG seed (default 1)\n"
              << "  --no-check : skip residual check\n";
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2) {
    if (rank == 0) usage(argv[0]);
    MPI_Finalize();
    return 1;
  }

  int64_t N = 0;
  try {
    N = (int64_t)std::stoll(argv[1]);
  } catch (...) {
    if (rank == 0) usage(argv[0]);
    MPI_Finalize();
    return 1;
  }

  uint64_t seed = 1;
  if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
    seed = (uint64_t)std::stoull(argv[2]);
  }

  bool do_check = true;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-check") do_check = false;
  }

  Dist1D dist;
  dist.init(N, rank, size);

  std::vector<double> A_local, b_local;
  generate_on_root_and_scatter(N, seed, dist, A_local, b_local, MPI_COMM_WORLD);

  Timers T;
  MPI_Barrier(MPI_COMM_WORLD);
  double t_total0=now_sec();

  bool ok = mpi_lu_elim_partial_pivot(N, dist, A_local, b_local, T,MPI_COMM_WORLD);
  if (!ok) {
    if (rank == 0) std::cerr << "Elimination failed.\n";
    MPI_Finalize();
    return 2;
  }

  std::vector<double> x;
  double t0 = now_sec();
  gather_and_backsolve(N, dist, A_local, b_local, x, MPI_COMM_WORLD);
  double t1 = now_sec();
  T.solve += (t1 - t0);

  MPI_Barrier(MPI_COMM_WORLD);
  double t_total1 = now_sec();
  double total = t_total1-t_total0;

  // Take the longest time among all ranks
  double total_max = 0.0;
  MPI_Reduce(&total, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Use critical path time as the actual runtime.
  double sec = total_max;

  // Approx Linpack flops
  long double flops = (2.0L / 3.0L) * (long double)N * (long double)N * (long double)N;
  long double gflops = (flops / sec) / 1.0e9L;

  double relres = 0.0;
  if (do_check) {
    relres = residual_check_distributed(N, seed, dist, x, MPI_COMM_WORLD);
  }

  double pivot_max = 0.0, swap_max = 0.0, bcast_max = 0.0, update_max = 0.0, solve_max = 0.0;

  MPI_Reduce(&T.pivot,  &pivot_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&T.swap,   &swap_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&T.bcast,  &bcast_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&T.update, &update_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&T.solve,  &solve_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "MiniHPL-MPI\n";
    std::cout << "MPI ranks     : " << size << "\n";
    std::cout << "N             : " << N << "\n";
    std::cout << "Time (s)      : " << sec << "\n";
    std::cout << "Perf (GF/s)   : " << (double)gflops << "\n";
    std::cout << "Breakdown (MAX over ranks)\n";
    std::cout << "  pivot_allreduce_s : " << pivot_max << "\n";
    std::cout << "  swap_rows_s       : " << swap_max << "\n";
    std::cout << "  bcast_pivot_s     : " << bcast_max << "\n";
    std::cout << "  update_s          : " << update_max << "\n";
    std::cout << "  gather_solve_s    : " << solve_max << "\n";
    std::cout << "CSV," << size << "," << N << "," << sec << ","
          << pivot_max << "," << swap_max << "," << bcast_max << ","
          << update_max << "," << solve_max << "\n";
    if (do_check) {
      std::cout << "RelResidual   : " << std::scientific << relres << "\n";
    } else {
      std::cout << "RelResidual   : (skipped)\n";
    }
  }

  MPI_Finalize();
  return 0;
}
