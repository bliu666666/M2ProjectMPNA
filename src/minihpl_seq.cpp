#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Row-major indexing
static inline std::size_t idx(std::size_t i, std::size_t j, std::size_t n) {
  return i * n + j;
}

// Generate a well-behaved random dense matrix A and vector b.
// keep a copy of A0 for residual check.
static void generate_system(std::size_t n, uint64_t seed,
                            std::vector<double>& A,
                            std::vector<double>& A0,
                            std::vector<double>& b) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  A.assign(n * n, 0.0);
  b.assign(n, 0.0);

  for (std::size_t i = 0; i < n; ++i) {
    double rowsum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      double v = dist(rng);
      A[idx(i, j, n)] = v;
      rowsum += std::abs(v);
    }
    // Make it more diagonally dominant to reduce singularity risk.
    A[idx(i, i, n)] += rowsum + 1.0;

    b[i] = dist(rng);
  }

  A0 = A;
}

// LU factorization with partial pivoting in-place on A.
// Produces L and U packed into A:
// - L has unit diagonal, stored in strict lower part of A
// - U stored in upper part (including diagonal)
// Pivot array p records row swaps: after factorization, PA = LU.
static bool lu_factorize_partial_pivot(std::size_t n,
                                       std::vector<double>& A,
                                       std::vector<int>& p) {
  p.resize(n);
  for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<int>(i);

  for (std::size_t k = 0; k < n; ++k) {
    // Find pivot row
    std::size_t piv = k;
    double maxabs = std::abs(A[idx(k, k, n)]);
    for (std::size_t i = k + 1; i < n; ++i) {
      double v = std::abs(A[idx(i, k, n)]);
      if (v > maxabs) { maxabs = v; piv = i; }
    }

    if (maxabs == 0.0 || !std::isfinite(maxabs)) {
      return false; // singular or NaN/Inf
    }

    // Swap rows in A and record permutation
    if (piv != k) {
      for (std::size_t j = 0; j < n; ++j) {
        std::swap(A[idx(k, j, n)], A[idx(piv, j, n)]);
      }
      std::swap(p[k], p[piv]);
    }

    // Elimination
    double Akk = A[idx(k, k, n)];
    for (std::size_t i = k + 1; i < n; ++i) {
      A[idx(i, k, n)] /= Akk;              // L(i,k)
      double Lik = A[idx(i, k, n)];
      // Update trailing submatrix
      for (std::size_t j = k + 1; j < n; ++j) {
        A[idx(i, j, n)] -= Lik * A[idx(k, j, n)];
      }
    }
  }
  return true;
}

// Solve LU x = Pb
static void lu_solve(std::size_t n,
                     const std::vector<double>& LU,
                     const std::vector<int>& p,
                     const std::vector<double>& b,
                     std::vector<double>& x) {
  x.assign(n, 0.0);

  // Apply permutation: y = Pb
  std::vector<double> y(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = b[static_cast<std::size_t>(p[i])];
  }

  // Forward substitution: solve Lz = y
  std::vector<double> z(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = y[i];
    for (std::size_t j = 0; j < i; ++j) {
      sum -= LU[idx(i, j, n)] * z[j];
    }
    z[i] = sum; // since Lii = 1
  }

  // Back substitution: solve Ux = z
  for (std::size_t ii = 0; ii < n; ++ii) {
    std::size_t i = n - 1 - ii;
    double sum = z[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      sum -= LU[idx(i, j, n)] * x[j];
    }
    double Uii = LU[idx(i, i, n)];
    x[i] = sum / Uii;
  }
}

// Compute relative residual
static double relative_residual(std::size_t n,
                                const std::vector<double>& A0,
                                const std::vector<double>& x,
                                const std::vector<double>& b) {
  double norm_r2 = 0.0;
  double normA_F2 = 0.0;
  double normx2 = 0.0;
  double normb2 = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    double ax = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      double aij = A0[idx(i, j, n)];
      ax += aij * x[j];
      normA_F2 += aij * aij;
    }
    double ri = ax - b[i];
    norm_r2 += ri * ri;
    normx2 += x[i] * x[i];
    normb2 += b[i] * b[i];
  }

  double norm_r = std::sqrt(norm_r2);
  double normA_F = std::sqrt(normA_F2);
  double normx = std::sqrt(normx2);
  double normb = std::sqrt(normb2);

  double denom = normA_F * normx + normb;
  if (denom == 0.0) return norm_r;
  return norm_r / denom;
}

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " N [seed]\n"
            << "  N    : matrix size (e.g., 512, 1024, 2048)\n"
            << "  seed : optional RNG seed (default 1)\n";
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(argv[0]); return 1; }

  std::size_t N = 0;
  try {
    N = static_cast<std::size_t>(std::stoull(argv[1]));
  } catch (...) {
    usage(argv[0]); return 1;
  }
  uint64_t seed = 1;
  if (argc >= 3) seed = static_cast<uint64_t>(std::stoull(argv[2]));

  std::vector<double> A, A0, b, x;
  generate_system(N, seed, A, A0, b);

  std::vector<int> p;

  // Time LU + solve
  auto t0 = std::chrono::high_resolution_clock::now();
  bool ok = lu_factorize_partial_pivot(N, A, p);
  if (!ok) {
    std::cerr << "LU factorization failed (singular/NaN).\n";
    return 2;
  }
  lu_solve(N, A, p, b, x);
  auto t1 = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();

  // use 2/3 N^3 as standard approximation.
  long double flops = (2.0L / 3.0L) * (long double)N * (long double)N * (long double)N;
  long double gflops = (flops / sec) / 1.0e9L;

  double relres = relative_residual(N, A0, x, b);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "MiniHPL-SEQ\n";
  std::cout << "N            : " << N << "\n";
  std::cout << "Time (s)     : " << sec << "\n";
  std::cout << "Perf (GF/s)  : " << (double)gflops << "\n";
  std::cout << "RelResidual  : " << std::scientific << relres << "\n";

  // Simple correctness threshold
  if (!(relres < 1e-8)) {
    std::cerr << "Warning: residual is larger than expected.\n";
  }
  return 0;
}
