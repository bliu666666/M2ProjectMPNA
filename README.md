# MiniHPL – A Reimplementation of HPL (MPI + BLAS3)

MiniHPL is a simplified, readable reimplementation of the High Performance Linpack (HPL) benchmark developed as part of an HPC course project.  
The goal is **not to beat HPL**, but to **understand where HPL performance comes from** by rebuilding the algorithm step‑by‑step and comparing against the official implementation.

The project demonstrates how algorithmic structure, communication patterns, and BLAS3 kernels interact in distributed-memory dense linear algebra.

---

## What is HPL?

HPL (High Performance Linpack) solves a dense linear system:

```
Ax = b
```

using LU factorization with partial pivoting (in the official version) and distributed block-cyclic matrices.  
It is the benchmark used for the **TOP500 supercomputer ranking**.

HPL performance is dominated by:
- matrix multiplication (DGEMM)
- communication hiding
- scheduling and overlap

MiniHPL reconstructs these mechanisms in a controlled and analyzable way.

---

## Project Objectives

This repository aims to:

1. Reimplement a simplified HPL-like solver
2. Study performance bottlenecks
3. Apply incremental optimizations
4. Compare with the official HPL benchmark

Instead of a black-box benchmark, MiniHPL provides a **white-box performance model**.

---

## Implemented Versions

### 1. `minihpl_mpi.cpp` (Baseline)

- 1D data distribution
- MPI communication
- naive triple nested loops
- memory-bound behavior
- poor scalability

Purpose:
Understand the cost of communication and memory traffic.

---

### 2. `minihpl_mpi_m2.cpp` (2D Block-Cyclic)

- 2D process grid (P×Q)
- panel factorization
- row and column communicators
- reduced communication scope

Purpose:
Reproduce the ScaLAPACK distribution used by HPL.

---

### 3. `minihpl_mpi_m2_blas.cpp` (Optimized – Final Version)

- 2D block-cyclic distribution
- BLAS Level 3 (DGEMM, DTRSM)
- cache blocking
- compute-bound execution

Key transformation:

```
Naive update:
A[i][j] -= L[i][k] * U[k][j]

Optimized:
C = C − A × B  (DGEMM)
```

This converts the code from **memory-bound** to **compute-bound**.

---

## Parallel Strategy

- Programming model: **MPI**
- Platform: distributed-memory HPC cluster
- Process grid: P×Q (e.g., 4×4 for 16 ranks)
- Data layout: **2D block-cyclic distribution**
- Communication:
  - row communicator (panel U broadcast)
  - column communicator (panel L broadcast)

This mirrors the ScaLAPACK/HPL design.

---

## Performance Instrumentation

MiniHPL includes detailed timing breakdown:

- Panel LU factorization
- Row broadcast
- Column broadcast
- TRSM
- GEMM update

This allows identifying where time is spent and why HPL is fast.

---

## Build

Requirements:
- C++17 compiler
- MPI (OpenMPI recommended)
- BLAS library (BLIS / OpenBLAS / MKL)

Example (BLIS):

```bash
module load openmpi
module load blis

mpicxx -O3 -march=native -std=c++17 minihpl_mpi_m2_blas.cpp -o minihpl_mpi -lblis
```

---

## Run

Example:

```bash
mpirun -np 16 ./minihpl_mpi 20736 192 1
```

Arguments:

```
N    : matrix size
NB   : block size
SEED : random seed
```

---

## Example Output

```
MiniHPL-MPI M3
Ranks=16 Grid=4x4
N=20736 NB=192
Time(s)=34.77
GF/s=170.9

Breakdown(MAX):
diagLU=0.078
bcastRow=0.108
bcastCol=0.090
trsm=0.056
gemm=34.59
```

Observation:
> More than 90% of execution time is spent in GEMM.

This matches the theoretical design of HPL.

---

## Comparison with Official HPL

MiniHPL reproduces:

- GFLOPS vs block size behavior
- optimal NB region
- GEMM dominance

But official HPL is still faster because it includes:

- look-ahead panel factorization
- nonblocking communication
- communication/computation overlap
- dynamic scheduling
- optimized pivoting

In other words:

> MiniHPL implements the algorithm  
> HPL implements the performance engineering.

---

## Key Findings

1. Performance is not determined by algorithmic complexity (both O(N³))
2. Performance is dominated by BLAS3 kernels
3. Communication hiding is critical
4. Cache blocking is essential
5. HPL is a scheduling problem as much as a linear algebra problem

---


## Possible Extensions

- Nonblocking MPI (MPI_Ibcast)
- Look-ahead panel
- NUMA-aware placement
- Multi-threaded BLAS integration
- GPU offload (cuBLAS)
- Pivoting support

---