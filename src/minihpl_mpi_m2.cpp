#include <mpi.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

struct Timers {
  double diag_lu = 0.0;
  double bcast_diag_row = 0.0; 
  double bcast_diag_col = 0.0; 
  double trsm_L = 0.0;         // Ukj = inv(Lkk) * Akj
  double trsm_U = 0.0;         // Lik = Aik * inv(Ukk)
  double bcast_Lik = 0.0;    
  double bcast_Ukj = 0.0;      
  double gemm = 0.0;           // Aij -= Lik*Ukj
};

static inline double now_sec() { return MPI_Wtime(); }

// ---------- choose process grid P x Q ----------
static void choose_grid(int size, int &P, int &Q) {
  int root = (int)std::sqrt((double)size);
  for (int p = root; p >= 1; --p) {
    if (size % p == 0) { P = p; Q = size / p; return; }
  }
  P = 1; Q = size;
}

// ---------- block-cyclic owner and local indexing ----------
struct Grid2D {
  int rank=0, size=1;
  int P=1, Q=1;
  int pr=0, pc=0;
  MPI_Comm cart = MPI_COMM_NULL;
  MPI_Comm row_comm = MPI_COMM_NULL; 
  MPI_Comm col_comm = MPI_COMM_NULL; 

  void init(MPI_Comm world) {
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &size);
    choose_grid(size, P, Q);

    int dims[2] = {P, Q};
    int periods[2] = {0, 0};
    MPI_Cart_create(world, 2, dims, periods, 1, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    pr = coords[0]; pc = coords[1];

    // row_comm: keep pc varying, pr fixed
    int remain_row[2] = {0, 1};
    MPI_Cart_sub(cart, remain_row, &row_comm);
    // col_comm: keep pr varying, pc fixed
    int remain_col[2] = {1, 0};
    MPI_Cart_sub(cart, remain_col, &col_comm);
  }

  // Owner of block (bi,bj)
  inline int owner_pr(int bi) const { return bi % P; }
  inline int owner_pc(int bj) const { return bj % Q; }
  inline bool owns_block(int bi, int bj) const {
    return owner_pr(bi) == pr && owner_pc(bj) == pc;
  }
  inline int row_root_in_rowcomm(int pc_root) const { return pc_root; } // ranks in row_comm are ordered by pc
  inline int col_root_in_colcomm(int pr_root) const { return pr_root; } // ranks in col_comm are ordered by pr
};

static inline int64_t blk_idx(int bi, int bj, int nb) { return (int64_t)bi * nb + bj; }

// ---------- deterministic diagonally dominant matrix generation per element ----------
static inline double hash_double(int64_t i, int64_t j, uint64_t seed) {
  // simple 64-bit mix to get pseudo-random in [-1,1]
  uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ULL ^ (uint64_t)j * 0xBF58476D1CE4E5B9ULL ^ seed;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27; x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  // map to [0,1)
  double u = (double)(x >> 11) * (1.0 / 9007199254740992.0);
  return 2.0*u - 1.0;
}

// ---------- local block storage (column-major inside each NBxNB block) ----------
struct LocalBlocks {
  int N=0, NB=0;
  int nb=0;                 // number of blocks per dimension
  int lnb_i=0, lnb_j=0;      // local block grid sizes
  std::vector<double> data; // contiguous storage for all local blocks
  std::vector<int> present; // mark whether block exists locally or not
  // local layout: blocks are stored in a dense local block grid [lbi][lbj] even if some are empty.

  void init(int N_, int NB_, const Grid2D &g) {
    N=N_; NB=NB_;
    nb = N / NB;

    // local block counts:
    // block rows owned: bi where bi%P==pr => count = ceil((nb-pr)/P)
    lnb_i = (nb - g.pr + g.P - 1) / g.P;
    lnb_j = (nb - g.pc + g.Q - 1) / g.Q;

    present.assign(lnb_i*lnb_j, 1); // by construction, every local index corresponds to an owned block
    data.assign((size_t)lnb_i*(size_t)lnb_j*(size_t)NB*(size_t)NB, 0.0);
  }

  inline double* block_ptr(int lbi, int lbj) {
    size_t offset = ((size_t)lbi*(size_t)lnb_j + (size_t)lbj) * (size_t)NB*(size_t)NB;
    return data.data() + offset;
  }
  inline const double* block_ptr(int lbi, int lbj) const {
    size_t offset = ((size_t)lbi*(size_t)lnb_j + (size_t)lbj) * (size_t)NB*(size_t)NB;
    return data.data() + offset;
  }

  // map global block index -> local block index (if owned)
  inline int lbi_of(int bi, const Grid2D &g) const { return (bi - g.pr) / g.P; }
  inline int lbj_of(int bj, const Grid2D &g) const { return (bj - g.pc) / g.Q; }
};

// Fill local blocks of A with diagonally dominant values
static void fill_A(LocalBlocks &A, const Grid2D &g, uint64_t seed) {
  int N=A.N, NB=A.NB, nb=A.nb;
  for(int lbi=0; lbi<A.lnb_i; ++lbi){
    int bi = g.pr + lbi*g.P;
    for(int lbj=0; lbj<A.lnb_j; ++lbj){
      int bj = g.pc + lbj*g.Q;
      double* B = A.block_ptr(lbi, lbj);
      // global indices range
      int64_t i0 = (int64_t)bi*NB;
      int64_t j0 = (int64_t)bj*NB;
      // column-major fill
      for(int jj=0;jj<NB;++jj){
        for(int ii=0;ii<NB;++ii){
          int64_t gi=i0+ii, gj=j0+jj;
          double v = hash_double(gi, gj, seed);
          B[ii + (size_t)jj*NB] = v;
        }
      }
      // add to diagonal blocks
      if(bi==bj){
        for(int d=0; d<NB; ++d){
          B[d + (size_t)d*NB] += (double)NB;
        }
      }
    }
  }
}

// ---------- Unblocked LU on a NBxNB block (no pivot), in-place column-major ----------
// Produces L and U packed in A.
static bool lu_nopiv_block(double* A, int NB) {
  for(int k=0;k<NB;++k){
    double Akk = A[k + (size_t)k*NB];
    if (Akk == 0.0 || !std::isfinite(Akk)) return false;
    for(int i=k+1;i<NB;++i){
      A[i + (size_t)k*NB] /= Akk; 
      double Lik = A[i + (size_t)k*NB];
      for(int j=k+1;j<NB;++j){
        A[i + (size_t)j*NB] -= Lik * A[k + (size_t)j*NB];
      }
    }
  }
  return true;
}

// ---------- naive TRSM/GEMM on blocks ----------
static void trsm_left_lower_unit(int NB, const double* L, double* B) {
  // Solve L * X = B, L lower unit, overwrite B <- X
  for(int col=0; col<NB; ++col){
    for(int i=0;i<NB;++i){
      double sum = B[i + (size_t)col*NB];
      for(int k=0;k<i;++k){
        sum -= L[i + (size_t)k*NB] * B[k + (size_t)col*NB];
      }
      // diag is 1
      B[i + (size_t)col*NB] = sum;
    }
  }
}

static void trsm_right_upper_nounit(int NB, const double* U, double* B) {
  // Solve X * U = B  => X = B * inv(U). Overwrite B <- X.
  for(int r=0;r<NB;++r){
    // back substitution on columns of U
    for(int k=NB-1; k>=0; --k){
      double sum = B[r + (size_t)k*NB];
      for(int j=k+1;j<NB;++j){
        sum -= B[r + (size_t)j*NB] * U[k + (size_t)j*NB]; // U(k,j)
      }
      sum /= U[k + (size_t)k*NB];
      B[r + (size_t)k*NB] = sum;
    }
  }
}

static void gemm_sub(int NB, const double* A, const double* B, double* C) {
  for(int j=0;j<NB;++j){
    for(int i=0;i<NB;++i){
      double sum=0.0;
      for(int k=0;k<NB;++k){
        sum += A[i + (size_t)k*NB] * B[k + (size_t)j*NB];
      }
      C[i + (size_t)j*NB] -= sum;
    }
  }
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);

  int rank=0,size=1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if(argc < 3){
    if(rank==0){
      std::cerr << "Usage: " << argv[0] << " N NB [seed]\n"
                << "  Requires N % NB == 0\n";
    }
    MPI_Finalize();
    return 1;
  }

  int N = std::stoi(argv[1]);
  int NB = std::stoi(argv[2]);
  uint64_t seed = 1;
  if(argc >= 4) seed = (uint64_t)std::stoull(argv[3]);

  if(N % NB != 0){
    if(rank==0) std::cerr << "Error: N % NB != 0\n";
    MPI_Finalize();
    return 2;
  }

  Grid2D g; g.init(MPI_COMM_WORLD);
  if(rank==0){
    std::cout << "MiniHPL-MPI M2 (2D block-cyclic, no BLAS, no pivot)\n";
    std::cout << "Ranks="<<size<<"  Grid="<<g.P<<"x"<<g.Q<<"\n";
  }

  LocalBlocks A;
  A.init(N, NB, g);
  fill_A(A, g, seed);

  int nb = A.nb;
  Timers T;

  MPI_Barrier(MPI_COMM_WORLD);
  double t_total0 = now_sec();

  // Workspace buffers for broadcasts of blocks
  std::vector<double> diag_block((size_t)NB*(size_t)NB, 0.0);
  std::vector<double> Lkk((size_t)NB*(size_t)NB, 0.0); 
  std::vector<double> Ukk((size_t)NB*(size_t)NB, 0.0);

  // Buffers for Lik and Ukj broadcasts
  std::vector<double> bufA((size_t)NB*(size_t)NB, 0.0);

  // Main blocked LU (no pivot)
  for(int k=0; k<nb; ++k){
    int owner_pr = k % g.P;
    int owner_pc = k % g.Q;

    // Diagonal block factorization on owner
    double t0 = now_sec();
    if(g.pr == owner_pr && g.pc == owner_pc){
      int lbi = A.lbi_of(k, g);
      int lbj = A.lbj_of(k, g);
      double* Akk = A.block_ptr(lbi, lbj);
      std::memcpy(diag_block.data(), Akk, (size_t)NB*(size_t)NB*sizeof(double));
      bool ok = lu_nopiv_block(diag_block.data(), NB);
      if(!ok){
        std::cerr << "Rank "<<rank<<": LU failed at k="<<k<<"\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      // write back
      std::memcpy(Akk, diag_block.data(), (size_t)NB*(size_t)NB*sizeof(double));
    }
    T.diag_lu += (now_sec()-t0);

    // Broadcast diagonal block factors to row and column
    int row_root = g.row_root_in_rowcomm(owner_pc);
    int col_root = g.col_root_in_colcomm(owner_pr);

    if(g.pr == owner_pr){
      int lbi = (g.pr == owner_pr && g.pc == owner_pc) ? A.lbi_of(k,g) : 0;
      int lbj = (g.pr == owner_pr && g.pc == owner_pc) ? A.lbj_of(k,g) : 0;
      if(g.pc == owner_pc){
        double* Akk = A.block_ptr(lbi, lbj);
        std::memcpy(Lkk.data(), Akk, (size_t)NB*(size_t)NB*sizeof(double));
      }
      t0 = now_sec();
      MPI_Bcast(Lkk.data(), NB*NB, MPI_DOUBLE, row_root, g.row_comm);
      T.bcast_diag_row += (now_sec()-t0);
    }

    if(g.pc == owner_pc){
      int lbi = (g.pr == owner_pr && g.pc == owner_pc) ? A.lbi_of(k,g) : 0;
      int lbj = (g.pr == owner_pr && g.pc == owner_pc) ? A.lbj_of(k,g) : 0;
      if(g.pr == owner_pr){
        double* Akk = A.block_ptr(lbi, lbj);
        std::memcpy(Ukk.data(), Akk, (size_t)NB*(size_t)NB*sizeof(double));
      }
      t0 = now_sec();
      MPI_Bcast(Ukk.data(), NB*NB, MPI_DOUBLE, col_root, g.col_comm);
      T.bcast_diag_col += (now_sec()-t0);
    }

    // RSM on row panel
    if(g.pr == owner_pr){
      t0 = now_sec();
      for(int j = k+1; j<nb; ++j){
        if(g.owns_block(k, j)){
          int lbi = A.lbi_of(k, g);
          int lbj = A.lbj_of(j, g);
          double* Akj = A.block_ptr(lbi, lbj);
          trsm_left_lower_unit(NB, Lkk.data(), Akj);
        }
      }
      T.trsm_L += (now_sec()-t0);
    }

    // TRSM on col panel
    if(g.pc == owner_pc){
      t0 = now_sec();
      for(int i = k+1; i<nb; ++i){
        if(g.owns_block(i, k)){
          int lbi = A.lbi_of(i, g);
          int lbj = A.lbj_of(k, g);
          double* Aik = A.block_ptr(lbi, lbj);
          trsm_right_upper_nounit(NB, Ukk.data(), Aik);
        }
      }
      T.trsm_U += (now_sec()-t0);
    }

    // Trailing update with SUMMA-like broadcasts:
    for(int i = k+1; i<nb; ++i){
      break;
    }

    t0 = now_sec();
    for(int i = k+1; i<nb; ++i){
      int pr_i = i % g.P;
      int root_pc_for_Lik = owner_pc;
      if(g.pr == pr_i){
        if(g.pc == root_pc_for_Lik && g.owns_block(i, k)){
          int lbi = A.lbi_of(i, g);
          int lbj = A.lbj_of(k, g);
          std::memcpy(bufA.data(), A.block_ptr(lbi, lbj), (size_t)NB*(size_t)NB*sizeof(double));
        }
        int row_root = g.row_root_in_rowcomm(root_pc_for_Lik);
        MPI_Bcast(bufA.data(), NB*NB, MPI_DOUBLE, row_root, g.row_comm);
      }

      for(int j = k+1; j<nb; ++j){
        int pc_j = j % g.Q;
        int root_pr_for_Ukj = owner_pr;
        std::vector<double> bufB((size_t)NB*(size_t)NB, 0.0);
        if(g.pc == pc_j){
          if(g.pr == root_pr_for_Ukj && g.owns_block(k, j)){
            int lbi = A.lbi_of(k, g);
            int lbj = A.lbj_of(j, g);
            std::memcpy(bufB.data(), A.block_ptr(lbi, lbj), (size_t)NB*(size_t)NB*sizeof(double));
          }
          int col_root = g.col_root_in_colcomm(root_pr_for_Ukj);
          MPI_Bcast(bufB.data(), NB*NB, MPI_DOUBLE, col_root, g.col_comm);
        }

        // Update Aij if owned by this rank
        if(g.owns_block(i, j)){
          int lbi = A.lbi_of(i, g);
          int lbj = A.lbj_of(j, g);
          double* Aij = A.block_ptr(lbi, lbj);

          if(g.pr == pr_i && g.pc == pc_j){
            gemm_sub(NB, bufA.data(), bufB.data(), Aij);
          }
        }
      }
    }
    T.gemm += (now_sec()-t0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_total1 = now_sec();
  double total = t_total1 - t_total0;

  // critical path time
  double total_max=0.0;
  MPI_Reduce(&total,&total_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  // breakdown max
  auto redmax=[&](double v){
    double m=0.0; MPI_Reduce(&v,&m,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); return m;
  };
  double lu_m = redmax(T.diag_lu);
  double bdr_m = redmax(T.bcast_diag_row);
  double bdc_m = redmax(T.bcast_diag_col);
  double trL_m = redmax(T.trsm_L);
  double trU_m = redmax(T.trsm_U);
  double gemm_m = redmax(T.gemm);

  long double flops = (2.0L/3.0L)*(long double)N*(long double)N*(long double)N;
  long double gflops = (flops/ (long double)total_max)/1.0e9L;

  if(rank==0){
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"N="<<N<<" NB="<<NB<<" Grid="<<g.P<<"x"<<g.Q<<"\n";
    std::cout<<"Time(s)="<<total_max<<"  GF/s="<<(double)gflops<<"\n";
    std::cout<<"Breakdown(MAX): diagLU="<<lu_m
             << " bcastDiagRow="<<bdr_m
             << " bcastDiagCol="<<bdc_m
             << " trsmL="<<trL_m
             << " trsmU="<<trU_m
             << " gemm="<<gemm_m
             << "\n";
    std::cout<<"CSV,"<<size<<","<<N<<","<<total_max<<","<<lu_m<<","<<bdr_m<<","<<bdc_m<<","<<trL_m<<","<<trU_m<<","<<gemm_m<<"\n";
  }

  if(g.row_comm!=MPI_COMM_NULL) MPI_Comm_free(&g.row_comm);
  if(g.col_comm!=MPI_COMM_NULL) MPI_Comm_free(&g.col_comm);
  if(g.cart!=MPI_COMM_NULL) MPI_Comm_free(&g.cart);

  MPI_Finalize();
  return 0;
}

