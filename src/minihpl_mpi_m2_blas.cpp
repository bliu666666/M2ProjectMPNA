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

#include <cblas.h>

struct Timers {
  double diag_lu = 0.0;
  double bcast_diag_row = 0.0;
  double bcast_diag_col = 0.0;
  double trsm_L = 0.0;
  double trsm_U = 0.0;
  double gemm = 0.0;
};

static inline double now_sec() { return MPI_Wtime(); }

static void choose_grid(int size, int &P, int &Q) {
  int root = (int)std::sqrt((double)size);
  for (int p = root; p >= 1; --p) if (size % p == 0) { P=p; Q=size/p; return; }
  P=1; Q=size;
}

struct Grid2D {
  int rank=0,size=1,P=1,Q=1,pr=0,pc=0;
  MPI_Comm cart=MPI_COMM_NULL,row_comm=MPI_COMM_NULL,col_comm=MPI_COMM_NULL;

  void init(MPI_Comm world){
    MPI_Comm_rank(world,&rank);
    MPI_Comm_size(world,&size);
    choose_grid(size,P,Q);
    int dims[2]={P,Q}, periods[2]={0,0};
    MPI_Cart_create(world,2,dims,periods,1,&cart);
    int coords[2]; MPI_Cart_coords(cart,rank,2,coords);
    pr=coords[0]; pc=coords[1];
    int rem_row[2]={0,1}; MPI_Cart_sub(cart,rem_row,&row_comm);
    int rem_col[2]={1,0}; MPI_Cart_sub(cart,rem_col,&col_comm);
  }
  inline int owner_pr(int bi) const { return bi % P; }
  inline int owner_pc(int bj) const { return bj % Q; }
  inline bool owns_block(int bi,int bj) const { return owner_pr(bi)==pr && owner_pc(bj)==pc; }
  inline int row_root(int pc_root) const { return pc_root; }
  inline int col_root(int pr_root) const { return pr_root; }
};

static inline double hash_double(int64_t i, int64_t j, uint64_t seed) {
  uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ULL ^ (uint64_t)j * 0xBF58476D1CE4E5B9ULL ^ seed;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27; x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  double u = (double)(x >> 11) * (1.0 / 9007199254740992.0);
  return 2.0*u - 1.0;
}

struct LocalBlocks {
  int N=0,NB=0,nb=0,lnb_i=0,lnb_j=0;
  std::vector<double> data;

  void init(int N_,int NB_, const Grid2D &g){
    N=N_; NB=NB_; nb=N/NB;
    lnb_i = (nb - g.pr + g.P - 1) / g.P;
    lnb_j = (nb - g.pc + g.Q - 1) / g.Q;
    data.assign((size_t)lnb_i*(size_t)lnb_j*(size_t)NB*(size_t)NB, 0.0);
  }
  inline double* block_ptr(int lbi,int lbj){
    size_t off=((size_t)lbi*(size_t)lnb_j+(size_t)lbj)*(size_t)NB*(size_t)NB;
    return data.data()+off;
  }
  inline int lbi_of(int bi,const Grid2D& g) const { return (bi-g.pr)/g.P; }
  inline int lbj_of(int bj,const Grid2D& g) const { return (bj-g.pc)/g.Q; }
};

static void fill_A(LocalBlocks &A, const Grid2D& g, uint64_t seed){
  int NB=A.NB;
  for(int lbi=0;lbi<A.lnb_i;++lbi){
    int bi=g.pr + lbi*g.P;
    for(int lbj=0;lbj<A.lnb_j;++lbj){
      int bj=g.pc + lbj*g.Q;
      double* B=A.block_ptr(lbi,lbj);
      int64_t i0=(int64_t)bi*NB, j0=(int64_t)bj*NB;
      for(int jj=0;jj<NB;++jj){
        for(int ii=0;ii<NB;++ii){
          int64_t gi=i0+ii, gj=j0+jj;
          B[ii + (size_t)jj*NB] = hash_double(gi,gj,seed);
        }
      }
      if(bi==bj){
        for(int d=0;d<NB;++d) B[d + (size_t)d*NB] += (double)NB;
      }
    }
  }
}

static bool lu_nopiv_block(double* A, int NB){
  for(int k=0;k<NB;++k){
    double Akk=A[k + (size_t)k*NB];
    if(Akk==0.0 || !std::isfinite(Akk)) return false;
    for(int i=k+1;i<NB;++i){
      A[i + (size_t)k*NB] /= Akk;
      double Lik=A[i + (size_t)k*NB];
      for(int j=k+1;j<NB;++j){
        A[i + (size_t)j*NB] -= Lik * A[k + (size_t)j*NB];
      }
    }
  }
  return true;
}

int main(int argc,char** argv){
  MPI_Init(&argc,&argv);
  int rank=0,size=1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if(argc<3){
    if(rank==0) std::cerr<<"Usage: "<<argv[0]<<" N NB [seed]  (N%NB==0)\n";
    MPI_Finalize(); return 1;
  }
  int N=std::stoi(argv[1]);
  int NB=std::stoi(argv[2]);
  uint64_t seed=1; if(argc>=4) seed=(uint64_t)std::stoull(argv[3]);
  if(N%NB!=0){ if(rank==0) std::cerr<<"Error: N%NB!=0\n"; MPI_Finalize(); return 2; }

  Grid2D g; g.init(MPI_COMM_WORLD);
  if(rank==0){
    std::cout<<"MiniHPL-MPI M3 (2D block-cyclic + BLAS3, no pivot)\n";
    std::cout<<"Ranks="<<size<<" Grid="<<g.P<<"x"<<g.Q<<"\n";
  }

  LocalBlocks A; A.init(N,NB,g);
  fill_A(A,g,seed);
  int nb=A.nb;

  Timers T;
  std::vector<double> diag((size_t)NB*(size_t)NB,0.0);
  std::vector<double> Lkk((size_t)NB*(size_t)NB,0.0);
  std::vector<double> Ukk((size_t)NB*(size_t)NB,0.0);

  MPI_Barrier(MPI_COMM_WORLD);
  double t_total0=now_sec();

  for(int k=0;k<nb;++k){
    int opr=k%g.P, opc=k%g.Q;

    // diag LU on owner
    double t0=now_sec();
    if(g.pr==opr && g.pc==opc){
      int lbi=A.lbi_of(k,g), lbj=A.lbj_of(k,g);
      double* Akk=A.block_ptr(lbi,lbj);
      std::memcpy(diag.data(),Akk,(size_t)NB*(size_t)NB*sizeof(double));
      if(!lu_nopiv_block(diag.data(),NB)){ MPI_Abort(MPI_COMM_WORLD,3); }
      std::memcpy(Akk,diag.data(),(size_t)NB*(size_t)NB*sizeof(double));
    }
    T.diag_lu += (now_sec()-t0);

    // bcast diag along row (Lkk) and col (Ukk)
    int row_root=g.row_root(opc);
    if(g.pr==opr){
      if(g.pc==opc){
        int lbi=A.lbi_of(k,g), lbj=A.lbj_of(k,g);
        std::memcpy(Lkk.data(), A.block_ptr(lbi,lbj), (size_t)NB*(size_t)NB*sizeof(double));
      }
      t0=now_sec();
      MPI_Bcast(Lkk.data(), NB*NB, MPI_DOUBLE, row_root, g.row_comm);
      T.bcast_diag_row += (now_sec()-t0);
    }

    int col_root=g.col_root(opr);
    if(g.pc==opc){
      if(g.pr==opr){
        int lbi=A.lbi_of(k,g), lbj=A.lbj_of(k,g);
        std::memcpy(Ukk.data(), A.block_ptr(lbi,lbj), (size_t)NB*(size_t)NB*sizeof(double));
      }
      t0=now_sec();
      MPI_Bcast(Ukk.data(), NB*NB, MPI_DOUBLE, col_root, g.col_comm);
      T.bcast_diag_col += (now_sec()-t0);
    }

    // TRSM row panel
    if(g.pr==opr){
      t0=now_sec();
      for(int j=k+1;j<nb;++j){
        if(g.owns_block(k,j)){
          int lbi=A.lbi_of(k,g), lbj=A.lbj_of(j,g);
          double* Akj=A.block_ptr(lbi,lbj);
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                      NB, NB, 1.0, Lkk.data(), NB, Akj, NB);
        }
      }
      T.trsm_L += (now_sec()-t0);
    }

    // TRSM col panel
    if(g.pc==opc){
      t0=now_sec();
      for(int i=k+1;i<nb;++i){
        if(g.owns_block(i,k)){
          int lbi=A.lbi_of(i,g), lbj=A.lbj_of(k,g);
          double* Aik=A.block_ptr(lbi,lbj);
          cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                      NB, NB, 1.0, Ukk.data(), NB, Aik, NB);
        }
      }
      T.trsm_U += (now_sec()-t0);
    }

    // Trailing update
    t0=now_sec();
    std::vector<double> Lik((size_t)NB*(size_t)NB,0.0);
    std::vector<double> Ukj((size_t)NB*(size_t)NB,0.0);

    for(int i=k+1;i<nb;++i){
      int pr_i = i % g.P;
      int root_pc_for_Lik = opc;

      if(g.pr==pr_i){
        if(g.pc==root_pc_for_Lik && g.owns_block(i,k)){
          int lbi=A.lbi_of(i,g), lbj=A.lbj_of(k,g);
          std::memcpy(Lik.data(), A.block_ptr(lbi,lbj), (size_t)NB*(size_t)NB*sizeof(double));
        }
        MPI_Bcast(Lik.data(), NB*NB, MPI_DOUBLE, g.row_root(root_pc_for_Lik), g.row_comm);
      }

      for(int j=k+1;j<nb;++j){
        int pc_j = j % g.Q;
        int root_pr_for_Ukj = opr;

        if(g.pc==pc_j){
          if(g.pr==root_pr_for_Ukj && g.owns_block(k,j)){
            int lbi=A.lbi_of(k,g), lbj=A.lbj_of(j,g);
            std::memcpy(Ukj.data(), A.block_ptr(lbi,lbj), (size_t)NB*(size_t)NB*sizeof(double));
          }
          MPI_Bcast(Ukj.data(), NB*NB, MPI_DOUBLE, g.col_root(root_pr_for_Ukj), g.col_comm);
        }

        if(g.owns_block(i,j)){
          int lbi=A.lbi_of(i,g), lbj=A.lbj_of(j,g);
          double* Aij=A.block_ptr(lbi,lbj);
          // Aij -= Lik * Ukj
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                      NB, NB, NB, -1.0, Lik.data(), NB, Ukj.data(), NB,
                      1.0, Aij, NB);
        }
      }
    }
    T.gemm += (now_sec()-t0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double total = now_sec()-t_total0;

  double total_max=0.0;
  MPI_Reduce(&total,&total_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  auto redmax=[&](double v){ double m=0.0; MPI_Reduce(&v,&m,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); return m; };
  double lu_m=redmax(T.diag_lu), bdr_m=redmax(T.bcast_diag_row), bdc_m=redmax(T.bcast_diag_col),
         trL_m=redmax(T.trsm_L), trU_m=redmax(T.trsm_U), gemm_m=redmax(T.gemm);

  long double flops=(2.0L/3.0L)*(long double)N*(long double)N*(long double)N;
  long double gflops=(flops/(long double)total_max)/1.0e9L;

  if(rank==0){
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"N="<<N<<" NB="<<NB<<" Grid="<<g.P<<"x"<<g.Q<<"\n";
    std::cout<<"Time(s)="<<total_max<<"  GF/s="<<(double)gflops<<"\n";
    std::cout<<"Breakdown(MAX): diagLU="<<lu_m
             <<" bcastDiagRow="<<bdr_m
             <<" bcastDiagCol="<<bdc_m
             <<" trsmL="<<trL_m
             <<" trsmU="<<trU_m
             <<" gemm="<<gemm_m<<"\n";
    std::cout<<"CSV,"<<size<<","<<N<<","<<total_max<<","<<lu_m<<","<<bdr_m<<","<<bdc_m<<","<<trL_m<<","<<trU_m<<","<<gemm_m<<"\n";
  }

  if(g.row_comm!=MPI_COMM_NULL) MPI_Comm_free(&g.row_comm);
  if(g.col_comm!=MPI_COMM_NULL) MPI_Comm_free(&g.col_comm);
  if(g.cart!=MPI_COMM_NULL) MPI_Comm_free(&g.cart);

  MPI_Finalize();
  return 0;
}

