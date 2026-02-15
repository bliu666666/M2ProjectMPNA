#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

struct Dist1D {
  int rank=0, size=1;
  int64_t N=0;
  std::vector<int> counts, displs;

  void init(int64_t n, int r, int s) {
    N=n; rank=r; size=s;
    counts.assign(size,0);
    displs.assign(size,0);
    int64_t base=N/size, rem=N%size;
    for(int p=0;p<size;++p) counts[p]=(int)(base + (p<rem?1:0));
    displs[0]=0;
    for(int p=1;p<size;++p) displs[p]=displs[p-1]+counts[p-1];
  }
  inline int owner(int64_t gi) const {
    for(int r=0;r<size;++r){
      int64_t s=displs[r], e=s+counts[r];
      if(gi>=s && gi<e) return r;
    }
    return -1;
  }
  inline int64_t local_index(int64_t gi) const { return gi - displs[rank]; }
  inline int64_t local_rows() const { return counts[rank]; }
};

static inline int64_t lidx(int64_t li, int64_t j, int64_t N) { return li*N + j; }

static void generate_on_root_and_scatter(int64_t N, uint64_t seed,
                                        const Dist1D& dist,
                                        std::vector<double>& A_local,
                                        std::vector<double>& b_local,
                                        MPI_Comm comm) {
  int rank=dist.rank, size=dist.size;
  int64_t lr=dist.local_rows();
  A_local.assign((size_t)lr*(size_t)N,0.0);
  b_local.assign((size_t)lr,0.0);

  std::vector<double> A_full, b_full;
  if(rank==0){
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(-1.0,1.0);
    A_full.assign((size_t)N*(size_t)N,0.0);
    b_full.assign((size_t)N,0.0);
    for(int64_t i=0;i<N;++i){
      double rowsum=0.0;
      for(int64_t j=0;j<N;++j){
        double v=U(rng);
        A_full[(size_t)i*(size_t)N+(size_t)j]=v;
        rowsum += std::abs(v);
      }
      A_full[(size_t)i*(size_t)N+(size_t)i] += rowsum + 1.0;
      b_full[(size_t)i]=U(rng);
    }
  }

  std::vector<int> scA(size), dpA(size), scB(size), dpB(size);
  for(int r=0;r<size;++r){
    scA[r]=dist.counts[r]*(int)N;
    dpA[r]=dist.displs[r]*(int)N;
    scB[r]=dist.counts[r];
    dpB[r]=dist.displs[r];
  }

  MPI_Scatterv(rank==0?A_full.data():nullptr, scA.data(), dpA.data(), MPI_DOUBLE,
               A_local.data(), (int)(lr*N), MPI_DOUBLE, 0, comm);
  MPI_Scatterv(rank==0?b_full.data():nullptr, scB.data(), dpB.data(), MPI_DOUBLE,
               b_local.data(), (int)lr, MPI_DOUBLE, 0, comm);
}

static void swap_global_rows(int64_t N, const Dist1D& dist,
                             std::vector<double>& A_local,
                             std::vector<double>& b_local,
                             int64_t ra, int64_t rb,
                             MPI_Comm comm) {
  if(ra==rb) return;
  int oa=dist.owner(ra), ob=dist.owner(rb);
  int rank=dist.rank;

  if(oa==ob){
    if(rank==oa){
      int64_t la=dist.local_index(ra), lb=dist.local_index(rb);
      for(int64_t j=0;j<N;++j) std::swap(A_local[lidx(la,j,N)], A_local[lidx(lb,j,N)]);
      std::swap(b_local[(size_t)la], b_local[(size_t)lb]);
    }
    return;
  }

  std::vector<double> bufA, bufB;
  double bA=0, bB=0;

  if(rank==oa){
    int64_t la=dist.local_index(ra);
    bufA.resize((size_t)N);
    for(int64_t j=0;j<N;++j) bufA[(size_t)j]=A_local[lidx(la,j,N)];
    bA=b_local[(size_t)la];

    bufB.resize((size_t)N);
    MPI_Sendrecv(bufA.data(), (int)N, MPI_DOUBLE, ob, 100,
                 bufB.data(), (int)N, MPI_DOUBLE, ob, 101,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&bA,1,MPI_DOUBLE,ob,200,
                 &bB,1,MPI_DOUBLE,ob,201,
                 comm, MPI_STATUS_IGNORE);

    for(int64_t j=0;j<N;++j) A_local[lidx(la,j,N)]=bufB[(size_t)j];
    b_local[(size_t)la]=bB;
  }

  if(rank==ob){
    int64_t lb=dist.local_index(rb);
    bufB.resize((size_t)N);
    for(int64_t j=0;j<N;++j) bufB[(size_t)j]=A_local[lidx(lb,j,N)];
    bB=b_local[(size_t)lb];

    bufA.resize((size_t)N);
    MPI_Sendrecv(bufB.data(), (int)N, MPI_DOUBLE, oa, 101,
                 bufA.data(), (int)N, MPI_DOUBLE, oa, 100,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&bB,1,MPI_DOUBLE,oa,201,
                 &bA,1,MPI_DOUBLE,oa,200,
                 comm, MPI_STATUS_IGNORE);

    for(int64_t j=0;j<N;++j) A_local[lidx(lb,j,N)]=bufA[(size_t)j];
    b_local[(size_t)lb]=bA;
  }
}

// Broadcast a set of global rows [k, k+w) to all ranks.
// For each row r, owner(r) provides it.
static void bcast_panel_rows(int64_t N, const Dist1D& dist,
                             const std::vector<double>& A_local,
                             const std::vector<double>& b_local,
                             int64_t k, int64_t w,
                             std::vector<double>& panelA, 
                             std::vector<double>& panelB, 
                             MPI_Comm comm) {
  panelA.assign((size_t)w*(size_t)N, 0.0);
  panelB.assign((size_t)w, 0.0);

  // For each row, fill locally on its owner then bcast from that owner.
  std::vector<double> rowbuf((size_t)N, 0.0);
  for(int64_t rr=0; rr<w; ++rr){
    int64_t gr = k + rr;
    int root = dist.owner(gr);

    if(dist.rank==root){
      int64_t lr = dist.local_index(gr);
      for(int64_t j=0;j<N;++j) rowbuf[(size_t)j] = A_local[lidx(lr,j,N)];
      panelB[(size_t)rr] = b_local[(size_t)lr];
    }
    MPI_Bcast(rowbuf.data(), (int)N, MPI_DOUBLE, root, comm);
    MPI_Bcast(&panelB[(size_t)rr], 1, MPI_DOUBLE, root, comm);

    // Store broadcasted rowbuf
    std::memcpy(&panelA[(size_t)rr*(size_t)N], rowbuf.data(), (size_t)N*sizeof(double));
  }
}

// Blocked elimination: panel width NB, 1D row distribution.
// Pivoting is done inside each panel using global MAXLOC per column.
static bool mpi_lu_blocked_panel(int64_t N, int NB,
                                 const Dist1D& dist,
                                 std::vector<double>& A_local,
                                 std::vector<double>& b_local,
                                 Timers& T,
                                 MPI_Comm comm) {
  int rank=dist.rank;

  for(int64_t k0=0; k0<N; k0+=NB){
    int64_t w = std::min<int64_t>(NB, N-k0);

    // Panel factorization column by column inside panel
    for(int64_t kk=0; kk<w; ++kk){
      int64_t k = k0 + kk;

      // pivot select (global)
      double t0 = now_sec();

      double local_max=-1.0;
      int local_idx=-1;
      int64_t start=dist.displs[rank], end=start+dist.counts[rank];

      for(int64_t i=std::max<int64_t>(k, start); i<end; ++i){
        int64_t li = dist.local_index(i);
        double v = std::abs(A_local[lidx(li, k, N)]);
        if(v>local_max){ local_max=v; local_idx=(int)i; }
      }

      struct { double val; int idx; } in, out;
      in.val=local_max; in.idx=local_idx;
      MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
      T.pivot += (now_sec()-t0);

      int64_t piv = (int64_t)out.idx;
      if(out.idx<0 || !(out.val>0.0) || !std::isfinite(out.val)){
        if(rank==0) std::cerr<<"Pivot failed at k="<<k<<"\n";
        return false;
      }

      // swap rows k <-> piv
      t0 = now_sec();
      swap_global_rows(N, dist, A_local, b_local, k, piv, comm);
      T.swap += (now_sec()-t0);

      // broadcast pivot row k
      // For stability in panel factorization we need row k now:
      std::vector<double> pivot_row((size_t)N,0.0);
      double pivot_b=0.0;
      int root = dist.owner(k);
      if(rank==root){
        int64_t lk=dist.local_index(k);
        for(int64_t j=0;j<N;++j) pivot_row[(size_t)j]=A_local[lidx(lk,j,N)];
        pivot_b = b_local[(size_t)lk];
      }
      t0 = now_sec();
      MPI_Bcast(pivot_row.data(), (int)N, MPI_DOUBLE, root, comm);
      MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, root, comm);
      T.bcast += (now_sec()-t0);

      double Akk = pivot_row[(size_t)k];
      if(Akk==0.0 || !std::isfinite(Akk)){
        if(rank==0) std::cerr<<"Singular/NaN pivot at k="<<k<<"\n";
        return false;
      }

      // eliminate below pivot within panel columns
      t0 = now_sec();
      for(int64_t i=std::max<int64_t>(k+1, start); i<end; ++i){
        int64_t li=dist.local_index(i);
        double Lik = A_local[lidx(li,k,N)]/Akk;
        A_local[lidx(li,k,N)] = Lik;
        // Only update within panel columns (k+1..k0+w-1) for panel factorization
        for(int64_t j=k+1; j<k0+w; ++j){
          A_local[lidx(li,j,N)] -= Lik * pivot_row[(size_t)j];
        }
        b_local[(size_t)li] -= Lik * pivot_b;
      }
      T.update += (now_sec()-t0);
    }

    // After panel factorization, broadcast the whole panel rows [k0, k0+w)
    // so everyone can update trailing matrix block in fewer bcasts conceptually.
    std::vector<double> panelA, panelB;
    double t0 = now_sec();
    bcast_panel_rows(N, dist, A_local, b_local, k0, w, panelA, panelB, comm);
    T.bcast += (now_sec()-t0);

    // Trailing matrix update for columns j >= k0+w
    // For each local row i >= k0+w:
    // A(i, j) -= sum_{t=0..w-1} L(i,k0+t) * U(k0+t, j)
    // b(i)    -= sum_{t=0..w-1} L(i,k0+t) * b(k0+t)
    t0 = now_sec();
    int64_t start=dist.displs[rank], end=start+dist.counts[rank];
    for(int64_t i=std::max<int64_t>(k0+w, start); i<end; ++i){
      int64_t li = dist.local_index(i);

      // update b using panelB
      double delta_b = 0.0;
      for(int64_t t=0; t<w; ++t){
        double Lik = A_local[lidx(li, k0+t, N)];
        delta_b += Lik * panelB[(size_t)t];
      }
      b_local[(size_t)li] -= delta_b;

      // update A trailing cols
      for(int64_t j=k0+w; j<N; ++j){
        double sum = 0.0;
        for(int64_t t=0; t<w; ++t){
          double Lik = A_local[lidx(li, k0+t, N)];
          double Ukj = panelA[(size_t)t*(size_t)N + (size_t)j];
          sum += Lik * Ukj;
        }
        A_local[lidx(li, j, N)] -= sum;
      }
    }
    T.update += (now_sec()-t0);
  }

  return true;
}

static void gather_and_backsolve(int64_t N, const Dist1D& dist,
                                 const std::vector<double>& A_local,
                                 const std::vector<double>& b_local,
                                 std::vector<double>& x,
                                 MPI_Comm comm) {
  int rank=dist.rank, size=dist.size;

  std::vector<int> rcA(size), dpA(size), rcB(size), dpB(size);
  for(int r=0;r<size;++r){
    rcA[r]=dist.counts[r]*(int)N;
    dpA[r]=dist.displs[r]*(int)N;
    rcB[r]=dist.counts[r];
    dpB[r]=dist.displs[r];
  }

  std::vector<double> A_full, b_full;
  if(rank==0){
    A_full.assign((size_t)N*(size_t)N,0.0);
    b_full.assign((size_t)N,0.0);
  }

  MPI_Gatherv(A_local.data(), (int)(dist.local_rows()*N), MPI_DOUBLE,
              rank==0?A_full.data():nullptr, rcA.data(), dpA.data(), MPI_DOUBLE, 0, comm);
  MPI_Gatherv(b_local.data(), (int)dist.local_rows(), MPI_DOUBLE,
              rank==0?b_full.data():nullptr, rcB.data(), dpB.data(), MPI_DOUBLE, 0, comm);

  if(rank==0){
    x.assign((size_t)N,0.0);
    for(int64_t ii=0; ii<N; ++ii){
      int64_t i=N-1-ii;
      double sum=b_full[(size_t)i];
      for(int64_t j=i+1;j<N;++j){
        sum -= A_full[(size_t)i*(size_t)N+(size_t)j]*x[(size_t)j];
      }
      double Uii = A_full[(size_t)i*(size_t)N+(size_t)i];
      x[(size_t)i]=sum/Uii;
    }
  } else {
    x.assign((size_t)N,0.0);
  }
  MPI_Bcast(x.data(), (int)N, MPI_DOUBLE, 0, comm);
}

static void usage(const char* p){
  std::cerr<<"Usage: "<<p<<" N NB [seed] [--no-check]\n";
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int rank=0,size=1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  if(argc<3){
    if(rank==0) usage(argv[0]);
    MPI_Finalize();
    return 1;
  }

  int64_t N = std::stoll(argv[1]);
  int NB = std::stoi(argv[2]);
  uint64_t seed = 1;
  bool do_check=false; 
  for(int i=3;i<argc;++i){
    std::string a=argv[i];
    if(a=="--check") do_check=true;
    else if(a.rfind("--",0)!=0) seed = (uint64_t)std::stoull(a);
  }

  Dist1D dist; dist.init(N,rank,size);
  std::vector<double> A_local,b_local;
  generate_on_root_and_scatter(N, seed, dist, A_local, b_local, MPI_COMM_WORLD);

  Timers T;
  MPI_Barrier(MPI_COMM_WORLD);
  double t_total0 = now_sec();

  bool ok = mpi_lu_blocked_panel(N, NB, dist, A_local, b_local, T, MPI_COMM_WORLD);
  if(!ok){
    if(rank==0) std::cerr<<"Factorization failed\n";
    MPI_Finalize();
    return 2;
  }

  std::vector<double> x;
  double t0=now_sec();
  gather_and_backsolve(N, dist, A_local, b_local, x, MPI_COMM_WORLD);
  T.solve += (now_sec()-t0);

  MPI_Barrier(MPI_COMM_WORLD);
  double t_total1 = now_sec();
  double total = t_total1 - t_total0;

  // critical path time
  double total_max=0.0;
  MPI_Reduce(&total,&total_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  double sec = total_max;

  // breakdown max
  double pivot_max=0, swap_max=0, bcast_max=0, update_max=0, solve_max=0;
  MPI_Reduce(&T.pivot,&pivot_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&T.swap,&swap_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&T.bcast,&bcast_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&T.update,&update_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&T.solve,&solve_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  long double flops = (2.0L/3.0L)*(long double)N*(long double)N*(long double)N;
  long double gflops = (flops/sec)/1.0e9L;

  if(rank==0){
    std::cout<<std::fixed<<std::setprecision(6);
    std::cout<<"MiniHPL-MPI M1 Block1D\n";
    std::cout<<"MPI ranks     : "<<size<<"\n";
    std::cout<<"N             : "<<N<<"\n";
    std::cout<<"NB            : "<<NB<<"\n";
    std::cout<<"Time (s)      : "<<sec<<"\n";
    std::cout<<"Perf (GF/s)   : "<<(double)gflops<<"\n";
    std::cout<<"Breakdown (MAX)\n";
    std::cout<<"pivot_s : "<<pivot_max<<"\n";
    std::cout<<"swap_s  : "<<swap_max<<"\n";
    std::cout<<"bcast_s : "<<bcast_max<<"\n";
    std::cout<<"update_s: "<<update_max<<"\n";
    std::cout<<"solve_s : "<<solve_max<<"\n";
    std::cout<<"CSV,"<<size<<","<<N<<","<<sec<<","<<pivot_max<<","<<swap_max<<","<<bcast_max<<","<<update_max<<","<<solve_max<<"\n";
  }

  MPI_Finalize();
  return 0;
}

