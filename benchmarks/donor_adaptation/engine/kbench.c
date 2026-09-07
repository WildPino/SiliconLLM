// E10 -- pre-registered in docs/research/donor_adaptation/briefs/BRIEF_E10_PACKED_KERNEL_BINDER.md
// (pushed a0366d0, before this file existed).
//
// Is the packed matvec core-bound or stream-bound at ~22.5 GB/s?  Same matvec SOURCE, unmodified:
// this file #includes donor_engine.c with its main() renamed out of the way, so the kernel under
// test is byte-for-byte the one the engine runs.  Row length n_in is held FIXED across every cell
// so per-row and per-call cost is identical; only the number of rows -- the footprint -- moves,
// across the 16 MB L3 boundary probe-3 located.
//
// G-K0 is the PLANTED CONTROL and is read FIRST: if the fp32 arm does not rise by >=2.0 from its
// L3-resident cell to its DRAM cell, this bench cannot see residency at all and every packed cell
// it produced is void.
#define main donor_engine_main
#include "donor_engine.c"
#undef main

#define NIN 3584                 /* held fixed: Coder-7B's D */
static const double MB = 1024.0*1024.0;

typedef struct { const char* tag; double mb; } cell_t;
static const cell_t CELLS[] = {
    {"4MB",4},{"8MB",8},{"12MB",12},{"24MB",24},{"48MB",48},{"512MB",512},{"2048MB",2048}
};
#define NCELL (int)(sizeof(CELLS)/sizeof(CELLS[0]))

/* one cell: returns GB/s of weight bytes delivered */
static double run_cell(int packed,double mb,double* out_gbps_bytes){
    const size_t row_bytes = packed ? (size_t)(NIN/2) : (size_t)NIN*4;
    size_t n_out = (size_t)(mb*MB/(double)row_bytes);
    if(n_out<64) n_out=64;
    const size_t wbytes = n_out*row_bytes;

    mat_t m; memset(&m,0,sizeof(m));
    m.out=(int)n_out; m.in=NIN; m.packed=packed; m.tm=NULL; m.Mpad=0;
    void* W = xmalloc(wbytes);
    float* scale = (float*)xmalloc(n_out*4);
    float* x = (float*)xmalloc((size_t)NIN*4);
    float* y = (float*)xmalloc(n_out*4);

    unsigned s=12345u;
    if(packed){ int8_t* c=(int8_t*)W; for(size_t i=0;i<wbytes;i++){ s=s*1664525u+1013904223u; c[i]=(int8_t)((s>>16)%9); } m.code=(const int8_t*)W; }
    else      { float* f=(float*)W;  for(size_t i=0;i<wbytes/4;i++){ s=s*1664525u+1013904223u; f[i]=(float)((int)(s>>20)%7-3)*0.01f; } m.f32=(const float*)W; }
    for(size_t o=0;o<n_out;o++) scale[o]=0.01f;
    m.scale=scale;
    for(int i=0;i<NIN;i++) x[i]=(float)((i%13)-6)*0.1f;

    /* ~8 GB of weight traffic per cell, at least 3 calls */
    long reps = (long)(8.0*1024.0*MB/(double)wbytes); if(reps<3) reps=3;

    matvec(&m,x,NULL,y);                      /* warm: allocates g_xe/g_xo, faults the pages in */
    double t0=now_s();
    for(long r=0;r<reps;r++) matvec(&m,x,NULL,y);
    double dt=now_s()-t0;

    double sink=0; for(size_t o=0;o<n_out;o+=997) sink+=y[o];
    if(sink==1.0e300) printf("");            /* keep y live */

    *out_gbps_bytes = (double)wbytes*(double)reps/dt/1e9;
    double gwps = (double)n_out*(double)NIN*(double)reps/dt/1e9;
    free(W); free(scale); free(x); free(y);
    return gwps;
}

int main(int argc,char** argv){
    int nrep = argc>1 ? atoi(argv[1]) : 3;
#ifdef _OPENMP
    omp_set_num_threads(6);
#endif
    printf("E10 kbench -- same matvec source, n_in=%d fixed, --mvacc %d, 6 threads\n",NIN,g_mvacc);
    printf("%-8s %-7s %10s %10s %10s   %s\n","arm","cell","GB/s med","G-w/s med","spread","reps");
    for(int packed=1;packed>=0;packed--){
        for(int c=0;c<NCELL;c++){
            double g[16],b[16];
            for(int r=0;r<nrep;r++) g[r]=run_cell(packed,CELLS[c].mb,&b[r]);
            for(int i=1;i<nrep;i++) for(int j=0;j<nrep-i;j++)
                if(b[j]>b[j+1]){ double t=b[j];b[j]=b[j+1];b[j+1]=t; t=g[j];g[j]=g[j+1];g[j+1]=t; }
            double med=b[nrep/2], gmed=g[nrep/2];
            double spread=(b[nrep-1]-b[0])/med*100.0;
            printf("%-8s %-7s %10.2f %10.2f %9.1f%%   %d\n",
                   packed?"packed":"fp32", CELLS[c].tag, med, gmed, spread, nrep);
            fflush(stdout);
        }
    }
    return 0;
}
