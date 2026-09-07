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
/* mode: 1 packed, 0 fp32, 2 lut (E11 -- brief 00d4446), 3 BLOCKED lut (E13 -- brief 89fd63e).
   Modes 2 and 3 are the SAME KERNEL over the SAME BYTES in two layouts, so their outputs must be
   bit-identical (G-M0b below).  ycap, when non-NULL, receives y. */
static double run_cell_m(int mode,double mb,double* out_gbps_bytes,float* ycap,size_t ycapn){
    const int packed = (mode==1);
    const int lutm   = (mode==2||mode==3);
    g_lutblk = (mode==3);                     /* read by build/TM_BASE/TM_STRIDE; set per cell */
    const size_t row_bytes = (mode!=0) ? (size_t)(NIN/2) : (size_t)NIN*4;
    size_t n_out = (size_t)(mb*MB/(double)row_bytes);
    if(n_out<64) n_out=64;
    const size_t wbytes = n_out*row_bytes;

    mat_t m; memset(&m,0,sizeof(m));
    m.out=(int)n_out; m.in=NIN; m.packed=packed; m.tm=NULL; m.Mpad=0;
    /* the LUT kernel writes 32-row tiles, so it needs the row count padded to 32 and the codes
       held TILE-major [in/2][Mpad] -- the same bytes as the packed arm, transposed. */
    const size_t Mpad = lutm ? ((n_out+31)/32)*32 : 0;
    void* W = xmalloc(lutm ? (size_t)(NIN/2)*Mpad : wbytes);
    float* scale = (float*)xmalloc(n_out*4);
    float* x = (float*)xmalloc((size_t)NIN*4);
    float* y = (float*)xmalloc(n_out*4);

    unsigned s=12345u;
    if(mode==1){ int8_t* c=(int8_t*)W; for(size_t i=0;i<wbytes;i++){ s=s*1664525u+1013904223u; c[i]=(int8_t)((s>>16)%9); } m.code=(const int8_t*)W; }
    else if(lutm){
        /* SAME rng stream as the packed arm, laid out tile-major, so G-L0 compares two spellings
           of ONE matrix and not two different matrices.  mode 3 permutes the SAME bytes into the
           blocked layout: tile b's T steps held contiguously instead of Mpad apart. */
        const size_t T=(size_t)(NIN/2);
        int8_t* t=(int8_t*)W; memset(t,4,T*Mpad);                 /* 4 = the (0,0) trit pair */
        for(size_t o=0;o<n_out;o++) for(size_t j=0;j<T;j++){
            s=s*1664525u+1013904223u;
            size_t at = g_lutblk ? ((o/32)*T + j)*32 + (o%32) : j*Mpad + o;
            t[at]=(int8_t)((s>>16)%9); }
        m.tm=(const int8_t*)W; m.Mpad=(int)Mpad;
    }
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
    if(ycap) for(size_t o=0;o<n_out&&o<ycapn;o++) ycap[o]=y[o];

    *out_gbps_bytes = (double)wbytes*(double)reps/dt/1e9;
    double gwps = (double)n_out*(double)NIN*(double)reps/dt/1e9;
    free(W); free(scale); free(x); free(y);
    return gwps;
}
static double run_cell(int packed,double mb,double* g){ return run_cell_m(packed,mb,g,NULL,0); }

int main(int argc,char** argv){
    int nrep = argc>1 ? atoi(argv[1]) : 3;
#ifdef _OPENMP
    omp_set_num_threads(6);
#endif
    printf("E10/E11 kbench -- same matvec source, n_in=%d fixed, --mvacc %d, 6 threads\n",NIN,g_mvacc);

    /* ---- G-L0: correctness of the LUT arm, read BEFORE any LUT timing (brief 00d4446 §4).
       Same rng stream, tile-major instead of row-major: two spellings of ONE matrix. */
    {
        const size_t ncap = 2340;                      /* the 4 MB cell's row count */
        float* yp=(float*)xmalloc(ncap*4); float* yl=(float*)xmalloc(ncap*4);
        double g;
        run_cell_m(1,4.0,&g,yp,ncap);
        int saveg=g_group; g_group=0;  run_cell_m(2,4.0,&g,yl,ncap);
        double e2=0,s2=0; for(size_t o=0;o<ncap;o++){ double d=yl[o]-yp[o]; e2+=d*d; s2+=(double)yp[o]*yp[o]; }
        double rel0=sqrt(e2/(s2>0?s2:1));
        g_group=32; run_cell_m(2,4.0,&g,yl,ncap);
        e2=0; for(size_t o=0;o<ncap;o++){ double d=yl[o]-yp[o]; e2+=d*d; }
        double rel32=sqrt(e2/(s2>0?s2:1));
        g_group=saveg;
        printf("G-L0  rel l2 lut-vs-packed: whole-vector %.3e   group32 %.3e   (gate <=0.20 and not ~0;\n"
               "      published: 1.40e-01 whole-vector, 3.10e-02 at G=32)\n",rel0,rel32);
        printf("G-L0  %s\n", (rel0<=0.20&&rel0>1e-4)?"PASS":"FAIL -- every LUT cell below is VOID");

        /* ---- G-M0b (E13): the blocked arm is a PERMUTATION of the same bytes with the same t
           order and the same accumulate tree, so it must be BIT-IDENTICAL to the plain LUT arm.
           Read before any blocked timing: if a single bit moves, the index arithmetic is wrong
           and every blocked cell below is void.  This is the microbench mirror of G-M0, which is
           the same claim made end-to-end on the engine's logits. */
        float* yb=(float*)xmalloc(ncap*4);
        g_group=0; run_cell_m(2,4.0,&g,yl,ncap);
        g_group=0; run_cell_m(3,4.0,&g,yb,ncap);
        size_t nbit=0; for(size_t o=0;o<ncap;o++) if(memcmp(&yl[o],&yb[o],4)!=0) nbit++;
        printf("G-M0b blocked vs tile-major, bitwise: %s (%zu/%zu floats differ)\n",
               nbit?"FAIL -- every blocked cell below is VOID":"PASS", nbit, ncap);
        g_group=saveg;
        free(yp); free(yl); free(yb);
    }

    printf("%-8s %-7s %10s %10s %10s   %s\n","arm","cell","GB/s med","G-w/s med","spread","reps");
    static const int MODES[4]={1,2,3,0};                 /* known-positive first, then the arms */
    static const char* TAGS[4]={"packed","lut","lutblk","fp32"};
    for(int mi=0;mi<4;mi++){
        const int mode = MODES[mi];
        const char* tag = TAGS[mi];
        for(int c=0;c<NCELL;c++){
            double g[16],b[16];
            for(int r=0;r<nrep;r++) g[r]=run_cell_m(mode,CELLS[c].mb,&b[r],NULL,0);
            for(int i=1;i<nrep;i++) for(int j=0;j<nrep-i;j++)
                if(b[j]>b[j+1]){ double t=b[j];b[j]=b[j+1];b[j+1]=t; t=g[j];g[j]=g[j+1];g[j+1]=t; }
            printf("%-8s %-7s %10.2f %10.2f %9.1f%%   %d\n",
                   tag, CELLS[c].tag, b[nrep/2], g[nrep/2], (b[nrep-1]-b[0])/b[nrep/2]*100.0, nrep);
            fflush(stdout);
        }
    }
    return 0;
}
