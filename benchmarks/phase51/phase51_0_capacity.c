// Phase 51.0 - Associative memory CAPACITY PROOF (standalone, model-independent).
//
// NO model, NO training, NO substrate. Pure VSA measurement: can a single superposed
// hypervector hold a sequence of REAL TinyStories tokens and let us read back token_{t-k}
// at distance k? This is the FEASIBILITY GATE before wiring any readout onto the store.
//
// Codebook: 1024 fixed random hypervectors, one per BPE-1024 token (reuses bpe1024.bin).
// Write (one memory vector): M = sum_{k=0..T-1} decay^k * permute^k( atom(token_{t-k}) )
//   where permute^k = cyclic-shift by k (the silicon-native invertible binding).
// Read at distance k: q = permute^{-k}(M); cleanup = nearest atom (argmax) over the 1024.
//
// Two algebras compared:
//   (a) bipolar {-1,+1}, real-sum superposition, cosine cleanup (clean capacity).
//   (b) binary  {0,1},   weighted-majority superposition, Hamming(XOR+popcount) cleanup
//       (silicon-native "free on CPU" version; position via bit-rotation = the permutation).
//
// Metric: accuracy(k) for k = 0..KMAX over grid D x decay. chance = 1/1024.
// k*(D,decay) = max distance with recall >= threshold (reports 50% and above-chance 2%).
//
// Controls (mandatory):
//   (i)   T=1 sanity  -> must be ~100% (binding works).
//   (ii)  key-shuffle -> permute the EXPONENT used at write only; read still uses -k.
//         Must collapse to chance => proves positional binding (not an artifact) carries info.
//   (iii) raw-cosine of the true atom vs post-cleanup accuracy (both reported).
//   (iv)  real-repetition split: accuracy on probe positions whose token repeats elsewhere
//         in the window (crosstalk vs constructive) vs unique-token positions.
//   (v)   theory line: VSA capacity ~ D/(2 ln vocab) printed next to measured k*.
//
// Build: gcc -O3 -march=native -mavx2 -mfma -lm -I . benchmarks/phase38-42/phase51_0_capacity.c -o bin/phase51_0_capacity.exe
// Run:   bin/phase51_0_capacity.exe <tinystories.txt> <bpe1024.bin> <out.csv> [--smoke]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "benchmarks/phase50/bpe_codec.h"

// ---- deterministic RNG (splitmix64) -----------------------------------------
static uint64_t rng_state;
static inline uint64_t sm64(void){
    uint64_t z=(rng_state+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
static inline double rngd(void){ return (double)(sm64()>>11)*(1.0/9007199254740992.0); }

#define VOCAB 1024
static const double CHANCE = 1.0/(double)VOCAB;
static const double LNVOCAB = 6.931471805599453; // ln(1024)

// ---- config -----------------------------------------------------------------
static int   g_Dlist[]  = {256,512,1024,2048,4096,8192};
static int   g_nD       = 6;
static double g_decaylist[] = {1.0, 0.99, 0.95, 0.9};
static int   g_nDecay   = 4;
static int   g_KMAX     = 200;       // probe distances 0..KMAX
static int   g_NSEQ     = 96;        // sequences averaged per config

// codebooks (regenerated per D)
static float*    g_cb_f = NULL;       // bipolar [VOCAB*D] in {-1,+1}
static uint8_t*  g_cb_b = NULL;       // binary  [VOCAB*D] in {0,1}
static uint64_t* g_cb_w = NULL;       // binary bit-packed [VOCAB*W]

static void make_codebook(int D){
    int W=(D+63)/64;
    g_cb_f=(float*)realloc(g_cb_f,(size_t)VOCAB*D*sizeof(float));
    g_cb_b=(uint8_t*)realloc(g_cb_b,(size_t)VOCAB*D);
    g_cb_w=(uint64_t*)realloc(g_cb_w,(size_t)VOCAB*W*sizeof(uint64_t));
    rng_state=0xCAFEF00D51000000ULL ^ (uint64_t)D; // fixed per D => reproducible
    for(int j=0;j<VOCAB;j++){
        uint64_t* wj=g_cb_w+(size_t)j*W; for(int w=0;w<W;w++) wj[w]=0;
        for(int d=0;d<D;d++){
            int bit=(int)(sm64()&1ULL);
            g_cb_f[(size_t)j*D+d]= bit? 1.0f : -1.0f;
            g_cb_b[(size_t)j*D+d]= (uint8_t)bit;
            if(bit) wj[d>>6]|=(1ULL<<(d&63));
        }
    }
}

// ---- token windows from REAL TinyStories val ---------------------------------
static int**  g_seq=NULL;            // [NSEQ][T] token ids
static int    g_T=0;                 // tokens per window = KMAX+1

static void build_windows(const char* path, const Bpe* B){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open fail %s\n",path); exit(1);}
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char* buf=(unsigned char*)malloc(fsz); if(fread(buf,1,fsz,f)!=(size_t)fsz){fprintf(stderr,"read fail\n");exit(1);} fclose(f);
    g_T=g_KMAX+1;
    long lo=(long)(0.90*fsz), hi=fsz-4000; // held-out tail region
    long span=hi-lo, stride=span/g_NSEQ;
    g_seq=(int**)malloc(sizeof(int*)*g_NSEQ);
    int need=g_T;
    uint32_t* tmp=(uint32_t*)malloc(sizeof(uint32_t)*8192);
    for(int s=0;s<g_NSEQ;s++){
        long off=lo+(long)s*stride;
        // advance to a space so chunking starts clean
        while(off<hi && !bpe_isspace(buf[off])) off++;
        while(off<hi && bpe_isspace(buf[off])) off++;
        long end=off+4000; if(end>fsz) end=fsz;
        size_t nt=bpe_encode_region(B,buf,off,end,tmp);
        if((int)nt<need){ // rare; pull more bytes
            end=off+8000; if(end>fsz) end=fsz; nt=bpe_encode_region(B,buf,off,end,tmp);
        }
        g_seq[s]=(int*)malloc(sizeof(int)*g_T);
        for(int t=0;t<g_T;t++) g_seq[s][t]=(t<(int)nt)? (int)tmp[t] : 0;
    }
    free(tmp); free(buf);
}

// ---- bipolar write/read ------------------------------------------------------
// distance-k token = x[T-1-k]; M[(i+k)%D] += w*atom[i].
static void bip_write(const int* x,int D,double decay,const int* expo,float* M){
    for(int d=0;d<D;d++) M[d]=0.0f;
    double w=1.0;
    for(int k=0;k<g_T;k++){
        int tok=x[g_T-1-k];
        const float* a=g_cb_f+(size_t)tok*D;
        int sh=expo? expo[k] : k;           // key-shuffle uses scrambled exponent
        float wf=(float)w;
        int lim=D-sh;
        for(int i=0;i<lim;i++)   M[i+sh]   += wf*a[i];
        for(int i=lim;i<D;i++)   M[i+sh-D] += wf*a[i];
        w*=decay;
    }
}
// read distance k: q[i]=M[(i+k)%D]; argmax_j dot(q,a_j). returns argmax & raw cosine of true.
static int bip_read(const float* M,int D,int k,int truetok,float* q,double* rawcos_out){
    int sh=k, lim=D-sh;
    for(int i=0;i<lim;i++) q[i]=M[i+sh];
    for(int i=lim;i<D;i++) q[i]=M[i+sh-D];
    double qn=0.0; for(int i=0;i<D;i++) qn+=(double)q[i]*q[i]; qn=sqrt(qn)+1e-30;
    int best=0; float bestdot=-1e30f;
    for(int j=0;j<VOCAB;j++){
        const float* a=g_cb_f+(size_t)j*D;
        float dot=0.0f; for(int i=0;i<D;i++) dot+=q[i]*a[i];
        if(dot>bestdot){ bestdot=dot; best=j; }
    }
    // raw cosine of the TRUE atom (no cleanup)
    const float* at=g_cb_f+(size_t)truetok*D; double dt=0.0;
    for(int i=0;i<D;i++) dt+=(double)q[i]*at[i];
    *rawcos_out = dt/(qn*sqrt((double)D));
    return best;
}

// ---- binary write/read (weighted majority + Hamming) -------------------------
static void bin_write(const int* x,int D,double decay,const int* expo,float* vote){
    for(int d=0;d<D;d++) vote[d]=0.0f;
    double w=1.0;
    for(int k=0;k<g_T;k++){
        int tok=x[g_T-1-k];
        const uint8_t* a=g_cb_b+(size_t)tok*D;
        int sh=expo? expo[k] : k; float wf=(float)w; int lim=D-sh;
        for(int i=0;i<lim;i++)   vote[i+sh]   += a[i]? wf : -wf;
        for(int i=lim;i<D;i++)   vote[i+sh-D] += a[i]? wf : -wf;
        w*=decay;
    }
}
static int bin_read(const float* vote,int D,int k,int truetok,uint8_t* Mbit,uint64_t* qpack,double* rawcos_out){
    int W=(D+63)/64;
    // threshold vote -> M bits
    for(int d=0;d<D;d++) Mbit[d]=(vote[d]>0.0f)?1:0;
    // q = rotate M by -k : q[i]=Mbit[(i+k)%D]; pack into words
    for(int w=0;w<W;w++) qpack[w]=0;
    int sh=k,lim=D-sh;
    for(int i=0;i<D;i++){ int src=(i<lim)? i+sh : i+sh-D; if(Mbit[src]) qpack[i>>6]|=(1ULL<<(i&63)); }
    int best=0; int bestham=D+1;
    for(int j=0;j<VOCAB;j++){
        const uint64_t* aw=g_cb_w+(size_t)j*W; int ham=0;
        for(int w=0;w<W;w++) ham+=__builtin_popcountll(qpack[w]^aw[w]);
        if(ham<bestham){ bestham=ham; best=j; }
    }
    // raw similarity of true atom = 1 - ham/D mapped to [-1,1] (bipolar-equivalent cosine)
    const uint64_t* tw=g_cb_w+(size_t)truetok*W; int ham=0;
    for(int w=0;w<W;w++) ham+=__builtin_popcountll(qpack[w]^tw[w]);
    *rawcos_out = 1.0 - 2.0*(double)ham/(double)D;
    return best;
}

// ---- per-window repetition mask (token at distance k repeats elsewhere) -------
static void rep_mask(const int* x,int* isrep){
    for(int k=0;k<g_T;k++){
        int tok=x[g_T-1-k]; int rep=0;
        for(int k2=0;k2<g_T;k2++){ if(k2!=k && x[g_T-1-k2]==tok){ rep=1; break; } }
        isrep[k]=rep;
    }
}

int main(int argc,char** argv){
    if(argc<4){ fprintf(stderr,"usage: %s <tinystories> <bpe1024.bin> <out.csv> [--smoke]\n",argv[0]); return 1; }
    const char* tspath=argv[1]; const char* bpepath=argv[2]; const char* csvpath=argv[3];
    int smoke=0; for(int i=4;i<argc;i++) if(!strcmp(argv[i],"--smoke")) smoke=1;
    if(smoke){
        static int Ds[]={256,512}; g_Dlist[0]=Ds[0]; g_Dlist[1]=Ds[1]; g_nD=2;
        g_decaylist[0]=1.0; g_decaylist[1]=0.95; g_nDecay=2;
        g_KMAX=40; g_NSEQ=8;
    }

    Bpe B; if(!bpe_load_file(&B,bpepath)){ fprintf(stderr,"bpe load fail\n"); return 1; }
    if(B.vocab!=VOCAB){ fprintf(stderr,"WARN bpe vocab=%d != %d\n",B.vocab,VOCAB); }
    build_windows(tspath,&B);
    printf("Phase 51.0 capacity proof | NSEQ=%d KMAX=%d T=%d | chance=%.5f\n",g_NSEQ,g_KMAX,g_T,CHANCE);

    FILE* csv=fopen(csvpath,"w");
    fprintf(csv,"algebra,D,decay,k,n,acc,raw_cos,n_rep,acc_rep,n_uniq,acc_uniq\n");

    // scratch (max D)
    int Dmax=g_Dlist[g_nD-1]; int Wmax=(Dmax+63)/64;
    float* M=(float*)malloc(sizeof(float)*Dmax);
    float* q=(float*)malloc(sizeof(float)*Dmax);
    float* vote=(float*)malloc(sizeof(float)*Dmax);
    uint8_t* Mbit=(uint8_t*)malloc(Dmax);
    uint64_t* qpack=(uint64_t*)malloc(sizeof(uint64_t)*Wmax);
    int* expo=(int*)malloc(sizeof(int)*g_T);     // shuffled exponent for control
    int* isrep=(int*)malloc(sizeof(int)*g_T);

    // per-k accumulators
    int* hit=(int*)malloc(sizeof(int)*g_T); int* tot=(int*)malloc(sizeof(int)*g_T);
    double* rc=(double*)malloc(sizeof(double)*g_T);
    int* hitR=(int*)malloc(sizeof(int)*g_T); int* totR=(int*)malloc(sizeof(int)*g_T);
    int* hitU=(int*)malloc(sizeof(int)*g_T); int* totU=(int*)malloc(sizeof(int)*g_T);

    const char* algn[2]={"bipolar","binary"};

    for(int ai=0; ai<2; ai++){
      for(int di=0; di<g_nD; di++){
        int D=g_Dlist[di]; make_codebook(D);
        double theory = (double)D/(2.0*LNVOCAB);
        clock_t t0=clock();
        for(int dci=0; dci<g_nDecay; dci++){
            double decay=g_decaylist[dci];
            for(int k=0;k<g_T;k++){ hit[k]=tot[k]=0; rc[k]=0; hitR[k]=totR[k]=0; hitU[k]=totU[k]=0; }
            for(int s=0;s<g_NSEQ;s++){
                const int* x=g_seq[s]; rep_mask(x,isrep);
                if(ai==0) bip_write(x,D,decay,NULL,M);
                else      bin_write(x,D,decay,NULL,vote);
                for(int k=0;k<g_T;k++){
                    int truetok=x[g_T-1-k]; double rcos; int pred;
                    if(ai==0) pred=bip_read(M,D,k,truetok,q,&rcos);
                    else      pred=bin_read(vote,D,k,truetok,Mbit,qpack,&rcos);
                    int ok=(pred==truetok);
                    hit[k]+=ok; tot[k]++; rc[k]+=rcos;
                    if(isrep[k]){ hitR[k]+=ok; totR[k]++; } else { hitU[k]+=ok; totU[k]++; }
                }
            }
            // write CSV + find k*
            int kstar50=-1, kstarAC=-1;
            for(int k=0;k<g_T;k++){
                double acc=(double)hit[k]/tot[k];
                double accR=totR[k]? (double)hitR[k]/totR[k] : 0.0;
                double accU=totU[k]? (double)hitU[k]/totU[k] : 0.0;
                fprintf(csv,"%s,%d,%.2f,%d,%d,%.4f,%.4f,%d,%.4f,%d,%.4f\n",
                    algn[ai],D,decay,k,tot[k],acc,rc[k]/tot[k],totR[k],accR,totU[k],accU);
                if(acc>=0.50) kstar50=k;
                if(acc>=0.02) kstarAC=k;
            }
            printf("  %-7s D=%5d decay=%.2f | k*@50%%=%-3d  k*@2%%=%-3d  acc(0)=%.3f acc(%d)=%.3f | theory~%.0f\n",
                algn[ai],D,decay,kstar50,kstarAC,(double)hit[0]/tot[0],g_KMAX,(double)hit[g_KMAX]/tot[g_KMAX],theory);
        }
        printf("    [%s D=%d done in %.1fs]\n",algn[ai],D,(double)(clock()-t0)/CLOCKS_PER_SEC);
      }
    }

    // ---- CONTROLS (run on a representative config: D=middle, decay=1.0) -------
    int Dc = smoke? g_Dlist[g_nD-1] : 1024; // representative dim
    make_codebook(Dc);
    printf("\n=== CONTROLS @ D=%d decay=1.0 ===\n",Dc);
    // (i) T=1 sanity: only the most-recent token written -> recall@0 must ~100%
    {
        int h=0,n=0;
        for(int s=0;s<g_NSEQ;s++){
            int one[1]; one[0]=g_seq[s][g_T-1]; int savedT=g_T; g_T=1;
            double rcos; int pred;
            // temporarily reuse writer with T=1
            if(1){ bip_write(one,Dc,1.0,NULL,M); pred=bip_read(M,Dc,0,one[0],q,&rcos);}
            g_T=savedT; h+=(pred==one[0]); n++;
        }
        printf("  (i)   T=1 sanity (bipolar): acc@0 = %.3f  (expect ~1.000)\n",(double)h/n);
    }
    // two repetition stats: (a) rep-fraction = probes touching a repeated token (context for
    // control iv); (b) collision-floor = mean mult(token)/T = E[position-blind guess accuracy]
    // = exactly what a bijective key-shuffle collapses to (Sum p_i^2 over the window bag).
    {
        long rep=0,nn=0; double cf=0;
        for(int s=0;s<g_NSEQ;s++){ const int* x=g_seq[s]; rep_mask(x,isrep);
            for(int k=0;k<g_T;k++){ rep+=isrep[k]; nn++;
                int tok=x[g_T-1-k],m=0; for(int j=0;j<g_T;j++) if(x[g_T-1-j]==tok) m++;
                cf+=(double)m/g_T; } }
        printf("  rep-fraction: %.4f of probes touch a repeated token | collision-floor(Sum p^2): %.4f\n",
            (double)rep/nn, cf/nn);
    }
    // (ii) key-shuffle: scramble the write exponent (still a bijection => every distance slot
    //      cleanly stores SOME token, just the wrong one). With real Zipf tokens this collapses
    //      to the REPETITION FLOOR, not uniform 1/1024 -> proves positional binding carries the
    //      ORDERED info (real recall is far above this floor).
    {
        int h=0,n=0;
        for(int s=0;s<g_NSEQ;s++){
            for(int k=0;k<g_T;k++) expo[k]=k;
            for(int k=g_T-1;k>0;k--){ int r=(int)(rngd()*(k+1)); int tmp=expo[k]; expo[k]=expo[r]; expo[r]=tmp; }
            bip_write(g_seq[s],Dc,1.0,expo,M);
            for(int k=0;k<g_T;k++){ double rcos; int pred=bip_read(M,Dc,k,g_seq[s][g_T-1-k],q,&rcos); h+=(pred==g_seq[s][g_T-1-k]); n++; }
        }
        printf("  (ii)  key-shuffle (bipolar): acc = %.4f  (-> position-blind floor: <= collision-floor,\n        attenuated by superposition interference; far below true low-k recall, well above chance)\n",(double)h/n);
    }
    // (ii-b) random-readout: read from a fresh random M -> anchors empirical uniform chance.
    {
        int h=0,n=0;
        for(int s=0;s<g_NSEQ;s++){
            for(int d=0;d<Dc;d++) M[d]=(sm64()&1ULL)?1.0f:-1.0f;
            for(int k=0;k<g_T;k++){ double rcos; int pred=bip_read(M,Dc,k,g_seq[s][g_T-1-k],q,&rcos); h+=(pred==g_seq[s][g_T-1-k]); n++; }
        }
        printf("  (ii-b)random-M (bipolar): acc = %.4f  chance=%.4f  (empirical uniform floor)\n",(double)h/n,CHANCE);
    }
    printf("  (iii) raw_cos vs acc: see CSV columns acc & raw_cos per k.\n");
    printf("  (iv)  repetition split: see CSV acc_rep (token repeats elsewhere) vs acc_uniq.\n");
    printf("  (v)   theory: VSA capacity ~ D/(2 ln vocab) = D/%.2f printed per config above.\n",2.0*LNVOCAB);

    fclose(csv);
    printf("\nCSV -> %s\n",csvpath);
    return 0;
}
