// Silicon Entropy Engine — CONSOLIDATED single-core inference engine (P4.3).
//
// One core, feature-flags, replacing the five stage engines (e1/e2/e3/e35/e4_engine.c, now archival).
// The model file's magic selects the MLP family: E1M1 (dense gated-dReLU) or E4M1 (MoE top-8 experts).
//
//   --mlp  fp32|lut     MLP matvec path: fp32 dequant reference | ternary pshufb-LUT (int8 activations)
//   --skip on|off       dense-LUT only: E3 exact activation-skip (gate-first, up row-skip, down tile-skip)
//   --exp  exact|fast   selective-scan exp: libm expf | exp256_ps poly (E3.5, deterministic, parity-gated)
//
// Stage equivalences (acceptance: bit-identical logit streams vs the archived stage engines):
//   E1      = E1M1 --mlp fp32 --exp exact          E3.5iso = E1M1 --mlp fp32 --exp fast
//   E2      = E1M1 --mlp lut --skip off --exp exact  E3.5   = E1M1 --mlp lut --skip on --exp fast
//   E3      = E1M1 --mlp lut --skip on  --exp exact
//   E4-ref  = E4M1 --mlp fp32 --exp exact          E4-full = E4M1 --mlp lut --exp fast
//
// Modes: --bpb | --logits (top-1 vs the fp32 config) | --dumplogits <file> (raw fp32 logit stream,
//        the parity-acceptance instrument) | --kselftest (synthetic kernel self-tests, no weights; CI)
// Protocol shared with the stage gates: val = last 10% of ids.u16; windows of --seq (512) with
// state_reset per window; --ntok tokens from --offset.
//
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/engine.c -o bin/engine -lm
// Law: correctness-first; NO -ffast-math; tok/s of this core are re-measured only at training end.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif

#define V    1024
#define D    256
#define N    96
#define H    8
#define HD   (D/H)
#define L    6
#define DN   512
#define DTR  16
#define CONV 4
#define WIN  128
#define SWA_LAYER 5
#define AQ   63
// dense (E1M1)
#define MLP_HID 1024
#define TUP  (D/2)
#define TDN  (MLP_HID/2)
// MoE (E4M1)
#define E    32
#define HID_E 128
#define KTOP 8
#define GH   (E*HID_E)
#define MPAD_GU ((GH+31)&~31)
#define MPAD_D  ((D+31)&~31)
#define TDE  (HID_E/2)
#define NLAYER (L+2)

// ---------------- shared primitives ----------------
static inline __m256 exp256_ps(__m256 x){                        // E3.5 poly-exp (Cephes, fixed coeffs, deterministic)
    const __m256 hi=_mm256_set1_ps(88.3762626647949f), lo=_mm256_set1_ps(-88.3762626647949f);
    x=_mm256_min_ps(_mm256_max_ps(x,lo),hi);
    __m256 fx=_mm256_fmadd_ps(x,_mm256_set1_ps(1.44269504088896341f),_mm256_set1_ps(0.5f)); fx=_mm256_floor_ps(fx);
    x=_mm256_fnmadd_ps(fx,_mm256_set1_ps(0.693359375f),x); x=_mm256_fnmadd_ps(fx,_mm256_set1_ps(-2.12194440e-4f),x);
    __m256 z=_mm256_mul_ps(x,x), p=_mm256_set1_ps(1.9875691500E-4f);
    p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(1.3981999507E-3f)); p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(8.3334519073E-3f));
    p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(4.1665795894E-2f)); p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(1.6666665459E-1f));
    p=_mm256_fmadd_ps(p,x,_mm256_set1_ps(5.0000001201E-1f)); p=_mm256_fmadd_ps(p,z,x); p=_mm256_add_ps(p,_mm256_set1_ps(1.0f));
    __m256i e=_mm256_cvttps_epi32(fx); e=_mm256_slli_epi32(_mm256_add_epi32(e,_mm256_set1_epi32(0x7f)),23);
    return _mm256_mul_ps(p,_mm256_castsi256_ps(e));
}
static inline float exp_approx1(float x){ float o[8]; _mm256_storeu_ps(o,exp256_ps(_mm256_set1_ps(x))); return o[0]; }
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
// Sequential per-row dotf. A blocked/multi-accumulator GEMV (2.3-2.6x in an isolated cache-resident microbench,
// gemv_bench.c) was tried and REVERTED: in-engine the per-token GEMVs are memory-bandwidth-bound (weights evicted
// between tokens, not FMA-latency-bound), so ILP does not help and the 4-strided access is ~4% SLOWER than this
// sequential stream (probe-3 rho-law). Lesson: microbench speedup (compute-bound) does not compose to engine speedup
// (memory-bound) -- measure end-to-end. The fp32 projections are floor-ish; the byte lever (ternary) failed on quality (Phase 61).
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

// ---------------- ternary LUT kernels (probe-1 lineage, bit-exact vs scalar-int) ----------------
static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_full(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static void matvec_lut_tileskip(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,const int* act,int na){
    for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int a=0;a<na;a++){ int t=act[a];
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
// windowed rows [row0,row0+M) of a tile-major block (the E4 per-expert access)
static void matvec_lut_rows(const int8_t* codes,const int8_t* lut,int32_t* y,int row0,int M,int Mpad,int T){
    for(int bb=0;bb<M;bb+=32){ int base=row0+bb; __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&bb+r<M;r++) y[bb+r]=tmp[r]; }
}
static void build_lut_t3(const int8_t* xq,int T,int8_t* lut){ for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
    for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1,w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } } }
static void bc_tm(const int8_t* Wt,int M,int K,int Mpad,int8_t* codes){ int T=K/2;
    for(int t=0;t<T;t++){ for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0; } }
static void bc_rm(const int8_t* Wt,int M,int K,int8_t* codes){ int T=K/2;
    for(int m=0;m<M;m++) for(int t=0;t<T;t++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)m*T+t]=(int8_t)((w0+1)*3+(w1+1)); } }
static void ref_t3(const int8_t* Wt,const int8_t* xq,int32_t* y,int M,int K){
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; } }
static float quant_i8(const float* x,int n,int8_t* xq){ float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; } float scale=amax/(float)AQ,inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; } return scale; }

// ---------------- model state ----------------
typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L];
static int g_moe=0;                                              // set by the weights magic
// dense (E1M1)
static float *gate_f[L],*up_f[L],*down_f[L];
static int8_t *gate_tm[L],*up_tm[L],*up_rm[L],*down_tm[L]; static float *gate_sc[L],*up_sc[L],*down_sc[L];
// MoE (E4M1)
static float *router_w[L],*router_b[L],*egate_f[L],*eup_f[L],*eWd_f[L];
static int8_t *egate_cd[L],*eup_cd[L],*eWd_cd[L]; static float *egate_sc[L],*eup_sc[L],*eWd_sc[L];
static int8_t *egate_wt[L],*eup_wt[L],*eWd_wt[L];                // kept for kernel self-tests vs scalar-int
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} return p; }
static int8_t* rdi8(FILE*f,size_t n){ int8_t*p=xmalloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short read i8\n");exit(1);} return p; }

static void load_backbone(FILE* f){                              // shared prefix of both formats up to the MLP tensors
    emb=rd(f,(size_t)V*D);
}
static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t h[16]; if(fread(h,4,16,f)!=16){fprintf(stderr,"short header\n");exit(1);}
    if(h[0]==0x45314D31) g_moe=0; else if(h[0]==0x45344D31) g_moe=1; else {fprintf(stderr,"bad magic %08x\n",h[0]);exit(1);}
    if(g_moe && ((int)h[11]!=E||(int)h[12]!=HID_E||(int)h[13]!=KTOP)){fprintf(stderr,"E/hid_e/k mismatch\n");exit(1);}
    if(!g_moe && (int)h[11]!=MLP_HID){fprintf(stderr,"mlp_hid mismatch\n");exit(1);}
    int has_packed=h[14];
    load_backbone(f);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);
            s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        mlp_n2[l]=rd(f,D);
        if(g_moe){ router_w[l]=rd(f,(size_t)E*D); router_b[l]=rd(f,E);
            egate_f[l]=rd(f,(size_t)GH*D); eup_f[l]=rd(f,(size_t)GH*D); eWd_f[l]=rd(f,(size_t)E*D*HID_E); }
        else { gate_f[l]=rd(f,(size_t)MLP_HID*D); up_f[l]=rd(f,(size_t)MLP_HID*D); down_f[l]=rd(f,(size_t)D*MLP_HID); }
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    if(!has_packed){fprintf(stderr,"engine needs packed ternary in the export\n");exit(1);}
    if(g_moe){
        for(int l=0;l<L;l++){
            egate_wt[l]=rdi8(f,(size_t)GH*D); egate_sc[l]=rd(f,GH);
            eup_wt[l]=rdi8(f,(size_t)GH*D); eup_sc[l]=rd(f,GH);
            eWd_wt[l]=rdi8(f,(size_t)E*D*HID_E); eWd_sc[l]=rd(f,(size_t)E*D);
            egate_cd[l]=xmalloc((size_t)TUP*MPAD_GU); bc_tm(egate_wt[l],GH,D,MPAD_GU,egate_cd[l]);
            eup_cd[l]=xmalloc((size_t)TUP*MPAD_GU); bc_tm(eup_wt[l],GH,D,MPAD_GU,eup_cd[l]);
            eWd_cd[l]=xmalloc((size_t)E*TDE*MPAD_D);
            for(int e=0;e<E;e++) bc_tm(eWd_wt[l]+(size_t)e*D*HID_E,D,HID_E,MPAD_D,eWd_cd[l]+(size_t)e*TDE*MPAD_D);
        }
    } else {
        for(int l=0;l<L;l++){ int8_t* gq=rdi8(f,(size_t)MLP_HID*D); gate_sc[l]=rd(f,MLP_HID);
            int8_t* uq=rdi8(f,(size_t)MLP_HID*D); up_sc[l]=rd(f,MLP_HID);
            int8_t* dq=rdi8(f,(size_t)D*MLP_HID); down_sc[l]=rd(f,D);
            int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
            gate_tm[l]=xmalloc((size_t)TUP*Mpg); bc_tm(gq,MLP_HID,D,Mpg,gate_tm[l]);
            up_tm[l]=xmalloc((size_t)TUP*Mpg); bc_tm(uq,MLP_HID,D,Mpg,up_tm[l]);
            up_rm[l]=xmalloc((size_t)MLP_HID*TUP); bc_rm(uq,MLP_HID,D,up_rm[l]);
            down_tm[l]=xmalloc((size_t)TDN*Mpd); bc_tm(dq,D,MLP_HID,Mpd,down_tm[l]);
            free(gq);free(uq);free(dq); }
    }
    long pos=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f); fclose(f);
    if(pos!=end) fprintf(stderr,"WARN %ld trailing bytes\n",end-pos);
    fprintf(stderr,"engine weights ok (%s)\n",g_moe?"E4M1 MoE":"E1M1 dense");
}
static void load_meta(const char* path){ FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"no meta\n");exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1); if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; } fclose(f); }
static void load_ids(const char* path){ FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"no ids\n");exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); nids=b/2; ids=xmalloc(b); if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f); }
static void state_reset(void){ memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0; }
static inline void rmsnorm(const float* in,const float* w,float* out){ float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i];
    float r=1.0f/sqrtf(ms/(float)D+1e-5f); for(int i=0;i<D;i++) out[i]=in[i]*r*w[i]; }

// ---------------- dense MLP: fp32 | LUT-full (E2) | LUT-skip (E3) ----------------
static int8_t xqb[MLP_HID]; static int8_t lutb[TDN*16]; static int32_t Sb[MLP_HID]; static int act_tiles[TDN];
static void mlp_dense(int l,const float* xn,float* out,int mlp_lut,int skip){
    float gh[MLP_HID],uh[MLP_HID];
    if(!mlp_lut){ matvec(gate_f[l],xn,gh,MLP_HID,D); matvec(up_f[l],xn,uh,MLP_HID,D);
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]); matvec(down_f[l],gh,out,D,MLP_HID); return; }
    int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
    float sa=quant_i8(xn,D,xqb); build_lut_t3(xqb,TUP,lutb);
    matvec_lut_full(gate_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);
    for(int i=0;i<MLP_HID;i++) gh[i]=(float)Sb[i]*sa*gate_sc[l][i];
    if(!skip){                                                    // E2: up full, relu*relu
        matvec_lut_full(up_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);
        for(int i=0;i<MLP_HID;i++) uh[i]=(float)Sb[i]*sa*up_sc[l][i];
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]);
    } else {                                                      // E3: exact up row-skip (gate<=0 -> h=0)
        for(int i=0;i<MLP_HID;i++){ if(gh[i]>0.0f){ const int8_t* cr=up_rm[l]+(size_t)i*TUP; int S=0;
                for(int t=0;t<TUP;t++) S+=lutb[t*16+cr[t]]; float u=(float)S*sa*up_sc[l][i]; gh[i]=gh[i]*reluf(u); } else gh[i]=0.0f; }
    }
    float sh=quant_i8(gh,MLP_HID,xqb); build_lut_t3(xqb,TDN,lutb);
    if(!skip){ matvec_lut_full(down_tm[l],lutb,Sb,D,Mpd,TDN); }
    else { int na=0; for(int t=0;t<TDN;t++){ if(xqb[2*t]||xqb[2*t+1]) act_tiles[na++]=t; }
        matvec_lut_tileskip(down_tm[l],lutb,Sb,D,Mpd,act_tiles,na); }
    for(int i=0;i<D;i++) out[i]=(float)Sb[i]*sh*down_sc[l][i];
}

// ---------------- MoE MLP: fp32 experts (E4-ref) | LUT experts (E4) ----------------
static void topk_sel(const float* p,int n,int k,int* idx,float* val){   // matches torch.topk (ties -> lower index)
    int used[E]; memset(used,0,sizeof(int)*n);
    for(int j=0;j<k;j++){ int bi=-1; float bv=-1e30f;
        for(int i=0;i<n;i++) if(!used[i]&&p[i]>bv){ bv=p[i]; bi=i; }
        used[bi]=1; idx[j]=bi; val[j]=p[bi]; }
}
static int8_t g_xq[D],g_lut[TUP*16],g_hq[HID_E],g_lutd[TDE*16]; static int32_t g_S[HID_E],g_Sd[D];
static void mlp_moe(int l,const float* xn,float* out,int mlp_lut){
    float rp[E]; for(int e=0;e<E;e++) rp[e]=dotf(router_w[l]+(size_t)e*D,xn,D)+router_b[l][e];
    float mx=-1e30f; for(int e=0;e<E;e++) if(rp[e]>mx)mx=rp[e];
    float Z=0; for(int e=0;e<E;e++){ rp[e]=expf(rp[e]-mx); Z+=rp[e]; } for(int e=0;e<E;e++) rp[e]/=Z;
    int idx[KTOP]; float wv[KTOP]; topk_sel(rp,E,KTOP,idx,wv);
    float ws=0; for(int j=0;j<KTOP;j++) ws+=wv[j]; for(int j=0;j<KTOP;j++) wv[j]/=ws;
    memset(out,0,D*4);
    float sa=0; if(mlp_lut){ sa=quant_i8(xn,D,g_xq); build_lut_t3(g_xq,TUP,g_lut); }
    for(int j=0;j<KTOP;j++){ int e=idx[j]; float tw=wv[j]; float he[HID_E];
        if(!mlp_lut){
            for(int i=0;i<HID_E;i++){ float gg=dotf(egate_f[l]+(size_t)(e*HID_E+i)*D,xn,D);
                float uu=dotf(eup_f[l]+(size_t)(e*HID_E+i)*D,xn,D); he[i]=reluf(gg)*reluf(uu)*tw; }
            for(int d=0;d<D;d++) out[d]+=dotf(eWd_f[l]+((size_t)e*D+d)*HID_E,he,HID_E);
        } else {
            matvec_lut_rows(egate_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);
            float gg[HID_E]; for(int i=0;i<HID_E;i++) gg[i]=(float)g_S[i]*sa*egate_sc[l][e*HID_E+i];
            matvec_lut_rows(eup_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);
            for(int i=0;i<HID_E;i++){ float uu=(float)g_S[i]*sa*eup_sc[l][e*HID_E+i]; he[i]=reluf(gg[i])*reluf(uu)*tw; }
            float sh=quant_i8(he,HID_E,g_hq); build_lut_t3(g_hq,TDE,g_lutd);
            matvec_lut_rows(eWd_cd[l]+(size_t)e*TDE*MPAD_D,g_lutd,g_Sd,0,D,MPAD_D,TDE);
            for(int d=0;d<D;d++) out[d]+=(float)g_Sd[d]*sh*eWd_sc[l][(size_t)e*D+d];
        }
    }
}

// ---------------- forward ----------------
typedef struct { double scan,scan_other,swa,mlp,head; } Tacc;
// cfg: mlp_lut (0 fp32 / 1 LUT), skip (dense only), exp_fast (0 exact / 1 poly)
static void forward_token(uint32_t tok,float* logits,int mlp_lut,int skip,int exp_fast,Tacc* T){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vvv[D],att[WIN],ao[D],tmp[D];
    double t0;
    memcpy(x,emb+(size_t)tok*D,D*4);
    for(int l=0;l<L;l++){
        rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);
        if(is_swa[l]){
            if(T)t0=now_s();
            matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vvv,xz+2*D,D*4);
            int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vvv,D*4);
            kvpos++; if(kvcnt<WIN) kvcnt++; memset(ao,0,D*4);
            for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float m2=-1e30f;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>m2)m2=sc; }
                float Zs=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-m2); Zs+=att[j]; } float zi=1.0f/Zs;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                    for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
            matvec(swa.o,ao,tmp,D,D); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->swa+=now_s()-t0;
        } else {
            SSML*s=&ssm[l];
            if(T)t0=now_s();
            matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
            float (*cb)[CONV]=convbuf[l];
            for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
            matvec(s->x_proj,xx,dbl,DTR+2*N,DN);
            const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
            for(int c=0;c<DN;c++) dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]);
            if(T){T->scan_other+=now_s()-t0; t0=now_s();}
            float (*hl)[N]=hstate[l];
            if(!exp_fast){
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                    for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; } y[c]=acc+s->Dskip[c]*xc; }
            } else {
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],dbx=dt[c]*xc;
                    __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx),vacc=_mm256_setzero_ps();
                    for(int j=0;j<N;j+=8){ __m256 ee=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                        __m256 hcj=_mm256_fmadd_ps(ee,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                        _mm256_storeu_ps(hc+j,hcj); vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+j),vacc); } y[c]=hsum256(vacc)+s->Dskip[c]*xc; }
            }
            if(T){T->scan+=now_s()-t0; t0=now_s();}
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->scan_other+=now_s()-t0;
        }
        rmsnorm(x, mlp_n2[l], xn);
        if(T)t0=now_s();
        if(g_moe) mlp_moe(l,xn,tmp,mlp_lut); else mlp_dense(l,xn,tmp,mlp_lut,skip);
        for(int i=0;i<D;i++) x[i]+=tmp[i];
        if(T)T->mlp+=now_s()-t0;
    }
    if(T)t0=now_s();
    rmsnorm(x,normf,xn); matvec(head,xn,logits,V,D);
    if(T)T->head+=now_s()-t0;
}

// ---------------- modes ----------------
static void dump_logits(const char* path,long seqW,long ntok,long offset,int mlp_lut,int skip,int exp_fast){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    FILE* f=fopen(path,"wb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    float* lg=xmalloc((size_t)V*4); long done=0,pos=offset;
    while(done<ntok){ state_reset();
        for(long t=0;t<seqW&&done<ntok;t++,done++){ forward_token(val[pos+t],lg,mlp_lut,skip,exp_fast,NULL);
            fwrite(lg,4,V,f); }
        pos+=seqW; }
    fclose(f); free(lg);
    printf("dumped %ld x %d fp32 logits -> %s\n",ntok,V,path);
}
static double run_bpb(long seqW,long eval_tok,int mlp_lut,int skip,int exp_fast,long* ntok_o){
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    long lim=eval_tok<(nval-1)?eval_tok:(nval-1); double bits=0; long nbytes=0,ntok=0,pos=0;
    float* logits=xmalloc((size_t)V*4); const double LN2=0.6931471805599453;
    while(pos+seqW+1<=lim){ state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],logits,mlp_lut,skip,exp_fast,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; for(int o=0;o<V;o++) if(logits[o]>mx) mx=logits[o];
            double se=0; for(int o=0;o<V;o++) se+=exp((double)logits[o]-mx);
            bits += (-((double)logits[tgt]-mx)+log(se))/LN2; nbytes+=id2len[tgt]; ntok++; }
        pos+=seqW; }
    free(logits); if(ntok_o)*ntok_o=ntok; return bits/(nbytes>0?nbytes:1);
}
static int gate_logits(long seqW,long ntok_target,int mlp_lut,int skip,int exp_fast){
    // top-1 agreement of the selected config vs the fp32+exact reference config (same binary)
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* l1=xmalloc((size_t)V*4); float* l2=xmalloc((size_t)V*4);
    long agree=0,tot=0,pos=0; static uint16_t a1[4096];
    while(tot<ntok_target){
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l1,0,0,0,NULL); float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(l1[o]>mx){mx=l1[o];am=o;} a1[t]=am; }
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l2,mlp_lut,skip,exp_fast,NULL); float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(l2[o]>mx){mx=l2[o];am=o;} if(am==a1[t]) agree++; tot++; }
        pos+=seqW;
    }
    double pct=100.0*agree/tot;
    printf("==== top-1 agreement (config vs fp32+exact ref, %ld tok) ====\n  agreement=%.4f%% (%ld/%ld)\n",tot,pct,agree,tot);
    free(l1); free(l2); return pct>=99.0?0:2;
}
static void timing(long ntok,int mlp_lut,int skip,int exp_fast){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    Tacc T={0,0,0,0,0}; state_reset(); double t0=now_s();
    for(long i=0;i<ntok;i++) forward_token(val[i%100000],lg,mlp_lut,skip,exp_fast,&T);
    double tot=now_s()-t0; double sc=1e6/ntok;
    printf("==== timing (%s, mlp=%s skip=%d exp=%s, %ld tok): %.1f tok/s | %.1f us/tok ====\n",
           g_moe?"MoE":"dense",mlp_lut?"lut":"fp32",skip,exp_fast?"fast":"exact",ntok,ntok/tot,1e6/(ntok/tot));
    printf("   scan %.1f  scan-other %.1f  SWA %.1f  MLP %.1f  head %.1f  (us/tok)\n",T.scan*sc,T.scan_other*sc,T.swa*sc,T.mlp*sc,T.head*sc);
    free(lg);
}

// ---------------- synthetic kernel self-tests (no weights; CI) ----------------
static int kernel_selftest(void){
    int rc=0; srand(45678); long checks=0; int worst=0;
    { // dense shapes: full / row-scalar / tile-skip vs scalar-int
        int dims[2][2]={{MLP_HID,D},{D,MLP_HID}};
        for(int c=0;c<2;c++){ int M=dims[c][0],K=dims[c][1],Mpad=(M+31)&~31,T=K/2;
            int8_t* Wt=xmalloc((size_t)M*K); int8_t* ctm=xmalloc((size_t)T*Mpad); int8_t* crm=xmalloc((size_t)M*T);
            int8_t* xq=xmalloc(K); int8_t* lut=xmalloc((size_t)T*16);
            int32_t* Sl=xmalloc((size_t)M*4); int32_t* Sr=xmalloc((size_t)M*4); int* act=xmalloc((size_t)T*sizeof(int));
            for(int trial=0;trial<24;trial++){
                for(size_t i=0;i<(size_t)M*K;i++) Wt[i]=(int8_t)(rand()%3-1);
                for(int k=0;k<K;k++){ int v=rand()%(2*AQ+1)-AQ; if(rand()%3==0)v=0; xq[k]=(int8_t)v; }
                bc_tm(Wt,M,K,Mpad,ctm); bc_rm(Wt,M,K,crm); build_lut_t3(xq,T,lut); ref_t3(Wt,xq,Sr,M,K);
                matvec_lut_full(ctm,lut,Sl,M,Mpad,T);
                for(int m=0;m<M;m++){ int d=abs(Sl[m]-Sr[m]); if(d>worst)worst=d; checks++; }
                for(int m=0;m<M;m+=7){ const int8_t* cr=crm+(size_t)m*T; int S=0; for(int t=0;t<T;t++) S+=lut[t*16+cr[t]];
                    int d=abs(S-Sr[m]); if(d>worst)worst=d; checks++; }
                int na=0; for(int t=0;t<T;t++) if(xq[2*t]||xq[2*t+1]) act[na++]=t;
                matvec_lut_tileskip(ctm,lut,Sl,M,Mpad,act,na);
                for(int m=0;m<M;m++){ int d=abs(Sl[m]-Sr[m]); if(d>worst)worst=d; checks++; }
            }
            free(Wt);free(ctm);free(crm);free(xq);free(lut);free(Sl);free(Sr);free(act);
        }
    }
    { // MoE shapes: windowed rows + per-expert blocks vs scalar-int
        int8_t* Wgu=xmalloc((size_t)GH*D); int8_t* cgu=xmalloc((size_t)TUP*MPAD_GU);
        int8_t xq[D],lut[TUP*16]; int32_t Sl[HID_E],Sr[HID_E];
        for(int trial=0;trial<8;trial++){
            for(size_t i=0;i<(size_t)GH*D;i++) Wgu[i]=(int8_t)(rand()%3-1);
            for(int k=0;k<D;k++) xq[k]=(int8_t)(rand()%(2*AQ+1)-AQ);
            bc_tm(Wgu,GH,D,MPAD_GU,cgu); build_lut_t3(xq,TUP,lut);
            for(int e=0;e<E;e+=5){ matvec_lut_rows(cgu,lut,Sl,e*HID_E,HID_E,MPAD_GU,TUP);
                ref_t3(Wgu+(size_t)e*HID_E*D,xq,Sr,HID_E,D);
                for(int i=0;i<HID_E;i++){ int d=abs(Sl[i]-Sr[i]); if(d>worst)worst=d; checks++; } }
        }
        free(Wgu);free(cgu);
    }
    printf("==== engine kernel self-test (synthetic, no weights): all LUT paths vs scalar-int ====\n");
    printf("  %ld checks | worst |S - S_ref| = %d  %s\n",checks,worst,worst==0?"PASS":"FAIL");
    if(worst!=0) rc=2;
    double worst_dom=0;
    for(long i=0;i<=3000000;i++){ float x=-30.0f+(float)i*(30.0f/3000000.0f);
        float a=exp_approx1(x); float r=expf(x); double rel=fabs((double)a-r)/(double)r; if(rel>worst_dom)worst_dom=rel; }
    printf("  exp256_ps vs libm on [-30,0]: max rel err = %.3e  %s (<=2e-6)\n",worst_dom,worst_dom<=2e-6?"PASS":"FAIL");
    if(worst_dom>2e-6) rc=2;
    return rc;
}

int main(int argc,char**argv){
    int mlp_lut=1,skip=1,exp_fast=1;                 // default = the full optimized config
    int do_bpb=0,do_logits=0,do_tm=0; long seqW=512,eval_tok=200000,ntok=10240,offset=0,timetok=3000;
    const char* wp=NULL; const char* dumpto=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--mlp")&&i+1<argc){ i++; mlp_lut=strcmp(argv[i],"fp32")?1:0; }
        else if(!strcmp(argv[i],"--skip")&&i+1<argc){ i++; skip=strcmp(argv[i],"off")?1:0; }
        else if(!strcmp(argv[i],"--exp")&&i+1<argc){ i++; exp_fast=strcmp(argv[i],"exact")?1:0; }
        else if(!strcmp(argv[i],"--bpb")) do_bpb=1;
        else if(!strcmp(argv[i],"--logits")) do_logits=1;
        else if(!strcmp(argv[i],"--timing")) do_tm=1;
        else if(!strcmp(argv[i],"--dumplogits")&&i+1<argc) dumpto=argv[++i];
        else if(!strcmp(argv[i],"--seq")&&i+1<argc) seqW=atol(argv[++i]);
        else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--ntok")&&i+1<argc) ntok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--offset")&&i+1<argc) offset=atol(argv[++i]);
        else if(!strcmp(argv[i],"--time-tok")&&i+1<argc) timetok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--kselftest")) return kernel_selftest();
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i];
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    if(!wp){ fprintf(stderr,"usage: engine --weights <model.bin> [--mlp fp32|lut] [--skip on|off] [--exp exact|fast]\n"
                            "              [--bpb] [--logits] [--timing] [--dumplogits <file>] [--ntok N] [--offset N]\n"); return 1; }
    load_weights(wp); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    fprintf(stderr,"engine loaded: %s | mlp=%s skip=%s exp=%s | ids=%ld\n",
            g_moe?"MoE":"dense",mlp_lut?"lut":"fp32",skip?"on":"off",exp_fast?"fast":"exact",nids);
    int rc=0;
    if(dumpto) dump_logits(dumpto,seqW,ntok,offset,mlp_lut,skip,exp_fast);
    if(do_logits) rc|=gate_logits(seqW,ntok,mlp_lut,skip,exp_fast);
    if(do_bpb){ long nt; double b=run_bpb(seqW,eval_tok,mlp_lut,skip,exp_fast,&nt);
        printf("==== BPB (%ld tok): %.6f ====\n",nt,b); }
    if(do_tm) timing(timetok,mlp_lut,skip,exp_fast);
    printf("STOP. engine run above. No commit.\n");
    return rc;
}
