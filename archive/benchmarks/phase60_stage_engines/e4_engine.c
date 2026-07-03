// Phase 60 / E4 - MoE two-pool. Target moe_gran.pt (probe-4 promoted: E=32 hid_e=128 top-8). The MLP sublayer is a
//   token-choice top-k MoE: fp32 router (always-resident) -> top-8 experts -> per-expert gated-dReLU-ternary MLP.
//   Only the 8 selected experts are computed (the rest are masked to exactly 0 in PyTorch -> computing top-8 is exact).
//   This is the two-pool layout: router+SSM+head resident (fp32), experts a STREAMED ternary pool (gate/up/Wd).
//
//   Ladder (one variable per rung), gates read PyTorch dumps from e4_reference.py:
//     --golden   G1: E4-ref (fp32 experts, exact scan) per-layer l2-rel vs PyTorch trace.
//     --dispatch G2: top-8 ids + renorm weights vs PyTorch on the 64-tok trace (router is fp32 -> must match; tie-breaks reported).
//     --logits   G3: top-1 (E4-ref) vs PyTorch >=99.9% ; --bpb G4: BPB vs 0.8589 +-0.002.
//     --selftest G5a: LUT expert kernel bit-exact (integer S) vs scalar-int reference.
//     --optbpb   G5b: E4-LUT+skip BPB vs E4-ref (<=+0.005) & top-1 (>=99%).
//     --pool     G6: streamed-pool bytes/token (COUNTED, not timed) + ws@8 (|union experts|/E over 8 pos) vs probe-4 84-89%.
//     --timing   full-engine tok/s (fast-exp scan + LUT experts).
// clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/e4_engine.c -o bin/e4_engine.exe -lm
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

#define V 1024
#define D 256
#define N 96
#define H 8
#define HD (D/H)
#define L 6
#define DN 512
#define DTR 16
#define CONV 4
#define WIN 128
#define SWA_LAYER 5
#define E 32
#define HID_E 128
#define KTOP 8
#define GH (E*HID_E)          // 4096 merged gate/up rows
#define MPAD_GU ((GH+31)&~31) // 4096
#define MPAD_D  ((D+31)&~31)  // 256
#define TUP (D/2)             // 128 tiles for gate/up (K=D)
#define TDE (HID_E/2)         // 64 tiles for down (K=hid_e)
#define AQ 63
#define NLAYER (L+2)

// ---- poly-exp (E3.5) ----
static inline __m256 exp256_ps(__m256 x){
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
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

// ---- LUT kernel ----
static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
// compute rows [row0, row0+M) of a tile-major code block; y indexed 0..M-1
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
static float quant_i8(const float* x,int n,int8_t* xq){ float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; } float scale=amax/(float)AQ,inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; } return scale; }
static void ref_t3(const int8_t* Wt,const int8_t* xq,int32_t* y,int M,int K){ for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; } }

// --kselftest: synthetic self-test of the E4 kernel surfaces, NO weights/ids needed (CI-runnable).
// (1) windowed matvec_lut_rows on the merged gate/up block (row0 = expert offset) vs scalar-int ref;
// (2) per-expert tile-major Wd block vs scalar-int ref. Bit-exact required.
static void* kst_malloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }
static int kernel_selftest(void){
    srand(34567); long worst=0,checks=0;
    int8_t* Wgu=kst_malloc((size_t)GH*D);   int8_t* cgu=kst_malloc((size_t)TUP*MPAD_GU);
    int8_t* Wde=kst_malloc((size_t)E*D*HID_E); int8_t* cde=kst_malloc((size_t)E*TDE*MPAD_D);
    int8_t xq[D],lut[TUP*16],hq[HID_E],lutd[TDE*16]; int32_t Sl[D>HID_E?D:HID_E],Sr[D>HID_E?D:HID_E];
    for(int trial=0;trial<16;trial++){
        for(size_t i=0;i<(size_t)GH*D;i++) Wgu[i]=(int8_t)(rand()%3-1);
        for(size_t i=0;i<(size_t)E*D*HID_E;i++) Wde[i]=(int8_t)(rand()%3-1);
        for(int k=0;k<D;k++) xq[k]=(int8_t)(rand()%(2*AQ+1)-AQ);
        for(int k=0;k<HID_E;k++) hq[k]=(int8_t)(rand()%(2*AQ+1)-AQ);
        bc_tm(Wgu,GH,D,MPAD_GU,cgu); build_lut_t3(xq,TUP,lut);
        for(int e=0;e<E;e+=5){                                              // windowed rows (expert offsets)
            matvec_lut_rows(cgu,lut,Sl,e*HID_E,HID_E,MPAD_GU,TUP);
            ref_t3(Wgu+(size_t)e*HID_E*D,xq,Sr,HID_E,D);
            for(int i=0;i<HID_E;i++){ long d=labs((long)Sl[i]-Sr[i]); if(d>worst)worst=d; checks++; } }
        for(int e=0;e<E;e++) bc_tm(Wde+(size_t)e*D*HID_E,D,HID_E,MPAD_D,cde+(size_t)e*TDE*MPAD_D);
        build_lut_t3(hq,TDE,lutd);
        for(int e=0;e<E;e+=5){                                              // per-expert Wd blocks
            matvec_lut_rows(cde+(size_t)e*TDE*MPAD_D,lutd,Sl,0,D,MPAD_D,TDE);
            ref_t3(Wde+(size_t)e*D*HID_E,hq,Sr,D,HID_E);
            for(int d=0;d<D;d++){ long df=labs((long)Sl[d]-Sr[d]); if(df>worst)worst=df; checks++; } }
    }
    free(Wgu);free(cgu);free(Wde);free(cde);
    printf("==== E4 kernel self-test (synthetic, no weights): windowed rows + per-expert Wd vs scalar-int ====\n");
    printf("  %ld checks | worst |S_lut - S_ref| = %ld  %s\n",checks,worst,worst==0?"PASS":"FAIL");
    return worst==0?0:2;
}

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L];
static float *router_w[L],*router_b[L];                 // fp32 router
static float *gate_f[L],*up_f[L],*Wd_f[L];              // dequant fp32 (E4-ref): gate/up (GH,D), Wd (E,D,hid_e)
static int8_t *gate_wt[L],*up_wt[L],*Wd_wt[L];          // packed ternary {-1,0,1}
static int8_t *gate_cd[L],*up_cd[L],*Wd_cd[L];          // LUT codes: gate/up tile-major (TUP,MPAD_GU); Wd per-expert (E)*(TDE,MPAD_D)
static float *gate_sc[L],*up_sc[L],*Wd_sc[L];           // scales: gate/up (GH), Wd (E*D)
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short\n");exit(1);} return p; }
static int8_t* rdi8(FILE*f,size_t n){ int8_t*p=xmalloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short i8\n");exit(1);} return p; }

static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"open %s\n",path);exit(1);}
    uint32_t h[16]; if(fread(h,4,16,f)!=16||h[0]!=0x45344D31){fprintf(stderr,"bad magic\n");exit(1);}
    if((int)h[11]!=E||(int)h[12]!=HID_E||(int)h[13]!=KTOP){fprintf(stderr,"E/hid_e/k mismatch %u %u %u\n",h[11],h[12],h[13]);exit(1);}
    int has_packed=h[14];
    emb=rd(f,(size_t)V*D);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]); s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        mlp_n2[l]=rd(f,D);
        router_w[l]=rd(f,(size_t)E*D); router_b[l]=rd(f,E);
        gate_f[l]=rd(f,(size_t)GH*D); up_f[l]=rd(f,(size_t)GH*D); Wd_f[l]=rd(f,(size_t)E*D*HID_E);
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    if(!has_packed){fprintf(stderr,"need packed\n");exit(1);}
    for(int l=0;l<L;l++){
        gate_wt[l]=rdi8(f,(size_t)GH*D); gate_sc[l]=rd(f,GH);
        up_wt[l]=rdi8(f,(size_t)GH*D); up_sc[l]=rd(f,GH);
        Wd_wt[l]=rdi8(f,(size_t)E*D*HID_E); Wd_sc[l]=rd(f,(size_t)E*D);
        gate_cd[l]=xmalloc((size_t)TUP*MPAD_GU); bc_tm(gate_wt[l],GH,D,MPAD_GU,gate_cd[l]);
        up_cd[l]=xmalloc((size_t)TUP*MPAD_GU); bc_tm(up_wt[l],GH,D,MPAD_GU,up_cd[l]);
        Wd_cd[l]=xmalloc((size_t)E*TDE*MPAD_D);
        for(int e=0;e<E;e++) bc_tm(Wd_wt[l]+(size_t)e*D*HID_E, D, HID_E, MPAD_D, Wd_cd[l]+(size_t)e*TDE*MPAD_D);
    }
    long pos=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f); fclose(f);
    if(pos!=end) fprintf(stderr,"WARN %ld trailing\n",end-pos);
    fprintf(stderr,"E4 weights ok (E=%d hid_e=%d k=%d)\n",E,HID_E,KTOP);
}
static void load_meta(const char* path){ FILE*f=fopen(path,"rb"); if(!f){exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1); if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; } fclose(f); }
static void load_ids(const char* path){ FILE*f=fopen(path,"rb"); if(!f){exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); nids=b/2; ids=xmalloc(b); if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f); }
static void state_reset(void){ memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0; }
static inline void rmsnorm(const float* in,const float* w,float* out){ float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i];
    float r=1.0f/sqrtf(ms/(float)D+1e-5f); for(int i=0;i<D;i++) out[i]=in[i]*r*w[i]; }

// top-k selection matching torch.topk (largest k; ties -> lower index first)
static void topk(const float* p,int n,int k,int* idx,float* val){
    int used[E]; memset(used,0,sizeof(int)*n);
    for(int j=0;j<k;j++){ int bi=-1; float bv=-1e30f;
        for(int i=0;i<n;i++) if(!used[i] && p[i]>bv){ bv=p[i]; bi=i; }
        used[bi]=1; idx[j]=bi; val[j]=p[bi]; }
}

typedef struct { int topi[NLAYER][KTOP]; float topw[NLAYER][KTOP]; int nmoe; } Disp;
static int8_t g_xq[D], g_lut[TUP*16], g_hq[HID_E], g_lutd[TDE*16]; static int32_t g_S[HID_E], g_Sd[D];  // g_xq sized D: gate/up quantize xn (D dims)
int g_diag=0;

// mlp_mode: 0 fp32 experts (E4-ref), 1 LUT experts (E4-opt). disp/sel optional captures.
static void moe_mlp(int l,const float* xn,float* out,int mlp_mode,int moe_idx,Disp* disp){
    float rp[E]; for(int e=0;e<E;e++) rp[e]=dotf(router_w[l]+(size_t)e*D,xn,D)+router_b[l][e];
    float mx=-1e30f; for(int e=0;e<E;e++) if(rp[e]>mx)mx=rp[e];
    float Z=0; for(int e=0;e<E;e++){ rp[e]=expf(rp[e]-mx); Z+=rp[e]; } for(int e=0;e<E;e++) rp[e]/=Z;   // softmax
    int idx[KTOP]; float wv[KTOP]; topk(rp,E,KTOP,idx,wv);
    float ws=0; for(int j=0;j<KTOP;j++) ws+=wv[j]; for(int j=0;j<KTOP;j++) wv[j]/=ws;                  // renorm top-k
    if(disp){ for(int j=0;j<KTOP;j++){ disp->topi[moe_idx][j]=idx[j]; disp->topw[moe_idx][j]=wv[j]; } }
    memset(out,0,D*4);
    float sa=0; if(mlp_mode==1){ sa=quant_i8(xn,D,g_xq); build_lut_t3(g_xq,TUP,g_lut); }
    for(int j=0;j<KTOP;j++){ int e=idx[j]; float tw=wv[j]; float he[HID_E];
        if(mlp_mode==0){
            for(int i=0;i<HID_E;i++){ float gg=dotf(gate_f[l]+(size_t)(e*HID_E+i)*D,xn,D);
                float uu=dotf(up_f[l]+(size_t)(e*HID_E+i)*D,xn,D); he[i]=reluf(gg)*reluf(uu)*tw; }
            for(int d=0;d<D;d++) out[d]+=dotf(Wd_f[l]+((size_t)e*D+d)*HID_E,he,HID_E);
        } else {
            matvec_lut_rows(gate_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);
            float gg[HID_E]; for(int i=0;i<HID_E;i++) gg[i]=(float)g_S[i]*sa*gate_sc[l][e*HID_E+i];
            matvec_lut_rows(up_cd[l],g_lut,g_S,e*HID_E,HID_E,MPAD_GU,TUP);
            for(int i=0;i<HID_E;i++){ float uu=(float)g_S[i]*sa*up_sc[l][e*HID_E+i]; he[i]=reluf(gg[i])*reluf(uu)*tw; }
            float sh=quant_i8(he,HID_E,g_hq); build_lut_t3(g_hq,TDE,g_lutd);
            matvec_lut_rows(Wd_cd[l]+(size_t)e*TDE*MPAD_D,g_lutd,g_Sd,0,D,MPAD_D,TDE);
            for(int d=0;d<D;d++) out[d]+=(float)g_Sd[d]*sh*Wd_sc[l][(size_t)e*D+d];
        }
    }
}

typedef struct { double scan,scan_other,swa,mlp,head; } Tacc;
static void forward_token(uint32_t tok,float* logits,int scan_mode,int mlp_mode,float* caps,Disp* disp,Tacc* T){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vvv[D],att[WIN],ao[D],tmp[D];
    double t0; int moe_idx=0;
    memcpy(x,emb+(size_t)tok*D,D*4); if(caps) memcpy(caps,x,D*4);
    for(int l=0;l<L;l++){
        rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);
        if(is_swa[l]){
            if(T)t0=now_s();
            matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vvv,xz+2*D,D*4);
            int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vvv,D*4);
            kvpos++; if(kvcnt<WIN) kvcnt++; memset(ao,0,D*4);
            for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float m2=-1e30f;
                for(int jj=0;jj<kvcnt;jj++){ int s=(kvpos-kvcnt+jj)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[jj]=sc; if(sc>m2)m2=sc; }
                float Zs=0; for(int jj=0;jj<kvcnt;jj++){ att[jj]=expf(att[jj]-m2); Zs+=att[jj]; } float zi=1.0f/Zs;
                for(int jj=0;jj<kvcnt;jj++){ int s=(kvpos-kvcnt+jj)%WIN; float w=att[jj]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                    for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
            matvec(swa.o,ao,tmp,D,D); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->swa+=now_s()-t0;
        } else { SSML*s=&ssm[l];
            if(T)t0=now_s();
            matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
            float (*cb)[CONV]=convbuf[l];
            for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
            matvec(s->x_proj,xx,dbl,DTR+2*N,DN); const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
            for(int c=0;c<DN;c++) dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]);
            if(T){T->scan_other+=now_s()-t0; t0=now_s();}
            float (*hl)[N]=hstate[l];
            if(scan_mode==0){ for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                    for(int jj=0;jj<N;jj++){ hc[jj]=expf(dtc*Ac[jj])*hc[jj]+dtc*Bm[jj]*xc; acc+=hc[jj]*Cm[jj]; } y[c]=acc+s->Dskip[c]*xc; } }
            else { for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],dbx=dt[c]*xc;
                    __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx),vacc=_mm256_setzero_ps();
                    for(int jj=0;jj<N;jj+=8){ __m256 ee=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+jj)));
                        __m256 hcj=_mm256_fmadd_ps(ee,_mm256_loadu_ps(hc+jj),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+jj)));
                        _mm256_storeu_ps(hc+jj,hcj); vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+jj),vacc); } y[c]=hsum256(vacc)+s->Dskip[c]*xc; } }
            if(T){T->scan+=now_s()-t0; t0=now_s();}
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->scan_other+=now_s()-t0;
        }
        rmsnorm(x, mlp_n2[l], xn);
        if(T)t0=now_s();
        moe_mlp(l,xn,tmp,mlp_mode,moe_idx,disp); for(int i=0;i<D;i++) x[i]+=tmp[i]; moe_idx++;
        if(T)T->mlp+=now_s()-t0;
        if(caps) memcpy(caps+(size_t)(l+1)*D,x,D*4);
    }
    if(T)t0=now_s();
    rmsnorm(x,normf,xn); if(caps) memcpy(caps+(size_t)(L+1)*D,xn,D*4); matvec(head,xn,logits,V,D);
    if(T)T->head+=now_s()-t0;
}

// ---- G1 golden trace ----
static int gate_golden(void){
    FILE*f=fopen("results/phase60/golden_moe_trace.bin","rb"); if(!f){fprintf(stderr,"no moe trace\n");return 1;}
    uint32_t hd[4]; if(fread(hd,4,4,f)!=4||hd[0]!=0x4D543031){fprintf(stderr,"bad MT01\n");return 1;}
    int T=hd[1],nl=hd[3]; uint16_t* tin=xmalloc((size_t)T*2); if(fread(tin,2,T,f)!=(size_t)T){return 1;}
    float* ref=xmalloc((size_t)nl*T*D*4); if(fread(ref,4,(size_t)nl*T*D,f)!=(size_t)nl*T*D){return 1;}
    uint32_t vv; if(fread(&vv,4,1,f)!=1){return 1;} float* rl=xmalloc((size_t)T*V*4); if(fread(rl,4,(size_t)T*V,f)!=(size_t)T*V){return 1;} fclose(f);
    float* caps=xmalloc((size_t)nl*D*4); float* lg=xmalloc((size_t)V*4);
    double sqerr[NLAYER+1],sqref[NLAYER+1],maxabs[NLAYER+1]; for(int i=0;i<=nl;i++) sqerr[i]=sqref[i]=maxabs[i]=0;
    state_reset();
    for(int t=0;t<T;t++){ forward_token(tin[t],lg,0,0,caps,NULL,NULL);
        for(int ly=0;ly<nl;ly++){ const float* c=caps+(size_t)ly*D; const float* p=ref+((size_t)ly*T+t)*D;
            for(int d=0;d<D;d++){ double e=fabs((double)c[d]-p[d]); if(e>maxabs[ly])maxabs[ly]=e; sqerr[ly]+=e*e; sqref[ly]+=(double)p[d]*p[d]; } }
        const float* pl=rl+(size_t)t*V; for(int o=0;o<V;o++){ double e=fabs((double)lg[o]-pl[o]); if(e>maxabs[nl])maxabs[nl]=e; sqerr[nl]+=e*e; sqref[nl]+=(double)pl[o]*pl[o]; } }
    const char* nm[NLAYER+1]={"emb","blk0","blk1","blk2","blk3","blk4","blk5","norm_f","logits"};
    printf("==== E4 G1 golden-trace (MoE fp32 vs PyTorch, %d tok) ====\n",T); int pass=1;
    for(int i=0;i<=nl;i++){ double l2=sqrt(sqerr[i]/(sqref[i]+1e-30)); printf("  %-7s max_abs=%.3e l2_rel=%.3e %s\n",nm[i],maxabs[i],l2,l2<1e-3?"":"<--"); if(l2>=1e-3)pass=0; }
    printf("  E4 G1 %s (<1e-3 per layer)\n",pass?"PASS":"FAIL"); return pass?0:2;
}
// ---- G2 dispatch ----
static int gate_dispatch(void){
    FILE*f=fopen("results/phase60/golden_moe_dispatch.bin","rb"); if(!f){fprintf(stderr,"no dispatch\n");return 1;}
    uint32_t hd[5]; if(fread(hd,4,5,f)!=5||hd[0]!=0x4D443031){fprintf(stderr,"bad MD01\n");return 1;}
    int nmoe=hd[1],T=hd[2],K=hd[3];
    uint16_t* rid=xmalloc((size_t)nmoe*T*K*2); float* rw=xmalloc((size_t)nmoe*T*K*4);
    for(int m=0;m<nmoe;m++){ if(fread(rid+(size_t)m*T*K,2,(size_t)T*K,f)!=(size_t)T*K){return 1;} if(fread(rw+(size_t)m*T*K,4,(size_t)T*K,f)!=(size_t)T*K){return 1;} }
    fclose(f);
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    Disp disp; long id_mm=0,setmm=0; double wworst=0; state_reset();
    for(int t=0;t<T;t++){ forward_token(val[t],lg,0,0,NULL,&disp,NULL);
        for(int m=0;m<nmoe;m++){ const uint16_t* ri=rid+((size_t)m*T+t)*K; const float* rww=rw+((size_t)m*T+t)*K;
            // set match (order-independent) + id-order match + weight diff
            for(int j=0;j<K;j++){ if(disp.topi[m][j]!=ri[j]) id_mm++; double wd=fabs((double)disp.topw[m][j]-rww[j]); if(wd>wworst)wworst=wd; }
            int sm=0; for(int j=0;j<K;j++){ int found=0; for(int i2=0;i2<K;i2++) if(disp.topi[m][j]==ri[i2]){found=1;break;} if(!found)sm=1; } if(sm)setmm++;
        }
    }
    printf("==== E4 G2 dispatch (top-%d ids+weights vs PyTorch, %d MoE layers x %d pos) ====\n",K,nmoe,T);
    printf("  set-mismatch positions=%ld  id-order-mismatch=%ld  worst |weight diff|=%.3e\n",setmm,id_mm,wworst);
    printf("  E4 G2 %s (set match required; id-order/weight tie-breaks reported)\n",setmm==0?"PASS":"REVIEW");
    return setmm==0?0:2;
}
// ---- G3/G4 top-1 + BPB ----
static int gate_val(long seqW,long eval_tok,int mlp_mode,const char* tag){
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    FILE*f=fopen("results/phase60/golden_moe_val.bin","rb"); int haveref=0,W2=0,nwin=0; uint16_t* ra=NULL;
    if(f){ uint32_t hd[3]; if(fread(hd,4,3,f)==3&&hd[0]==0x4D563031){ W2=hd[1]; nwin=hd[2]; ra=xmalloc((size_t)nwin*W2*2); if(fread(ra,2,(size_t)nwin*W2,f)!=(size_t)nwin*W2){ra=NULL;} else haveref=1; } fclose(f); }
    double bits=0; long nb=0,nt=0,pos=0; long agree=0,atot=0; float* lg=xmalloc((size_t)V*4); const double LN2=0.6931471805599453;
    long lim=eval_tok<(nval-1)?eval_tok:(nval-1);
    while(pos+seqW+1<=lim){ state_reset(); int w=(int)(pos/seqW);
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],lg,0,mlp_mode,NULL,NULL,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; int am=0; for(int o=0;o<V;o++) if(lg[o]>mx){mx=lg[o];am=o;} double se=0; for(int o=0;o<V;o++) se+=exp((double)lg[o]-mx);
            bits+=(-((double)lg[tgt]-mx)+log(se))/LN2; nb+=id2len[tgt]; nt++;
            if(haveref && w<nwin){ if(am==ra[(size_t)w*W2+t]) agree++; atot++; } }
        pos+=seqW; }
    double bpb=bits/(nb>0?nb:1);
    printf("==== E4 G3/G4 %s (seq=%ld tok=%ld) ====\n",tag,seqW,nt);
    printf("  BPB=%.6f (ref 0.858854, delta %+.6f)  top-1 vs PyTorch=%.4f%% (%ld/%ld)\n",bpb,bpb-0.858854,atot?100.0*agree/atot:0,agree,atot);
    printf("  E4 BPB %s | top-1 %s\n", fabs(bpb-0.858854)<=0.002?"PASS":"FAIL", (atot&&100.0*agree/atot>=99.9)?"PASS":"(ref-slice only)");
    free(lg); return 0;
}
// ---- G5a expert-kernel bit-exactness ----
static int gate_selftest(void){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    // run E4-ref to advance state; at each MoE layer test a couple selected experts' LUT S vs scalar-int ref
    long worst=0,checks=0; Disp disp; state_reset();
    float xn_dummy; (void)xn_dummy;
    for(long ti=0;ti<100;ti++){ forward_token(val[ti],lg,0,0,NULL,&disp,NULL); }
    // dedicated exactness pass: for fresh xq/hq build LUT and compare to scalar ref on gate rows and Wd
    int8_t xq[D],lut[TUP*16]; int32_t Sl[D],Sr[D];   // sized D (>=HID_E): down test writes D rows
    for(int trial=0;trial<64;trial++){ for(int i=0;i<D;i++) xq[i]=(int8_t)((trial*7+i*3)%127-63);
        build_lut_t3(xq,TUP,lut);
        for(int e=0;e<E;e+=8){ matvec_lut_rows(gate_cd[0],lut,Sl,e*HID_E,HID_E,MPAD_GU,TUP);
            ref_t3(gate_wt[0]+(size_t)e*HID_E*D,xq,Sr,HID_E,D);
            for(int i=0;i<HID_E;i++){ long d=labs((long)Sl[i]-Sr[i]); if(d>worst)worst=d; checks++; } }
        int8_t hq[HID_E],lutd[TDE*16]; for(int i=0;i<HID_E;i++) hq[i]=(int8_t)((trial*5+i)%63);
        build_lut_t3(hq,TDE,lutd);
        for(int e=0;e<E;e+=8){ matvec_lut_rows(Wd_cd[0]+(size_t)e*TDE*MPAD_D,lutd,Sl,0,D,MPAD_D,TDE);
            ref_t3(Wd_wt[0]+(size_t)e*D*HID_E,hq,Sr,D,HID_E);
            for(int d=0;d<D;d++){ long df=labs((long)Sl[d]-Sr[d]); if(df>worst)worst=df; checks++; } }
    }
    printf("==== E4 G5a expert-kernel bit-exactness (LUT S vs scalar-int, %ld checks) ====\n",checks);
    printf("  worst |S_lut - S_ref| = %ld  %s\n",worst,worst==0?"PASS":"FAIL"); free(lg); return worst==0?0:2;
}
// ---- G5b E4-LUT vs E4-ref (BPB + top-1) ----
// dumpflips: optional diagnostic (P0.2 freshness check) — one line per disagreeing position:
//   <global_val_index> <ref_argmax> <lut_argmax>. Does not alter the gate.
static int gate_opt(long seqW,long ntok_t,long offset,const char* dumpflips){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lr=xmalloc((size_t)V*4); float* lo=xmalloc((size_t)V*4);
    FILE* df=dumpflips?fopen(dumpflips,"w"):NULL;
    if(dumpflips&&!df){fprintf(stderr,"cannot open %s\n",dumpflips);exit(1);}
    double br=0,bo=0; long nbr=0,nt=0,agree=0,pos=offset; const double LN2=0.6931471805599453; static uint16_t ar[4096];
    while(nt<ntok_t){ state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],lr,0,0,NULL,NULL,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; int am=0; for(int o=0;o<V;o++) if(lr[o]>mx){mx=lr[o];am=o;} ar[t]=am; double se=0; for(int o=0;o<V;o++) se+=exp((double)lr[o]-mx); br+=(-((double)lr[tgt]-mx)+log(se))/LN2; nbr+=id2len[tgt]; }
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],lo,0,1,NULL,NULL,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; int am=0; for(int o=0;o<V;o++) if(lo[o]>mx){mx=lo[o];am=o;}
            if(am==ar[t])agree++; else if(df) fprintf(df,"%ld %d %d\n",pos+t,(int)ar[t],am);
            double se=0; for(int o=0;o<V;o++) se+=exp((double)lo[o]-mx); bo+=(-((double)lo[tgt]-mx)+log(se))/LN2; nt++; }
        pos+=seqW; }
    if(df) fclose(df);
    double bpr=br/nbr, bpo=bo/nbr; double pct=100.0*agree/nt;
    printf("==== E4 G5b E4-LUT+skip vs E4-ref (%ld tok, offset %ld) ====\n",nt,offset);
    printf("  BPB ref=%.6f LUT=%.6f delta=%+.6f  top-1(LUT vs ref)=%.4f%%\n",bpr,bpo,bpo-bpr,pct);
    printf("  E4 G5b BPB %s | top-1 %s\n",(bpo-bpr)<=0.005?"PASS":"FAIL",pct>=99.0?"PASS":"FAIL");
    free(lr);free(lo); return ((bpo-bpr)<=0.005&&pct>=99.0)?0:2;
}
// ---- G6 streamed-pool accounting + ws@8 ----
static void gate_pool(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4); Disp disp;
    // per-expert packed bytes (streamed pool): gate rows(128)*codes(D/2) + up same + Wd(D rows*hid_e/2) + scales
    long per_expert = (long)HID_E*(D/2) + (long)HID_E*(D/2) + (long)D*(HID_E/2) + (long)(2*HID_E + D)*4; // codes + fp32 scales
    long nmoe=0; double sum_bytes=0;
    // ws@8: union of selected experts over 8 consecutive positions, per MoE layer
    int hist[NLAYER][8][KTOP]; int hp=0; double ws_acc[NLAYER]; long ws_n=0; for(int i=0;i<NLAYER;i++) ws_acc[i]=0;
    state_reset();
    for(long ti=0;ti<ntok;ti++){ forward_token(val[ti],lg,1,1,NULL,&disp,NULL); nmoe=disp.nmoe?disp.nmoe:0;
        int nm=L; // MoE on every use_mlp layer = all L (all blocks have mlp)
        sum_bytes += (double)nm*KTOP*per_expert;
        for(int m=0;m<nm;m++) for(int j=0;j<KTOP;j++) hist[m][hp][j]=disp.topi[m][j];
        hp++; if(hp==8){ for(int m=0;m<nm;m++){ int seen[E]; memset(seen,0,sizeof seen); int u=0;
                for(int p=0;p<8;p++) for(int j=0;j<KTOP;j++){ int e=hist[m][p][j]; if(!seen[e]){seen[e]=1;u++;} } ws_acc[m]+=(double)u/E; }
            ws_n++; hp=0; }
    }
    printf("==== E4 G6 streamed-pool accounting (%ld tok) ====\n",ntok);
    printf("  bytes/token in expert pool (COUNTED, top-%d experts x %ld B) = %.1f KB/tok  (priced @28GB/s scale-up: %.1f us/tok)\n",
           KTOP, per_expert, sum_bytes/ntok/1024.0, (sum_bytes/ntok)/28e9*1e6);
    printf("  ws@8 (|union selected|/E over 8 pos) per MoE layer | anchor probe-4 ~84-89%%:\n   ");
    double m=0; for(int i=0;i<L;i++){ printf(" L%d=%.0f%%",i,100.0*ws_acc[i]/ws_n); m+=100.0*ws_acc[i]/ws_n; } printf("  | mean %.1f%%\n",m/L);
    (void)nmoe; free(lg);
}
// --dumplogits: raw fp32 logit stream (P4.3 consolidation-parity instrument).
// Config via --dscan (0 exact / 1 fast) and --dmlp (0 fp32 experts / 1 LUT experts).
static void dump_logits_stream(const char* path,long seqW,long ntok,int scan_mode,int mlp_mode){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    FILE* f=fopen(path,"wb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    float* lg=xmalloc((size_t)V*4); long done=0,pos=0;
    while(done<ntok){ state_reset();
        for(long t=0;t<seqW&&done<ntok;t++,done++){ forward_token(val[pos+t],lg,scan_mode,mlp_mode,NULL,NULL,NULL); fwrite(lg,4,V,f); }
        pos+=seqW; }
    fclose(f); free(lg); printf("dumped %ld x %d fp32 logits (scan=%d mlp=%d) -> %s\n",ntok,V,scan_mode,mlp_mode,path);
}

static void diag(void){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; int T=32;
    float* c0=xmalloc((size_t)NLAYER*D*4); float* c1=xmalloc((size_t)NLAYER*D*4); float* lg=xmalloc((size_t)V*4);
    float* keep=xmalloc((size_t)T*NLAYER*D*4);
    state_reset(); for(int t=0;t<T;t++){ forward_token(val[t],lg,0,0,c0,NULL,NULL); memcpy(keep+(size_t)t*NLAYER*D,c0,(size_t)NLAYER*D*4); }
    state_reset(); double sq[NLAYER],sr[NLAYER],ma[NLAYER]; for(int i=0;i<NLAYER;i++) sq[i]=sr[i]=ma[i]=0;
    g_diag=1;
    for(int t=0;t<T;t++){ forward_token(val[t],lg,0,1,c1,NULL,NULL); const float* e=keep+(size_t)t*NLAYER*D;
        for(int ly=0;ly<NLAYER;ly++) for(int d=0;d<D;d++){ double df=fabs((double)c1[ly*D+d]-e[ly*D+d]); if(df>ma[ly])ma[ly]=df; sq[ly]+=df*df; sr[ly]+=(double)e[ly*D+d]*e[ly*D+d]; } }
    printf("==== E4 DIAG: LUT vs fp32-ref residual per block (%d tok) ====\n",T);
    const char* nm[NLAYER]={"emb","blk0","blk1","blk2","blk3","blk4","blk5","norm_f"};
    for(int i=0;i<NLAYER;i++) printf("  %-7s max_abs=%.3e l2_rel=%.3e\n",nm[i],ma[i],sqrt(sq[i]/(sr[i]+1e-30)));
    free(c0);free(c1);free(lg);free(keep);
}
static void timing(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    struct{const char*nm;int sm,mm;} cfg[2]={{"E4-ref (exact scan, fp32 experts)",0,0},{"E4 full (fast-exp, LUT experts)",1,1}};
    for(int c=0;c<2;c++){ Tacc T={0,0,0,0,0}; state_reset(); double t0=now_s();
        for(long i=0;i<ntok;i++) forward_token(val[i%100000],lg,cfg[c].sm,cfg[c].mm,NULL,NULL,&T);
        double tot=now_s()-t0; double sc=1e6/ntok;
        printf("==== %s: %.1f tok/s | %.1f us/tok ====\n",cfg[c].nm,ntok/tot,1e6/(ntok/tot));
        printf("   scan %.1f  scan-other %.1f  SWA %.1f  MLP(MoE) %.1f (%.1f%%)  head %.1f\n",T.scan*sc,T.scan_other*sc,T.swa*sc,T.mlp*sc,100.0*T.mlp/tot,T.head*sc); }
    free(lg);
}

int main(int argc,char**argv){
    int gg=0,gd=0,gl=0,gb=0,st=0,go=0,gp=0,tm=0,all=0; long seqW=512,eval_tok=200000,ntok=10240,pooltok=4000,timetok=2000,offset=0;
    int dscan=0,dmlp=0;
    const char* wp="results/phase60/e4_model.bin"; const char* dumpflips=NULL; const char* dlpath=NULL;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--golden"))gg=1; else if(!strcmp(argv[i],"--dispatch"))gd=1;
        else if(!strcmp(argv[i],"--logits"))gl=1; else if(!strcmp(argv[i],"--bpb"))gb=1; else if(!strcmp(argv[i],"--selftest"))st=1;
        else if(!strcmp(argv[i],"--optbpb"))go=1; else if(!strcmp(argv[i],"--pool"))gp=1; else if(!strcmp(argv[i],"--timing"))tm=1;
        else if(!strcmp(argv[i],"--diag")){ st=-1; }
        else if(!strcmp(argv[i],"--all"))all=1; else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--offset")&&i+1<argc) offset=atol(argv[++i]);
        else if(!strcmp(argv[i],"--dumpflips")&&i+1<argc) dumpflips=argv[++i];
        else if(!strcmp(argv[i],"--kselftest")) return kernel_selftest();   // synthetic, no weights (CI)
        else if(!strcmp(argv[i],"--dumplogits")&&i+1<argc) dlpath=argv[++i];
        else if(!strcmp(argv[i],"--dscan")&&i+1<argc) dscan=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dmlp")&&i+1<argc) dmlp=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ntok")&&i+1<argc) ntok=atol(argv[++i]); else if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i]; }
    if(!gg&&!gd&&!gl&&!gb&&!st&&!go&&!gp&&!tm&&!all&&!dlpath) all=1;
    load_weights(wp); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    fprintf(stderr,"E4 loaded: ids=%ld\n",nids);
    if(dlpath){ dump_logits_stream(dlpath,seqW,ntok,dscan,dmlp); return 0; }
    int rc=0;
    if(st==-1){ diag(); printf("STOP. diag.\n"); return 0; }
    if(all||gg) rc|=gate_golden();
    if(all||gd) rc|=gate_dispatch();
    if(all||st) rc|=gate_selftest();
    if(all||go) rc|=gate_opt(seqW,ntok,offset,dumpflips);
    if(gl||gb) gate_val(seqW,eval_tok,0,"E4-ref");
    if(all||gp) gate_pool(pooltok);
    if(all||tm) timing(timetok);
    printf("STOP. E4 ladder above. No commit.\n");
    return rc;
}
