// Phase 60 / E2 - ternary pshufb-LUT MLP. ONE VARIABLE vs E1: the 3 MLP matvecs (gate/up/down) use the probe-1
//   ternary LUT kernel (g=2 base-3, 9-entry int8 LUT, pshufb 32 rows/instr) instead of fp32 FMA. Everything else
//   (emb, SSM scan, SWA, head, norms, decode) is byte-for-byte the E1 fp32 core.
//
//   The LUT needs QUANTIZED activations. Per-token absmax -> int8 in [-63,63] (for ternary g=2, a LUT entry
//   w0*x0+w1*x1 with |w|<=1 stays <= 2*63=126 < 127, so it fits int8 -> the bit-exact pshufb kernel is untouched;
//   63 is the widest range that fits, i.e. best activation precision). Dequant: y_real[m] = S[m] * act_scale * w_scale[m],
//   where S[m] = integer dot(wq[m,:], xq[:]) and w_scale[m] = the BitLinear158 per-row absmean scale (from the export).
//
//   Activation quantization is a NEW inference-time error (the model trained with fp activations). It is MEASURED here.
//   Reads results/phase60/e1_model.bin (magic E1M1: fp32 core + packed ternary already present).
//
//   Modes:
//     --selftest   G1(E2): LUT integer S bit-exact vs scalar integer reference, all 3 matvecs x all layers.
//     --bpb        G2: BPB with LUT MLP vs the in-binary E1 fp32 path (delta must be <= +0.005).
//     --logits     G3: top-1 agreement LUT vs E1 fp32 over >=10K val tokens (same binary, only MLP path differs).
//     --timing     G4: per-component time breakdown (ssm-scan/ssm-other/SWA/MLP/head), fp32 vs LUT MLP (Amdahl-explicit).
//     --gen        greedy sample with LUT MLP (eyeball).
//
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/e2_engine.c -o bin/e2_engine.exe -lm
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
#define MLP_HID 1024
#define AQ 63                      // activation quant range [-AQ,AQ]

static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

// ---------------- ternary LUT kernel (probe-1, bit-exact) ----------------
static inline void acc_add_i8x32(__m256i* acc, __m256i p){
    __m128i lo=_mm256_castsi256_si128(p), hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo));
    acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi));
    acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_t3(const int8_t* codes, const int8_t* lut, int32_t* y, int M, int Mpad, int K){
    int T=K/2;
    for(int base=0; base<M; base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base));
            acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx));
        }
        int32_t tmp[32];
        _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32 && base+r<M;r++) y[base+r]=tmp[r];
    }
}
static void build_lut_t3(const int8_t* xq, int K, int8_t* lut){    // 9 used entries (0..8), g=2 base-3
    int T=K/2;
    for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
        for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1, w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } }
}
static void build_codes(const int8_t* Wt, int M, int K, int Mpad, int8_t* codes){  // Wt in {-1,0,1}, row-major [m*K+k]
    int T=K/2;
    for(int t=0;t<T;t++){
        for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t], w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0;
    }
}
static void ref_t3(const int8_t* Wt, const int8_t* xq, int32_t* y, int M, int K){   // scalar integer reference
    for(int m=0;m<M;m++){ long s=0; for(int k=0;k<K;k++) s+=(long)Wt[(size_t)m*K+k]*xq[k]; y[m]=(int32_t)s; }
}
static float quant_i8(const float* x, int n, int8_t* xq){       // per-token absmax -> [-AQ,AQ]; returns scale
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; }
    float scale=amax/(float)AQ, inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; }
    return scale;
}

// --kselftest: synthetic kernel self-test, NO weights/ids needed (CI-runnable).
// Random ternary weights + int8 activations in [-AQ,AQ]; LUT path vs scalar-int reference, bit-exact required.
static void* xmalloc(size_t n);
static int kernel_selftest(void){
    srand(12345); int worst=0; long checks=0;
    int dims[2][2]={{MLP_HID,D},{D,MLP_HID}};           // both (M,K) shapes the engine uses
    for(int c=0;c<2;c++){ int M=dims[c][0],K=dims[c][1],Mpad=(M+31)&~31;
        int8_t* Wt=xmalloc((size_t)M*K); int8_t* codes=xmalloc((size_t)(K/2)*Mpad);
        int8_t* xq=xmalloc(K); int8_t* lut=xmalloc((size_t)(K/2)*16);
        int32_t* Sl=xmalloc((size_t)M*4); int32_t* Sr=xmalloc((size_t)M*4);
        for(int trial=0;trial<32;trial++){
            for(size_t i=0;i<(size_t)M*K;i++) Wt[i]=(int8_t)(rand()%3-1);
            for(int k=0;k<K;k++) xq[k]=(int8_t)(rand()%(2*AQ+1)-AQ);
            build_codes(Wt,M,K,Mpad,codes); build_lut_t3(xq,K,lut);
            matvec_lut_t3(codes,lut,Sl,M,Mpad,K); ref_t3(Wt,xq,Sr,M,K);
            for(int m=0;m<M;m++){ int d=abs(Sl[m]-Sr[m]); if(d>worst)worst=d; checks++; }
        }
        free(Wt);free(codes);free(xq);free(lut);free(Sl);free(Sr);
    }
    printf("==== E2 kernel self-test (synthetic, no weights): pshufb-LUT vs scalar-int ====\n");
    printf("  %ld dot-products | worst |S_lut - S_ref| = %d  %s\n",checks,worst,worst==0?"PASS":"FAIL");
    return worst==0?0:2;
}

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
typedef struct { int8_t* codes; int8_t* wt; float* ws; int M,K,Mpad; } LutW;   // wt kept for the bit-exact self-test
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L], *mlp_gate_f[L], *mlp_up_f[L], *mlp_down_f[L];          // fp32 dequant (E1 path)
static LutW lg[L], lu[L], ld[L];                                               // LUT gate/up/down (E2 path)
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} return p; }
static int8_t* rdi8(FILE*f,size_t n){ int8_t*p=xmalloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short read i8\n");exit(1);} return p; }

static void setup_lut(LutW* w, int8_t* wt, float* ws, int M, int K){
    w->wt=wt; w->ws=ws; w->M=M; w->K=K; w->Mpad=(M+31)&~31;
    w->codes=xmalloc((size_t)(K/2)*w->Mpad); build_codes(wt,M,K,w->Mpad,w->codes);
}
static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t h[16]; if(fread(h,4,16,f)!=16||h[0]!=0x45314D31){fprintf(stderr,"bad magic\n");exit(1);}
    if((int)h[11]!=MLP_HID){fprintf(stderr,"mlp_hid mismatch\n");exit(1);}
    int has_packed=h[14];
    emb=rd(f,(size_t)V*D);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);
            s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        mlp_n2[l]=rd(f,D);
        mlp_gate_f[l]=rd(f,(size_t)MLP_HID*D); mlp_up_f[l]=rd(f,(size_t)MLP_HID*D); mlp_down_f[l]=rd(f,(size_t)D*MLP_HID);
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    if(!has_packed){fprintf(stderr,"E2 needs packed ternary (re-run e1_export.py)\n");exit(1);}
    for(int l=0;l<L;l++){
        int8_t* gq=rdi8(f,(size_t)MLP_HID*D); float* gs=rd(f,MLP_HID);
        int8_t* uq=rdi8(f,(size_t)MLP_HID*D); float* us=rd(f,MLP_HID);
        int8_t* dq=rdi8(f,(size_t)D*MLP_HID); float* ds=rd(f,D);
        setup_lut(&lg[l],gq,gs,MLP_HID,D); setup_lut(&lu[l],uq,us,MLP_HID,D); setup_lut(&ld[l],dq,ds,D,MLP_HID);
    }
    long pos=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f); fclose(f);
    if(pos!=end) fprintf(stderr,"WARN %ld trailing bytes\n",end-pos);
    fprintf(stderr,"E2 weights ok (fp32 core + packed ternary; act quant [-%d,%d])\n",AQ,AQ);
}
static void load_meta(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1);
        if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; } fclose(f);
}
static void load_ids(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){exit(1);} fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET);
    nids=b/2; ids=xmalloc(b); if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f);
}
static void state_reset(void){
    memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0;
}
static inline void rmsnorm(const float* in,const float* w,float* out){
    float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i]; float r=1.0f/sqrtf(ms/(float)D+1e-5f);
    for(int i=0;i<D;i++) out[i]=in[i]*r*w[i];
}

// timing accumulators (NULL = off)
typedef struct { double scan,scan_other,swa,mlp,head; } Tacc;
static int8_t g_xq[MLP_HID]; static int8_t g_lut[(MLP_HID/2)*16]; static int32_t g_S[MLP_HID];

// gated dReLU MLP via LUT (mlp_mode=1) or fp32 (mlp_mode=0). Writes into x (residual add done by caller? no, here).
static void mlp_forward(int l,const float* xn,float* out,int mlp_mode){
    float gh[MLP_HID],uh[MLP_HID];
    if(mlp_mode==0){
        matvec(mlp_gate_f[l],xn,gh,MLP_HID,D); matvec(mlp_up_f[l],xn,uh,MLP_HID,D);
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]);
        matvec(mlp_down_f[l],gh,out,D,MLP_HID);
    } else {
        float sa=quant_i8(xn,D,g_xq); build_lut_t3(g_xq,D,g_lut);
        matvec_lut_t3(lg[l].codes,g_lut,g_S,MLP_HID,lg[l].Mpad,D);
        for(int i=0;i<MLP_HID;i++) gh[i]=(float)g_S[i]*sa*lg[l].ws[i];
        matvec_lut_t3(lu[l].codes,g_lut,g_S,MLP_HID,lu[l].Mpad,D);   // gate & up share xn -> same LUT
        for(int i=0;i<MLP_HID;i++) uh[i]=(float)g_S[i]*sa*lu[l].ws[i];
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]);
        float sh=quant_i8(gh,MLP_HID,g_xq); build_lut_t3(g_xq,MLP_HID,g_lut);
        matvec_lut_t3(ld[l].codes,g_lut,g_S,D,ld[l].Mpad,MLP_HID);
        for(int i=0;i<D;i++) out[i]=(float)g_S[i]*sh*ld[l].ws[i];
    }
}
static void forward_token(uint32_t tok,float* logits,int mlp_mode,Tacc* T){
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
            for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float mx=-1e30f;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>mx)mx=sc; }
                float Z=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-mx); Z+=att[j]; } float zi=1.0f/Z;
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
            for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; }
                y[c]=acc+s->Dskip[c]*xc; }
            if(T){T->scan+=now_s()-t0; t0=now_s();}
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->scan_other+=now_s()-t0;
        }
        rmsnorm(x, mlp_n2[l], xn);
        if(T)t0=now_s();
        mlp_forward(l,xn,tmp,mlp_mode); for(int i=0;i<D;i++) x[i]+=tmp[i];
        if(T)T->mlp+=now_s()-t0;
    }
    if(T)t0=now_s();
    rmsnorm(x,normf,xn); matvec(head,xn,logits,V,D);
    if(T)T->head+=now_s()-t0;
}

// ---------------- G1(E2) kernel bit-exactness ----------------
static int gate_selftest(void){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],tmp[D],gh[MLP_HID];
    int8_t xq[MLP_HID],lut[(MLP_HID/2)*16]; int32_t Slut[MLP_HID],Sref[MLP_HID];
    long ntok=200; int worst=0; long checks=0;
    state_reset();
    for(long ti=0;ti<ntok;ti++){ uint32_t tok=val[ti];
        memcpy(x,emb+(size_t)tok*D,D*4);
        for(int l=0;l<L;l++){
            rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);
            if(is_swa[l]){ /* replicate SWA to keep state correct */
                float q[D],kk[D],vvv[D],att[WIN],ao[D];
                matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vvv,xz+2*D,D*4);
                int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vvv,D*4);
                kvpos++; if(kvcnt<WIN) kvcnt++; memset(ao,0,D*4);
                for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float mx=-1e30f;
                    for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>mx)mx=sc; }
                    float Z=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-mx); Z+=att[j]; } float zi=1.0f/Z;
                    for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                        for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
                matvec(swa.o,ao,tmp,D,D); for(int i=0;i<D;i++) x[i]+=tmp[i];
            } else { SSML*s=&ssm[l];
                matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
                float (*cb)[CONV]=convbuf[l];
                for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                    float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
                matvec(s->x_proj,xx,dbl,DTR+2*N,DN); const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
                for(int c=0;c<DN;c++) dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]);
                float (*hl)[N]=hstate[l];
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                    for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; } y[c]=acc+s->Dskip[c]*xc; }
                for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
                matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            }
            rmsnorm(x, mlp_n2[l], xn);
            // --- the actual self-test: LUT S vs scalar-int S for gate, up (input xn), then down (input h) ---
            float sa=quant_i8(xn,D,xq); build_lut_t3(xq,D,lut);
            matvec_lut_t3(lg[l].codes,lut,Slut,MLP_HID,lg[l].Mpad,D); ref_t3(lg[l].wt,xq,Sref,MLP_HID,D);
            for(int i=0;i<MLP_HID;i++){ int d=abs(Slut[i]-Sref[i]); if(d>worst)worst=d; checks++; }
            float gg[MLP_HID],uu[MLP_HID];
            for(int i=0;i<MLP_HID;i++) gg[i]=(float)Slut[i]*sa*lg[l].ws[i];
            matvec_lut_t3(lu[l].codes,lut,Slut,MLP_HID,lu[l].Mpad,D); ref_t3(lu[l].wt,xq,Sref,MLP_HID,D);
            for(int i=0;i<MLP_HID;i++){ int d=abs(Slut[i]-Sref[i]); if(d>worst)worst=d; checks++; }
            for(int i=0;i<MLP_HID;i++) uu[i]=(float)Slut[i]*sa*lu[l].ws[i];
            for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gg[i])*reluf(uu[i]);
            float sh=quant_i8(gh,MLP_HID,xq); build_lut_t3(xq,MLP_HID,lut);
            matvec_lut_t3(ld[l].codes,lut,Slut,D,ld[l].Mpad,MLP_HID); ref_t3(ld[l].wt,xq,Sref,D,MLP_HID);
            for(int i=0;i<D;i++){ int d=abs(Slut[i]-Sref[i]); if(d>worst)worst=d; checks++; }
            for(int i=0;i<D;i++) tmp[i]=(float)Slut[i]*sh*ld[l].ws[i];
            for(int i=0;i<D;i++) x[i]+=tmp[i];
        }
    }
    printf("==== G1(E2) kernel bit-exactness (LUT integer S vs scalar-int ref) ====\n");
    printf("  %ld dot-products checked over %ld tokens x %d layers x 3 matvecs | worst |S_lut - S_ref| = %d\n",checks,ntok,L,worst);
    printf("  G1(E2) %s (0 = bit-exact, probe-1 discipline)\n",worst==0?"PASS":"FAIL");
    return worst==0?0:2;
}

// ---------------- G2 BPB (LUT) vs E1 fp32, in the same binary ----------------
static double run_bpb(int mlp_mode,long seqW,long eval_tok,long* ntok_o){
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    long lim=eval_tok<(nval-1)?eval_tok:(nval-1); double bits=0; long nbytes=0,ntok=0,pos=0;
    float* logits=xmalloc((size_t)V*4); const double LN2=0.6931471805599453;
    while(pos+seqW+1<=lim){ state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],logits,mlp_mode,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; for(int o=0;o<V;o++) if(logits[o]>mx) mx=logits[o];
            double se=0; for(int o=0;o<V;o++) se+=exp((double)logits[o]-mx);
            bits += (-((double)logits[tgt]-mx)+log(se))/LN2; nbytes+=id2len[tgt]; ntok++; }
        pos+=seqW; }
    free(logits); if(ntok_o)*ntok_o=ntok; return bits/(nbytes>0?nbytes:1);
}
static int gate_bpb(long seqW,long eval_tok){
    long nt; double e1=run_bpb(0,seqW,eval_tok,&nt); double e2=run_bpb(1,seqW,eval_tok,&nt);
    printf("==== G2 BPB parity (LUT vs E1 fp32, seq=%ld eval-tok=%ld, %ld tok) ====\n",seqW,eval_tok,nt);
    printf("  E1 fp32 MLP BPB=%.6f | E2 LUT MLP BPB=%.6f | delta=%+.6f  G2 %s (<=+0.005)\n",
           e1,e2,e2-e1,(e2-e1)<=0.005?"PASS":"FAIL");
    return (e2-e1)<=0.005?0:2;
}

// ---------------- G3 top-1 agreement LUT vs E1 fp32 ----------------
// dumpflips: optional diagnostic (P0.2 freshness check) — writes one line per disagreeing position:
//   <global_val_index> <fp32_argmax> <lut_argmax>. Does not alter the gate.
static int gate_logits(long seqW,long ntok_target,long offset,const char* dumpflips){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* l1=xmalloc((size_t)V*4); float* l2=xmalloc((size_t)V*4);
    FILE* df=dumpflips?fopen(dumpflips,"w"):NULL;
    if(dumpflips&&!df){fprintf(stderr,"cannot open %s\n",dumpflips);exit(1);}
    long agree=0,tot=0,pos=offset;
    while(tot<ntok_target){
        state_reset();
        // run both paths on the same window; states are independent (separate reset per path) -> re-run per path
        // simpler: run fp32 for the window capturing argmax, then LUT for the window.
        static uint16_t a1[4096];
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l1,0,NULL); float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(l1[o]>mx){mx=l1[o];am=o;} a1[t]=am; }
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l2,1,NULL); float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(l2[o]>mx){mx=l2[o];am=o;}
            if(am==a1[t]) agree++; else if(df) fprintf(df,"%ld %d %d\n",pos+t,(int)a1[t],am);
            tot++; }
        pos+=seqW;
    }
    if(df) fclose(df);
    double pct=100.0*agree/tot;
    printf("==== G3 top-1 agreement (LUT vs E1 fp32, %ld tok, offset %ld) ====\n",tot,offset);
    printf("  agreement=%.4f%% (%ld/%ld)  G3 %s (>=99%%)\n",pct,agree,tot,pct>=99.0?"PASS":"FAIL");
    free(l1); free(l2); return pct>=99.0?0:2;
}

// --dumplogits: raw fp32 logit stream (P4.3 consolidation-parity instrument); mlp_mode 1 = the E2 LUT config.
static void dump_logits(const char* path,long seqW,long ntok,int mlp_mode){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    FILE* f=fopen(path,"wb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    float* lg=xmalloc((size_t)V*4); long done=0,pos=0;
    while(done<ntok){ state_reset();
        for(long t=0;t<seqW&&done<ntok;t++,done++){ forward_token(val[pos+t],lg,mlp_mode,NULL); fwrite(lg,4,V,f); }
        pos+=seqW; }
    fclose(f); free(lg); printf("dumped %ld x %d fp32 logits (mlp_mode=%d) -> %s\n",ntok,V,mlp_mode,path);
}

// ---------------- G4 timing breakdown (Amdahl) ----------------
static void timing(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* logits=xmalloc((size_t)V*4);
    for(int mode=0;mode<2;mode++){ Tacc T={0,0,0,0,0};
        state_reset(); double t0=now_s();
        for(long i=0;i<ntok;i++) forward_token(val[i%100000],logits,mode,&T);
        double tot=now_s()-t0;
        double us=1e6/(ntok/tot);
        printf("==== timing (%s MLP, %ld tok) : %.1f tok/s | %.1f us/tok ====\n",mode?"LUT":"fp32",ntok,ntok/tot,us);
        double sc=1e6/ntok;
        printf("   ssm-scan(exp) %.1f us  ssm-other %.1f us  SWA %.1f us  MLP %.1f us  head %.1f us  (per tok)\n",
               T.scan*sc,T.scan_other*sc,T.swa*sc,T.mlp*sc,T.head*sc);
        printf("   MLP share=%.1f%%  scan share=%.1f%%\n",100.0*T.mlp/tot,100.0*(T.scan+T.scan_other)/tot);
    }
    free(logits);
}

int main(int argc,char**argv){
    int st=0,gb=0,gl=0,tm=0,all=0; long seqW=512,eval_tok=200000,ntok=10240,timetok=3000,offset=0;
    const char* wp="results/phase60/e1_model.bin"; const char* dumpflips=NULL; const char* dlpath=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--selftest")) st=1; else if(!strcmp(argv[i],"--bpb")) gb=1;
        else if(!strcmp(argv[i],"--logits")) gl=1; else if(!strcmp(argv[i],"--timing")) tm=1;
        else if(!strcmp(argv[i],"--all")) all=1;
        else if(!strcmp(argv[i],"--seq")&&i+1<argc) seqW=atol(argv[++i]);
        else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--ntok")&&i+1<argc) ntok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--time-tok")&&i+1<argc) timetok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--offset")&&i+1<argc) offset=atol(argv[++i]);
        else if(!strcmp(argv[i],"--dumpflips")&&i+1<argc) dumpflips=argv[++i];
        else if(!strcmp(argv[i],"--kselftest")) return kernel_selftest();   // synthetic, no weights (CI)
        else if(!strcmp(argv[i],"--dumplogits")&&i+1<argc) dlpath=argv[++i];
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i];
    }
    if(!st&&!gb&&!gl&&!tm&&!all&&!dlpath) all=1;
    load_weights(wp); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    fprintf(stderr,"E2 loaded: ids=%ld\n",nids);
    if(dlpath){ dump_logits(dlpath,seqW,ntok,1); return 0; }               // E2 config = LUT MLP
    int rc=0;
    if(all||st) rc|=gate_selftest();
    if(all||gl) rc|=gate_logits(seqW,ntok,offset,dumpflips);
    if(all||gb) rc|=gate_bpb(seqW,eval_tok);
    if(all||tm) timing(timetok);
    printf("STOP. E2 gates above. No commit.\n");
    return rc;
}
