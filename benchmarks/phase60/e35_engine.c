// Phase 60 / E3.5 - fast-exp scan path (deterministic, versioned, fixed-coefficient AVX2 poly-exp).
//   The exact-exp selective scan is 83-91% of per-token time (E2 G4). Replace libm expf in the scan recurrence
//   with exp256_ps (Cephes-style fixed-coefficient poly, 8-wide), which also lets the inner N=96 loop vectorize.
//   LAWFUL under Phase 35: that rule bans NON-DETERMINISTIC compiler reordering (-ffast-math), not an explicit,
//   versioned, deterministic approximation with pre-registered parity gates. ONE VARIABLE = the scan-exp path.
//
//   Domain-aware: the scan arg is dt*A with dt=softplus(.)>0 and A=-exp(A_log)<0 -> ALWAYS <= 0, exp in (0,1].
//   The approximation error is characterized on THAT observed domain (dumped from the run), not a generic range.
//
//   To isolate the variable cleanly, the GATES compare (fast-exp + fp32 MLP) vs E1 (exact-exp + fp32 MLP) -- the
//   MLP path is held at the exact fp32 dequant so the only change is the scan exp. The FULL-STACK profile (LUT+skip
//   MLP + fast-exp = the real optimized engine) is reported separately for reading E4/E5.
//
//   Modes: --range | --golden | --bpb | --logits | --timing | --all       Build like the others.
// clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/e35_engine.c -o bin/e35_engine.exe -lm
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
#define MLP_HID 1024
#define AQ 63
#define TUP (D/2)
#define TDN (MLP_HID/2)
#define NLAYER (L+2)

// ---- deterministic fixed-coefficient poly-exp (Cephes, from phase55) ----
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
static inline float exp_approx1(float x){ float o[8]; _mm256_storeu_ps(o,exp256_ps(_mm256_set1_ps(x))); return o[0]; }

static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

// ---- LUT kernel + skip (from E3) ----
static inline void acc_add_i8x32(__m256i* acc, __m256i p){
    __m128i lo=_mm256_castsi256_si128(p), hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo));
    acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi));
    acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_full(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    for(int base=0;base<M;base+=32){ __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static void matvec_lut_tileskip(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,const int* act,int na){
    for(int base=0;base<M;base+=32){ __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int a=0;a<na;a++){ int t=act[a]; __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static void build_lut_t3(const int8_t* xq,int T,int8_t* lut){ for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
    for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1,w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } } }
static void bc_tm(const int8_t* Wt,int M,int K,int Mpad,int8_t* codes){ int T=K/2;
    for(int t=0;t<T;t++){ for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0; } }
static void bc_rm(const int8_t* Wt,int M,int K,int8_t* codes){ int T=K/2;
    for(int m=0;m<M;m++) for(int t=0;t<T;t++){ int w0=Wt[(size_t)m*K+2*t],w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)m*T+t]=(int8_t)((w0+1)*3+(w1+1)); } }
static float quant_i8(const float* x,int n,int8_t* xq){ float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; } float scale=amax/(float)AQ,inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; } return scale; }

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L],*gate_f[L],*up_f[L],*down_f[L];                 // fp32 dequant MLP (E1 path)
static int8_t *gate_tm[L],*up_rm[L],*down_tm[L]; static float *gate_sc[L],*up_sc[L],*down_sc[L];   // LUT (E3 path)
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short\n");exit(1);} return p; }
static int8_t* rdi8(FILE*f,size_t n){ int8_t*p=xmalloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short i8\n");exit(1);} return p; }

// range-of-exp-arg accumulator
static double g_argmin=1e30,g_argmax=-1e30; static int g_collect=0;

static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){exit(1);}
    uint32_t h[16]; if(fread(h,4,16,f)!=16||h[0]!=0x45314D31){fprintf(stderr,"bad magic\n");exit(1);}
    int has_packed=h[14];
    emb=rd(f,(size_t)V*D);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);
            s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        mlp_n2[l]=rd(f,D); gate_f[l]=rd(f,(size_t)MLP_HID*D); up_f[l]=rd(f,(size_t)MLP_HID*D); down_f[l]=rd(f,(size_t)D*MLP_HID);
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    if(!has_packed){fprintf(stderr,"need packed\n");exit(1);}
    for(int l=0;l<L;l++){ int8_t* gq=rdi8(f,(size_t)MLP_HID*D); gate_sc[l]=rd(f,MLP_HID);
        int8_t* uq=rdi8(f,(size_t)MLP_HID*D); up_sc[l]=rd(f,MLP_HID);
        int8_t* dq=rdi8(f,(size_t)D*MLP_HID); down_sc[l]=rd(f,D);
        int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
        gate_tm[l]=xmalloc((size_t)TUP*Mpg); bc_tm(gq,MLP_HID,D,Mpg,gate_tm[l]);
        up_rm[l]=xmalloc((size_t)MLP_HID*TUP); bc_rm(uq,MLP_HID,D,up_rm[l]);
        down_tm[l]=xmalloc((size_t)TDN*Mpd); bc_tm(dq,D,MLP_HID,Mpd,down_tm[l]);
        free(gq);free(uq);free(dq); }
    fclose(f); fprintf(stderr,"E3.5 weights ok\n");
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

static int8_t xqb[MLP_HID]; static int8_t lutb[TDN*16]; static int32_t Sb[MLP_HID]; static int act_tiles[TDN];

// mlp_mode: 0 = fp32 dequant (E1), 2 = LUT+skip (E3)
static void mlp_forward(int l,const float* xn,float* out,int mlp_mode){
    float gh[MLP_HID],uh[MLP_HID];
    if(mlp_mode==0){ matvec(gate_f[l],xn,gh,MLP_HID,D); matvec(up_f[l],xn,uh,MLP_HID,D);
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]); matvec(down_f[l],gh,out,D,MLP_HID); return; }
    int Mpg=(MLP_HID+31)&~31,Mpd=(D+31)&~31;
    float sa=quant_i8(xn,D,xqb); build_lut_t3(xqb,TUP,lutb);
    matvec_lut_full(gate_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);
    for(int i=0;i<MLP_HID;i++) gh[i]=(float)Sb[i]*sa*gate_sc[l][i];
    for(int i=0;i<MLP_HID;i++){ if(gh[i]>0.0f){ const int8_t* cr=up_rm[l]+(size_t)i*TUP; int S=0;
            for(int t=0;t<TUP;t++) S+=lutb[t*16+cr[t]]; float u=(float)S*sa*up_sc[l][i]; gh[i]=gh[i]*reluf(u); } else gh[i]=0.0f; }
    float sh=quant_i8(gh,MLP_HID,xqb); build_lut_t3(xqb,TDN,lutb);
    int na=0; for(int t=0;t<TDN;t++){ if(xqb[2*t]||xqb[2*t+1]) act_tiles[na++]=t; }
    matvec_lut_tileskip(down_tm[l],lutb,Sb,D,Mpd,act_tiles,na);
    for(int i=0;i<D;i++) out[i]=(float)Sb[i]*sh*down_sc[l][i];
}

typedef struct { double scan,scan_other,swa,mlp,head; } Tacc;
// scan_mode: 0 exact expf, 1 fast exp256_ps (vectorized inner loop)
static void forward_token(uint32_t tok,float* logits,int scan_mode,int mlp_mode,int cap_layers,float* caps,Tacc* T){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vvv[D],att[WIN],ao[D],tmp[D];
    double t0;
    memcpy(x,emb+(size_t)tok*D,D*4);
    if(caps) memcpy(caps,x,D*4);
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
            if(scan_mode==0){
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                    for(int j=0;j<N;j++){ float arg=dtc*Ac[j];
                        if(g_collect){ if(arg<g_argmin)g_argmin=arg; if(arg>g_argmax)g_argmax=arg; }
                        hc[j]=expf(arg)*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; }
                    y[c]=acc+s->Dskip[c]*xc; }
            } else {
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],dbx=dt[c]*xc;
                    __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx),vacc=_mm256_setzero_ps();
                    for(int j=0;j<N;j+=8){
                        __m256 e=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                        __m256 hcj=_mm256_fmadd_ps(e,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                        _mm256_storeu_ps(hc+j,hcj); vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+j),vacc); }
                    y[c]=hsum256(vacc)+s->Dskip[c]*xc; }
            }
            if(T){T->scan+=now_s()-t0; t0=now_s();}
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->scan_other+=now_s()-t0;
        }
        rmsnorm(x, mlp_n2[l], xn);
        if(T)t0=now_s();
        mlp_forward(l,xn,tmp,mlp_mode); for(int i=0;i<D;i++) x[i]+=tmp[i];
        if(T)T->mlp+=now_s()-t0;
        if(caps) memcpy(caps+(size_t)(l+1)*D,x,D*4);
    }
    if(T)t0=now_s();
    rmsnorm(x,normf,xn); if(caps) memcpy(caps+(size_t)(L+1)*D,xn,D*4);
    matvec(head,xn,logits,V,D);
    if(T)T->head+=now_s()-t0;
    (void)cap_layers;
}

// ---- range-aware error characterization ----
static void gate_range(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    g_collect=1; g_argmin=1e30; g_argmax=-1e30; state_reset();
    for(long i=0;i<ntok;i++) forward_token(val[i],lg,0,0,0,NULL,NULL);   // exact scan, collect arg domain
    g_collect=0;
    // max rel-err of exp256 vs expf on the observed domain
    double worst=0,at=0; int steps=200000;
    for(int k=0;k<=steps;k++){ double xx=g_argmin+(g_argmax-g_argmin)*k/steps;
        double a=exp_approx1((float)xx), e=exp((double)xx); double r=fabs(a-e)/(e>1e-30?e:1e-30); if(r>worst){worst=r;at=xx;} }
    printf("==== E3.5 range-aware exp characterization (%ld tok) ====\n",ntok);
    printf("  observed scan-arg domain dt*A in [%.4f, %.4f]  (always <=0, exp in (0,1])\n",g_argmin,g_argmax);
    printf("  max rel-err exp256_ps vs libm expf on that domain = %.3e (at arg=%.4f)\n",worst,at);
    free(lg);
}

// ---- golden trace error profile (fast vs exact, fp32 MLP) ----
static void gate_golden(void){
    FILE*f=fopen("results/phase60/golden_trace.bin","rb"); if(!f){fprintf(stderr,"no golden_trace.bin\n");return;}
    uint32_t hd[4]; if(fread(hd,4,4,f)!=4){return;} int Tt=hd[1]; fclose(f);
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* cx=xmalloc((size_t)NLAYER*D*4); float* ce=xmalloc((size_t)NLAYER*D*4); float* lg=xmalloc((size_t)V*4);
    double sqerr[NLAYER],sqref[NLAYER],maxabs[NLAYER]; for(int i=0;i<NLAYER;i++) sqerr[i]=sqref[i]=maxabs[i]=0;
    state_reset(); float* keepE=xmalloc((size_t)Tt*NLAYER*D*4);
    for(int t=0;t<Tt;t++){ forward_token(val[t],lg,0,0,0,ce,NULL); memcpy(keepE+(size_t)t*NLAYER*D,ce,(size_t)NLAYER*D*4); }
    state_reset();
    for(int t=0;t<Tt;t++){ forward_token(val[t],lg,1,0,0,cx,NULL); const float* e=keepE+(size_t)t*NLAYER*D;
        for(int ly=0;ly<NLAYER;ly++) for(int d=0;d<D;d++){ double df=fabs((double)cx[ly*D+d]-e[ly*D+d]);
            if(df>maxabs[ly])maxabs[ly]=df; sqerr[ly]+=df*df; sqref[ly]+=(double)e[ly*D+d]*e[ly*D+d]; } }
    const char* nm[NLAYER]={"emb","blk0","blk1","blk2","blk3","blk4","blk5","norm_f"};
    printf("==== E3.5 golden-trace error profile (fast-exp vs exact, fp32 MLP, %d tok) ====\n",Tt);
    for(int i=0;i<NLAYER;i++) printf("  %-7s max_abs=%.3e  l2_rel=%.3e\n",nm[i],maxabs[i],sqrt(sqerr[i]/(sqref[i]+1e-30)));
    free(cx);free(ce);free(lg);free(keepE);
}

static double run_bpb(int scan_mode,int mlp_mode,long seqW,long eval_tok,long* nto){
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    long lim=eval_tok<(nval-1)?eval_tok:(nval-1); double bits=0; long nb=0,nt=0,pos=0;
    float* lg=xmalloc((size_t)V*4); const double LN2=0.6931471805599453;
    while(pos+seqW+1<=lim){ state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],lg,scan_mode,mlp_mode,0,NULL,NULL); int tgt=val[pos+t+1];
            double mx=-1e30; for(int o=0;o<V;o++) if(lg[o]>mx)mx=lg[o]; double se=0; for(int o=0;o<V;o++) se+=exp((double)lg[o]-mx);
            bits+=(-((double)lg[tgt]-mx)+log(se))/LN2; nb+=id2len[tgt]; nt++; } pos+=seqW; }
    free(lg); if(nto)*nto=nt; return bits/(nb>0?nb:1);
}
static int gate_bpb(long seqW,long eval_tok){
    long nt; double e1=run_bpb(0,0,seqW,eval_tok,&nt); double ef=run_bpb(1,0,seqW,eval_tok,&nt);
    printf("==== E3.5 BPB (fp32 MLP; exact vs fast exp, %ld tok) ====\n",nt);
    printf("  exact(E1)=%.6f  fast=%.6f  delta=%+.6f  %s (<=+0.001)\n",e1,ef,ef-e1,(ef-e1)<=0.001?"PASS":"FAIL");
    return (ef-e1)<=0.001?0:2;
}
static int gate_logits(long seqW,long ntok_t){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* le=xmalloc((size_t)V*4); float* lf=xmalloc((size_t)V*4);
    long agree=0,tot=0,pos=0; static uint16_t ae[4096];
    while(tot<ntok_t){ state_reset(); for(long t=0;t<seqW;t++){ forward_token(val[pos+t],le,0,0,0,NULL,NULL);
            float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(le[o]>mx){mx=le[o];am=o;} ae[t]=am; }
        state_reset(); for(long t=0;t<seqW;t++){ forward_token(val[pos+t],lf,1,0,0,NULL,NULL);
            float mx=-1e30f; int am=0; for(int o=0;o<V;o++) if(lf[o]>mx){mx=lf[o];am=o;} if(am==ae[t])agree++; tot++; } pos+=seqW; }
    double pct=100.0*agree/tot;
    printf("==== E3.5 top-1 (fast vs exact, fp32 MLP, %ld tok) ====\n",tot);
    printf("  agreement=%.4f%% (%ld/%ld)  %s (>=99.9%%)\n",pct,agree,tot,pct>=99.9?"PASS":"FAIL");
    free(le);free(lf); return pct>=99.9?0:2;
}
static void timing(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    struct { const char* nm; int sm,mm; } cfg[3]={{"E1 exact-exp + fp32 MLP",0,0},{"fast-exp + fp32 MLP",1,0},{"fast-exp + LUT+skip MLP (full engine)",1,2}};
    for(int c=0;c<3;c++){ Tacc T={0,0,0,0,0}; state_reset(); double t0=now_s();
        for(long i=0;i<ntok;i++) forward_token(val[i%100000],lg,cfg[c].sm,cfg[c].mm,0,NULL,&T);
        double tot=now_s()-t0; double sc=1e6/ntok;
        printf("==== %s: %.1f tok/s | %.1f us/tok ====\n",cfg[c].nm,ntok/tot,1e6/(ntok/tot));
        printf("   scan %.1f us (%.1f%%)  scan-other %.1f  SWA %.1f  MLP %.1f  head %.1f\n",
               T.scan*sc,100.0*T.scan/tot,T.scan_other*sc,T.swa*sc,T.mlp*sc,T.head*sc); }
    free(lg);
}

int main(int argc,char**argv){
    int rg=0,gd=0,gb=0,gl=0,tm=0,all=0; long seqW=512,eval_tok=100000,ntok=10240,timetok=3000,rangetok=2000;
    const char* wp="results/phase60/e1_model.bin";
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--range"))rg=1; else if(!strcmp(argv[i],"--golden"))gd=1;
        else if(!strcmp(argv[i],"--bpb"))gb=1; else if(!strcmp(argv[i],"--logits"))gl=1; else if(!strcmp(argv[i],"--timing"))tm=1;
        else if(!strcmp(argv[i],"--all"))all=1; else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--ntok")&&i+1<argc) ntok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i]; }
    if(!rg&&!gd&&!gb&&!gl&&!tm&&!all) all=1;
    load_weights(wp); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    fprintf(stderr,"E3.5 loaded: ids=%ld\n",nids);
    int rc=0;
    if(all||rg) gate_range(rangetok);
    if(all||gd) gate_golden();
    if(all||gl) rc|=gate_logits(seqW,ntok);
    if(all||gb) rc|=gate_bpb(seqW,eval_tok);
    if(all||tm) timing(timetok);
    printf("STOP. E3.5 gates above. No commit.\n");
    return rc;
}
