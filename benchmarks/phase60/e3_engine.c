// Phase 60 / E3 - activation-sparsity SKIP (predictor-free) on the E2 LUT path. ONE VARIABLE vs E2: the skip
//   mechanism. Gate-first: compute gate (full LUT, it IS the predictor), then
//     - UP row-skip: only compute up[i] for units with gate[i] > 0 (gate<=0 -> relu(gate)=0 -> h[i]=0 anyway).
//       up codes are stored ROW-MAJOR (each hidden row's tiles contiguous) so an active-row dot is a contiguous
//       read and skipped rows are simply not visited (rho-safe, no strided gather).
//     - DOWN tile-skip: only accumulate down over hidden tiles where hq != 0 (zero activation contributes exactly 0).
//       down codes stay TILE-MAJOR (codes[tile*Mpad+row]) so a tile = a contiguous Mpad block -> tile-skip is a
//       contiguous-chunk skip, not a strided gather (rho-safe; the Architect's point #4, already satisfied by layout).
//
//   EXACT: skipped up-rows land on h[i]=0 (they would be zero regardless); skipped down-tiles add integer 0. The
//   integer dot S is associative, so the result is BIT-IDENTICAL to E2 (stronger than "numerically identical").
//   Reads results/phase60/e1_model.bin. Compares the E2 full-LUT path (mode 1) and the E3 skip path (mode 2) in-binary.
//
//   Modes: --bitid (logit bit-identity E3 vs E2) | --sparsity (per-layer gate%/h%) | --timing (MLP time E2 vs E3) | --all
// Build: clang -O3 -mavx2 -mfma -march=znver2 benchmarks/phase60/e3_engine.c -o bin/e3_engine.exe -lm
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
#define TUP (D/2)          // up/gate tiles (K=D)
#define TDN (MLP_HID/2)    // down tiles (K=hid)

static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline void matvec(const float*W,const float*x,float*y,int out,int in){ for(int o=0;o<out;o++) y[o]=dotf(W+(size_t)o*in,x,in); }
static inline float silu(float x){ return x/(1.0f+expf(-x)); }
static inline float softplus(float x){ return x>20.0f?x:log1pf(expf(x)); }
static inline float reluf(float x){ return x>0.0f?x:0.0f; }

// ---- LUT kernel (probe-1) ----
static inline void acc_add_i8x32(__m256i* acc, __m256i p){
    __m128i lo=_mm256_castsi256_si128(p), hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo));
    acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi));
    acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_full(const int8_t* codes, const int8_t* lut, int32_t* y, int M, int Mpad, int T){
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
// down tile-skip: iterate only active tiles (contiguous Mpad blocks each), rho-safe
static void matvec_lut_tileskip(const int8_t* codes, const int8_t* lut, int32_t* y, int M, int Mpad, const int* act, int na){
    for(int base=0; base<M; base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int a=0;a<na;a++){ int t=act[a];
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
static void build_lut_t3(const int8_t* xq, int T, int8_t* lut){
    for(int t=0;t<T;t++){ int8_t x0=xq[2*t],x1=xq[2*t+1];
        for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1, w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; } }
}
static void build_codes_tilemajor(const int8_t* Wt, int M, int K, int Mpad, int8_t* codes){
    int T=K/2;
    for(int t=0;t<T;t++){ for(int m=0;m<M;m++){ int w0=Wt[(size_t)m*K+2*t], w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)t*Mpad+m]=(int8_t)((w0+1)*3+(w1+1)); }
        for(int m=M;m<Mpad;m++) codes[(size_t)t*Mpad+m]=0; }
}
static void build_codes_rowmajor(const int8_t* Wt, int M, int K, int8_t* codes){   // codes[m*T + t]
    int T=K/2;
    for(int m=0;m<M;m++) for(int t=0;t<T;t++){ int w0=Wt[(size_t)m*K+2*t], w1=Wt[(size_t)m*K+2*t+1]; codes[(size_t)m*T+t]=(int8_t)((w0+1)*3+(w1+1)); }
}
static float quant_i8(const float* x, int n, int8_t* xq){
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    if(amax==0.0f){ memset(xq,0,n); return 0.0f; }
    float scale=amax/(float)AQ, inv=1.0f/scale;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; } return scale;
}

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L];
static int8_t *gate_tm[L], *up_tm[L], *up_rm[L], *down_tm[L];   // gate/up tile-major (E2), up row-major (E3), down tile-major
static float *gate_sc[L], *up_sc[L], *down_sc[L];
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short\n");exit(1);} return p; }
static int8_t* rdi8(FILE*f,size_t n){ int8_t*p=xmalloc(n); if(fread(p,1,n,f)!=n){fprintf(stderr,"short i8\n");exit(1);} return p; }

// sparsity counters (accumulated during skip runs)
static double acc_gate_active[L], acc_h_active[L], acc_downtiles[L]; static long acc_ntok=0;

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
        mlp_n2[l]=rd(f,D); rd(f,(size_t)MLP_HID*D); rd(f,(size_t)MLP_HID*D); rd(f,(size_t)D*MLP_HID);  // skip fp32 dequant
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    if(!has_packed){fprintf(stderr,"need packed ternary\n");exit(1);}
    for(int l=0;l<L;l++){
        int8_t* gq=rdi8(f,(size_t)MLP_HID*D); gate_sc[l]=rd(f,MLP_HID);
        int8_t* uq=rdi8(f,(size_t)MLP_HID*D); up_sc[l]=rd(f,MLP_HID);
        int8_t* dq=rdi8(f,(size_t)D*MLP_HID); down_sc[l]=rd(f,D);
        int Mpg=(MLP_HID+31)&~31, Mpd=(D+31)&~31;
        gate_tm[l]=xmalloc((size_t)TUP*Mpg); build_codes_tilemajor(gq,MLP_HID,D,Mpg,gate_tm[l]);
        up_tm[l]  =xmalloc((size_t)TUP*Mpg); build_codes_tilemajor(uq,MLP_HID,D,Mpg,up_tm[l]);
        up_rm[l]  =xmalloc((size_t)MLP_HID*TUP); build_codes_rowmajor(uq,MLP_HID,D,up_rm[l]);
        down_tm[l]=xmalloc((size_t)TDN*Mpd); build_codes_tilemajor(dq,D,MLP_HID,Mpd,down_tm[l]);
        free(gq); free(uq); free(dq);
    }
    long pos=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f); fclose(f);
    (void)pos;(void)end;
    fprintf(stderr,"E3 weights ok (gate/up tile+up row-major + down tile; act[-%d,%d])\n",AQ,AQ);
}
static void load_meta(const char* path){ FILE*f=fopen(path,"rb"); if(!f){exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1); if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; } fclose(f); }
static void load_ids(const char* path){ FILE*f=fopen(path,"rb"); if(!f){exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); nids=b/2; ids=xmalloc(b); if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f); }
static void state_reset(void){ memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0; }
static inline void rmsnorm(const float* in,const float* w,float* out){
    float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i]; float r=1.0f/sqrtf(ms/(float)D+1e-5f);
    for(int i=0;i<D;i++) out[i]=in[i]*r*w[i]; }

static int8_t xqb[MLP_HID]; static int8_t lutb[TDN*16]; static int32_t Sb[MLP_HID]; static int act_tiles[TDN];

// mlp_mode: 1 = E2 full LUT, 2 = E3 skip
static void mlp_forward(int l,const float* xn,float* out,int mlp_mode,int count){
    int Mpg=(MLP_HID+31)&~31, Mpd=(D+31)&~31;
    float gh[MLP_HID],uh[MLP_HID];
    float sa=quant_i8(xn,D,xqb); build_lut_t3(xqb,TUP,lutb);
    matvec_lut_full(gate_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);            // gate: always full (the predictor)
    for(int i=0;i<MLP_HID;i++) gh[i]=(float)Sb[i]*sa*gate_sc[l][i];
    if(mlp_mode==1){
        matvec_lut_full(up_tm[l],lutb,Sb,MLP_HID,Mpg,TUP);
        for(int i=0;i<MLP_HID;i++) uh[i]=(float)Sb[i]*sa*up_sc[l][i];
        for(int i=0;i<MLP_HID;i++) gh[i]=reluf(gh[i])*reluf(uh[i]);
    } else {
        int ga=0;
        for(int i=0;i<MLP_HID;i++){
            if(gh[i]>0.0f){ const int8_t* cr=up_rm[l]+(size_t)i*TUP; int S=0;
                for(int t=0;t<TUP;t++) S+=lutb[t*16+cr[t]];
                float u=(float)S*sa*up_sc[l][i]; gh[i]=gh[i]*reluf(u); ga++;
            } else gh[i]=0.0f;                                       // gate<=0 -> h=0 (exact; up skipped)
        }
        if(count) acc_gate_active[l]+=ga;
    }
    // down: quant h, then full (E2) or tile-skip (E3)
    float sh=quant_i8(gh,MLP_HID,xqb); build_lut_t3(xqb,TDN,lutb);
    if(mlp_mode==1){
        matvec_lut_full(down_tm[l],lutb,Sb,D,Mpd,TDN);
    } else {
        int na=0,hact=0;
        for(int t=0;t<TDN;t++){ int nz=(xqb[2*t]!=0)||(xqb[2*t+1]!=0); if(nz) act_tiles[na++]=t; }
        for(int i=0;i<MLP_HID;i++) if(xqb[i]!=0) hact++;
        matvec_lut_tileskip(down_tm[l],lutb,Sb,D,Mpd,act_tiles,na);
        if(count){ acc_h_active[l]+=hact; acc_downtiles[l]+=na; }
    }
    for(int i=0;i<D;i++) out[i]=(float)Sb[i]*sh*down_sc[l][i];
}

typedef struct { double scan,scan_other,swa,mlp,head; } Tacc;
static void forward_token(uint32_t tok,float* logits,int mlp_mode,int count,Tacc* T){
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
                for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; } y[c]=acc+s->Dskip[c]*xc; }
            if(T){T->scan+=now_s()-t0; t0=now_s();}
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
            if(T)T->scan_other+=now_s()-t0;
        }
        rmsnorm(x, mlp_n2[l], xn);
        if(T)t0=now_s();
        mlp_forward(l,xn,tmp,mlp_mode,count); for(int i=0;i<D;i++) x[i]+=tmp[i];
        if(T)T->mlp+=now_s()-t0;
    }
    if(T)t0=now_s();
    rmsnorm(x,normf,xn); matvec(head,xn,logits,V,D);
    if(T)T->head+=now_s()-t0;
}

// ---- G1(E3) logit bit-identity E3 vs E2 ----
static int gate_bitid(long seqW,long ntok_target){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr;
    float* l2=xmalloc((size_t)V*4); float* l3=xmalloc((size_t)V*4); float* keep=xmalloc((size_t)seqW*V*4);
    long tot=0,mism=0; double worst=0; long pos=0;
    while(tot<ntok_target){
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l2,1,0,NULL); memcpy(keep+(size_t)t*V,l2,(size_t)V*4); }
        state_reset();
        for(long t=0;t<seqW;t++){ forward_token(val[pos+t],l3,2,0,NULL);
            const float* e2=keep+(size_t)t*V;
            for(int o=0;o<V;o++){ double d=fabs((double)l3[o]-e2[o]); if(d>worst)worst=d; if(l3[o]!=e2[o]) mism++; }
            tot++; }
        pos+=seqW;
    }
    printf("==== G1(E3) logit bit-identity (E3 skip vs E2 full-LUT, %ld tok) ====\n",tot);
    printf("  mismatched logit values=%ld / %ld  worst |diff|=%.3e  G1(E3) %s (skip is exact -> must be 0)\n",
           mism,tot*(long)V,worst,mism==0?"PASS":"FAIL");
    free(l2);free(l3);free(keep); return mism==0?0:2;
}

// ---- sparsity readout ----
static void sparsity(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    for(int l=0;l<L;l++){acc_gate_active[l]=acc_h_active[l]=acc_downtiles[l]=0;} acc_ntok=0;
    state_reset();
    for(long i=0;i<ntok;i++){ forward_token(val[i],lg,2,1,NULL); acc_ntok++; }
    printf("==== E3 activation sparsity per layer (%ld tok) | anchors probe-2: gate 79%%, h 92%% ====\n",ntok);
    printf("  layer  gate_sparsity%%  h_sparsity(hq==0)%%  down_tiles_used%%(of %d)\n",TDN);
    double sg=0,sh=0,std=0;
    for(int l=0;l<L;l++){ double ga=acc_gate_active[l]/acc_ntok, ha=acc_h_active[l]/acc_ntok, dt=acc_downtiles[l]/acc_ntok;
        double gs=100.0*(1.0-ga/MLP_HID), hs=100.0*(1.0-ha/MLP_HID), du=100.0*dt/TDN;
        printf("   %d      %6.1f          %6.1f              %6.1f\n",l,gs,hs,du); sg+=gs; sh+=hs; std+=du; }
    printf("  mean   gate %.1f%%  h %.1f%%  down-tiles-used %.1f%% (=> down tile-skip %.1f%%)\n",
           sg/L,sh/L,std/L,100.0-std/L);
    free(lg);
}

// ---- timing E2 full vs E3 skip ----
static void timing(long ntok){
    long ntr=(long)(nids*0.9); uint16_t* val=ids+ntr; float* lg=xmalloc((size_t)V*4);
    for(int mode=1;mode<=2;mode++){ Tacc T={0,0,0,0,0}; state_reset(); double t0=now_s();
        for(long i=0;i<ntok;i++) forward_token(val[i%100000],lg,mode,0,&T);
        double tot=now_s()-t0; double sc=1e6/ntok;
        printf("==== timing (%s, %ld tok): %.1f tok/s | %.1f us/tok | MLP %.1f us (%.1f%%) ====\n",
               mode==1?"E2 full-LUT":"E3 skip",ntok,ntok/tot,1e6/(ntok/tot),T.mlp*sc,100.0*T.mlp/tot);
    }
    free(lg);
}

int main(int argc,char**argv){
    int bi=0,sp=0,tm=0,all=0; long seqW=512,ntok=5120,sptok=4000,timetok=3000;
    const char* wp="results/phase60/e1_model.bin";
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--bitid"))bi=1; else if(!strcmp(argv[i],"--sparsity"))sp=1;
        else if(!strcmp(argv[i],"--timing"))tm=1; else if(!strcmp(argv[i],"--all"))all=1;
        else if(!strcmp(argv[i],"--ntok")&&i+1<argc) ntok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i]; }
    if(!bi&&!sp&&!tm&&!all) all=1;
    load_weights(wp); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    fprintf(stderr,"E3 loaded: ids=%ld\n",nids);
    int rc=0;
    if(all||bi) rc|=gate_bitid(seqW,ntok);
    if(all||sp) sparsity(sptok);
    if(all||tm) timing(timetok);
    printf("STOP. E3 gates above. No commit.\n");
    return rc;
}
