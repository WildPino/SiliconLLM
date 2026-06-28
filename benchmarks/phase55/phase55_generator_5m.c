// Phase 55 - 5M Arch-A C inference engine (quality bump). fp32.
//   D=256 N=96 H=8 L=6 (5 SSM + 1 SWA@5, win128, expand2, dt_rank16) + per-block MLP (hidden from header).
//   Loads archA_5m_weights.bin (magic ARCB), meta.bin, ids.u16. Same locked decode (rep1.2/win128/no-top-p),
//   worst-of-32 gate (word+byte), bench. Mirrors phase55_generator.c (the rep-penalty set()-dedup fix is carried).
//
// Build: gcc -O3 -march=native -mavx2 -mfma benchmarks/phase55/phase55_generator_5m.c -o bin/phase55_generator_5m.exe -lm
// Run:   bin/phase55_generator_5m.exe --bpb --exp fast --seq 512 --eval-tok 200000 --weights results/phase55/archA_5m_weights.bin
//        bin/phase55_generator_5m.exe --gen --seq 512 --gen-bytes 2000 --ksamples 32 --weights results/phase55/archA_5m_weights.bin
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <immintrin.h>

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
#define MLP_HID_MAX 1024

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

typedef struct { float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm; } SSML;
typedef struct { float *qkv,*o,*norm; } SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L], *mlp_fc1[L], *mlp_fc2[L]; static int g_use_mlp=1, g_mlp_hid=MLP_HID_MAX;
static float (*hstate)[DN][N]; static float (*convbuf)[DN][CONV]; static float *kring,*vring; static int kvpos,kvcnt;
static unsigned char* id2bytes[V]; static int id2len[V]; static uint16_t* ids; static long nids;

static void* xmalloc(size_t n){ void*p=malloc(n); if(!p){fprintf(stderr,"OOM %zu\n",n);exit(1);} return p; }
static float* rd(FILE*f,size_t n){ float*p=xmalloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read %zu\n",n);exit(1);} return p; }

static void load_weights(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t h[12]; if(fread(h,4,12,f)!=12){fprintf(stderr,"bad header\n");exit(1);}
    if(h[0]!=0x41524342){fprintf(stderr,"bad magic %08x (need ARCB/5M)\n",h[0]);exit(1);}
    if((int)h[2]!=D||(int)h[3]!=N||(int)h[5]!=L||(int)h[7]!=DTR){fprintf(stderr,"dim mismatch: D%u N%u L%u dtr%u vs compiled %d/%d/%d/%d\n",h[2],h[3],h[5],h[7],D,N,L,DTR);exit(1);}
    g_mlp_hid=h[11]; g_use_mlp=(g_mlp_hid>0); if(g_mlp_hid>MLP_HID_MAX){fprintf(stderr,"mlp_hid %d>max\n",g_mlp_hid);exit(1);}
    emb=rd(f,(size_t)V*D);
    for(int l=0;l<L;l++){ is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){ swa.norm=rd(f,D); swa.qkv=rd(f,(size_t)3*D*D); swa.o=rd(f,(size_t)D*D); }
        else { SSML*s=&ssm[l]; s->norm=rd(f,D);
            s->in_proj=rd(f,(size_t)2*DN*D); s->conv_w=rd(f,(size_t)DN*CONV); s->conv_b=rd(f,DN);
            s->x_proj=rd(f,(size_t)(DTR+2*N)*DN); s->dt_proj=rd(f,(size_t)DN*DTR); s->dt_b=rd(f,DN);
            s->A=rd(f,(size_t)DN*N); for(int i=0;i<DN*N;i++) s->A[i]=-expf(s->A[i]);
            s->Dskip=rd(f,DN); s->out_proj=rd(f,(size_t)D*DN); }
        if(g_use_mlp){ mlp_n2[l]=rd(f,D); mlp_fc1[l]=rd(f,(size_t)g_mlp_hid*D); mlp_fc2[l]=rd(f,(size_t)D*g_mlp_hid); }
    }
    normf=rd(f,D); head=rd(f,(size_t)V*D);
    long pos=ftell(f); fseek(f,0,SEEK_END); long end=ftell(f); fclose(f);
    if(pos!=end) fprintf(stderr,"WARN: %ld trailing bytes (layout mismatch?)\n",end-pos);
}
static void load_meta(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    uint32_t mg,vv,nt; if(fread(&mg,4,1,f)!=1||fread(&vv,4,1,f)!=1||fread(&nt,4,1,f)!=1){exit(1);}
    if(mg!=0x54444D50){fprintf(stderr,"bad meta magic\n");exit(1);}
    unsigned char el[V]; if(fread(el,1,vv,f)!=vv){exit(1);}
    for(uint32_t t=0;t<vv;t++){ id2len[t]=el[t]; id2bytes[t]=xmalloc(el[t]+1); if(fread(id2bytes[t],1,el[t],f)!=(size_t)el[t]){exit(1);} id2bytes[t][el[t]]=0; }
    fclose(f);
}
static void load_ids(const char* path){
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);exit(1);}
    fseek(f,0,SEEK_END); long b=ftell(f); fseek(f,0,SEEK_SET); nids=b/2; ids=xmalloc(b);
    if(fread(ids,2,nids,f)!=(size_t)nids){exit(1);} fclose(f);
}
static void state_reset(void){
    memset(hstate,0,(size_t)L*DN*N*4); memset(convbuf,0,(size_t)L*DN*CONV*4);
    memset(kring,0,(size_t)WIN*D*4); memset(vring,0,(size_t)WIN*D*4); kvpos=0; kvcnt=0;
}
static inline void rmsnorm(const float* in,const float* w,float* out){
    float ms=0; for(int i=0;i<D;i++) ms+=in[i]*in[i]; float r=1.0f/sqrtf(ms/D+1e-5f);
    for(int i=0;i<D;i++) out[i]=in[i]*r*w[i];
}
static void forward_token(uint32_t tok,int fast_exp,float* logits){
    float x[D],xn[D],xz[2*DN],xx[DN],z[DN],dbl[DTR+2*N],dt[DN],y[DN],q[D],kk[D],vv[D],att[WIN],ao[D],tmp[D],mh[MLP_HID_MAX];
    memcpy(x,emb+(size_t)tok*D,D*4);
    for(int l=0;l<L;l++){
        rmsnorm(x, is_swa[l]?swa.norm:ssm[l].norm, xn);
        if(is_swa[l]){
            matvec(swa.qkv,xn,xz,3*D,D); memcpy(q,xz,D*4); memcpy(kk,xz+D,D*4); memcpy(vv,xz+2*D,D*4);
            int slot=kvpos%WIN; memcpy(kring+(size_t)slot*D,kk,D*4); memcpy(vring+(size_t)slot*D,vv,D*4);
            kvpos++; if(kvcnt<WIN) kvcnt++;
            memset(ao,0,D*4);
            for(int hh=0;hh<H;hh++){ const float* qh=q+hh*HD; float mx=-1e30f;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float sc=dotf(qh,kring+(size_t)s*D+hh*HD,HD)/sqrtf((float)HD); att[j]=sc; if(sc>mx)mx=sc; }
                float Z=0; for(int j=0;j<kvcnt;j++){ att[j]=expf(att[j]-mx); Z+=att[j]; } float zi=1.0f/Z;
                for(int j=0;j<kvcnt;j++){ int s=(kvpos-kvcnt+j)%WIN; float w=att[j]*zi; const float* vh=vring+(size_t)s*D+hh*HD;
                    for(int d=0;d<HD;d++) ao[hh*HD+d]+=w*vh[d]; } }
            matvec(swa.o,ao,tmp,D,D); for(int i=0;i<D;i++) x[i]+=tmp[i];
        } else {
            SSML*s=&ssm[l];
            matvec(s->in_proj,xn,xz,2*DN,D); memcpy(xx,xz,DN*4); memcpy(z,xz+DN,DN*4);
            float (*cb)[CONV]=convbuf[l];
            for(int c=0;c<DN;c++){ for(int t=0;t<CONV-1;t++) cb[c][t]=cb[c][t+1]; cb[c][CONV-1]=xx[c];
                float acc=s->conv_b[c]; const float* w=s->conv_w+(size_t)c*CONV; for(int t=0;t<CONV;t++) acc+=w[t]*cb[c][t]; xx[c]=silu(acc); }
            matvec(s->x_proj,xx,dbl,DTR+2*N,DN);
            const float* Bm=dbl+DTR; const float* Cm=dbl+DTR+N;
            for(int c=0;c<DN;c++) dt[c]=softplus(dotf(s->dt_proj+(size_t)c*DTR,dbl,DTR)+s->dt_b[c]);
            float (*hl)[N]=hstate[l];
            if(fast_exp){
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],dbx=dt[c]*xc;
                    __m256 vdtc=_mm256_set1_ps(dtc),vdbx=_mm256_set1_ps(dbx),vacc=_mm256_setzero_ps();
                    for(int j=0;j<N;j+=8){
                        __m256 e=exp256_ps(_mm256_mul_ps(vdtc,_mm256_loadu_ps(Ac+j)));
                        __m256 hcj=_mm256_fmadd_ps(e,_mm256_loadu_ps(hc+j),_mm256_mul_ps(vdbx,_mm256_loadu_ps(Bm+j)));
                        _mm256_storeu_ps(hc+j,hcj); vacc=_mm256_fmadd_ps(hcj,_mm256_loadu_ps(Cm+j),vacc); }
                    y[c]=hsum256(vacc)+s->Dskip[c]*xc; }
            } else {
                for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
                    for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; }
                    y[c]=acc+s->Dskip[c]*xc; }
            }
            for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
            matvec(s->out_proj,y,tmp,D,DN); for(int i=0;i<D;i++) x[i]+=tmp[i];
        }
        if(g_use_mlp){                                   // per-block MLP sublayer: x += fc2(silu(fc1(norm2(x))))
            rmsnorm(x, mlp_n2[l], xn);
            matvec(mlp_fc1[l],xn,mh,g_mlp_hid,D); for(int i=0;i<g_mlp_hid;i++) mh[i]=silu(mh[i]);
            matvec(mlp_fc2[l],mh,tmp,D,g_mlp_hid); for(int i=0;i<D;i++) x[i]+=tmp[i];
        }
    }
    rmsnorm(x,normf,xn); matvec(head,xn,logits,V,D);
}

// ---------------- gate metrics (identical to phase55_generator.c) ----------------
static const char* NAMES[]={"lily","max","mom","mommy","mum","mummy","mia","tim","tom","ben","sam","sue","dad","daddy","anna","lucy","jack","sara","my","spot","bella","leo","amy",0};
static int is_name(const char* w){ for(int i=0;NAMES[i];i++) if(!strcmp(w,NAMES[i])) return 1; return 0; }
typedef struct { int topBi,altLp,runWst; double nameWst; int ok; } WM;
typedef struct { int wsRun,chRun,nonPrint; double wsFrac; } BG;
static BG byte_guard(const unsigned char* b,long n){
    int mWs=0,mCh=0,ws=0,ch=0,wc=0,nonp=0,prev=-1;
    for(long i=0;i<n;i++){ int x=b[i];
        if(x==32||x==9||x==10||x==13){ wc++; ws++; if(ws>mWs)mWs=ws; ch=0; }
        else { ws=0; ch=(x==prev)?ch+1:1; if(ch>mCh)mCh=ch; if(x<32||x>126) nonp++; } prev=x; }
    BG g; g.wsRun=mWs; g.chRun=mCh; g.nonPrint=nonp; g.wsFrac=n?(double)wc/n:0; return g;
}
static WM word_metrics(const unsigned char* b,long n){
    char (*tok)[64]=xmalloc(sizeof(char[64])*(n+1)); int nt=0; char cur[64]; int cl=0;
    for(long i=0;i<=n;i++){ int x=(i<n)?b[i]:' '; int al=(x>='a'&&x<='z')||(x>='A'&&x<='Z');
        if(al){ if(cl<63) cur[cl++]=(char)tolower(x); } else { if(cl>=2){ cur[cl]=0; strcpy(tok[nt++],cur); } cl=0; } }
    WM m; m.ok=(nt>=4); if(!m.ok){ free(tok); return m; }
    int nm=0; for(int j=0;j<nt;j++) if(is_name(tok[j])) nm++; m.nameWst=(double)nm/nt*100.0;
    int run=1,mr=1; for(int j=1;j<nt;j++){ run=(!strcmp(tok[j],tok[j-1]))?run+1:1; if(run>mr)mr=run; } m.runWst=mr;
    int topbi=0; for(int j=0;j+1<nt;j++){ int cnt=1; for(int kk=j+1;kk+1<nt;kk++) if(!strcmp(tok[kk],tok[j])&&!strcmp(tok[kk+1],tok[j+1])) cnt++; if(cnt>topbi)topbi=cnt; }
    m.topBi=topbi; int al=0,am=0; for(int j=0;j+2<nt;j++){ al=(!strcmp(tok[j],tok[j+2]))?al+1:0; if(al>am)am=al; } m.altLp=am;
    free(tok); return m;
}
static uint64_t rng_s;
static inline uint64_t sm64(){ uint64_t z=(rng_s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline float rfloat(){ return (float)((sm64()>>40)/(double)(1ULL<<24)); }

int main(int argc,char**argv){
    int do_bpb=0,do_gen=0,do_bench=0,mode=1,greedy=0; long seqW=512,eval_tok=200000,gen_bytes=2000,K=32,bench_tok=20000; float rep=1.2f; int repwin=128;
    const char* wpath="results/phase55/archA_5m_weights.bin";
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--bpb")) do_bpb=1; else if(!strcmp(argv[i],"--gen")) do_gen=1; else if(!strcmp(argv[i],"--bench")) do_bench=1;
        else if(!strcmp(argv[i],"--exp")&&i+1<argc){ const char*m=argv[++i]; mode=!strcmp(m,"exact")?0:1; }
        else if(!strcmp(argv[i],"--seq")&&i+1<argc) seqW=atol(argv[++i]);
        else if(!strcmp(argv[i],"--eval-tok")&&i+1<argc) eval_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--gen-bytes")&&i+1<argc) gen_bytes=atol(argv[++i]);
        else if(!strcmp(argv[i],"--ksamples")&&i+1<argc) K=atol(argv[++i]);
        else if(!strcmp(argv[i],"--rep")&&i+1<argc) rep=atof(argv[++i]);
        else if(!strcmp(argv[i],"--rep-win")&&i+1<argc) repwin=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--bench-tok")&&i+1<argc) bench_tok=atol(argv[++i]);
        else if(!strcmp(argv[i],"--greedy")) greedy=1;
        else if(!strcmp(argv[i],"--weights")&&i+1<argc) wpath=argv[++i];
    }
    if(!do_bpb&&!do_gen&&!do_bench) do_bpb=1;
    load_weights(wpath); load_meta("results/phase55/meta.bin"); load_ids("results/phase55/ids.u16");
    hstate=calloc(L,sizeof(*hstate)); convbuf=calloc(L,sizeof(*convbuf)); kring=calloc((size_t)WIN*D,4); vring=calloc((size_t)WIN*D,4);
    long ntr=(long)(nids*0.9); long nval=nids-ntr; uint16_t* val=ids+ntr;
    fprintf(stderr,"5M loaded: ids=%ld (val=%ld) | D%d N%d L%d mlp_hid%d | weights ok\n",nids,nval,D,N,L,g_mlp_hid);
    float* logits=xmalloc((size_t)V*4); const float LN2=0.6931471805599453f;

    if(do_bench){
        printf("==== Phase 55 5M bench: tok/s (AVX2 single-core, fp32, %s exp) ====\n", mode?"fast":"exact");
        state_reset(); uint32_t tk=val[0]; long warm=bench_tok/10; clock_t t0=0;
        for(long s=0;s<warm+bench_tok;s++){ if(s==warm) t0=clock(); forward_token(tk,mode,logits);
            float mx=-1e30f; uint32_t bt=0; for(int o=0;o<V;o++) if(logits[o]>mx){mx=logits[o];bt=o;} tk=bt; }
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC, tps=bench_tok/el;
        printf("  5M: %.0f tok/s | %.2f us/tok (expected ~600-900; headroom vs 1.46M's 3300)\n", tps, 1e6/tps);
    }
    if(do_bpb){
        printf("==== Phase 55 5M val BPB (window=%ld, eval-tok=%ld, %s exp) ====\n",seqW,eval_tok,mode?"fast":"exact");
        long lim = eval_tok<(nval-1)?eval_tok:(nval-1); double bits=0; long nbytes=0,ntok=0,pos=0;
        while(pos+seqW+1<=lim){ state_reset();
            for(long t=0;t<seqW;t++){ forward_token(val[pos+t],mode,logits); int tgt=val[pos+t+1];
                float mx=-1e30f; for(int o=0;o<V;o++) if(logits[o]>mx) mx=logits[o];
                double se=0; for(int o=0;o<V;o++) se+=exp((double)(logits[o]-mx));
                bits += (-((double)logits[tgt]-mx)+log(se))/LN2; nbytes+=id2len[tgt]; ntok++; }
            pos+=seqW; }
        printf("  val BPB=%.4f bits/byte (over %ld tok / %ld bytes; bits/tok=%.4f) -- compare to train-run val BPB (overfit watch)\n",
               bits/(nbytes>0?nbytes:1), ntok, nbytes, bits/(ntok>0?ntok:1));
    }
    if(do_gen){
        int fe=mode; float temps[2]={0.65f,0.55f};
        const char* dirbase="results/phase55/human_5m";
        system("cmd /c \"if not exist results\\phase55\\human_5m mkdir results\\phase55\\human_5m\" >NUL 2>&1");
        printf("==== Phase 55 5M GATE 3: closed-loop gen, locked decode (rep=%.2f win=%d, NO top-p), %s exp, K=%ld/temp ====\n",rep,repwin,fe?"fast":"exact",K);
        printf("  bars: wsRun<=6 chRun<=8 wsFrac<=0.2502 nonPrint<=135 | topBi<=8 altLp<=2 runWst<=2 nameWst<=9\n");
        unsigned char* out=xmalloc(gen_bytes+64); unsigned char* worst=xmalloc(gen_bytes+64); uint16_t* gentok=xmalloc((size_t)(gen_bytes+64)*2);
        for(int ti=0;ti<2;ti++){ float temp=temps[ti];
            int wTopBi=0,wAltLp=0,wRunWst=0,wWsRun=0,wChRun=0,wNonP=0; double wWsFrac=0,wName=0; int worstScore=-1; long worstLen=0;
            for(long k=0;k<K;k++){
                rng_s=0x55AA0000ULL ^ ((uint64_t)ti<<20) ^ (uint64_t)k*0x100000001B3ULL;
                long seedpos=(long)(sm64()%(nval-seqW-1)); state_reset(); long olen=0,gl=0;
                for(int sI=0;sI<16;sI++){ uint16_t tk=val[seedpos+sI]; forward_token(tk,fe,logits); gentok[gl++]=tk;
                    for(int z=0;z<id2len[tk]&&olen<gen_bytes;z++) out[olen++]=id2bytes[tk][z]; }
                while(olen<gen_bytes){
                    if(rep!=1.0f){ long lo=gl>repwin?gl-repwin:0; static int seen[V]; static int stamp=0; stamp++;
                        for(long p=lo;p<gl;p++){ int tt=gentok[p]; if(seen[tt]!=stamp){ seen[tt]=stamp; float v=logits[tt]; logits[tt]= v>0? v/rep : v*rep; } } }
                    int tok;
                    if(greedy){ float mx=-1e30f; tok=0; for(int o=0;o<V;o++) if(logits[o]>mx){mx=logits[o];tok=o;} }
                    else { float mx=-1e30f; for(int o=0;o<V;o++){ logits[o]/=temp; if(logits[o]>mx)mx=logits[o]; }
                        float Z=0; for(int o=0;o<V;o++){ logits[o]=expf(logits[o]-mx); Z+=logits[o]; }
                        float u=rfloat()*Z,acc=0; tok=V-1; for(int o=0;o<V;o++){ acc+=logits[o]; if(u<=acc){tok=o;break;} } }
                    gentok[gl++]=(uint16_t)tok; for(int z=0;z<id2len[tok]&&olen<gen_bytes;z++) out[olen++]=id2bytes[tok][z];
                    forward_token((uint16_t)tok,fe,logits);
                }
                BG bg=byte_guard(out,olen); WM wm=word_metrics(out,olen);
                if(bg.wsRun>wWsRun)wWsRun=bg.wsRun; if(bg.chRun>wChRun)wChRun=bg.chRun; if(bg.nonPrint>wNonP)wNonP=bg.nonPrint; if(bg.wsFrac>wWsFrac)wWsFrac=bg.wsFrac;
                if(wm.ok){ if(wm.topBi>wTopBi)wTopBi=wm.topBi; if(wm.altLp>wAltLp)wAltLp=wm.altLp; if(wm.runWst>wRunWst)wRunWst=wm.runWst; if(wm.nameWst>wName)wName=wm.nameWst; }
                int score=wm.ok?(wm.topBi+10*wm.altLp+10*wm.runWst):0; if(score>worstScore){ worstScore=score; worstLen=olen; memcpy(worst,out,olen); }
                if(k<5){ char fn[300]; snprintf(fn,sizeof fn,"%s/s%ld_T%.2f.txt",dirbase,k+1,temp); FILE*wf=fopen(fn,"wb"); if(wf){ fwrite(out,1,olen,wf); fclose(wf); } }
            }
            { char fn[300]; snprintf(fn,sizeof fn,"%s/worst_T%.2f.txt",dirbase,temp); FILE*wf=fopen(fn,"wb"); if(wf){ fwrite(worst,1,worstLen,wf); fclose(wf); } }
            printf("  T%.2f WORST-OF-%ld | topBi=%d altLp=%d runWst=%d nameWst=%.1f | wsRun=%d chRun=%d wsFrac=%.4f nonPrint=%d\n",
                   temp,K,wTopBi,wAltLp,wRunWst,wName,wWsRun,wChRun,wWsFrac,wNonP);
        }
        printf("  samples (first 5/temp + worst) -> %s/  (architect reads semantics/referents). STOP gate 3.\n",dirbase);
    }
    printf("STOP. no commit.\n");
    return 0;
}
