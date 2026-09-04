// donor_engine.c -- run a pretrained Qwen2.5 donor on our own runtime.
//
// This is the piece the donor-adaptation programme never had: engine.c is an SSM engine and has
// no attention organ, no RoPE and no KV cache, so no donor could ever execute on it. This is a
// transformer runtime in the same style and with the same numeric conventions -- ternary weight
// codes with one fp32 scale per output row, fp32 norms/biases/embeddings, fp32 activations.
//
// It reads the flat binary written by qwen_export.py, which supports TWO weight modes on purpose:
//   quant=0 fp32     -> proves this runtime CORRECT against PyTorch before quantization is in play
//   quant=1 ternary  -> the format engine.c actually consumes
// Building the ternary path first would leave a bug in this file indistinguishable from the cost
// of the conversion, so the parity gate runs on fp32 and only then moves to ternary.
//
// Modes:
//   --logits <ids.bin> <n>   dump fp32 logits for the first n positions      (parity gate)
//   --bpb <slice.bin>        bits per UTF-8 byte over an exported eval slice (quality)
//   --bench <n>              time n single-token decode steps with a warm KV  (speed)
//
// Build:
//   clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp donor_engine.c -o donor_engine.exe -lm

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_WIN32)
#include <windows.h>
static double now_s(void){ LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart/(double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9; }
#endif
#include <immintrin.h>

// per-organ wall-clock accounting: profile before optimising, always.
enum { T_QKV=0, T_ATTN, T_O, T_FFN, T_HEAD, T_NORM, T_N };
static double g_t[T_N]; static const char* g_tn[T_N]={"qkv_proj","attention","o_proj","ffn","head","norm+glue"};
static int g_prof=0;
#define TIC double _t0=g_prof?now_s():0.0
#define TOC(k) do{ if(g_prof) g_t[k]+=now_s()-_t0; }while(0)

static void* xmalloc(size_t n){ void* p=malloc(n); if(!p){ fprintf(stderr,"OOM %zu\n",n); exit(1);} return p; }
static void die(const char* m){ fprintf(stderr,"FATAL: %s\n",m); exit(1); }

// ------------------------------------------------------------------ weight matrix (either mode)
typedef struct {
    int out, in, packed;
    const float* f32;      // quant==0
    const int8_t* code;    // quant==1, [out, in] row-major, values in {-1,0,+1}
    const float* scale;    // quant==1, [out]
} mat_t;

typedef struct {
    const float* in_norm;
    mat_t q, k, v, o;
    const float *qb, *kb, *vb;
    const float* post_norm;
    mat_t gate, up, down;
} layer_t;

typedef struct {
    int D, F, L, NH, NKV, HD, V, tied, quant;
    float rms_eps, rope_theta;
    const float* embed;
    layer_t* lay;
    const float* final_norm;
    mat_t head;              // only when tied==0
    char* blob;
    size_t blob_bytes;
} model_t;

// ------------------------------------------------------------------ kernels
// y[o] = sum_i W[o][i] * x[i]   (+ bias).  Both modes; ternary is codes*scale, which is exactly
// what the exporter's `deq = wq * scale` means, computed without materialising deq.
// De-interleaved activations for the packed path: byte j of a row holds the weights for input
// features 2j (low trit) and 2j+1 (high trit), so the kernel needs x split by parity. Built once
// per matvec call, outside the parallel region -- O(n_in) against the loop's O(n_out * n_in).
static float *g_xe=NULL,*g_xo=NULL; static int g_xcap=0;

static void matvec(const mat_t* m, const float* x, const float* bias, float* y){
    const int n_out=m->out, n_in=m->in;   // NOT "OUT"/"IN": windows.h defines those as SAL macros
    if(m->packed){
        const int H=n_in/2;
        if(H>g_xcap){ free(g_xe); free(g_xo); g_xe=xmalloc((size_t)H*4); g_xo=xmalloc((size_t)H*4); g_xcap=H; }
        for(int j=0;j<H;j++){ g_xe[j]=x[2*j]; g_xo[j]=x[2*j+1]; }
        // pshufb tables: index is the packed byte v in [0,8];  low trit = v%3 - 1, high = v/3 - 1
        const __m128i TLO=_mm_setr_epi8(-1,0,1,-1,0,1,-1,0,1,0,0,0,0,0,0,0);
        const __m128i THI=_mm_setr_epi8(-1,-1,-1,0,0,0,1,1,1,0,0,0,0,0,0,0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int o=0;o<n_out;o++){
            const int8_t* c=m->code+(size_t)o*H;
            __m256 acc=_mm256_setzero_ps(); int j=0;
            for(;j+8<=H;j+=8){
                __m128i b=_mm_loadl_epi64((const __m128i*)(c+j));   // 8 packed bytes
                __m128i lo=_mm_shuffle_epi8(TLO,b), hi=_mm_shuffle_epi8(THI,b);
                acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)),
                                    _mm256_loadu_ps(g_xe+j),acc);
                acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)),
                                    _mm256_loadu_ps(g_xo+j),acc);
            }
            float s[8]; _mm256_storeu_ps(s,acc);
            float t=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
            for(;j<H;j++){ int v=(uint8_t)c[j]; int t0=v%3-1, t1=(v/3)-1; t+=(float)t0*g_xe[j]+(float)t1*g_xo[j]; }
            t*=m->scale[o];
            y[o]=bias?t+bias[o]:t;
        }
        return;
    }
    if(m->f32){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int o=0;o<n_out;o++){
            const float* w=m->f32+(size_t)o*n_in;
            __m256 acc=_mm256_setzero_ps(); int i=0;
            for(;i+8<=n_in;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i),acc);
            float s[8]; _mm256_storeu_ps(s,acc);
            float t=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
            for(;i<n_in;i++) t+=w[i]*x[i];
            y[o]=bias?t+bias[o]:t;
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int o=0;o<n_out;o++){
        const int8_t* c=m->code+(size_t)o*n_in;
        __m256 acc=_mm256_setzero_ps(); int i=0;
        for(;i+8<=n_in;i+=8){
            // 8 ternary codes -> int32 -> float, then FMA against the activations
            __m128i c8=_mm_loadl_epi64((const __m128i*)(c+i));
            __m256i ci=_mm256_cvtepi8_epi32(c8);
            acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(ci),_mm256_loadu_ps(x+i),acc);
        }
        float s[8]; _mm256_storeu_ps(s,acc);
        float t=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
        for(;i<n_in;i++) t+=(float)c[i]*x[i];
        t*=m->scale[o];
        y[o]=bias?t+bias[o]:t;
    }
}

static void rmsnorm(const float* x,const float* w,int n,float eps,float* y){
    double ss=0.0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float inv=(float)(1.0/sqrt(ss/(double)n+(double)eps));
    for(int i=0;i<n;i++) y[i]=x[i]*inv*w[i];
}

static float silu(float x){ return x/(1.0f+expf(-x)); }

// HF "rotate_half": q'[j] = q[j]cos - q[j+h]sin ; q'[j+h] = q[j+h]cos + q[j]sin,  h = HD/2
static void rope(float* v,int n_heads,int HD,int pos,float theta){
    const int h=HD/2;
    for(int head=0;head<n_heads;head++){
        float* p=v+(size_t)head*HD;
        for(int j=0;j<h;j++){
            float f=(float)(pos*pow((double)theta,-2.0*(double)j/(double)HD));
            float c=cosf(f), s=sinf(f);
            float a=p[j], b=p[j+h];
            p[j]=a*c-b*s; p[j+h]=b*c+a*s;
        }
    }
}

// ------------------------------------------------------------------ model loading
static const char* rd(const char** p, size_t n){ const char* q=*p; *p+=n; return q; }

static void read_mat(const char** p, mat_t* m, int out, int in, int quant){
    m->out=out; m->in=in; m->packed=(quant==2); m->f32=NULL; m->code=NULL; m->scale=NULL;
    if(quant==0){ m->f32=(const float*)rd(p,(size_t)out*in*4); }
    else if(quant==1){ m->code=(const int8_t*)rd(p,(size_t)out*in);
                       m->scale=(const float*)rd(p,(size_t)out*4); }
    else { m->code=(const int8_t*)rd(p,(size_t)out*(in/2));   // base-3 g=2, 2 trits/byte
           m->scale=(const float*)rd(p,(size_t)out*4); }
}

static void load(model_t* M,const char* path){
    FILE* f=fopen(path,"rb"); if(!f) die("cannot open weights");
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    M->blob=xmalloc((size_t)sz); M->blob_bytes=(size_t)sz;
    if(fread(M->blob,1,(size_t)sz,f)!=(size_t)sz) die("short read");
    fclose(f);
    const char* p=M->blob;
    if(memcmp(p,"QWENDON1",8)) die("bad magic -- not a QWENDON1 file");
    p+=8;
    const int32_t* hi=(const int32_t*)p; p+=9*4;
    M->D=hi[0]; M->F=hi[1]; M->L=hi[2]; M->NH=hi[3]; M->NKV=hi[4];
    M->HD=hi[5]; M->V=hi[6]; M->tied=hi[7]; M->quant=hi[8];
    const float* hf=(const float*)p; p+=2*4;
    M->rms_eps=hf[0]; M->rope_theta=hf[1];
    const int QO=M->NH*M->HD, KVO=M->NKV*M->HD;
    fprintf(stderr,"  D=%d F=%d L=%d heads=%d/%d hd=%d V=%d tied=%d quant=%s eps=%g theta=%g\n",
            M->D,M->F,M->L,M->NH,M->NKV,M->HD,M->V,M->tied,M->quant==2?"packed(2 trits/byte)":M->quant?"ternary":"fp32",
            M->rms_eps,M->rope_theta);
    M->embed=(const float*)rd(&p,(size_t)M->V*M->D*4);
    M->lay=xmalloc((size_t)M->L*sizeof(layer_t));
    for(int l=0;l<M->L;l++){
        layer_t* L=&M->lay[l];
        L->in_norm=(const float*)rd(&p,(size_t)M->D*4);
        read_mat(&p,&L->q,QO,M->D,M->quant);  L->qb=(const float*)rd(&p,(size_t)QO*4);
        read_mat(&p,&L->k,KVO,M->D,M->quant); L->kb=(const float*)rd(&p,(size_t)KVO*4);
        read_mat(&p,&L->v,KVO,M->D,M->quant); L->vb=(const float*)rd(&p,(size_t)KVO*4);
        read_mat(&p,&L->o,M->D,QO,M->quant);
        L->post_norm=(const float*)rd(&p,(size_t)M->D*4);
        read_mat(&p,&L->gate,M->F,M->D,M->quant);
        read_mat(&p,&L->up,  M->F,M->D,M->quant);
        read_mat(&p,&L->down,M->D,M->F,M->quant);
    }
    M->final_norm=(const float*)rd(&p,(size_t)M->D*4);
    if(!M->tied) read_mat(&p,&M->head,M->V,M->D,M->quant);
    size_t used=(size_t)(p-M->blob);
    if(used!=M->blob_bytes){
        fprintf(stderr,"  FATAL: consumed %zu of %zu bytes -- layout mismatch\n",used,M->blob_bytes);
        exit(1);
    }
    fprintf(stderr,"  layout OK: consumed exactly %zu bytes\n",used);
}

// ------------------------------------------------------------------ state
typedef struct {
    float *x,*xb,*xb2,*q,*k,*v,*att,*attout,*hb,*hb2,*logits;
    float *kcache,*vcache;      // [L][maxseq][KVO]
    int maxseq;
} state_t;

static void state_init(state_t* s,const model_t* M,int maxseq){
    const int QO=M->NH*M->HD, KVO=M->NKV*M->HD;
    s->maxseq=maxseq;
    s->x=xmalloc((size_t)M->D*4); s->xb=xmalloc((size_t)M->D*4); s->xb2=xmalloc((size_t)M->D*4);
    s->q=xmalloc((size_t)QO*4); s->k=xmalloc((size_t)KVO*4); s->v=xmalloc((size_t)KVO*4);
    s->att=xmalloc((size_t)M->NH*maxseq*4);   // per-head scratch, no malloc in the hot loop
    s->attout=xmalloc((size_t)QO*4);         // QO need not equal D on every donor
    s->hb=xmalloc((size_t)M->F*4); s->hb2=xmalloc((size_t)M->F*4);
    s->logits=xmalloc((size_t)M->V*4);
    s->kcache=xmalloc((size_t)M->L*maxseq*KVO*4);
    s->vcache=xmalloc((size_t)M->L*maxseq*KVO*4);
}

// one token at position `pos`; logits land in s->logits
static void forward(const model_t* M,state_t* s,int token,int pos){
    const int D=M->D, F=M->F, NH=M->NH, NKV=M->NKV, HD=M->HD;
    const int QO=NH*HD, KVO=NKV*HD, GQA=NH/NKV;
    memcpy(s->x,M->embed+(size_t)token*D,(size_t)D*4);

    for(int l=0;l<M->L;l++){
        const layer_t* L=&M->lay[l];
        { TIC; rmsnorm(s->x,L->in_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
        { TIC;
          matvec(&L->q,s->xb,L->qb,s->q);
          matvec(&L->k,s->xb,L->kb,s->k);
          matvec(&L->v,s->xb,L->vb,s->v);
          rope(s->q,NH,HD,pos,M->rope_theta);
          rope(s->k,NKV,HD,pos,M->rope_theta);
          TOC(T_QKV); }

        float* kc=s->kcache+((size_t)l*s->maxseq+pos)*KVO;
        float* vc=s->vcache+((size_t)l*s->maxseq+pos)*KVO;
        memcpy(kc,s->k,(size_t)KVO*4);
        memcpy(vc,s->v,(size_t)KVO*4);

        const float inv=1.0f/sqrtf((float)HD);
        TIC;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int h=0;h<NH;h++){
            const int kvh=h/GQA;
            float* a=s->att+(size_t)h*s->maxseq;      // preallocated, disjoint per head
            const float* qh=s->q+(size_t)h*HD;
            float mx=-1e30f;
            for(int t=0;t<=pos;t++){
                const float* kt=s->kcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
                float d=0.0f; for(int i=0;i<HD;i++) d+=qh[i]*kt[i];
                d*=inv; a[t]=d; if(d>mx) mx=d;
            }
            float sum=0.0f;
            for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }
            float rs=1.0f/sum;
            float* out=s->attout+(size_t)h*HD;
            for(int i=0;i<HD;i++) out[i]=0.0f;
            for(int t=0;t<=pos;t++){
                const float* vt=s->vcache+((size_t)l*s->maxseq+t)*KVO+(size_t)kvh*HD;
                float w=a[t]*rs;
                for(int i=0;i<HD;i++) out[i]+=w*vt[i];
            }
        }
        TOC(T_ATTN);
        { TIC; matvec(&L->o,s->attout,NULL,s->xb2);
          for(int i=0;i<D;i++) s->x[i]+=s->xb2[i]; TOC(T_O); }

        { TIC; rmsnorm(s->x,L->post_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
        { TIC;
          matvec(&L->gate,s->xb,NULL,s->hb);
          matvec(&L->up,  s->xb,NULL,s->hb2);
          for(int i=0;i<F;i++) s->hb[i]=silu(s->hb[i])*s->hb2[i];
          matvec(&L->down,s->hb,NULL,s->xb2);
          for(int i=0;i<D;i++) s->x[i]+=s->xb2[i];
          TOC(T_FFN); }
    }
    { TIC; rmsnorm(s->x,M->final_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
    TIC;
    if(M->tied){
        mat_t h={M->V,D,0,M->embed,NULL,NULL};   // packed=0: the tied head reads the fp32 embedding
        matvec(&h,s->xb,NULL,s->logits);
    } else {
        matvec(&M->head,s->xb,NULL,s->logits);
    }
    TOC(T_HEAD);
}

// ------------------------------------------------------------------ modes
static int32_t* read_ids(const char* path,long* n){
    FILE* f=fopen(path,"rb"); if(!f) die("cannot open ids");
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    int32_t* v=xmalloc((size_t)sz); if(fread(v,1,(size_t)sz,f)!=(size_t)sz) die("short ids read");
    fclose(f); *n=sz/4; return v;
}

int main(int argc,char** argv){
    const char* wp=NULL; const char* mode=NULL; const char* arg2=NULL; long arg3=0;
    const char* logout=NULL;
    int threads=1, seqlen=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--weights")&&i+1<argc) wp=argv[++i];
        else if(!strcmp(argv[i],"--threads")&&i+1<argc) threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--seqlen")&&i+1<argc) seqlen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--logits")&&i+3<argc){ mode="logits"; arg2=argv[++i]; arg3=atol(argv[++i]); logout=argv[++i]; }
        else if(!strcmp(argv[i],"--bpb")&&i+1<argc){ mode="bpb"; arg2=argv[++i]; }
        else if(!strcmp(argv[i],"--bench")&&i+1<argc){ mode="bench"; arg3=atol(argv[++i]); }
        else if(!strcmp(argv[i],"--profile")){ g_prof=1; }
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    if(!wp||!mode){ fprintf(stderr,
        "usage: donor_engine --weights <bin> [--threads N] [--seqlen N]\n"
        "                    (--logits <ids.bin> <n> | --bpb <ids.bin> | --bench <n>)\n"); return 1; }
#ifdef _OPENMP
    if(threads<1) threads=1; omp_set_num_threads(threads); omp_set_dynamic(0);
#endif
    model_t M; load(&M,wp);

    if(!strcmp(mode,"bench")){
        state_t s; state_init(&s,&M,arg3+2);
        forward(&M,&s,1,0);                                   // warm
        double t0=now_s();
        for(long i=0;i<arg3;i++) forward(&M,&s,1+(int)(i%100),(int)(i+1));
        double dt=now_s()-t0;
        printf("BENCH  %ld tokens  %.3f s  %.2f tok/s  (threads=%d, %s)\n",
               arg3,dt,arg3/dt,threads,M.quant==2?"packed":M.quant?"ternary":"fp32");
        if(g_prof){
            double tot=0; for(int k=0;k<T_N;k++) tot+=g_t[k];
            printf("  organ        ms/token   %% of total\n");
            for(int k=0;k<T_N;k++)
                printf("  %-11s %9.3f   %5.1f%%\n",g_tn[k],g_t[k]/arg3*1e3,100.0*g_t[k]/tot);
            printf("  %-11s %9.3f   (organs summed; wall %.3f ms/token)\n",
                   "TOTAL",tot/arg3*1e3,dt/arg3*1e3);
        }
        return 0;
    }

    long n=0; int32_t* ids=read_ids(arg2,&n);
    int SL = seqlen>0 ? seqlen : (int)n;
    if(n%SL){ fprintf(stderr,"ids length %ld not divisible by seqlen %d\n",n,SL); return 1; }
    long nseq=n/SL;
    state_t s; state_init(&s,&M,SL+1);

    if(!strcmp(mode,"logits")){
        // to a FILE, never stdout: on Windows stdout is text mode and mangles binary data
        FILE* lo=fopen(logout,"wb"); if(!lo) die("cannot open logits output");
        long m2 = arg3<n?arg3:n;
        for(long i=0;i<m2;i++){ forward(&M,&s,ids[i],(int)i); fwrite(s.logits,4,(size_t)M.V,lo); }
        fclose(lo);
        fprintf(stderr,"wrote %ld logit vectors of %d floats to %s\n",m2,M.V,logout);
        return 0;
    }

    // --bpb: total NLL over next-token prediction, per sequence
    double tot_nats=0.0; long npred=0;
    double t0=now_s();
    for(long q=0;q<nseq;q++){
        const int32_t* seq=ids+q*SL;
        for(int t=0;t<SL;t++){
            forward(&M,&s,seq[t],t);
            if(t+1<SL){
                float mx=-1e30f; for(int i=0;i<M.V;i++) if(s.logits[i]>mx) mx=s.logits[i];
                double sum=0.0; for(int i=0;i<M.V;i++) sum+=exp((double)(s.logits[i]-mx));
                tot_nats += -((double)(s.logits[seq[t+1]]-mx) - log(sum));
                npred++;
            }
        }
        fprintf(stderr,"  seq %ld/%ld  running nats/token %.6f  (%.1fs)\n",
                q+1,nseq,tot_nats/(double)npred,now_s()-t0);
    }
    printf("NATS_TOTAL %.10f\nN_PREDICTED %ld\nNATS_PER_TOKEN %.10f\n",
           tot_nats,npred,tot_nats/(double)npred);
    return 0;
}
