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
enum { T_QKV=0, T_ROPE, T_ATTN, T_O, T_FFN, T_HEAD, T_NORM, T_N };
static double g_t[T_N]; static const char* g_tn[T_N]={"qkv_proj","rope","attention","o_proj","ffn","head","norm+glue"};
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
    const int8_t* tm;      // --lut only: the SAME packed bytes, tile-major [in/2][Mpad]
    int Mpad;              // out rounded up to 32, the tile width the LUT kernel writes
} mat_t;

typedef struct {
    const float* in_norm;
    mat_t q, k, v, o;
    const float *qb, *kb, *vb;
    const float* post_norm;
    mat_t gate, up, down;
    // --fuse: q|k|v and gate|up concatenated by OUTPUT ROW. They share their input, so three
    // matvecs become one and two become one. The point is not fewer instructions -- it is fewer
    // OpenMP regions and bigger ones: k_proj and v_proj are 128 output rows, which across 6
    // threads is 21 rows and ~4.6 us of real work per fork.
    mat_t qkv, gateup;
    const float* qkvb;
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

// ---------------------------------------------------------------- the LUT path (--lut)
// probe-1's pshufb-LUT, which is what engine.c's dense path uses and what R1 s4.2 identified as the
// only lever left on the FFN term. Two changes at once, and they must not be conflated:
//   (a) LAYOUT. The same packed bytes, transposed to tile-major [t][Mpad], so 32 output rows share
//       one broadcast table and the shuffle result IS the partial product -- no FMA, no converts.
//   (b) NUMERICS. Activations are quantized to int8 (AQ=63) so the table entries fit a byte.
//       max |entry| = 2*63 = 126 <= 127; that bound is WHY AQ is 63 and not 127.
// (a) is exactly lossless. (b) is NOT -- it is the first activation quantization this programme has
// ever run on a donor, BRIEF_T2 s5 records it as untested. Its cost is measured, not assumed.
#define AQ 63
static int      g_lut=0, g_lutdiag=0, g_lutnodown=0, g_lutnohead=0;  // --lut* family
static long     g_dn[2]={0,0}; static double g_dcrest[2]={0,0},g_drel[2]={0,0},g_dcmax[2]={0,0};
static int8_t  *g_xq=NULL,*g_lutab=NULL; static int g_lutcap=0;

static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p), hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo));
    acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi));
    acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
// x -> int8 with one scale for the whole vector; returns the scale (0 for an all-zero input).
// g_clip > 0 sets the grid from k*rms instead of amax, saturating whatever lies beyond. The donor's
// activations have a crest factor (amax/rms) of 8-70, so an amax grid spends its 63 steps covering
// outliers and leaves ~4-8 of them for the bulk. Clipping trades a few saturated features for
// resolution on the rest. k=0 keeps the amax behaviour, which is what engine.c does on a model
// TRAINED for this format and is therefore the right default there and the wrong one here.
static float g_clip=0.0f;
static float quant_i8(const float* x,int n,int8_t* xq){
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(x[i]); if(a>amax) amax=a; }
    if(amax==0.0f){ memset(xq,0,(size_t)n); return 0.0f; }
    float lim=amax;
    if(g_clip>0.0f){ double s2=0; for(int i=0;i<n;i++) s2+=(double)x[i]*x[i];
        float r=(float)sqrt(s2/n)*g_clip; if(r>0.0f&&r<lim) lim=r; }
    float s=lim/(float)AQ, inv=1.0f/s;
    for(int i=0;i<n;i++){ int v=(int)lrintf(x[i]*inv); if(v>AQ)v=AQ; if(v<-AQ)v=-AQ; xq[i]=(int8_t)v; }
    return s;
}
// One 16-byte table per input PAIR. Index is this runtime's packed byte v: low trit = v%3-1 pairs
// with input 2t, high trit = v/3-1 with 2t+1. (engine.c uses the opposite digit order; the byte
// layout here is the one qwen_export.py already writes, so the tile-major copy is a pure transpose.)
static void build_lut(const int8_t* xq,int T,int8_t* lut){
    for(int t=0;t<T;t++){ int x0=xq[2*t],x1=xq[2*t+1];
        for(int v=0;v<16;v++) lut[t*16+v]=(int8_t)(v<9 ? (v%3-1)*x0+(v/3-1)*x1 : 0); }
}
// Grouped variant: the input is cut into ng contiguous groups of gp PAIRS, each carrying its own
// activation scale, and the int32 tile accumulator is folded to float at every group boundary.
// This is what the crest numbers argue for: outlier channels in an LLM are persistent, so cutting
// the input by channel index confines them to a few groups instead of setting the grid for all of
// them. The extra cost is ng foldings of 32 lanes per tile -- O(ng * M) against the kernel's
// O(T * Mpad), i.e. nothing.
static void matvec_lut_g(const int8_t* codes,const int8_t* lut,float* y,int M,int Mpad,int T,
                         int gp,const float* gs,int ng){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int base=0;base<Mpad;base+=32){
        float f[32]; for(int r=0;r<32;r++) f[r]=0.0f;
        for(int g=0;g<ng;g++){
            int t0=g*gp, t1=t0+gp; if(t1>T) t1=T; if(t0>=t1) break;
            __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),
                            _mm256_setzero_si256(),_mm256_setzero_si256()};
            for(int t=t0;t<t1;t++){
                __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
                __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base));
                acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx));
            }
            int32_t tmp[32];
            _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]);  _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
            _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
            for(int r=0;r<32;r++) f[r]+=(float)tmp[r]*gs[g];
        }
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=f[r];
    }
}
static void matvec_lut(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int base=0;base<Mpad;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),
                        _mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){
            __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base));
            acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx));
        }
        int32_t tmp[32];
        _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]);  _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r];
    }
}
// Transpose the packed row-major bytes into tile-major. Pure data movement: no re-encoding, so a
// --lut model and a --quant packed model hold bit-identical WEIGHTS and differ only in activations.
static void build_tm(mat_t* m){
    if(!m->packed||m->tm) return;
    const int H=m->in/2; m->Mpad=((m->out+31)/32)*32;
    int8_t* tm=xmalloc((size_t)H*m->Mpad);
    for(int t=0;t<H;t++){
        for(int o=0;o<m->out;o++) tm[(size_t)t*m->Mpad+o]=m->code[(size_t)o*H+t];
        for(int o=m->out;o<m->Mpad;o++) tm[(size_t)t*m->Mpad+o]=4;   // code 4 == (0,0), a no-op row
    }
    m->tm=tm;
}

static int32_t* g_i32b=NULL; static int g_i32cap=0;
static int32_t* g_i32(int n){ if(n>g_i32cap){ free(g_i32b); g_i32b=xmalloc((size_t)n*4); g_i32cap=n; } return g_i32b; }
static float* g_f32b=NULL; static int g_f32cap=0;
static float* g_f32(int n){ if(n>g_f32cap){ free(g_f32b); g_f32b=xmalloc((size_t)n*4); g_f32cap=n; } return g_f32b; }
static int g_group=0, g_gscap=0; static float* g_gs=NULL;   // --lut-group
static char g_grouplbl[16]="";

static void matvec(const mat_t* m, const float* x, const float* bias, float* y){
    const int n_out=m->out, n_in=m->in;   // NOT "OUT"/"IN": windows.h defines those as SAL macros
    if(m->tm){                                                    // --lut
        const int H=n_in/2;
        if(H>g_lutcap){ free(g_xq); free(g_lutab);
            g_xq=xmalloc((size_t)H*2); g_lutab=xmalloc((size_t)H*16); g_lutcap=H; }
        // --lut-group G: G input channels share one scale (G must be even so no PAIR straddles a
        // boundary). G=0 keeps one scale for the whole vector, which is engine.c's convention.
        int gsz = g_group>0 ? g_group : n_in;
        if(gsz>n_in) gsz=n_in;
        int ng = (n_in+gsz-1)/gsz;
        if(ng>g_gscap){ free(g_gs); g_gs=xmalloc((size_t)ng*4); g_gscap=ng; }
        for(int g=0;g<ng;g++){ int o0=g*gsz, len=(o0+gsz<=n_in)?gsz:(n_in-o0);
            g_gs[g]=quant_i8(x+o0,len,g_xq+o0); }
        float sx=g_gs[0];
        if(g_lutdiag){
            // Measure the damage on the INPUT SIDE ONLY. This is arithmetic on x and its int8
            // round-trip; the kernel is not involved, so it separates "int8 activations are
            // lossy on a donor" from "the kernel is wrong".
            // crest is measured WITHIN each group, because that is what sets the grid: with
            // --lut-group the whole-vector amax/rms no longer describes the resolution anyone gets.
            double s2=0,e2=0,cs=0,cmax=0; int nc=0;
            for(int g=0;g<ng;g++){ int o0=g*gsz, len=(o0+gsz<=n_in)?gsz:(n_in-o0);
                double gs2=0,gam=0;
                for(int i=o0;i<o0+len;i++){ double v=x[i]; gs2+=v*v; if(fabs(v)>gam) gam=fabs(v); }
                double grms=sqrt(gs2/len);
                if(grms>0){ double c=gam/grms; cs+=c; nc++; if(c>cmax) cmax=c; } }
            for(int i=0;i<n_in;i++){ double v=x[i],r=(double)g_xq[i]*g_gs[i/gsz], d=v-r;
                s2+=v*v; e2+=d*d; }
            double rel=sqrt(e2/(s2>0?s2:1));
            int b = (n_in==4864)?1:0;                     // down_proj is the only in=4864 organ
            g_dn[b]++; g_drel[b]+=rel;
            g_dcrest[b]+=(nc?cs/nc:0); if(cmax>g_dcmax[b]) g_dcmax[b]=cmax;
        }
        build_lut(g_xq,H,g_lutab);
        if(ng==1){
            int32_t* acc=(int32_t*)g_i32(n_out);
            matvec_lut(m->tm,g_lutab,acc,n_out,m->Mpad,H);
            for(int o=0;o<n_out;o++){ float t=(float)acc[o]*sx*m->scale[o]; y[o]=bias?t+bias[o]:t; }
        } else {
            float* acc=g_f32(n_out);
            matvec_lut_g(m->tm,g_lutab,acc,n_out,m->Mpad,H,gsz/2,g_gs,ng);
            for(int o=0;o<n_out;o++){ float t=acc[o]*m->scale[o]; y[o]=bias?t+bias[o]:t; }
        }
        return;
    }
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
// rope used to compute pow()+cosf()+sinf() PER HEAD PER LAYER, i.e. NH+NKV heads x L layers x
// HD/2 elements of transcendentals per token -- 12,288 double-precision pow() calls per token on
// this donor, for 32 distinct values. The angle depends only on (pos, j): not on the head, not on
// the layer. Hoisted to one table per token. Bit-identical by construction: the same expression,
// the same order, the same float rounding -- verified against a 64-token logit dump.
#define ROPE_MAX_HD 512
static void rope_table(float* cs,int HD,int pos,float theta){
    const int h=HD/2;
    for(int j=0;j<h;j++){
        float f=(float)(pos*pow((double)theta,-2.0*(double)j/(double)HD));
        cs[j]=cosf(f); cs[h+j]=sinf(f);
    }
}
static void rope(float* v,int n_heads,int HD,const float* cs){
    const int h=HD/2;
    for(int head=0;head<n_heads;head++){
        float* p=v+(size_t)head*HD;
        for(int j=0;j<h;j++){
            float c=cs[j], s=cs[h+j];
            float a=p[j], b=p[j+h];
            p[j]=a*c-b*s; p[j+h]=b*c+a*s;
        }
    }
}

// ------------------------------------------------------------------ model loading
static const char* rd(const char** p, size_t n){ const char* q=*p; *p+=n; return q; }

static void read_mat(const char** p, mat_t* m, int out, int in, int quant){
    m->out=out; m->in=in; m->packed=(quant==2); m->f32=NULL; m->code=NULL; m->scale=NULL;
    m->tm=NULL; m->Mpad=0;
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

// ---------------------------------------------------------------- --fuse
// Concatenate matrices that share an input, by output row. Pure data movement: the codes, the
// scales and the biases are copied unchanged, so a fused model computes bit-identical results
// and the only thing that changes is how many OpenMP regions a token opens (169 -> 97).
// It costs one extra copy of the fused weights in RAM (~117 MB on Qwen2.5-0.5B packed).
static int g_fuse=0;
static void fuse_mats(mat_t* dst,const mat_t* const* src,int n,int quant){
    int out=0, in=src[0]->in;
    for(int i=0;i<n;i++){ if(src[i]->in!=in) die("--fuse: inputs differ"); out+=src[i]->out; }
    dst->out=out; dst->in=in; dst->packed=(quant==2); dst->tm=NULL; dst->Mpad=0;
    dst->f32=NULL; dst->code=NULL; dst->scale=NULL;
    if(quant==0){
        float* w=xmalloc((size_t)out*in*4); size_t o=0;
        for(int i=0;i<n;i++){ memcpy(w+o,src[i]->f32,(size_t)src[i]->out*in*4); o+=(size_t)src[i]->out*in; }
        dst->f32=w;
    } else {
        size_t rb=(quant==2)?(size_t)(in/2):(size_t)in;
        int8_t* c=xmalloc((size_t)out*rb); float* sc=xmalloc((size_t)out*4);
        size_t bo=0; int so=0;
        for(int i=0;i<n;i++){
            memcpy(c+bo,src[i]->code,(size_t)src[i]->out*rb); bo+=(size_t)src[i]->out*rb;
            memcpy(sc+so,src[i]->scale,(size_t)src[i]->out*4); so+=src[i]->out;
        }
        dst->code=c; dst->scale=sc;
    }
}
static float* cat3f(const float* a,int na,const float* b,int nb,const float* c,int nc){
    float* r=xmalloc((size_t)(na+nb+nc)*4);
    memcpy(r,a,(size_t)na*4); memcpy(r+na,b,(size_t)nb*4); memcpy(r+na+nb,c,(size_t)nc*4);
    return r;
}
static void build_fused(model_t* M){
    size_t bytes=0;
    for(int l=0;l<M->L;l++){ layer_t* L=&M->lay[l];
        const mat_t* qkv[3]={&L->q,&L->k,&L->v};
        const mat_t* gu[2]={&L->gate,&L->up};
        fuse_mats(&L->qkv,qkv,3,M->quant);
        fuse_mats(&L->gateup,gu,2,M->quant);
        L->qkvb=cat3f(L->qb,L->q.out,L->kb,L->k.out,L->vb,L->v.out);
        size_t rb=(M->quant==2)?(size_t)(M->D/2):(M->quant?(size_t)M->D:(size_t)M->D*4);
        bytes+=(size_t)(L->qkv.out+L->gateup.out)*rb;
    }
    fprintf(stderr,"  --fuse: q|k|v and gate|up concatenated, +%.1f MB, matvec calls/token %d -> %d\n",
            bytes/1048576.0, M->L*7+1, M->L*4+1);
}

// ------------------------------------------------------------------ state
typedef struct {
    float *x,*xb,*xb2,*q,*k,*v,*att,*attout,*hb,*hb2,*logits,*qkvbuf,*gubuf;
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
    s->qkvbuf=xmalloc((size_t)(QO+2*KVO)*4);      // --fuse scratch; q|k|v land contiguous,
    s->gubuf =xmalloc((size_t)(2*M->F)*4);        // gate|up likewise, and both are read in place
}

// one token at position `pos`; logits land in s->logits
static void forward(const model_t* M,state_t* s,int token,int pos){
    const int D=M->D, F=M->F, NH=M->NH, NKV=M->NKV, HD=M->HD;
    const int QO=NH*HD, KVO=NKV*HD, GQA=NH/NKV;
    memcpy(s->x,M->embed+(size_t)token*D,(size_t)D*4);

    float ropecs[ROPE_MAX_HD];
    { TIC;
      if(HD>ROPE_MAX_HD) die("head_dim > ROPE_MAX_HD: raise the constant");
      rope_table(ropecs,HD,pos,M->rope_theta);   // once per TOKEN, not per head per layer
      TOC(T_ROPE); }

    for(int l=0;l<M->L;l++){
        const layer_t* L=&M->lay[l];
        { TIC; rmsnorm(s->x,L->in_norm,D,M->rms_eps,s->xb); TOC(T_NORM); }
        float *qp=s->q,*kp=s->k,*vp=s->v;
        { TIC;
          if(g_fuse){
              matvec(&L->qkv,s->xb,L->qkvb,s->qkvbuf);   // one region instead of three
              qp=s->qkvbuf; kp=s->qkvbuf+QO; vp=s->qkvbuf+QO+KVO;   // read in place, no copy
          } else {
              matvec(&L->q,s->xb,L->qb,s->q);
              matvec(&L->k,s->xb,L->kb,s->k);
              matvec(&L->v,s->xb,L->vb,s->v);
          }
          TOC(T_QKV); }
        { TIC;
          // split out of T_QKV on purpose: rope was inside it, so the per-organ GB/s the
          // ledger derived for qkv charged rope's transcendentals to the weights.
          rope(qp,NH,HD,ropecs);
          rope(kp,NKV,HD,ropecs);
          TOC(T_ROPE); }

        float* kc=s->kcache+((size_t)l*s->maxseq+pos)*KVO;
        float* vc=s->vcache+((size_t)l*s->maxseq+pos)*KVO;
        memcpy(kc,kp,(size_t)KVO*4);
        memcpy(vc,vp,(size_t)KVO*4);

        const float inv=1.0f/sqrtf((float)HD);
        TIC;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for(int h=0;h<NH;h++){
            const int kvh=h/GQA;
            float* a=s->att+(size_t)h*s->maxseq;      // preallocated, disjoint per head
            const float* qh=qp+(size_t)h*HD;
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
          if(g_fuse){
              matvec(&L->gateup,s->xb,NULL,s->gubuf);            // one region instead of two
              const float* g=s->gubuf; const float* u=s->gubuf+F;
              for(int i=0;i<F;i++) s->hb[i]=silu(g[i])*u[i];
          } else {
              matvec(&L->gate,s->xb,NULL,s->hb);
              matvec(&L->up,  s->xb,NULL,s->hb2);
              for(int i=0;i<F;i++) s->hb[i]=silu(s->hb[i])*s->hb2[i];
          }
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

// ---------------------------------------------------------------- --selftest-lut
// The kernel is checked against a SCALAR INTEGER reference over the same trits and the same int8
// activations, so any disagreement is the kernel and not the numerics. Case B is the planted
// control: it feeds the reference the OPPOSITE digit order and requires the comparison to FIRE.
// A gate never seen to trip is decoration (feedback_planted_controls).
static int selftest_lut(void){
    const int M=100, K=258;                       // M not a multiple of 32, K/2 odd: exercise padding
    const int H=K/2, Mpad=((M+31)/32)*32;
    int8_t* W=xmalloc((size_t)M*K); int8_t* code=xmalloc((size_t)M*H);
    int8_t* tm=xmalloc((size_t)H*Mpad); float* x=xmalloc((size_t)K*4);
    int8_t* xq=xmalloc((size_t)K); int8_t* lut=xmalloc((size_t)H*16);
    int32_t* y=xmalloc((size_t)Mpad*4); unsigned r=12345u;
    for(int i=0;i<M*K;i++){ r=r*1103515245u+12345u; W[i]=(int8_t)((int)((r>>16)%3)-1); }
    for(int i=0;i<K;i++){ r=r*1103515245u+12345u; x[i]=((float)(r>>8&0xFFFF)/32768.0f-1.0f)*3.7f; }
    for(int m=0;m<M;m++) for(int t=0;t<H;t++)                       // this runtime's byte: low=even
        code[(size_t)m*H+t]=(int8_t)((W[(size_t)m*K+2*t]+1)+3*(W[(size_t)m*K+2*t+1]+1));
    for(int t=0;t<H;t++){ for(int m=0;m<M;m++) tm[(size_t)t*Mpad+m]=code[(size_t)m*H+t];
                          for(int m=M;m<Mpad;m++) tm[(size_t)t*Mpad+m]=4; }
    quant_i8(x,K,xq); build_lut(xq,H,lut); matvec_lut(tm,lut,y,M,Mpad,H);
    long bad=0, badB=0;
    for(int m=0;m<M;m++){
        long a=0,b=0;
        for(int k=0;k<K;k++) a+=(long)W[(size_t)m*K+k]*xq[k];
        for(int t=0;t<H;t++) b+=(long)W[(size_t)m*K+2*t]*xq[2*t+1]+(long)W[(size_t)m*K+2*t+1]*xq[2*t];
        if(y[m]!=(int32_t)a) bad++;
        if(y[m]!=(int32_t)b) badB++;
    }
    printf("selftest-lut  A kernel vs scalar-int : %s (%ld/%d rows differ)\n",bad?"FAIL":"PASS",bad,M);
    printf("selftest-lut  B planted (swapped digits) must DIFFER: %s (%ld/%d rows differ)\n",
           badB?"PASS":"FAIL -- the comparison cannot tell a wrong kernel from a right one",badB,M);
    // Case C: the GROUPED kernel. Three groups with three different scales, against a scalar
    // float reference built the same way. Grouping changes the fold, not the trits, so a
    // disagreement here is the fold.
    const int NG=3, GP=(H+NG-1)/NG;
    float gs[3]={0.011f,0.25f,3.0f}; float* yf=xmalloc((size_t)Mpad*4);
    matvec_lut_g(tm,lut,yf,M,Mpad,H,GP,gs,NG);
    double worstC=0;
    for(int m=0;m<M;m++){
        double ref=0;
        for(int t=0;t<H;t++){ int g=t/GP; if(g>=NG) g=NG-1;
            ref+=gs[g]*((double)W[(size_t)m*K+2*t]*xq[2*t]+(double)W[(size_t)m*K+2*t+1]*xq[2*t+1]); }
        double den=fabs(ref)>1e-6?fabs(ref):1e-6, e=fabs(yf[m]-ref)/den;
        if(e>worstC) worstC=e;
    }
    printf("selftest-lut  C grouped kernel (3 scales) vs scalar float : %s (worst rel %.2e)\n",
           worstC<1e-5?"PASS":"FAIL",worstC);
    int ok=(bad==0&&badB>0&&worstC<1e-5);
    printf("selftest-lut  VERDICT: %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}

// --lut-diag: what int8 does to the ACTIVATIONS, measured on x alone. crest = amax/rms is the
// number that decides whether an amax-scaled int8 grid is usable: the grid has AQ=63 steps
// between 0 and amax, so a vector whose typical element sits at rms gets only 63/crest of them.
static void lut_diag_report(void){
    static const char* nm[2]={"all organs with in=896 (q,k,v,o,gate,up,head)","down_proj (in=4864)"};
    fprintf(stderr,"\n  --lut-diag: activation int8 round-trip, AQ=%d, amax scale per %s\n",
            AQ, g_group>0?g_grouplbl:"whole vector");
    fprintf(stderr,"  %-46s %8s %9s %9s %10s\n","organ group","calls","crest avg","crest max","rel err");
    for(int b=0;b<2;b++){ if(!g_dn[b]) continue;
        fprintf(stderr,"  %-46s %8ld %9.1f %9.1f %9.4f\n",nm[b],g_dn[b],
                g_dcrest[b]/g_dn[b],g_dcmax[b],g_drel[b]/g_dn[b]);
        fprintf(stderr,"  %-46s %8s effective levels at rms (in-group): %.1f of %d\n","","",
                63.0/(g_dcrest[b]/g_dn[b]),AQ);
    }
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
        else if(!strcmp(argv[i],"--lut")){ g_lut=1; }
        else if(!strcmp(argv[i],"--lut-diag")){ g_lut=1; g_lutdiag=1; atexit(lut_diag_report); }
        else if(!strcmp(argv[i],"--lut-clip")&&i+1<argc){ g_clip=(float)atof(argv[++i]); }
        else if(!strcmp(argv[i],"--lut-no-down")){ g_lutnodown=1; }
        else if(!strcmp(argv[i],"--lut-no-head")){ g_lutnohead=1; }
        else if(!strcmp(argv[i],"--fuse")){ g_fuse=1; }
        else if(!strcmp(argv[i],"--lut-group")&&i+1<argc){ g_group=atoi(argv[++i]);
            if(g_group&1) die("--lut-group must be even: a 2-trit LUT pair may not straddle a group");
            snprintf(g_grouplbl,sizeof g_grouplbl,"%d channels",g_group); }
        else if(!strcmp(argv[i],"--selftest-lut")){ return selftest_lut(); }
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    if(!wp||!mode){ fprintf(stderr,
        "usage: donor_engine --weights <bin> [--threads N] [--seqlen N]\n"
        "                    (--logits <ids.bin> <n> | --bpb <ids.bin> | --bench <n>)\n"); return 1; }
#ifdef _OPENMP
    if(threads<1) threads=1; omp_set_num_threads(threads); omp_set_dynamic(0);
#endif
    model_t M; load(&M,wp);
    if(g_fuse) build_fused(&M);          // BEFORE --lut: the tile-major copy must be of the
                                         // fused matrices, or the LUT path would see the old ones
    if(g_lut){
        if(M.quant!=2) die("--lut requires a --quant packed model (the tile-major copy is a transpose of those bytes)");
        size_t tmb=0; double t0=now_s();
        for(int l=0;l<M.L;l++){ layer_t* L=&M.lay[l];
            mat_t* mm[7];
            if(g_fuse){ mm[0]=&L->qkv; mm[1]=&L->o; mm[2]=&L->gateup; mm[3]=&L->down;
                        mm[4]=mm[5]=mm[6]=NULL; }
            else      { mm[0]=&L->q; mm[1]=&L->k; mm[2]=&L->v; mm[3]=&L->o;
                        mm[4]=&L->gate; mm[5]=&L->up; mm[6]=&L->down; }
            // down_proj is the last entry either way. Its input is the SwiGLU product, whose
            // crest factor (amax/rms) is 15 on average and 70 at worst -- an amax-scaled int8
            // grid leaves it ~4 usable levels of 63, and --lut-diag measures its round-trip
            // error at 2x every other organ's under a per-vector scale (--lut-group 32 removes
            // that asymmetry). --lut-no-down leaves it on the fp32-activation packed kernel.
            const int ndown = g_fuse ? 3 : 6;
            for(int j=0;j<7&&mm[j];j++){ if(j==ndown&&g_lutnodown) continue;
                                  build_tm(mm[j]); tmb+=(size_t)(mm[j]->in/2)*mm[j]->Mpad; } }
        if(!M.tied&&!g_lutnohead){ build_tm(&M.head); tmb+=(size_t)(M.head.in/2)*M.head.Mpad; }
        fprintf(stderr,"  --lut: tile-major replica built, %.1f MB, %.2f s (activations int8, AQ=%d)\n",
                tmb/1048576.0,now_s()-t0,AQ);
    }

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
