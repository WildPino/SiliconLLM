/* Synthetic protected-projection lower screen for pinned gpt-oss-120b geometry.
 * No source weights, no attention/KV, no experts, no LM head. See protocol.
 * Compile with clang -O3 -mavx2 -mfma -fopenmp and the pinned project header.
 */
#include <assert.h>
#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2 1
#include "../phase60/strat01_q4k_q8k.h"
static void require(int ok,const char *why) {
    if(!ok){fprintf(stderr,"core_failure:%s\n",why);exit(2);}
}
static void *alloc(size_t n,size_t width) {
    void *p=calloc(n,width);require(p!=NULL,"binding_allocation");return p;
}
#include "meth388_switch_thread_binding.h"

enum { LAYERS=36, D=2880, PAD=3072, Q=4096, KV=512, OIN=4096,
       ROUTES=128, TOPK=4, NB=PAD/256, FIXTURES=8, REPS=3 };
typedef struct { uint8_t *qkv, *out; float *router, *rbias, *qkv_bias, *out_bias,
    *anorm, *mnorm, *sink; } Layer;
static Layer layers[LAYERS];
static uint64_t seed=UINT64_C(0x120b20261015);
static uint64_t step(void) { seed^=seed>>12;seed^=seed<<25;seed^=seed>>27;
    return seed*UINT64_C(2685821657736338717); }
static float unit(void) { return (float)((step()>>40)&0xffffffu)/16777216.0f; }
static void *owned(size_t n) { void *p=_aligned_malloc(n,64);if(!p){fprintf(stderr,"allocation failed\n");exit(2);}return p; }
static double seconds(void) { static LARGE_INTEGER frequency={0};LARGE_INTEGER tick;
    if(!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&tick);return (double)tick.QuadPart/(double)frequency.QuadPart; }
static uint64_t mix(uint64_t h,const void *pointer,size_t count) {
    const uint8_t *p=(const uint8_t*)pointer;for(size_t i=0;i<count;i++){h^=p[i];h*=UINT64_C(1099511628211);}return h;
}
static void fill_q4(uint8_t *p,size_t bytes) {
    assert(bytes%144u==0);
    for(size_t b=0;b<bytes/144u;b++) {
        uint8_t *v=p+b*144u;
        /* Valid positive finite binary16 d=2^-10, dmin=2^-12. */
        v[0]=0;v[1]=0x14;v[2]=0;v[3]=0x0c;
        /* All eight six-bit scales/minima in valid Q4_K packing. */
        for(int j=0;j<12;j++)v[4+j]=(uint8_t)(step()&0xffu);
        for(int j=0;j<128;j++)v[16+j]=(uint8_t)(step()&0xffu);
    }
}
static void init(void) {
    const size_t qkv_bytes=(size_t)(Q+2*KV)*NB*144u;
    const size_t out_bytes=(size_t)D*(OIN/256)*144u;
    for(int l=0;l<LAYERS;l++) {
        Layer *a=&layers[l];a->qkv=owned(qkv_bytes);a->out=owned(out_bytes);
        a->router=owned((size_t)ROUTES*D*sizeof(float));a->rbias=owned(ROUTES*sizeof(float));
        a->qkv_bias=owned((Q+2*KV)*sizeof(float));a->out_bias=owned(D*sizeof(float));
        a->anorm=owned(D*sizeof(float));a->mnorm=owned(D*sizeof(float));a->sink=owned(64*sizeof(float));
        fill_q4(a->qkv,qkv_bytes);fill_q4(a->out,out_bytes);
        for(size_t j=0;j<(size_t)ROUTES*D;j++)a->router[j]=(unit()-.5f)*.002f;
        for(int j=0;j<ROUTES;j++)a->rbias[j]=(unit()-.5f)*.02f;
        for(int j=0;j<Q+2*KV;j++)a->qkv_bias[j]=(unit()-.5f)*.002f;
        for(int j=0;j<D;j++)a->out_bias[j]=(unit()-.5f)*.002f;
        for(int j=0;j<D;j++){a->anorm[j]=.9f+.2f*unit();a->mnorm[j]=.9f+.2f*unit();}
        for(int j=0;j<64;j++)a->sink[j]=(unit()-.5f)*.01f;
    }
}
static void norm(const float *x,const float *weight,float *y) {
    double sum=0;for(int i=0;i<D;i++)sum+=(double)x[i]*x[i];
    float scale=1.0f/sqrtf((float)(sum/D)+1e-5f);
    for(int i=0;i<D;i++)y[i]=x[i]*scale*weight[i];
}
static void q8input(const float *x,int count,strat01_q8_k_block *out) {
    float padded[OIN]={0};int packed=count==D?PAD:count;
    assert(count==D || count==OIN);
    memcpy(padded,x,(size_t)count*sizeof(float));
    if(!strat01_quantize_q8_k_row(padded,(unsigned)packed,out)){fprintf(stderr,"invalid Q8 input\n");exit(3);}
}
static void project(const uint8_t *weights,int rows,int cols,
                    const strat01_q8_k_block *input,float *output,int reference) {
    const size_t stride=(size_t)(cols/256)*144u;
    if(reference) {
        for(int r=0;r<rows;r++)output[r]=strat01_q4k_q8k_dot_reference_generic(weights+(size_t)r*stride,input,(unsigned)cols);
    } else {
        #pragma omp parallel num_threads(3)
        {
            require(omp_get_num_threads()==3,"project_team");worker_bind();
            #pragma omp for schedule(static)
            for(int r=0;r<rows;r++)output[r]=strat01_q4k_q8k_dot(weights+(size_t)r*stride,input,(unsigned)cols);
        }
    }
}
static void route(const Layer *a,const float *input,int *choice,float *prob) {
    float logits[ROUTES];
    #pragma omp parallel num_threads(3)
    {
        require(omp_get_num_threads()==3,"router_team");worker_bind();
        #pragma omp for schedule(static)
        for(int r=0;r<ROUTES;r++) {
            const float *w=a->router+(size_t)r*D;float sum=a->rbias[r];
            for(int i=0;i<D;i++)sum+=w[i]*input[i];
            logits[r]=sum;
        }
    }
    /* Stable top-k, descending logit, low index wins ties. */
    for(int k=0;k<TOPK;k++) {
        int best=-1;for(int r=0;r<ROUTES;r++) {
            int used=0;for(int j=0;j<k;j++)used|=(choice[j]==r);
            if(!used && (best<0 || logits[r]>logits[best]))best=r;
        } choice[k]=best;
    }
    float z=0;for(int k=0;k<TOPK;k++){prob[k]=expf(logits[choice[k]]-logits[choice[0]]);z+=prob[k];}
    for(int k=0;k<TOPK;k++)prob[k]/=z;
}
static void route_scalar_control(const Layer *a,const float *input,
                                 const int *choice,const float *prob,int layer) {
    float logits[ROUTES],scores[TOPK],mass=0;int indices[ROUTES];
    for(int r=0;r<ROUTES;r++) {
        float value=a->rbias[r];const float *w=a->router+(size_t)r*D;
        for(int i=0;i<D;i++)value+=w[i]*input[i];
        logits[r]=value;indices[r]=r;
    }
    /* Independent stable partial insertion sort over all 128 candidates. */
    for(int k=0;k<TOPK;k++)for(int r=k+1;r<ROUTES;r++)
        if(logits[indices[r]]>logits[indices[k]]) {
            int temp=indices[k];indices[k]=indices[r];indices[r]=temp;
        }
    for(int k=0;k<TOPK;k++) {
        scores[k]=expf(logits[indices[k]]-logits[indices[0]]);mass+=scores[k];
    }
    float observed=0;
    for(int k=0;k<TOPK;k++) {
        observed+=prob[k];
        if(choice[k]!=indices[k]||fabsf(prob[k]-scores[k]/mass)>1e-6f) {
            fprintf(stderr,"router control failed layer=%d slot=%d\n",layer,k);exit(6);
        }
    }
    if(!isfinite(observed)||fabsf(observed-1.0f)>1e-6f)exit(6);
}
static void compare(const float *active,const float *scalar,int n,int layer,const char *name) {
    for(int i=0;i<n;i++) {
        float limit=2e-4f+2e-4f*fabsf(scalar[i]);
        if(!isfinite(active[i])||!isfinite(scalar[i])||fabsf(active[i]-scalar[i])>limit) {
            fprintf(stderr,"numerical control failed layer=%d %s row=%d active=%g scalar=%g\n",
                    layer,name,i,active[i],scalar[i]);exit(5);
        }
    }
}
static uint64_t forward(int fixture,int validate,double *operator_seconds) {
    float x[D],ax[D],mx[D],qkv[Q+2*KV],out[D],oin[OIN];
    strat01_q8_k_block inq[NB],outq[OIN/256];
    uint64_t digest=UINT64_C(1469598103934665603);
    double operator_sum=0;
    for(int i=0;i<D;i++)x[i]=sinf((float)(i+1)*(fixture+1)*.001f)*.03f;
    for(int l=0;l<LAYERS;l++) {
        Layer *a=&layers[l];int selected[TOPK];float probability[TOPK];
        double began=operator_seconds?seconds():0;
        norm(x,a->anorm,ax);q8input(ax,D,inq);
        project(a->qkv,Q+2*KV,PAD,inq,qkv,0);
        for(int i=0;i<Q+2*KV;i++)qkv[i]+=a->qkv_bias[i];
        /* Deliberately favorable: Q becomes O input; no RoPE, KV attention,
           sink softmax, cache reads or causal masking is charged. K/V projection
           output is nevertheless fully computed and hashed. */
        for(int i=0;i<OIN;i++)oin[i]=qkv[i]*.01f;
        q8input(oin,OIN,outq);
        project(a->out,D,OIN,outq,out,0);
        for(int i=0;i<D;i++)out[i]+=a->out_bias[i];
        if(validate) {
            float *scalar=owned(sizeof(qkv));
            project(a->qkv,Q+2*KV,PAD,inq,scalar,1);
            for(int i=0;i<Q+2*KV;i++)scalar[i]+=a->qkv_bias[i];
            compare(qkv,scalar,Q+2*KV,l,"QKV");_aligned_free(scalar);
            scalar=owned(sizeof(out));project(a->out,D,OIN,outq,scalar,1);
            for(int i=0;i<D;i++)scalar[i]+=a->out_bias[i];
            compare(out,scalar,D,l,"O");_aligned_free(scalar);
        }
        for(int i=0;i<D;i++)x[i]+=out[i];
        norm(x,a->mnorm,mx);route(a,mx,selected,probability);
        if(operator_seconds)operator_sum+=seconds()-began;
        if(validate)route_scalar_control(a,mx,selected,probability,l);
        for(int i=0;i<Q+2*KV;i++)if(!isfinite(qkv[i]))exit(4);
        for(int i=0;i<D;i++)if(!isfinite(out[i]))exit(4);
        for(int k=0;k<TOPK;k++)if(!isfinite(probability[k]))exit(4);
        digest=mix(digest,qkv,sizeof(qkv));digest=mix(digest,out,sizeof(out));
        digest=mix(digest,selected,sizeof(selected));digest=mix(digest,probability,sizeof(probability));
    }
    if(operator_seconds)*operator_seconds=operator_sum;
    return digest;
}
int main(int argc,char **argv) {
    if(argc!=3){fprintf(stderr,"usage: protected_core_120b_cpu.exe 3 RUN_ID\n");return 2;}
    int threads=atoi(argv[1]),run=atoi(argv[2]);if(threads!=3||run<0||run>2)return 2;
    worker_setup(3);
    require(worker_masks[0]==1&&worker_masks[1]==4&&worker_masks[2]==16,"physical_0_2_4_binding");
    init();
    /* Untimed scalar-vs-AVX2 checks every QKV/O output row, every layer. */
    uint64_t control=forward(0,1,NULL);
    fprintf(stdout,"{\"kind\":\"control\",\"run\":%d,\"threads\":%d,\"hash\":\"%016" PRIx64 "\",\"q4_bytes\":%zu,\"router_f32_bytes\":%zu",
            run,threads,control,(size_t)LAYERS*((size_t)(Q+2*KV)*NB+(size_t)D*(OIN/256))*144u,
            (size_t)LAYERS*ROUTES*D*sizeof(float));worker_json();puts("}");fflush(stdout);
    for(int rep=0;rep<REPS;rep++)for(int fixture=0;fixture<FIXTURES;fixture++) {
        double operator_time=0,began=seconds();uint64_t h=forward(fixture,0,&operator_time);
        double elapsed=seconds()-began;
        fprintf(stdout,"{\"kind\":\"timed\",\"run\":%d,\"threads\":%d,\"rep\":%d,\"fixture\":%d,\"operator_seconds\":%.9f,\"wall_seconds\":%.9f,\"hash\":\"%016" PRIx64 "\"}\n",run,threads,rep,fixture,operator_time,elapsed,h);fflush(stdout);
    }
    printf("{\"kind\":\"end\",\"run\":%d",run);worker_json();puts("}");fflush(stdout);
    return 0;
}
