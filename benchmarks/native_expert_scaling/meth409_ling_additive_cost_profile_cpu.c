// M408 Ling source-sized additive operator stream: ALL values synthetic.
// Cost rejection screen; no causal attention/cache or pretrained quality claim.
#define _WIN32_WINNT 0x0601
#define WIN32_LEAN_AND_MEAN
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>
#include <psapi.h>
#include <bcrypt.h>
#include "../phase60/strat01_q6k_q8k_avx2.h"
#include "../phase60/strat01_f32_dot_reference_generic.h"
#define LAYERS 20
#define D 2048
#define PARENTS 256
#define BANKS 256
#define TOPK 8
#define HEADS 16
#define KV_HEADS 4
#define HD 128
#define H 512
#define SH 512
#define DH 5120
#define MAX_INPUT 5120
#define V 157184
#define QKV 3072
#define HEAD_ROW_BYTES ((D/256)*210)
typedef struct { int8_t q[MAX_INPUT];int32_t sum_x;float s;unsigned n; } Input;
typedef struct {
    unsigned d,o,b,nactive,nquery,ids[32];uint64_t seed;
    uint8_t *w;int8_t *palettes;float *s,*y;const Input *input;
} Matrix;
typedef struct {
    Matrix qkv,wo,g,u,down,sg,su,sd;
    float an[D],fn[D],qn[HD],kn[HD],router[PARENTS*D],bias[PARENTS];
    float ra[D],rf[D],context[D],xa[D],xf[D],qkn[D+KV_HEADS*HD],zg[TOPK*DH],sz[SH],mixture[D];
    Input qa,qf,qo,qz[TOPK],qs;
    unsigned parent[TOPK];float probs[PARENTS],gates[TOPK];
} Layer;
static Layer layers[LAYERS];static uint8_t *head,*embedding;
static unsigned current_token;static float embedding_row[D],final_norm[D],head_input[D],head_logits[V];
static strat01_q8_k_block hq[D/256];
static uint64_t allocated,active_i8,active_scales,code_capacity,active_palettes;
static double started;static int requested_threads;
static uint64_t executed_coeff,executed_scales,executed_palettes,executed_router;
/* M409_BEGIN */// M409 inserts timing/accounting only.
static double component_seconds[6];
static uint64_t component_calls[6],component_coeff[4];
/* M409_END */static void die(const char *s){fprintf(stderr,"M408 apparatus failure:%s\n",s);exit(2);}
static void require(int ok,const char *s){if(!ok)die(s);}
static void *alloc(size_t n,size_t w){void *p=calloc(n,w);require(p!=NULL,"worker_alloc");return p;}
#include "meth388_switch_thread_binding.h"
static double clock_s(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart/(double)f.QuadPart;}
static uint64_t mix(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static float fixture(uint64_t x){return ((int)(mix(x)&511)-256)*(1.0f/256.0f);}
static void *allocate(uint64_t n){if(!n || n>UINT64_C(6)*1024*1024*1024)die("allocation range");void *p=_aligned_malloc((size_t)n,64);if(!p)die("allocation failed");allocated+=n;return p;}
static uint64_t peak_rss(void){PROCESS_MEMORY_COUNTERS p;memset(&p,0,sizeof(p));p.cb=sizeof(p);require(GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p)),"rss");return p.PeakWorkingSetSize;}
static void guards(void){require(clock_s()-started<=120 && peak_rss()<=UINT64_C(6)*1024*1024*1024,"process_2min_6GiB");}
static int8_t palette_value(uint64_t seed,uint64_t j){return (int8_t)((int)(mix(j+seed+1234567)%127)-63);}
static void make_matrix(Matrix *m,unsigned d,unsigned o,unsigned banks,unsigned active,unsigned queries,uint64_t seed){
    if(d%32 || d>MAX_INPUT || !o || o%4 || active>32 || !queries || queries>active)die("matrix geometry");
    m->d=d;m->o=o;m->b=banks;m->nactive=active;m->nquery=queries;m->seed=seed;
    uint64_t bytes=(uint64_t)d*o*banks/4;
    m->w=allocate(bytes);m->palettes=allocate((uint64_t)banks*4096);m->s=allocate((uint64_t)o*banks*4);m->y=allocate((uint64_t)o*active*4);
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)bytes/8;j++){uint64_t z=mix((uint64_t)j+seed);memcpy(m->w+j*8,&z,8);}
    for(uint64_t j=0;j<(uint64_t)banks*4096;j++)m->palettes[j]=palette_value(seed,j);
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)o*banks;j++)m->s[j]=(1.0f+(float)(mix((uint64_t)j+seed)&127)/128.0f)/(126.0f*sqrtf((float)d));
    for(unsigned i=0;i<active;i++)m->ids[i]=i;
    active_i8+=(uint64_t)d*o*active;active_scales+=(uint64_t)o*active*4;code_capacity+=bytes*4;active_palettes+=(uint64_t)active*4096;
}
static int quantize(const float *x,unsigned n,Input *q){
    if(!n||n>MAX_INPUT||n%32)return 0;float max=0;
    for(unsigned j=0;j<n;j++){if(!isfinite(x[j]))return 0;float a=fabsf(x[j]);if(a>max)max=a;}
    q->n=n;q->s=max>0?max/63.0f:1.0f;if(!isfinite(q->s)||q->s<=0)return 0;
    float inverse=max>0?63.0f/max:1.0f;if(!isfinite(inverse))return 0;
    for(unsigned j=0;j<n;j++){float v=nearbyintf(x[j]*inverse);if(v>63)v=63;if(v< -63)v=-63;q->q[j]=(int8_t)v;}
    q->sum_x=0;for(unsigned j=0;j<n;j++)q->sum_x+=q->q[j];return 1;
}
// Each group of eight decoded weights has two U8 palette indices.
static __m128i decode8(const uint8_t *w,const int8_t *p){
    return _mm_add_epi8(_mm_loadl_epi64((const __m128i*)(p+(unsigned)w[0]*8)),_mm_loadl_epi64((const __m128i*)(p+2048+(unsigned)w[1]*8)));
}
static __m256i decode32(const uint8_t *w,const int8_t *p){
    __m128i lo=_mm_unpacklo_epi64(decode8(w,p),decode8(w+2,p)),hi=_mm_unpacklo_epi64(decode8(w+4,p),decode8(w+6,p));
    return _mm256_xor_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1),_mm256_set1_epi8((char)128));
}
static int32_t horizontal(__m256i v){int32_t values[8];_mm256_storeu_si256((__m256i*)values,v);int32_t sum=0;for(unsigned i=0;i<8;i++)sum+=values[i];return sum;}
static int32_t idot(const uint8_t *w,const int8_t *p,const Input *x,unsigned n){
    __m256i a=_mm256_setzero_si256(),ones=_mm256_set1_epi16(1);
    for(unsigned j=0;j<n;j+=32){__m256i pairs=_mm256_maddubs_epi16(decode32(w+j/4,p),_mm256_loadu_si256((const __m256i*)(x->q+j)));a=_mm256_add_epi32(a,_mm256_madd_epi16(pairs,ones));}
    return horizontal(a)-128*x->sum_x;
}
static void idot4(const uint8_t *w,const int8_t *p,const Input *x,unsigned n,int32_t result[4]){
    __m256i a[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()},ones=_mm256_set1_epi16(1);
    for(unsigned j=0;j<n;j+=32){
        __m256i input=_mm256_loadu_si256((const __m256i*)(x->q+j));
        #pragma clang loop unroll(full)
        for(unsigned r=0;r<4;r++){__m256i pairs=_mm256_maddubs_epi16(decode32(w+(size_t)r*n/4+j/4,p),input);a[r]=_mm256_add_epi32(a[r],_mm256_madd_epi16(pairs,ones));}
    }
    for(unsigned r=0;r<4;r++)result[r]=horizontal(a[r])-128*x->sum_x;
}
static void apply_tile(Matrix *m,const Input *input,unsigned tile){
    worker_bind();
    unsigned r=tile*4,k=r/m->o,row=r%m->o,bank=m->ids[k],query=m->nquery==1?0:k;const Input *x=&input[query];int32_t sums[4];
    idot4(m->w+((size_t)bank*m->o+row)*m->d/4,m->palettes+(size_t)bank*4096,x,m->d,sums);
    for(unsigned j=0;j<4;j++)m->y[r+j]=(float)sums[j]*(x->s*m->s[(size_t)bank*m->o+row+j]);
}
static __attribute__((noinline)) int apply(Matrix *m,const Input *input){
    for(unsigned k=0;k<m->nactive;k++)if(m->ids[k]>=m->b)return 0;
    for(unsigned k=0;k<m->nquery;k++)if(input[k].n!=m->d)return 0;m->input=input;unsigned tiles=m->o*m->nactive/4;
    if((uint64_t)m->d*m->o*m->nactive<32768){for(unsigned t=0;t<tiles;t++)apply_tile(m,input,t);}
    else{
        #pragma omp parallel for schedule(static)
        for(int t=0;t<(int)tiles;t++)apply_tile(m,input,(unsigned)t);
    }
    return 1;
}
static void checked_apply(Matrix *m,const Input *q){/* M409_BEGIN */
unsigned cat=m->b==PARENTS?2:(m->d==DH||m->o==DH)?1:(m->d==H||m->o==H)?3:0;
double component_start=clock_s();
/* M409_END */if(!apply(m,q))die("bank/input range");executed_coeff+=(uint64_t)m->d*m->o*m->nactive;executed_scales+=(uint64_t)m->o*m->nactive*4;executed_palettes+=(uint64_t)m->nactive*4096;/* M409_BEGIN */
component_seconds[cat]+=clock_s()-component_start;component_calls[cat]++;
component_coeff[cat]+=(uint64_t)m->d*m->o*m->nactive;
/* M409_END */}
static void checked_quant(const float *x,unsigned n,Input *q){if(!quantize(x,n,q))die("activation quantization");}
static void norm(const float *x,const float *w,unsigned n,float *y){double sum=0;for(unsigned j=0;j<n;j++)sum+=(double)x[j]*x[j];float r=(float)(1.0/sqrt(sum/n+1e-6));for(unsigned j=0;j<n;j++)y[j]=(x[j]*r)*w[j];}
static void initialize(void){
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];uint64_t z=(uint64_t)(l+1)*UINT64_C(10000000000);
        for(unsigned j=0;j<D;j++){s->an[j]=1+.1f*fixture(z+j);s->fn[j]=1+.1f*fixture(z+3000+j);}
        for(unsigned j=0;j<HD;j++){s->qn[j]=1+.1f*fixture(z+6000+j);s->kn[j]=1+.1f*fixture(z+7000+j);}
        if(l){
            #pragma omp parallel for schedule(static)
            for(int r=0;r<PARENTS;r++)for(unsigned j=0;j<D;j++)s->router[(size_t)r*D+j]=fixture(z+100000+(size_t)r*D+j)/sqrtf((float)D);
            for(unsigned r=0;r<PARENTS;r++)s->bias[r]=.01f*fixture(z+8000+r);
        }
        make_matrix(&s->qkv,D,QKV,1,1,1,z+1);make_matrix(&s->wo,D,D,1,1,1,z+2);
        if(l){
            make_matrix(&s->g,D,H,BANKS,TOPK,1,z+3);make_matrix(&s->u,D,H,BANKS,TOPK,1,z+4);make_matrix(&s->down,H,D,BANKS,TOPK,TOPK,z+5);
            make_matrix(&s->sg,D,SH,1,1,1,z+6);make_matrix(&s->su,D,SH,1,1,1,z+7);make_matrix(&s->sd,SH,D,1,1,1,z+8);
        }else{make_matrix(&s->g,D,DH,1,1,1,z+3);make_matrix(&s->u,D,DH,1,1,1,z+4);make_matrix(&s->down,DH,D,1,1,1,z+5);}
    }
    head=allocate((uint64_t)V*HEAD_ROW_BYTES);
    #pragma omp parallel for schedule(static)
    for(int64_t b=0;b<(int64_t)V*(D/256);b++){
        uint8_t *p=head+(size_t)b*210;for(unsigned j=0;j<26;j++){uint64_t v=mix((uint64_t)b*31+j+900000);memcpy(p+j*8,&v,8);}
        for(unsigned j=0;j<16;j++)p[192+j]=(uint8_t)(int8_t)((int)(mix((uint64_t)b*19+j)%17)-8);
        p[208]=0;p[209]=0x18; // finite positive F16 scale 2^-9
    }
    embedding=allocate((uint64_t)V*D*2);
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)V*D;j++){
        float value=fixture((uint64_t)j+UINT64_C(1234567890123));uint32_t bits;memcpy(&bits,&value,4);
        uint16_t upper=(uint16_t)(bits>>16);memcpy(embedding+j*2,&upper,2);
    }
    for(unsigned j=0;j<D;j++)final_norm[j]=1+.1f*fixture(700000+j);
    require(active_i8==779091968 && active_scales==2560000 && active_palettes==2277376,"active_407_ledger");
    require(allocated==UINT64_C(4931534848),"dynamic_payload_output_ledger");guards();
}
static void prepare(unsigned token){
    current_token=token;
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];uint64_t seed=(uint64_t)token*UINT64_C(1000000000)+l*UINT64_C(10000000);
        for(unsigned j=0;j<D;j++){s->ra[j]=fixture(seed+j);s->rf[j]=fixture(seed+10000+j);s->context[j]=fixture(seed+20000+j);}
    }
}
typedef struct {float score;unsigned id;} Rank;
static int rank_compare(const void *a,const void *b){
    const Rank *x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;
    return x->id<y->id?-1:x->id>y->id?1:0;
}
static void macro_router(Layer *s){/* M409_BEGIN */double component_start=clock_s();/* M409_END */
    executed_router+=(uint64_t)(PARENTS*D+PARENTS)*4;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<PARENTS;r++){worker_bind();float v=strat01_f32_dot_reference_generic(s->router+(size_t)r*D,s->xf,D);s->probs[r]=1.0f/(1.0f+expf(-v));}
    float groups[8];unsigned keep[8]={0};
    for(unsigned g=0;g<8;g++){
        float a=-INFINITY,b=-INFINITY;
        for(unsigned j=0;j<32;j++){unsigned id=g*32+j;float v=s->probs[id]+s->bias[id];if(v>a){b=a;a=v;}else if(v>b)b=v;}
        groups[g]=a+b;
    }
    for(unsigned k=0;k<4;k++){unsigned best=UINT32_MAX;for(unsigned g=0;g<8;g++)if(!keep[g]&&(best==UINT32_MAX||groups[g]>groups[best]))best=g;keep[best]=1;}
    for(unsigned k=0;k<TOPK;k++){
        unsigned best=UINT32_MAX;
        for(unsigned r=0;r<PARENTS;r++)if(keep[r/32]){
            int used=0;for(unsigned j=0;j<k;j++)if(s->parent[j]==r)used=1;
            if(!used&&(best==UINT32_MAX||s->probs[r]+s->bias[r]>s->probs[best]+s->bias[best]))best=r;
        }
        require(best<PARENTS,"router_selected_id");s->parent[k]=best;
    }
    float total=0;for(unsigned k=0;k<TOPK;k++)total+=s->probs[s->parent[k]];
    require(total>0,"router_mass");
    for(unsigned k=0;k<TOPK;k++)s->gates[k]=(s->probs[s->parent[k]]/(total+1e-20f))*2.5f;/* M409_BEGIN */component_seconds[4]+=clock_s()-component_start;component_calls[4]++;/* M409_END */
}
static float silu(float x){return x/(1.0f+expf(-x));}
static void execute(void){
/* M409_BEGIN */    memset(component_seconds,0,sizeof(component_seconds));memset(component_calls,0,sizeof(component_calls));memset(component_coeff,0,sizeof(component_coeff));
/* M409_END */    executed_coeff=executed_scales=executed_palettes=executed_router=0;
    for(unsigned j=0;j<D;j++){uint16_t bits;memcpy(&bits,embedding+((size_t)current_token*D+j)*2,2);uint32_t f=(uint32_t)bits<<16;memcpy(embedding_row+j,&f,4);layers[0].ra[j]=embedding_row[j];}
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];norm(s->ra,s->an,D,s->xa);checked_quant(s->xa,D,&s->qa);checked_apply(&s->qkv,&s->qa);
        for(unsigned h=0;h<HEADS;h++)norm(s->qkv.y+h*HD,s->qn,HD,s->qkn+h*HD);
        for(unsigned h=0;h<KV_HEADS;h++)norm(s->qkv.y+D+h*HD,s->kn,HD,s->qkn+D+h*HD);
        // Context is a declared fixture; no attention/RoPE/cache/composition.
        checked_quant(s->context,D,&s->qo);checked_apply(&s->wo,&s->qo);
        norm(s->rf,s->fn,D,s->xf);checked_quant(s->xf,D,&s->qf);
        if(l){macro_router(s);for(unsigned k=0;k<TOPK;k++)s->g.ids[k]=s->u.ids[k]=s->down.ids[k]=s->parent[k];}
        checked_apply(&s->g,&s->qf);checked_apply(&s->u,&s->qf);
        for(unsigned k=0;k<s->g.nactive;k++){
            for(unsigned j=0;j<s->g.o;j++)s->zg[k*s->g.o+j]=silu(s->g.y[k*s->g.o+j])*s->u.y[k*s->g.o+j];
            checked_quant(s->zg+k*s->g.o,s->g.o,&s->qz[k]);
        }
        checked_apply(&s->down,s->qz);
        if(l){checked_apply(&s->sg,&s->qf);checked_apply(&s->su,&s->qf);
            for(unsigned j=0;j<SH;j++)s->sz[j]=silu(s->sg.y[j])*s->su.y[j];
            checked_quant(s->sz,SH,&s->qs);checked_apply(&s->sd,&s->qs);}
        for(unsigned j=0;j<D;j++){float v=l?s->sd.y[j]:s->down.y[j];if(l)for(unsigned k=0;k<TOPK;k++)v+=s->gates[k]*s->down.y[k*D+j];s->mixture[j]=v;}
    }
/* M409_BEGIN */    double head_start=clock_s();
/* M409_END */    Layer *s=&layers[LAYERS-1];for(unsigned j=0;j<D;j++)head_input[j]=(s->xf[j]+s->wo.y[j])+s->mixture[j];
    norm(head_input,final_norm,D,head_input);require(strat01_quantize_q8_k_row(head_input,D,hq),"head_quant");
    #pragma omp parallel for schedule(static)
    for(int r=0;r<V;r++){worker_bind();head_logits[r]=strat01_q6k_q8k_dot_avx2(head+(size_t)r*HEAD_ROW_BYTES,hq,D);}/* M409_BEGIN */component_seconds[5]+=clock_s()-head_start;component_calls[5]++;/* M409_END */
}
static Matrix *matrix(Layer *s,unsigned i){Matrix *m[]={&s->qkv,&s->wo,&s->g,&s->u,&s->down,&s->sg,&s->su,&s->sd};return m[i];}
static void part(BCRYPT_HASH_HANDLE h,FILE *f,const void *p,size_t n){
    require(n<=UINT32_MAX&&BCryptHashData(h,(PUCHAR)p,(ULONG)n,0)>=0,"hash_part");if(f)require(fwrite(p,1,n,f)==n,"write_output");
}
static void floats(BCRYPT_HASH_HANDLE h,FILE *f,const float *p,unsigned n){for(unsigned j=0;j<n;j++)require(isfinite(p[j]),"finite_output");part(h,f,p,(size_t)n*4);}
static void output(const char *prefix,unsigned rep,unsigned token,char sha[65]){
    BCRYPT_ALG_HANDLE a=NULL;BCRYPT_HASH_HANDLE h=NULL;uint8_t digest[32];FILE *f=NULL;
    if(rep==0){char path[2048];require(snprintf(path,sizeof(path),"%s.input%u.bin",prefix,token)>0,"output_path");f=fopen(path,"wb");require(f!=NULL,"output_open");}
    require(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,NULL,0)>=0&&BCryptCreateHash(a,&h,NULL,0,NULL,0,0)>=0,"output_hash_create");
    const uint32_t dims[]={token,D,LAYERS,PARENTS,TOPK,V};part(h,f,"S408OUT1",8);part(h,f,dims,sizeof(dims));
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];unsigned count=l?8:5;part(h,f,&count,4);
        for(unsigned i=0;i<count;i++){Matrix *m=matrix(s,i);uint32_t shape[]={m->d,m->o,m->b,m->nactive,m->nquery};part(h,f,shape,sizeof(shape));part(h,f,m->ids,m->nactive*4);floats(h,f,m->y,m->o*m->nactive);}
        floats(h,f,s->qkn,D+KV_HEADS*HD);floats(h,f,s->mixture,D);
        if(l){part(h,f,s->parent,TOPK*4);floats(h,f,s->probs,PARENTS);floats(h,f,s->gates,TOPK);}
    }
    floats(h,f,head_input,D);floats(h,f,head_logits,V);
    require(BCryptFinishHash(h,digest,32,0)>=0,"output_hash_finish");BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);
    if(f)require(fclose(f)==0,"output_close");for(unsigned j=0;j<32;j++)sprintf(sha+j*2,"%02x",digest[j]);sha[64]=0;
}
static int64_t scalar_dot(const uint8_t *w,const int8_t *p,const int8_t *x,unsigned n){
    int64_t sum=0;for(unsigned j=0;j<n;j++){unsigned group=j/8,lane=j%8;int weight=(int)p[(unsigned)w[group*2]*8+lane]+(int)p[2048+(unsigned)w[group*2+1]*8+lane];sum+=(int64_t)weight*x[j];}return sum;
}
static double decoded_head(unsigned row){
    const uint8_t *raw=head+(size_t)row*HEAD_ROW_BYTES;double total=0;
    for(unsigned b=0;b<D/256;b++,raw+=210){
        double d=strat01_q4k_q8k_fp16le(raw+208)*hq[b].d;
        for(unsigned j=0;j<256;j++){
            unsigned half=j/128,phase=(j%128)/32,lane=j%32;
            unsigned ql=raw[half*64+lane+(phase&1)*32];if(phase>=2)ql>>=4;else ql&=15;
            int q=(int)(ql|(((raw[128+half*32+lane]>>(phase*2))&3)<<4))-32;
            int scale=((const int8_t*)raw)[192+half*8+phase*2+lane/16];total+=d*(double)scale*q*hq[b].qs[j];
        }
    }
    return total;
}
static float reference_f32(const float *x,const float *y,unsigned n){double sum=0;for(unsigned j=0;j<n;j++){volatile float product=x[j]*y[j];sum+=(double)product;}return (float)sum;}
static void route_control(Layer *s){
    Rank groups[8],rank[PARENTS];float probs[PARENTS];unsigned allowed[8]={0};
    for(unsigned g=0;g<8;g++){
        Rank local[32];for(unsigned j=0;j<32;j++){unsigned r=g*32+j;float v=reference_f32(s->router+(size_t)r*D,s->xf,D);probs[r]=1/(1+expf(-v));require(memcmp(probs+r,s->probs+r,4)==0,"router_scalar_probability");local[j]=(Rank){probs[r]+s->bias[r],r};}
        qsort(local,32,sizeof(Rank),rank_compare);groups[g]=(Rank){local[0].score+local[1].score,g};
    }
    qsort(groups,8,sizeof(Rank),rank_compare);for(unsigned j=0;j<4;j++)allowed[groups[j].id]=1;
    for(unsigned r=0;r<PARENTS;r++)rank[r]=(Rank){allowed[r/32]?probs[r]+s->bias[r]:-INFINITY,r};
    qsort(rank,PARENTS,sizeof(Rank),rank_compare);float total=0;for(unsigned k=0;k<TOPK;k++)total+=probs[rank[k].id];
    for(unsigned k=0;k<TOPK;k++){
        require(rank[k].id==s->parent[k],"router_group_selection");float gate=(probs[rank[k].id]/(total+1e-20f))*2.5f;
        require(memcmp(&gate,s->gates+k,4)==0,"unbiased_source_scaled_gate");
        require(s->g.ids[k]==rank[k].id&&s->u.ids[k]==rank[k].id&&s->down.ids[k]==rank[k].id,"selected_bank_binding");
    }
}
static void set_weight_lane(int8_t *p,unsigned lane,int value){
    int first=value>63?63:value< -63?-63:value;p[lane]=(int8_t)first;p[2048+lane]=(int8_t)(value-first);
}
static void selftest(void){
    uint8_t w[4*MAX_INPUT/4];int8_t p[4096];Input x;x.n=MAX_INPUT;memset(w,0,sizeof(w));memset(p,63,sizeof(p));
    memset(x.q,63,MAX_INPUT);x.sum_x=63*MAX_INPUT;
    if(idot(w,p,&x,MAX_INPUT)!=scalar_dot(w,p,x.q,MAX_INPUT)||idot(w,p,&x,MAX_INPUT)!=126*63*MAX_INPUT)die("integer maximum");
    memset(p,-63,sizeof(p));if(idot(w,p,&x,MAX_INPUT)!=-126*63*MAX_INPUT)die("integer minimum");
    uint64_t products=0,pairs=0;x.n=32;
    for(int weight=-126;weight<=126;weight++)for(int value=-63;value<=63;value++){
        memset(p,0,sizeof(p));for(unsigned j=0;j<8;j++)set_weight_lane(p,j,weight);
        memset(x.q,value,32);x.sum_x=32*value;int32_t sums[4];idot4(w,p,&x,32,sums);
        for(unsigned r=0;r<4;r++)if(sums[r]!=scalar_dot(w+r*8,p,x.q,32)||sums[r]!=idot(w+r*8,p,&x,32)||sums[r]!=(int64_t)32*weight*value)die("exhaustive decoded/input tile lane");products++;
    }
    const int weights[11]={-126,-125,-64,-63,-1,0,1,63,64,125,126};
    const int values[11]={-63,-62,-3,-2,-1,0,1,2,3,62,63};
    for(unsigned a=0;a<11;a++)for(unsigned b=0;b<11;b++)for(unsigned c=0;c<11;c++)for(unsigned d=0;d<11;d++){
        memset(p,0,sizeof(p));set_weight_lane(p,0,weights[a]);set_weight_lane(p,1,weights[b]);memset(x.q,0,32);
        x.q[0]=(int8_t)values[c];x.q[1]=(int8_t)values[d];x.sum_x=values[c]+values[d];int16_t paired[16];
        _mm256_storeu_si256((__m256i*)paired,_mm256_maddubs_epi16(decode32(w,p),_mm256_loadu_si256((const __m256i*)x.q)));
        if(paired[0]!=(weights[a]+128)*values[c]+(weights[b]+128)*values[d]||idot(w,p,&x,32)!=scalar_dot(w,p,x.q,32))die("mixed decoded pair/correction");pairs++;
    }
    memset(p,63,sizeof(p));memset(x.q,63,32);x.sum_x=32*63;Input bad=x;bad.sum_x++;
    if(idot(w,p,&bad,32)==scalar_dot(w,p,x.q,32))die("bias negative control");
    int64_t clean=scalar_dot(w,p,x.q,32);p[0]--;
    if(idot(w,p,&x,32)==clean)die("palette fault undetected");p[0]++;
    p[8]=62;w[0]=1;if(idot(w,p,&x,32)==clean)die("code fault undetected");w[0]=0;p[8]=63;
    memset(x.q,127,32);x.sum_x=32*127;if(idot(w,p,&x,32)==scalar_dot(w,p,x.q,32))die("unsupported S8 saturation fault undetected");
    if(products!=32131||pairs!=14641)die("complete integer controls");
    float zero[32]={0},tie[32]={63,.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f};Input quant;
    if(!quantize(zero,32,&quant)||quant.s!=1)die("zero quantizer");for(unsigned j=0;j<32;j++)if(quant.q[j])die("zero codes");
    if(!quantize(tie,32,&quant)||quant.q[1]!=0||quant.q[2]!=2||quant.q[3]!=2||quant.q[4]!=0||quant.q[5]!=-2||quant.q[6]!=-2)die("ties-even quantizer");
    tie[0]=NAN;if(quantize(tie,32,&quant))die("nonfinite quantizer accepted");
    prepare(13);execute();uint64_t exact=0,bank_checks=0,palette_checks=0;double e2=0,n2=0,he2=0,hn2=0;
    for(unsigned j=0;j<D;j++){uint16_t bits;memcpy(&bits,embedding+((size_t)13*D+j)*2,2);uint32_t f=(uint32_t)bits*65536;float value;memcpy(&value,&f,4);if(!isfinite(value)||memcmp(&value,embedding_row+j,4))die("embedding decode control");}
    for(unsigned l=1;l<LAYERS;l++)route_control(&layers[l]);
    for(unsigned l=0;l<LAYERS;l++)for(unsigned i=0;i<(l?8U:5U);i++){
        Matrix *m=matrix(&layers[l],i);
        for(unsigned bank=0;bank<m->b;bank++)for(unsigned edge=0;edge<2;edge++){
            size_t byte=(size_t)bank*m->d*m->o/4+(edge?(size_t)m->d*m->o/4-8:0);uint64_t expected=mix(byte/8+m->seed);
            if(memcmp(m->w+byte,&expected,8))die("stored code bank offset");bank_checks++;
            size_t position=(size_t)bank*4096+(edge?4096-8:0);
            for(unsigned j=0;j<8;j++)if(m->palettes[position+j]!=palette_value(m->seed,position+j))die("stored palette bank offset");palette_checks++;
        }
        for(unsigned k=0;k<m->nactive;k++)for(unsigned t=0;t<8;t++){
            unsigned r=t*(m->o-1)/7;const Input *in=&m->input[m->nquery==1?0:k];const uint8_t *weight=m->w+((size_t)m->ids[k]*m->o+r)*m->d/4;const int8_t *pal=m->palettes+(size_t)m->ids[k]*4096;
            int64_t sum=scalar_dot(weight,pal,in->q,m->d);if(sum!=idot(weight,pal,in,m->d))die("exact integer dot");
            float sc=m->s[(size_t)m->ids[k]*m->o+r],v=(float)sum*(in->s*sc);if(memcmp(&v,m->y+k*m->o+r,4))die("scalar output");
            double ref=(double)sum*in->s*sc,e=v-ref;e2+=e*e;n2+=ref*ref;exact++;
        }
    }
    for(unsigned r=0;r<64;r++){
        unsigned row=r*(V-1)/63;double v=decoded_head(row),e=head_logits[row]-v;he2+=e*e;hn2+=v*v;
    }
    Matrix *m=&layers[1].g;unsigned saved=m->ids[0];m->ids[0]=m->b;int rejected=!apply(m,&layers[1].qf);m->ids[0]=saved;
    int detected=0;unsigned sentinel_row=0,sentinel_byte=0;
    for(unsigned r=0;r<64&&!detected;r++)for(unsigned j=0;j<128&&!detected;j++){
        unsigned row=r*(V-1)/63;size_t offset=(size_t)row*HEAD_ROW_BYTES+j;uint8_t old=head[offset];double before=decoded_head(row);
        float native_before=strat01_q6k_q8k_dot_avx2(head+(size_t)row*HEAD_ROW_BYTES,hq,D);head[offset]^=1;
        double after=decoded_head(row);float native_after=strat01_q6k_q8k_dot_avx2(head+(size_t)row*HEAD_ROW_BYTES,hq,D);head[offset]=old;
        if(before!=after&&native_before!=native_after){detected=1;sentinel_row=row;sentinel_byte=j;}
    }
    if(!rejected||!detected||!(n2>0)||!(hn2>0)||sqrt(e2/n2)>1e-6||sqrt(he2/hn2)>1e-5)die("numeric/negative controls");
    printf("{\"event\":\"selftest\",\"exact_integer_rows\":%llu,\"stored_bank_offset_checks\":%llu,\"palette_bank_offset_checks\":%llu,\"integer_relative_l2\":%.17g,\"synthetic_head64_relative_l2\":%.17g,\"reference_route_layers\":19,\"bad_bank_rejected\":true,\"packed_head_change_detected\":true,\"sentinel_row\":%u,\"sentinel_byte\":%u,\"integer_extrema_and_quantizer_checks\":true,\"exhaustive_decoded_input_cases\":32131,\"tile_lane_cases\":128524,\"mixed_adjacent_pairs\":14641,\"row_bias_error_detected\":true,\"palette_fault_detected\":true,\"code_fault_detected\":true,\"S8_saturation_fault_detected\":true,\"embedding2048_exact\":true}\n",(unsigned long long)exact,(unsigned long long)bank_checks,(unsigned long long)palette_checks,sqrt(e2/n2),sqrt(he2/hn2),sentinel_row,sentinel_byte);fflush(stdout);
}

int main(int argc,char **argv){
    require(argc==3,"usage: THREADS OUTPUT_PREFIX");requested_threads=atoi(argv[1]);require(requested_threads==3||requested_threads==6,"profile_threads");
    fesetround(FE_TONEAREST);started=clock_s();worker_setup(requested_threads);initialize();
    printf("{\"event\":\"ready\",\"all_values_synthetic\":true,\"layers\":20,\"dimension\":2048,\"stored_banks\":256,\"selected\":8,\"threads\":%d,\"active_coefficients\":%llu,\"active_scales\":%llu,\"active_palettes\":%llu,\"stored_coded_coefficients\":%llu,\"allocated_dynamic_bytes\":%llu,\"setup_seconds\":%.9f",requested_threads,(unsigned long long)active_i8,(unsigned long long)active_scales,(unsigned long long)active_palettes,(unsigned long long)code_capacity,(unsigned long long)allocated,clock_s()-started);worker_json();puts("}");fflush(stdout);
    selftest();unsigned char selected[LAYERS][BANKS];memset(selected,0,sizeof(selected));
    for(unsigned rep=0;rep<3;rep++)for(unsigned token=0;token<10;token++){
        prepare(token);double t=clock_s();execute();double seconds=clock_s()-t;
/* M409_BEGIN */        double component_sum=0;for(unsigned c=0;c<6;c++)component_sum+=component_seconds[c];
        const uint64_t expected_calls[6]={40,3,57,57,19,1};
        const uint64_t expected_coeff[4]={209715200,31457280,478150656,59768832};
        for(unsigned c=0;c<6;c++)require(component_calls[c]==expected_calls[c]&&component_seconds[c]>0,"component_calls_seconds");
        for(unsigned c=0;c<4;c++)require(component_coeff[c]==expected_coeff[c],"component_coefficients");
        require(component_sum<=seconds+1e-8,"nonoverlapping_component_accounting");
/* M409_END */        uint64_t addressed=executed_coeff/4+executed_scales+executed_palettes+executed_router+(uint64_t)V*HEAD_ROW_BYTES+356352+4096;
        require(executed_coeff==779091968&&executed_scales==2560000&&executed_palettes==2277376&&executed_router==39865344&&addressed==503905280,"executed_full_descriptor");
        char sha[65];output(argv[2],rep,token,sha);
        for(unsigned l=1;l<LAYERS;l++)for(unsigned k=0;k<TOPK;k++)selected[l][layers[l].parent[k]]=1;
/* M409_BEGIN */        printf("{\"event\":\"cost\",\"rep\":%u,\"input\":%u,\"warmup\":%s,\"component_seconds\":[",rep,token,token<2?"true":"false");
        for(unsigned c=0;c<6;c++)printf("%s%.12f",c?",":"",component_seconds[c]);
        printf("],\"calls\":[");for(unsigned c=0;c<6;c++)printf("%s%llu",c?",":"",(unsigned long long)component_calls[c]);
        printf("],\"coded_coefficients\":[");for(unsigned c=0;c<4;c++)printf("%s%llu",c?",":"",(unsigned long long)component_coeff[c]);
        printf("],\"remainder_seconds\":%.12f}\n",seconds-component_sum);
/* M409_END */        printf("{\"event\":\"operator\",\"rep\":%u,\"input\":%u,\"warmup\":%s,\"seconds\":%.12f,\"complete_output_sha256\":\"%s\",\"executed_coefficients\":%llu,\"addressed_descriptor_bytes\":%llu}",rep,token,token<2?"true":"false",seconds,sha,(unsigned long long)executed_coeff,(unsigned long long)addressed);puts("");fflush(stdout);guards();
    }
    unsigned lo=BANKS,hi=0;for(unsigned l=1;l<LAYERS;l++){unsigned n=0;for(unsigned j=0;j<BANKS;j++)n+=selected[l][j];if(n<lo)lo=n;if(n>hi)hi=n;}
    printf("{\"event\":\"finished\",\"minimum_selected_union\":%u,\"maximum_selected_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f",lo,hi,(unsigned long long)peak_rss(),clock_s()-started);worker_json();puts("}");return 0;
}
