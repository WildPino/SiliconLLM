// Exact biased-U8/two-part activation dot on the unchanged303 fixture bank.
// Actual source Q6 head/router/norms; no learned children or causal LM claim.
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

#define LAYERS 26
#define D 1536
#define PARENTS 64
#define CHILDREN 10
#define BANKS 640
#define HEADS 8
#define H 128
#define SH 256
#define DH 1024
#define V 128256
#define QDIM 16
typedef struct { int8_t q[D],hi[D],lo[D];int32_t sum_x;float s;unsigned n; } Input;
typedef struct {
    unsigned d,o,b,nactive,nquery,ids[8];uint64_t seed;
    uint8_t *w;float *s,*y;const Input *input;
} Matrix;
typedef struct {
    Matrix q,kv,kb,vb,wo,g,u,down,sg,su,sd,cq;
    float an[D],fn[D],kn[512],router[PARENTS*D],bias[PARENTS];
    float ra[D],rf[D],latent[HEADS*512],xa[D],xf[D],kvnorm[512],zg[4*DH],sz[SH],mixture[D];
    Input qa,qf,qk[HEADS],qv[HEADS],qo,qz[4],qs;
    unsigned parent[4],child[4],flat[4];float probs[PARENTS],gates[4],*keys;
} Layer;
static Layer layers[LAYERS];
static uint8_t *head;
static float final_norm[D],head_input[D],head_logits[V];
static strat01_q8_k_block hq[D/256];
static uint64_t allocated,active_i8,active_scales,code_capacity;
static double started;

static void die(const char *s){fprintf(stderr,"M305 apparatus failure: %s\n",s);exit(2);}
static double clock_s(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart/(double)f.QuadPart;}
static uint64_t mix(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static float fixture(uint64_t x){return ((int)(mix(x)&511)-256)*(1.0f/256.0f);}
static void *allocate(uint64_t n){if(!n || n>UINT64_C(24)*1024*1024*1024)die("allocation range");void *p=_aligned_malloc((size_t)n,64);if(!p)die("allocation failed");allocated+=n;return p;}
static uint64_t peak_rss(void){PROCESS_MEMORY_COUNTERS p;memset(&p,0,sizeof(p));p.cb=sizeof(p);if(!GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p)))die("RSS query");return p.PeakWorkingSetSize;}
static void guards(void){if(clock_s()-started>600 || peak_rss()>UINT64_C(24)*1024*1024*1024)die("resource stop");}
static void read_exact(FILE *f,void *p,size_t n){if(fread(p,1,n,f)!=n)die("short input");}
static void payload(FILE *f,uint64_t offset,void *p,size_t n){if(_fseeki64(f,(int64_t)offset,SEEK_SET))die("seek");read_exact(f,p,n);}
static void verify_sha(const void **parts,const size_t *sizes,unsigned count,const uint8_t expected[32]){
    BCRYPT_ALG_HANDLE a=NULL;BCRYPT_HASH_HANDLE h=NULL;uint8_t value[32];
    if(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,NULL,0)<0 || BCryptCreateHash(a,&h,NULL,0,NULL,0,0)<0)die("SHA create");
    for(unsigned i=0;i<count;i++)if(sizes[i]>UINT32_MAX || BCryptHashData(h,(PUCHAR)parts[i],(ULONG)sizes[i],0)<0)die("SHA data");
    if(BCryptFinishHash(h,value,32,0)<0)die("SHA finish");BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);
    if(memcmp(value,expected,32))die("source payload SHA mismatch");
}
static void make_matrix(Matrix *m,unsigned d,unsigned o,unsigned banks,unsigned active,unsigned queries,uint64_t seed){
    if(d%32 || d>D || !o || active>8 || !queries || queries>active)die("matrix geometry");
    m->d=d;m->o=o;m->b=banks;m->nactive=active;m->nquery=queries;m->seed=seed;
    uint64_t bytes=(uint64_t)d*o*banks;
    m->w=allocate(bytes);m->s=allocate((uint64_t)o*banks*4);m->y=allocate((uint64_t)o*active*4);
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)bytes/8;j++){
        uint64_t z=mix((uint64_t)j+seed);int8_t q[8];memcpy(q,&z,8);for(unsigned k=0;k<8;k++)m->w[j*8+k]=(uint8_t)((q[k]==-128?0:q[k])+128);
    }
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)o*banks;j++)m->s[j]=(1.0f+(float)(mix((uint64_t)j+seed)&127)/128.0f)/(127.0f*sqrtf((float)d));
    for(unsigned i=0;i<active;i++)m->ids[i]=i;
    active_i8+=(uint64_t)d*o*active;active_scales+=(uint64_t)o*active*4;code_capacity+=bytes;
}
static void initialize(const char *spec,const char *model){
    FILE *f=fopen(spec,"rb"),*source=fopen(model,"rb");if(!f||!source)die("source/spec open");
    char magic[8];uint32_t dims[6];uint64_t file_bytes,head_offset,head_bytes,norm_offset;uint8_t head_sha[32],norm_sha[32];
    read_exact(f,magic,8);read_exact(f,dims,sizeof(dims));const uint32_t wanted[6]={LAYERS,PARENTS,CHILDREN,H,SH,HEADS};
    if(memcmp(magic,"M303SPC1",8)||memcmp(dims,wanted,sizeof(dims)))die("catalogue geometry");
    read_exact(f,&file_bytes,8);read_exact(f,&head_offset,8);read_exact(f,&head_bytes,8);read_exact(f,head_sha,32);read_exact(f,&norm_offset,8);read_exact(f,norm_sha,32);
    if(_fseeki64(source,0,SEEK_END) || (uint64_t)_ftelli64(source)!=file_bytes || file_bytes!=6474702976 || head_bytes!=161602560)die("source file geometry");
    head=allocate(head_bytes);payload(source,head_offset,head,(size_t)head_bytes);
    const void *hp[1]={head};size_t hs[1]={(size_t)head_bytes};verify_sha(hp,hs,1,head_sha);
    payload(source,norm_offset,final_norm,sizeof(final_norm));const void *np[1]={final_norm};size_t ns[1]={sizeof(final_norm)};verify_sha(np,ns,1,norm_sha);
    for(unsigned j=0;j<D;j++)if(!isfinite(final_norm[j]))die("nonfinite final norm");
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];uint64_t off[5];uint8_t sha[32];read_exact(f,off,sizeof(off));read_exact(f,sha,32);
        payload(source,off[0],s->an,sizeof(s->an));payload(source,off[1],s->fn,sizeof(s->fn));payload(source,off[2],s->kn,sizeof(s->kn));
        const void *p[5]={s->an,s->fn,s->kn,s->router,s->bias};size_t n[5]={sizeof(s->an),sizeof(s->fn),sizeof(s->kn),0,0};
        if(l){payload(source,off[3],s->router,sizeof(s->router));payload(source,off[4],s->bias,sizeof(s->bias));n[3]=sizeof(s->router);n[4]=sizeof(s->bias);}
        else if(off[3]||off[4])die("dense first router");verify_sha(p,n,5,sha);
        for(unsigned j=0;j<D;j++)if(!isfinite(s->an[j])||!isfinite(s->fn[j]))die("nonfinite source norm");
        for(unsigned j=0;j<512;j++)if(!isfinite(s->kn[j]))die("nonfinite KV norm");
        if(l)for(unsigned j=0;j<PARENTS*D;j++)if(!isfinite(s->router[j]))die("nonfinite source router");
        if(l)for(unsigned j=0;j<PARENTS;j++)if(!isfinite(s->bias[j]))die("nonfinite source bias");
        uint64_t z=(uint64_t)(l+1)*UINT64_C(10000000000);
        make_matrix(&s->q,D,D,1,1,1,z+1);make_matrix(&s->kv,D,576,1,1,1,z+2);
        make_matrix(&s->kb,128,512,HEADS,HEADS,HEADS,z+3);make_matrix(&s->vb,512,192,HEADS,HEADS,HEADS,z+4);make_matrix(&s->wo,D,D,1,1,1,z+5);
        if(l){
            make_matrix(&s->g,D,H,BANKS,4,1,z+6);make_matrix(&s->u,D,H,BANKS,4,1,z+7);make_matrix(&s->down,H,D,BANKS,4,4,z+8);
            make_matrix(&s->sg,D,SH,1,1,1,z+9);make_matrix(&s->su,D,SH,1,1,1,z+10);make_matrix(&s->sd,SH,D,1,1,1,z+11);
            make_matrix(&s->cq,D,QDIM,1,1,1,z+12);s->keys=allocate((uint64_t)PARENTS*CHILDREN*QDIM*4);
            for(unsigned j=0;j<PARENTS*CHILDREN*QDIM;j++)s->keys[j]=fixture(z+13+j);
        }else{make_matrix(&s->g,D,DH,1,1,1,z+6);make_matrix(&s->u,D,DH,1,1,1,z+7);make_matrix(&s->down,DH,D,1,1,1,z+8);}
    }
    if(fgetc(f)!=EOF)die("catalogue trailing data");fclose(f);fclose(source);
    if(active_i8!=273571840 || active_scales!=1902656)die("complete integer counts");guards();
}
static void split_input(Input *q){
    q->sum_x=0;
    for(unsigned j=0;j<q->n;j++){
        int x=q->q[j],h=x/2;q->hi[j]=(int8_t)h;q->lo[j]=(int8_t)(x-2*h);q->sum_x+=x;
    }
}
static int quantize(const float *x,unsigned n,Input *q){
    if(!n||n>D||n%32)return 0;float max=0;
    for(unsigned j=0;j<n;j++){if(!isfinite(x[j]))return 0;float a=fabsf(x[j]);if(a>max)max=a;}
    q->n=n;q->s=max>0?max/127.0f:1.0f;if(!isfinite(q->s)||q->s<=0)return 0;
    float inverse=max>0?127.0f/max:1.0f;if(!isfinite(inverse))return 0;
    for(unsigned j=0;j<n;j++){float v=nearbyintf(x[j]*inverse);if(v>127)v=127;if(v< -127)v=-127;q->q[j]=(int8_t)v;}
    split_input(q);return 1;
}
// Adjacent-pair high <=32130, low <=510; all accumulators and correction fit I32.
static int32_t idot(const uint8_t *w,const Input *x,unsigned n){
    __m256i hi=_mm256_setzero_si256(),lo=hi,ones=_mm256_set1_epi16(1);
    for(unsigned j=0;j<n;j+=32){
        __m256i weight=_mm256_loadu_si256((const __m256i*)(w+j));
        __m256i hp=_mm256_maddubs_epi16(weight,_mm256_loadu_si256((const __m256i*)(x->hi+j)));
        __m256i lp=_mm256_maddubs_epi16(weight,_mm256_loadu_si256((const __m256i*)(x->lo+j)));
        hi=_mm256_add_epi32(hi,_mm256_madd_epi16(hp,ones));lo=_mm256_add_epi32(lo,_mm256_madd_epi16(lp,ones));
    }
    int32_t v[8];_mm256_storeu_si256((__m256i*)v,_mm256_add_epi32(_mm256_slli_epi32(hi,1),lo));
    int32_t total=0;for(unsigned j=0;j<8;j++)total+=v[j];return total-128*x->sum_x;
}
static int apply(Matrix *m,const Input *input){
    for(unsigned k=0;k<m->nactive;k++)if(m->ids[k]>=m->b)return 0;
    for(unsigned k=0;k<m->nquery;k++)if(input[k].n!=m->d)return 0;m->input=input;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<(int)(m->o*m->nactive);r++){
        unsigned k=(unsigned)r/m->o,row=(unsigned)r%m->o,bank=m->ids[k],query=m->nquery==1?0:k;
        const Input *x=&input[query];int32_t sum=idot(m->w+((size_t)bank*m->o+row)*m->d,x,m->d);
        m->y[r]=(float)sum*(x->s*m->s[(size_t)bank*m->o+row]);
    }
    return 1;
}
static void checked_apply(Matrix *m,const Input *q){if(!apply(m,q))die("bank/input range");}
static void checked_quant(const float *x,unsigned n,Input *q){if(!quantize(x,n,q))die("activation quantization");}
static void norm(const float *x,const float *w,unsigned n,float *y){double sum=0;for(unsigned j=0;j<n;j++)sum+=(double)x[j]*x[j];float r=(float)(1.0/sqrt(sum/n+1e-6));for(unsigned j=0;j<n;j++)y[j]=(x[j]*r)*w[j];}
static void prepare(unsigned token){
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];uint64_t seed=(uint64_t)token*UINT64_C(1000000000)+l*UINT64_C(10000000);
        for(unsigned j=0;j<D;j++){s->ra[j]=fixture(seed+j);s->rf[j]=fixture(seed+10000+j);}
        for(unsigned j=0;j<HEADS*512;j++)s->latent[j]=fixture(seed+20000+j);
    }
}
static void macro_router(Layer *s){
    #pragma omp parallel for schedule(static)
    for(int r=0;r<PARENTS;r++){float v=strat01_f32_dot_reference_generic(s->router+(size_t)r*D,s->xf,D);s->probs[r]=1.0f/(1.0f+expf(-v));}
    for(unsigned k=0;k<4;k++){
        unsigned best=UINT32_MAX;
        for(unsigned r=0;r<PARENTS;r++){int used=0;for(unsigned j=0;j<k;j++)if(s->parent[j]==r)used=1;
            if(!used&&(best==UINT32_MAX || s->probs[r]+s->bias[r]>s->probs[best]+s->bias[best]))best=r;}
        s->parent[k]=best;
    }
    float total=0;for(unsigned k=0;k<4;k++)total+=s->probs[s->parent[k]];if(!(total>0))die("zero source gate");
    for(unsigned k=0;k<4;k++)s->gates[k]=s->probs[s->parent[k]]/total;
}
static void child_router(Layer *s){
    for(unsigned k=0;k<4;k++){
        unsigned p=s->parent[k],best=0;float value=-INFINITY;
        for(unsigned c=0;c<CHILDREN;c++){
            float v=strat01_f32_dot_reference_generic(s->keys+((size_t)p*CHILDREN+c)*QDIM,s->cq.y,QDIM);
            if(v>value){value=v;best=c;}
        }
        s->child[k]=best;s->flat[k]=p*CHILDREN+best;
        s->g.ids[k]=s->u.ids[k]=s->down.ids[k]=s->flat[k];
    }
}
static float silu(float x){return x/(1.0f+expf(-x));}
static void execute(void){
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];norm(s->ra,s->an,D,s->xa);checked_quant(s->xa,D,&s->qa);
        checked_apply(&s->q,&s->qa);checked_apply(&s->kv,&s->qa);norm(s->kv.y,s->kn,512,s->kvnorm);
        for(unsigned h=0;h<HEADS;h++){checked_quant(s->q.y+h*192,128,&s->qk[h]);checked_quant(s->latent+h*512,512,&s->qv[h]);}
        checked_apply(&s->kb,s->qk);checked_apply(&s->vb,s->qv);checked_quant(s->vb.y,D,&s->qo);checked_apply(&s->wo,&s->qo);
        norm(s->rf,s->fn,D,s->xf);checked_quant(s->xf,D,&s->qf);
        if(l){macro_router(s);checked_apply(&s->cq,&s->qf);child_router(s);}
        checked_apply(&s->g,&s->qf);checked_apply(&s->u,&s->qf);
        for(unsigned k=0;k<s->g.nactive;k++){
            for(unsigned j=0;j<s->g.o;j++)s->zg[k*s->g.o+j]=silu(s->g.y[k*s->g.o+j])*s->u.y[k*s->g.o+j];
            checked_quant(s->zg+k*s->g.o,s->g.o,&s->qz[k]);
        }
        checked_apply(&s->down,s->qz);
        if(l){
            checked_apply(&s->sg,&s->qf);checked_apply(&s->su,&s->qf);
            for(unsigned j=0;j<SH;j++)s->sz[j]=silu(s->sg.y[j])*s->su.y[j];
            checked_quant(s->sz,SH,&s->qs);checked_apply(&s->sd,&s->qs);
        }
        for(unsigned j=0;j<D;j++){
            float v=l?s->sd.y[j]:s->down.y[j];if(l)for(unsigned k=0;k<4;k++)v+=s->gates[k]*s->down.y[k*D+j];
            s->mixture[j]=v;
        }
    }
    Layer *s=&layers[LAYERS-1];for(unsigned j=0;j<D;j++)head_input[j]=(s->xf[j]+s->wo.y[j])+s->mixture[j];
    norm(head_input,final_norm,D,head_input);if(!strat01_quantize_q8_k_row(head_input,D,hq))die("head Q8_K quantization");
    #pragma omp parallel for schedule(static)
    for(int r=0;r<V;r++)head_logits[r]=strat01_q6k_q8k_dot_avx2(head+(size_t)r*(D/256)*210,hq,D);
}
static Matrix *matrix(Layer *s,unsigned i){Matrix *m[]={&s->q,&s->kv,&s->kb,&s->vb,&s->wo,&s->g,&s->u,&s->down,&s->sg,&s->su,&s->sd,&s->cq};return m[i];}
static uint64_t output_hash(void){
    uint64_t h=UINT64_C(0x123456789abcdef0);
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];for(unsigned i=0;i<(l?12U:8U);i++){Matrix *m=matrix(s,i);for(unsigned j=0;j<m->o*m->nactive;j++){uint32_t u;memcpy(&u,m->y+j,4);if(!isfinite(m->y[j]))die("nonfinite core output");h=mix(h^u);}}
        for(unsigned j=0;j<512;j++){uint32_t u;memcpy(&u,s->kvnorm+j,4);if(!isfinite(s->kvnorm[j]))die("nonfinite KV norm output");h=mix(h^u);}
        for(unsigned j=0;j<D;j++){uint32_t u;memcpy(&u,s->mixture+j,4);if(!isfinite(s->mixture[j]))die("nonfinite mixture");h=mix(h^u);}
        if(l)for(unsigned k=0;k<4;k++){h=mix(h^s->parent[k]);h=mix(h^s->child[k]);}
    }
    for(unsigned j=0;j<V;j++){uint32_t u;memcpy(&u,head_logits+j,4);if(!isfinite(head_logits[j]))die("nonfinite head output");h=mix(h^u);}return h;
}
static int64_t scalar_dot(const uint8_t *w,const int8_t *x,unsigned n){int64_t s=0;for(unsigned j=0;j<n;j++)s+=(int64_t)((int)w[j]-128)*x[j];return s;}
static double decoded_head(unsigned row){
    const uint8_t *raw=head+(size_t)row*6*210;double total=0;
    for(unsigned b=0;b<6;b++,raw+=210){
        double d=strat01_q4k_q8k_fp16le(raw+208)*hq[b].d;
        for(unsigned j=0;j<256;j++){
            unsigned half=j/128,phase=(j%128)/32,lane=j%32;
            unsigned ql=raw[half*64+lane+(phase&1)*32];if(phase>=2)ql>>=4;else ql&=15;
            int q=(int)(ql|(((raw[128+half*32+lane]>>(phase*2))&3)<<4))-32;
            int scale=((const int8_t*)raw)[192+half*8+phase*2+lane/16];
            total+=d*(double)scale*q*hq[b].qs[j];
        }
    }
    return total;
}
typedef struct {float score;unsigned id;} Rank;
static int rank_compare(const void *a,const void *b){
    const Rank *x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;
    return x->id<y->id?-1:x->id>y->id?1:0;
}
static float reference_f32(const float *x,const float *y,unsigned n){
    double sum=0;for(unsigned j=0;j<n;j++){volatile float product=x[j]*y[j];sum+=(double)product;}return (float)sum;
}
static void route_control(Layer *s){
    Rank rank[PARENTS];float probs[PARENTS];
    for(unsigned r=0;r<PARENTS;r++){
        float v=reference_f32(s->router+(size_t)r*D,s->xf,D);probs[r]=1.0f/(1.0f+expf(-v));
        rank[r]=(Rank){probs[r]+s->bias[r],r};if(memcmp(probs+r,s->probs+r,4))die("reference macro probability");
    }
    qsort(rank,PARENTS,sizeof(Rank),rank_compare);float total=0;for(unsigned k=0;k<4;k++)total+=probs[rank[k].id];
    for(unsigned k=0;k<4;k++){
        if(rank[k].id!=s->parent[k])die("reference macro selection");
        float gate=probs[rank[k].id]/total;if(memcmp(&gate,s->gates+k,4))die("reference macro gate");
        Rank child[CHILDREN];for(unsigned c=0;c<CHILDREN;c++)child[c]=(Rank){reference_f32(s->keys+((size_t)rank[k].id*CHILDREN+c)*QDIM,s->cq.y,QDIM),c};
        qsort(child,CHILDREN,sizeof(Rank),rank_compare);
        if(child[0].id!=s->child[k]||s->flat[k]!=rank[k].id*CHILDREN+child[0].id)die("reference child selection");
        if(s->g.ids[k]!=s->flat[k]||s->u.ids[k]!=s->flat[k]||s->down.ids[k]!=s->flat[k])die("selected bank binding");
    }
}
static void selftest(void){
    uint8_t w[D];Input x;x.n=D;for(unsigned j=0;j<D;j++){x.q[j]=127;w[j]=255;}split_input(&x);
    if(idot(w,&x,D)!=scalar_dot(w,x.q,D)||idot(w,&x,D)!=24774144)die("integer maximum");
    for(unsigned j=0;j<D;j++)w[j]=1;if(idot(w,&x,D)!=-24774144)die("integer minimum");
    uint64_t products=0,pairs=0;x.n=32;
    for(int weight=-127;weight<=127;weight++)for(int value=-127;value<=127;value++){
        for(unsigned j=0;j<32;j++){w[j]=(uint8_t)(weight+128);x.q[j]=(int8_t)value;}split_input(&x);
        if(idot(w,&x,32)!=scalar_dot(w,x.q,32)||idot(w,&x,32)!=(int64_t)32*weight*value)die("exhaustive weight/input");products++;
    }
    const int weights[7]={-127,-126,-1,0,1,126,127},values[11]={-127,-126,-3,-2,-1,0,1,2,3,126,127};
    for(unsigned a=0;a<7;a++)for(unsigned b=0;b<7;b++)for(unsigned c=0;c<11;c++)for(unsigned d=0;d<11;d++){
        memset(w,128,32);memset(x.q,0,32);w[0]=(uint8_t)(weights[a]+128);w[1]=(uint8_t)(weights[b]+128);x.q[0]=(int8_t)values[c];x.q[1]=(int8_t)values[d];split_input(&x);
        __m256i weight=_mm256_loadu_si256((const __m256i*)w);int16_t hp[16],lp[16];
        _mm256_storeu_si256((__m256i*)hp,_mm256_maddubs_epi16(weight,_mm256_loadu_si256((const __m256i*)x.hi)));
        _mm256_storeu_si256((__m256i*)lp,_mm256_maddubs_epi16(weight,_mm256_loadu_si256((const __m256i*)x.lo)));
        if(hp[0]!=(int)w[0]*x.hi[0]+(int)w[1]*x.hi[1]||lp[0]!=(int)w[0]*x.lo[0]+(int)w[1]*x.lo[1]||idot(w,&x,32)!=scalar_dot(w,x.q,32))die("mixed adjacent pair saturation/correction");pairs++;
    }
    memset(w,255,32);memset(x.q,127,32);split_input(&x);int16_t direct[16];
    _mm256_storeu_si256((__m256i*)direct,_mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)w),_mm256_loadu_si256((const __m256i*)x.q)));
    if(direct[0]!=32767||direct[0]==2*255*127||idot(w,&x,32)!=scalar_dot(w,x.q,32))die("unsplit saturation negative control");
    Input bad=x;bad.sum_x++;if(idot(w,&bad,32)==scalar_dot(w,x.q,32))die("row bias negative control");
    if(products!=65025||pairs!=5929)die("complete integer unit controls");
    float zero[32]={0},tie[32]={127,.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f};Input q;
    if(!quantize(zero,32,&q)||q.s!=1)die("zero quantizer");for(unsigned j=0;j<32;j++)if(q.q[j])die("zero codes");
    if(!quantize(tie,32,&q)||q.q[1]!=0||q.q[2]!=2||q.q[3]!=2||q.q[4]!=0||q.q[5]!=-2||q.q[6]!=-2)die("ties-even quantizer");
    tie[0]=NAN;if(quantize(tie,32,&q))die("nonfinite quantizer accepted");
    prepare(13);execute();uint64_t exact=0,bank_checks=0;double e2=0,n2=0,he2=0,hn2=0;
    for(unsigned l=1;l<LAYERS;l++)route_control(&layers[l]);
    for(unsigned l=0;l<LAYERS;l++)for(unsigned i=0;i<(l?12U:8U);i++){
        Matrix *m=matrix(&layers[l],i);
        for(unsigned bank=0;bank<m->b;bank++)for(unsigned edge=0;edge<2;edge++){
            size_t byte=(size_t)bank*m->d*m->o+(edge?(size_t)m->d*m->o-8:0);uint64_t z=mix(byte/8+m->seed);int8_t decoded[8];uint8_t expected[8];memcpy(decoded,&z,8);for(unsigned k=0;k<8;k++)expected[k]=(uint8_t)((decoded[k]==-128?0:decoded[k])+128);
            if(memcmp(m->w+byte,expected,8))die("stored bank offset");bank_checks++;
        }
        for(unsigned k=0;k<m->nactive;k++)for(unsigned t=0;t<8;t++){
            unsigned r=t*(m->o-1)/7;const Input *in=&m->input[m->nquery==1?0:k];const uint8_t *weight=m->w+((size_t)m->ids[k]*m->o+r)*m->d;
            int64_t s=scalar_dot(weight,in->q,m->d);if(s!=idot(weight,in,m->d))die("exact integer dot");
            float sc=m->s[(size_t)m->ids[k]*m->o+r],v=(float)s*(in->s*sc);if(memcmp(&v,m->y+k*m->o+r,4))die("scalar output");
            double ref=(double)s*in->s*sc,e=v-ref;e2+=e*e;n2+=ref*ref;exact++;
        }
    }
    for(unsigned r=0;r<64;r++){
        unsigned row=r*(V-1)/63;double v=decoded_head(row),e=head_logits[row]-v;he2+=e*e;hn2+=v*v;
    }
    Matrix *m=&layers[1].g;unsigned saved=m->ids[0];m->ids[0]=m->b;int rejected=!apply(m,&layers[1].qf);m->ids[0]=saved;
    int detected=0;unsigned sentinel_row=0,sentinel_byte=0;
    for(unsigned r=0;r<64&&!detected;r++)for(unsigned j=0;j<128&&!detected;j++){
        unsigned row=r*(V-1)/63;size_t offset=(size_t)row*6*210+j;uint8_t old=head[offset];double before=decoded_head(row);
        float native_before=strat01_q6k_q8k_dot_avx2(head+(size_t)row*6*210,hq,D);head[offset]^=1;
        double after=decoded_head(row);float native_after=strat01_q6k_q8k_dot_avx2(head+(size_t)row*6*210,hq,D);head[offset]=old;
        if(before!=after&&native_before!=native_after){detected=1;sentinel_row=row;sentinel_byte=j;}
    }
    if(!rejected||!detected||!(n2>0)||!(hn2>0)||sqrt(e2/n2)>1e-6||sqrt(he2/hn2)>1e-5)die("numeric/negative controls");
    printf("{\"event\":\"selftest\",\"exact_i8_rows\":%llu,\"stored_bank_offset_checks\":%llu,\"i8_relative_l2\":%.17g,\"real_head64_relative_l2\":%.17g,\"reference_route_layers\":25,\"bad_bank_rejected\":true,\"packed_head_change_detected\":true,\"sentinel_row\":%u,\"sentinel_byte\":%u,\"integer_extrema_and_quantizer_checks\":true,\"exhaustive_weight_input_cases\":65025,\"mixed_adjacent_pairs\":5929,\"unsplit_saturation_negative\":true,\"row_bias_error_detected\":true}\n",(unsigned long long)exact,(unsigned long long)bank_checks,sqrt(e2/n2),sqrt(he2/hn2),sentinel_row,sentinel_byte);fflush(stdout);
}
int main(int argc,char **argv){
    if(argc!=3)die("usage: SPEC MODEL");fesetround(FE_TONEAREST);omp_set_dynamic(0);omp_set_num_threads(6);started=clock_s();initialize(argv[1],argv[2]);
    printf("{\"event\":\"ready\",\"parent_count\":64,\"children_per_parent\":10,\"functions_per_layer\":640,\"threads\":6,\"allocated_bytes\":%llu,\"active_i8_coefficients\":%llu,\"active_row_scale_bytes\":%llu,\"stored_i8_coefficients\":%llu,\"initialization_seconds\":%.9f}\n",(unsigned long long)allocated,(unsigned long long)active_i8,(unsigned long long)active_scales,(unsigned long long)code_capacity,clock_s()-started);fflush(stdout);
    selftest();unsigned char selected[26][BANKS];memset(selected,0,sizeof(selected));
    for(unsigned rep=0;rep<3;rep++){
        for(unsigned token=0;token<10;token++){
            prepare(token);double t=clock_s();execute();double seconds=clock_s()-t;uint64_t hash=output_hash();
            for(unsigned l=1;l<LAYERS;l++)for(unsigned k=0;k<4;k++)selected[l][layers[l].flat[k]]=1;
            printf("{\"event\":\"operator\",\"rep\":%u,\"input\":%u,\"warmup\":%s,\"seconds\":%.12f,\"output_route_hash\":\"%016llx\"}\n",rep,token,token<2?"true":"false",seconds,(unsigned long long)hash);fflush(stdout);
        }
        guards();
    }
    unsigned lo=BANKS,hi=0;for(unsigned l=1;l<LAYERS;l++){unsigned n=0;for(unsigned j=0;j<BANKS;j++)n+=selected[l][j];if(n<lo)lo=n;if(n>hi)hi=n;}
    printf("{\"event\":\"finished\",\"minimum_selected_child_union\":%u,\"maximum_selected_child_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f}\n",lo,hi,(unsigned long long)peak_rss(),clock_s()-started);return 0;
}
