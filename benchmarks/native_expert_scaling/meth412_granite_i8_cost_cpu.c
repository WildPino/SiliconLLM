// M412: actual M411 source0 geometry, ALL tensor values synthetic.
// Independent layer/attention-context fixtures; this is not model inference.
#define _WIN32_WINNT 0x0601
#define WIN32_LEAN_AND_MEAN
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <limits.h>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>
#include <psapi.h>
#include <bcrypt.h>
#include "../phase60/strat01_f32_dot_reference_generic.h"
#define LAYERS 24
#define D 1024
#define H 512
#define BANKS 32
#define TOPK 8
#define V 49152
#define HEAD_SEED UINT64_C(1234567890123)
typedef struct {int16_t q[D];float s;unsigned n;} Input;
typedef struct {
    unsigned d,o,b,nactive,nquery,ids[TOPK];uint64_t seed;
    int8_t *w;float *s,*y;const Input *input;
} Matrix;
typedef struct {
    Matrix q,k,v,o,gu,down;
    float an[D],fn[D],router[BANKS*D];
    float ra[D],context[D],xa[D],rf[D],xf[D],zg[TOPK*H],mixture[D],residual[D];
    float scores[BANKS],gates[TOPK];unsigned parent[TOPK];
    Input qa,qo,qf,qz[TOPK];
} Layer;
static Layer layers[LAYERS];static Matrix head;static Input qhead;
static uint16_t *embedding;
static float embedding_row[D],final_norm[D],head_residual[D],head_input[D];
static uint64_t allocated,active_i8,active_scales,stored_i8;
static uint64_t executed_coeff,executed_scales,executed_router;
static double started;static int requested_threads;static unsigned current_token;
static void die(const char *s){fprintf(stderr,"M412 apparatus failure:%s\n",s);exit(2);}
static void require(int ok,const char *s){if(!ok)die(s);}
static void *alloc(size_t n,size_t w){void *p=calloc(n,w);require(p!=NULL,"worker_alloc");return p;}
#include "meth388_switch_thread_binding.h"
static double clock_s(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart/(double)f.QuadPart;}
static uint64_t mix(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static float fixture(uint64_t x){return ((int)(mix(x)&511)-256)*(1.0f/256.0f);}
static void *allocate(uint64_t n){require(n&&n<=UINT64_C(6)*1024*1024*1024,"allocation_range");void *p=_aligned_malloc((size_t)n,64);require(p!=NULL,"allocation");allocated+=n;return p;}
static uint64_t peak_rss(void){PROCESS_MEMORY_COUNTERS p;memset(&p,0,sizeof(p));p.cb=sizeof(p);require(GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p)),"rss");return p.PeakWorkingSetSize;}
static void guards(void){require(clock_s()-started<=120&&peak_rss()<=UINT64_C(6)*1024*1024*1024,"process_2min_6GiB");}
/* For cols<=4096 each I32 lane has <=512 products: 512*127*32767
   =2130641408<INT32_MAX. Global sum can reach17045131264: use I64. */
static int64_t head_integer_dot(const int8_t *a,const int16_t *b,int n){
    worker_bind();
    require(n>0&&n<=4096,"head_integer_bound");__m256i acc=_mm256_setzero_si256();int i=0;
    for(;i+16<=n;i+=16){__m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a+i)));__m256i y=_mm256_loadu_si256((const __m256i *)(b+i));acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x,y));}
    int32_t values[8];_mm256_storeu_si256((__m256i *)values,acc);int64_t sum=0;for(int j=0;j<8;j++)sum+=(int64_t)values[j];for(;i<n;i++)sum+=(int64_t)a[i]*(int64_t)b[i];return sum;
}
static float head_activation_codes(const float *x,int16_t *codes,int cols){
    require(cols>0&&cols<=4096&&fegetround()==FE_TONEAREST,"head_activation_contract");float maximum=0.f;
    for(int i=0;i<cols;i++){require(isfinite(x[i]),"head_activation_finite");float v=fabsf(x[i]);if(v>maximum)maximum=v;}
    float scale=maximum==0.f?1.f:maximum/32767.f;require(isfinite(scale)&&scale>0.f,"head_activation_scale");
    for(int i=0;i<cols;i++){float divided=x[i]/scale;long code=lrintf(divided);if(code>32767)code=32767;if(code< -32767)code= -32767;codes[i]=(int16_t)code;}return scale;
}
static int8_t expected_code(uint64_t seed,uint64_t position){
    unsigned raw=(unsigned)((mix(seed+position/8)>>(8*(position%8)))&255);
    int value=raw<128?(int)raw:(int)raw-256;return (int8_t)(value==-128?-127:value);
}
static float expected_scale(uint64_t seed,uint64_t row,unsigned d){return (1.f+(float)(mix(row+seed)&127)/128.f)/(127.f*sqrtf((float)d));}
static void make_matrix(Matrix *m,unsigned d,unsigned o,unsigned banks,unsigned active,unsigned queries,uint64_t seed){
    require(d&&d<=D&&d%16==0&&o&&active<=TOPK&&active<=banks&&(queries==1||queries==active),"matrix_geometry");
    m->d=d;m->o=o;m->b=banks;m->nactive=active;m->nquery=queries;m->seed=seed;
    uint64_t bytes=(uint64_t)d*o*banks;
    m->w=allocate(bytes);m->s=allocate((uint64_t)o*banks*4);m->y=allocate((uint64_t)o*active*4);
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)bytes/8;j++){
        uint64_t raw=mix((uint64_t)j+seed);int8_t values[8];memcpy(values,&raw,8);
        for(unsigned k=0;k<8;k++)if(values[k]==-128)values[k]=-127;memcpy(m->w+j*8,values,8);
    }
    #pragma omp parallel for schedule(static)
    for(int64_t j=0;j<(int64_t)o*banks;j++)m->s[j]=expected_scale(seed,(uint64_t)j,d);
    for(unsigned k=0;k<active;k++)m->ids[k]=k;
    active_i8+=(uint64_t)d*o*active;active_scales+=(uint64_t)o*active*4;stored_i8+=bytes;
}
static int quantize(const float *x,unsigned n,Input *q){
    if(!n||n>D)return 0;for(unsigned j=0;j<n;j++)if(!isfinite(x[j]))return 0;
    q->n=n;q->s=head_activation_codes(x,q->q,(int)n);return 1;
}
static void checked_quant(const float *x,unsigned n,Input *q){require(quantize(x,n,q),"quantize");}
static int apply(Matrix *m,const Input *input){
    for(unsigned k=0;k<m->nactive;k++)if(m->ids[k]>=m->b)return 0;
    for(unsigned k=0;k<m->nquery;k++)if(input[k].n!=m->d)return 0;m->input=input;
    #pragma omp parallel for schedule(static)
    for(int row=0;row<(int)(m->o*m->nactive);row++){
        unsigned k=(unsigned)row/m->o,r=(unsigned)row%m->o;size_t index=(size_t)m->ids[k]*m->o+r;
        const Input *x=input+(m->nquery==1?0:k);int64_t value=head_integer_dot(m->w+index*m->d,x->q,(int)m->d);
        m->y[row]=(float)(((double)value*(double)m->s[index])*(double)x->s);
    }
    return 1;
}
static void checked_apply(Matrix *m,const Input *x){require(apply(m,x),"bank/input_range");executed_coeff+=(uint64_t)m->d*m->o*m->nactive;executed_scales+=(uint64_t)m->o*m->nactive*4;}
static void norm(const float *x,const float *w,float *y){double sum=0;for(unsigned j=0;j<D;j++)sum+=(double)x[j]*x[j];float r=(float)(1./sqrt(sum/D+1e-6));for(unsigned j=0;j<D;j++)y[j]=(x[j]*r)*w[j];}
static float silu(float x){return x/(1.f+expf(-x));}
static void initialize(void){
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];uint64_t z=(uint64_t)(l+1)*UINT64_C(10000000000);
        for(unsigned j=0;j<D;j++){s->an[j]=1+.1f*fixture(z+j);s->fn[j]=1+.1f*fixture(z+3000+j);}
        #pragma omp parallel for schedule(static)
        for(int r=0;r<BANKS;r++)for(unsigned j=0;j<D;j++)s->router[(size_t)r*D+j]=fixture(z+100000+(size_t)r*D+j)/sqrtf((float)D);
        make_matrix(&s->q,D,D,1,1,1,z+1);make_matrix(&s->k,D,H,1,1,1,z+2);make_matrix(&s->v,D,H,1,1,1,z+3);
        make_matrix(&s->o,D,D,1,1,1,z+4);make_matrix(&s->gu,D,2*H,BANKS,TOPK,1,z+5);make_matrix(&s->down,H,D,BANKS,TOPK,TOPK,z+6);
    }
    make_matrix(&head,D,V,1,1,1,HEAD_SEED);embedding=allocate((uint64_t)V*D*2);
    // TWO target precision copies of ONE synthetic underlying head/embed fixture.
    #pragma omp parallel for schedule(static)
    for(int64_t row=0;row<V;row++)for(unsigned j=0;j<D;j++){
        float value=(float)head.w[(size_t)row*D+j]*head.s[row];uint32_t bits;memcpy(&bits,&value,4);embedding[(size_t)row*D+j]=(uint16_t)(bits>>16);
    }
    for(unsigned j=0;j<D;j++)final_norm[j]=1+.1f*fixture(700000+j);
    require(active_i8==427819008&&active_scales==2064384&&stored_i8==1333788672,"411_matrix_ledger");
    require(allocated==UINT64_C(1443299328),"dynamic_payload_output_ledger");guards();
}
static void prepare(unsigned token){
    current_token=token;for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];uint64_t seed=(uint64_t)token*UINT64_C(1000000000)+l*UINT64_C(10000000);
        for(unsigned j=0;j<D;j++){s->ra[j]=fixture(seed+j);s->context[j]=fixture(seed+20000+j);}
    }
}
typedef struct {float score;unsigned id;} Rank;
static int rank_compare(const void *a,const void *b){const Rank *x=a,*y=b;if(x->score>y->score)return -1;if(x->score<y->score)return 1;return x->id<y->id?-1:x->id>y->id?1:0;}
static void select_routes(const float scores[BANKS],unsigned ids[TOPK],float gates[TOPK]){
    unsigned ranked[TOPK];float p[TOPK],total=0;
    for(unsigned k=0;k<TOPK;k++){
        unsigned best=UINT32_MAX;for(unsigned r=0;r<BANKS;r++){
            int used=0;for(unsigned j=0;j<k;j++)if(ranked[j]==r)used=1;
            if(!used&&(best==UINT32_MAX||scores[r]>scores[best]))best=r;
        }ranked[k]=best;
    }
    for(unsigned k=0;k<TOPK;k++){p[k]=expf(scores[ranked[k]]-scores[ranked[0]]);total+=p[k];}
    // Source order: softmax in top-logit order; accumulation sorted by expert ID.
    for(unsigned k=0;k<TOPK;k++){ids[k]=ranked[k];gates[k]=p[k]/total;}
    for(unsigned i=0;i<TOPK;i++)for(unsigned j=i+1;j<TOPK;j++)if(ids[j]<ids[i]){
        unsigned old=ids[i];ids[i]=ids[j];ids[j]=old;float g=gates[i];gates[i]=gates[j];gates[j]=g;
    }
}
static void route(Layer *s){
    executed_router+=(uint64_t)BANKS*D*4;
    #pragma omp parallel for schedule(static)
    for(int r=0;r<BANKS;r++){worker_bind();s->scores[r]=strat01_f32_dot_reference_generic(s->router+(size_t)r*D,s->xf,D);}
    select_routes(s->scores,s->parent,s->gates);
    for(unsigned k=0;k<TOPK;k++)s->gu.ids[k]=s->down.ids[k]=s->parent[k];
}
static void execute(void){
    executed_coeff=executed_scales=executed_router=0;
    for(unsigned j=0;j<D;j++){uint32_t bits=(uint32_t)embedding[(size_t)current_token*D+j]<<16;float x;memcpy(&x,&bits,4);embedding_row[j]=x*12.f;layers[0].ra[j]=embedding_row[j];}
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];norm(s->ra,s->an,s->xa);checked_quant(s->xa,D,&s->qa);
        checked_apply(&s->q,&s->qa);checked_apply(&s->k,&s->qa);checked_apply(&s->v,&s->qa);
        // No causal attention, attention scale .015625, RoPE or KV/cache here.
        checked_quant(s->context,D,&s->qo);checked_apply(&s->o,&s->qo);
        for(unsigned j=0;j<D;j++)s->rf[j]=s->ra[j]+.22f*s->o.y[j];
        norm(s->rf,s->fn,s->xf);checked_quant(s->xf,D,&s->qf);route(s);checked_apply(&s->gu,&s->qf);
        for(unsigned k=0;k<TOPK;k++){
            for(unsigned j=0;j<H;j++)s->zg[k*H+j]=silu(s->gu.y[k*2*H+j])*s->gu.y[k*2*H+H+j];
            checked_quant(s->zg+k*H,H,&s->qz[k]);
        }checked_apply(&s->down,s->qz);
        for(unsigned j=0;j<D;j++){float v=0;for(unsigned k=0;k<TOPK;k++)v+=s->gates[k]*s->down.y[k*D+j];s->mixture[j]=v;s->residual[j]=s->rf[j]+.22f*v;}
    }
    memcpy(head_residual,layers[LAYERS-1].residual,sizeof(head_residual));norm(head_residual,final_norm,head_input);
    checked_quant(head_input,D,&qhead);checked_apply(&head,&qhead);for(unsigned r=0;r<V;r++)head.y[r]/=6.f;
}
static Matrix *matrix(Layer *s,unsigned i){Matrix *m[]={&s->q,&s->k,&s->v,&s->o,&s->gu,&s->down};return m[i];}
static void part(BCRYPT_HASH_HANDLE h,FILE *f,const void *p,size_t n){require(n<=UINT32_MAX&&BCryptHashData(h,(PUCHAR)p,(ULONG)n,0)>=0,"hash_part");if(f)require(fwrite(p,1,n,f)==n,"write_output");}
static void floats(BCRYPT_HASH_HANDLE h,FILE *f,const float *p,unsigned n){for(unsigned j=0;j<n;j++)require(isfinite(p[j]),"finite_output");part(h,f,p,(size_t)n*4);}
static void input_part(BCRYPT_HASH_HANDLE h,FILE *f,const Input *x){part(h,f,&x->n,4);floats(h,f,&x->s,1);part(h,f,x->q,x->n*2);}
static void matrix_part(BCRYPT_HASH_HANDLE h,FILE *f,const Matrix *m){uint32_t shape[]={m->d,m->o,m->b,m->nactive,m->nquery};part(h,f,shape,sizeof(shape));part(h,f,m->ids,m->nactive*4);floats(h,f,m->y,m->o*m->nactive);}
static void output(const char *prefix,unsigned rep,unsigned token,char sha[65]){
    BCRYPT_ALG_HANDLE a=NULL;BCRYPT_HASH_HANDLE h=NULL;uint8_t digest[32];FILE *f=NULL;
    if(rep==0){char path[2048];int n=snprintf(path,sizeof(path),"%s.input%u.bin",prefix,token);require(n>0&&(size_t)n<sizeof(path),"output_path");f=fopen(path,"wb");require(f!=NULL,"output_open");}
    require(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,NULL,0)>=0&&BCryptCreateHash(a,&h,NULL,0,NULL,0,0)>=0,"output_hash_create");
    const uint32_t dims[]={token,D,LAYERS,BANKS,TOPK,V,H};part(h,f,"S412OUT1",8);part(h,f,dims,sizeof(dims));floats(h,f,embedding_row,D);
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];for(unsigned i=0;i<6;i++)matrix_part(h,f,matrix(s,i));
        const float *values[]={s->ra,s->context,s->xa,s->rf,s->xf,s->mixture,s->residual};for(unsigned i=0;i<7;i++)floats(h,f,values[i],D);
        floats(h,f,s->zg,TOPK*H);part(h,f,s->parent,TOPK*4);floats(h,f,s->scores,BANKS);floats(h,f,s->gates,TOPK);
        input_part(h,f,&s->qa);input_part(h,f,&s->qo);input_part(h,f,&s->qf);for(unsigned k=0;k<TOPK;k++)input_part(h,f,&s->qz[k]);
    }
    floats(h,f,head_residual,D);floats(h,f,head_input,D);input_part(h,f,&qhead);matrix_part(h,f,&head);
    require(BCryptFinishHash(h,digest,32,0)>=0,"output_hash_finish");BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);
    if(f)require(fclose(f)==0,"output_close");for(unsigned j=0;j<32;j++)sprintf(sha+j*2,"%02x",digest[j]);sha[64]=0;
}
static int64_t scalar_dot(const int8_t *w,const int16_t *x,unsigned n){int64_t sum=0;for(unsigned j=0;j<n;j++)sum+=(int64_t)w[j]*(int64_t)x[j];return sum;}
static float reference_f32(const float *x,const float *y){double sum=0;for(unsigned j=0;j<D;j++){volatile float p=x[j]*y[j];sum+=(double)p;}return (float)sum;}
static void route_control(Layer *s){
    Rank ranked[BANKS];float p[TOPK],g[BANKS]={0},total=0;
    for(unsigned r=0;r<BANKS;r++){float v=reference_f32(s->router+(size_t)r*D,s->xf);require(!memcmp(&v,s->scores+r,4),"router_scalar_logit");ranked[r]=(Rank){v,r};}
    qsort(ranked,BANKS,sizeof(Rank),rank_compare);for(unsigned k=0;k<TOPK;k++){p[k]=expf(ranked[k].score-ranked[0].score);total+=p[k];}
    unsigned selected[BANKS]={0};for(unsigned k=0;k<TOPK;k++){g[ranked[k].id]=p[k]/total;selected[ranked[k].id]=1;}
    unsigned k=0;for(unsigned r=0;r<BANKS;r++)if(selected[r]){
        require(k<TOPK&&s->parent[k]==r&&!memcmp(&g[r],s->gates+k,4)&&s->gu.ids[k]==r&&s->down.ids[k]==r,"selected_softmax_sorted_bank");k++;
    }require(k==TOPK,"route_count");
}
static void norm_control(const float *x,const float *w,const float *y){
    long double square=0;for(unsigned j=0;j<D;j++)square+=(long double)x[j]*x[j];
    long double inv=1.L/sqrtl(square/D+1e-6L);double e2=0,n2=0;
    for(unsigned j=0;j<D;j++){long double ref=(long double)x[j]*inv*w[j];double e=(double)((long double)y[j]-ref);e2+=e*e;n2+=(double)(ref*ref);}
    require(n2>0&&sqrt(e2/n2)<=1e-6,"independent_norm");
}
static void quant_control(const float *x,const Input *q){
    float maximum=0;for(unsigned j=0;j<q->n;j++)if(fabsf(x[j])>maximum)maximum=fabsf(x[j]);
    float scale=maximum==0?1:maximum/32767.f;require(!memcmp(&scale,&q->s,4),"scalar_quant_scale");
    for(unsigned j=0;j<q->n;j++){
        // Round the actually specified F32 quotient via independent floor/parity.
        volatile float divided=x[j]/scale;double v=fabs((double)divided),lo=floor(v),fraction=v-lo;
        if(fraction>.5||(fraction==.5&&fmod(lo,2.)!=0))lo++;
        if(lo>32767)lo=32767;int code=(int)lo;if(divided<0)code=-code;require(q->q[j]==code,"scalar_nearest_even_code");
    }
}
static void matrix_control(Matrix *m,int is_head,uint64_t *rows,uint64_t *edges,double *e2,double *n2){
    for(unsigned b=0;b<m->b;b++)for(unsigned e=0;e<2;e++){
        size_t pos=(size_t)b*m->d*m->o+(e?(size_t)m->d*m->o-8:0);
        for(unsigned j=0;j<8;j++)require(m->w[pos+j]==expected_code(m->seed,pos+j),"stored_code_bank_offset");
        size_t row=(size_t)b*m->o+(e?m->o-1:0);float scale=expected_scale(m->seed,row,m->d);
        require(!memcmp(&scale,m->s+row,4),"stored_scale_bank_offset");(*edges)++;
    }
    for(unsigned k=0;k<m->nactive;k++)for(unsigned t=0;t<8;t++){
        unsigned row=t*(m->o-1)/7;size_t index=(size_t)m->ids[k]*m->o+row;const Input *x=m->input+(m->nquery==1?0:k);
        int64_t sum=scalar_dot(m->w+index*m->d,x->q,m->d);require(sum==head_integer_dot(m->w+index*m->d,x->q,(int)m->d),"scalar_I64_sum");
        double ref=((double)sum*(double)m->s[index])*(double)x->s;float v=(float)ref;if(is_head){v/=6.f;ref/=6.;}
        require(!memcmp(&v,m->y+k*m->o+row,4),"scalar_scaled_output");double err=(double)v-ref;*e2+=err*err;*n2+=ref*ref;(*rows)++;
    }
}
static void selftest(void){
    int8_t w[4096];int16_t x[4096];for(unsigned j=0;j<4096;j++){w[j]=127;x[j]=32767;}
    int64_t limit=UINT64_C(4096)*127*32767;require(limit>INT32_MAX&&head_integer_dot(w,x,4096)==limit&&scalar_dot(w,x,4096)==limit,"I64_global_max");
    require((int64_t)(int32_t)limit!=limit,"I32_truncation_negative");memset(w,-127,sizeof(w));require(head_integer_dot(w,x,4096)==-limit,"I64_global_min");
    const int values[]={-32767,-32766,-16384,-127,-1,0,1,127,16384,32766,32767};unsigned cases=0;
    for(int weight=-127;weight<=127;weight++)for(unsigned v=0;v<11;v++){
        for(unsigned j=0;j<31;j++){w[j]=(int8_t)(j%2?-weight:weight);x[j]=(int16_t)(j%3?-values[v]:values[v]);}
        require(head_integer_dot(w,x,31)==scalar_dot(w,x,31),"all_weight_representative_A16_mixed_tail");cases++;
    }require(cases==2805,"integer_cases");
    float zero[D]={0},ties[D]={32767,.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f};Input q;
    checked_quant(zero,D,&q);require(q.s==1,"zero_scale");for(unsigned j=0;j<D;j++)require(q.q[j]==0,"zero_code");
    checked_quant(ties,D,&q);quant_control(ties,&q);require(q.q[1]==0&&q.q[2]==2&&q.q[3]==2&&q.q[4]==0&&q.q[5]==-2&&q.q[6]==-2,"tie_codes");
    ties[0]=NAN;require(!quantize(ties,D,&q),"nonfinite_quant_rejected");ties[0]=INFINITY;require(!quantize(ties,D,&q),"infinite_quant_rejected");
    float scores[BANKS]={0},gates[TOPK];unsigned ids[TOPK];select_routes(scores,ids,gates);for(unsigned k=0;k<TOPK;k++)require(ids[k]==k&&gates[k]==.125f,"router_tie_control");
    prepare(13);execute();uint64_t rows=0,edges=0;double e2=0,n2=0;
    for(unsigned j=0;j<D;j++){
        float a=(float)expected_code(HEAD_SEED,(uint64_t)13*D+j)*expected_scale(HEAD_SEED,13,D);uint32_t bits;memcpy(&bits,&a,4);bits&=UINT32_C(0xffff0000);memcpy(&a,&bits,4);a*=12.f;
        require(!memcmp(&a,embedding_row+j,4),"BF16_lookup_times12");
    }
    for(unsigned l=0;l<LAYERS;l++){
        Layer *s=&layers[l];route_control(s);norm_control(s->ra,s->an,s->xa);norm_control(s->rf,s->fn,s->xf);
        quant_control(s->xa,&s->qa);quant_control(s->context,&s->qo);quant_control(s->xf,&s->qf);
        for(unsigned k=0;k<TOPK;k++)quant_control(s->zg+k*H,&s->qz[k]);
        for(unsigned i=0;i<6;i++)matrix_control(matrix(s,i),0,&rows,&edges,&e2,&n2);
        for(unsigned j=0;j<D;j++){
            float rf=s->ra[j]+.22f*s->o.y[j],m=0;for(unsigned k=0;k<TOPK;k++)m+=s->gates[k]*s->down.y[k*D+j];float residual=rf+.22f*m;
            require(!memcmp(&rf,s->rf+j,4)&&!memcmp(&m,s->mixture+j,4)&&!memcmp(&residual,s->residual+j,4),"residual_sorted_mixture_control");
        }
        for(unsigned k=0;k<TOPK;k++)for(unsigned j=0;j<H;j++){
            double a=s->gu.y[k*2*H+j],b=s->gu.y[k*2*H+H+j],ref=(a/(1.+exp(-a)))*b;
            require(fabs(s->zg[k*H+j]-ref)<=1e-6*fmax(1.,fabs(ref)),"packed_SiLU_gate_up");
        }
    }
    norm_control(head_residual,final_norm,head_input);quant_control(head_input,&qhead);matrix_control(&head,1,&rows,&edges,&e2,&n2);
    double he2=0,hn2=0;for(unsigned t=0;t<64;t++){
        unsigned r=t*(V-1)/63;int64_t sum=scalar_dot(head.w+(size_t)r*D,qhead.q,D);double ref=(((double)sum*head.s[r])*qhead.s)/6.;double e=head.y[r]-ref;he2+=e*e;hn2+=ref*ref;
    }
    Matrix *m=&layers[1].gu;unsigned saved=m->ids[0];m->ids[0]=m->b;int rejected=!apply(m,&layers[1].qf);m->ids[0]=saved;require(rejected,"invalid_bank_negative");
    int code_detected=0,scale_detected=0,head_detected=0;
    for(unsigned r=0;r<64&&(!code_detected||!scale_detected||!head_detected);r++){
        size_t index=(size_t)m->ids[0]*m->o+r;int64_t sum=scalar_dot(m->w+index*D,m->input->q,D);float before=(float)(((double)sum*m->s[index])*m->input->s);
        float old_scale=m->s[index];m->s[index]*=1.01f;float after=(float)(((double)sum*m->s[index])*m->input->s);m->s[index]=old_scale;if(before!=after)scale_detected=1;
        for(unsigned j=0;j<D&&(!code_detected||!head_detected);j++){
            int8_t old=m->w[index*D+j];m->w[index*D+j]=(int8_t)(old==127?126:old+1);
            int64_t altered=head_integer_dot(m->w+index*D,m->input->q,D);m->w[index*D+j]=old;if(altered!=sum)code_detected=1;
            int8_t old_head=head.w[(size_t)r*D+j];int64_t clean=scalar_dot(head.w+(size_t)r*D,qhead.q,D);head.w[(size_t)r*D+j]^=1;
            int64_t corrupt=head_integer_dot(head.w+(size_t)r*D,qhead.q,D);head.w[(size_t)r*D+j]=old_head;
            float a=(float)(((double)clean*head.s[r])*qhead.s)/6.f,b=(float)(((double)corrupt*head.s[r])*qhead.s)/6.f;if(clean!=corrupt&&a!=b)head_detected=1;
        }
    }
    require(rows==3848&&edges==3266&&n2>0&&hn2>0&&sqrt(e2/n2)<=1e-6&&sqrt(he2/hn2)<=1e-6&&code_detected&&scale_detected&&head_detected,"numeric_negatives");
    printf("{\"event\":\"selftest\",\"exact_integer_rows\":%llu,\"stored_bank_offset_checks\":%llu,\"integer_relative_l2\":%.17g,\"synthetic_head64_relative_l2\":%.17g,\"reference_route_layers\":24,\"all_weight_representative_A16_cases\":2805,\"bad_bank_rejected\":true,\"head_bit_change_detected\":true,\"code_fault_detected\":true,\"row_scale_fault_detected\":true,\"I32_global_truncation_detected\":true,\"zero_ties_nonfinite_quantizer\":true,\"scalar_all_fixture_activation_codes\":true,\"router_ties_and_sorted_softmax\":true,\"independent_norm_SiLU_residual_controls\":true,\"BF16_lookup1024_times12_exact\":true}\n",(unsigned long long)rows,(unsigned long long)edges,sqrt(e2/n2),sqrt(he2/hn2));fflush(stdout);
}
int main(int argc,char **argv){
    require(argc==3,"usage_THREADS_OUTPUT_PREFIX");requested_threads=atoi(argv[1]);require(requested_threads==3||requested_threads==6,"profile_threads");
    require(fesetround(FE_TONEAREST)==0,"round_mode");started=clock_s();worker_setup(requested_threads);initialize();
    printf("{\"event\":\"ready\",\"all_values_synthetic\":true,\"layers\":24,\"dimension\":1024,\"stored_banks\":32,\"selected\":8,\"threads\":%d,\"active_coefficients\":%llu,\"active_scales\":%llu,\"stored_coded_coefficients\":%llu,\"allocated_dynamic_bytes\":%llu,\"setup_seconds\":%.9f",requested_threads,(unsigned long long)active_i8,(unsigned long long)active_scales,(unsigned long long)stored_i8,(unsigned long long)allocated,clock_s()-started);worker_json();puts("}");fflush(stdout);
    selftest();unsigned char selected[LAYERS][BANKS];memset(selected,0,sizeof(selected));
    for(unsigned rep=0;rep<3;rep++)for(unsigned token=0;token<10;token++){
        prepare(token);double t=clock_s();execute();double seconds=clock_s()-t;
        uint64_t descriptor=executed_coeff+executed_scales+executed_router+200704+2048;
        require(executed_coeff==427819008&&executed_scales==2064384&&executed_router==3145728&&descriptor==433231872,"executed_411_full_descriptor");
        char sha[65];output(argv[2],rep,token,sha);for(unsigned l=0;l<LAYERS;l++)for(unsigned k=0;k<TOPK;k++)selected[l][layers[l].parent[k]]=1;
        printf("{\"event\":\"operator\",\"rep\":%u,\"input\":%u,\"warmup\":%s,\"seconds\":%.12f,\"complete_output_sha256\":\"%s\",\"executed_coefficients\":%llu,\"weight_descriptor_bytes\":%llu}\n",rep,token,token<2?"true":"false",seconds,sha,(unsigned long long)executed_coeff,(unsigned long long)descriptor);fflush(stdout);guards();
    }
    unsigned lo=BANKS,hi=0;for(unsigned l=0;l<LAYERS;l++){unsigned n=0;for(unsigned j=0;j<BANKS;j++)n+=selected[l][j];if(n<lo)lo=n;if(n>hi)hi=n;}
    printf("{\"event\":\"finished\",\"minimum_selected_union\":%u,\"maximum_selected_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f",lo,hi,(unsigned long long)peak_rss(),clock_s()-started);worker_json();puts("}");return 0;
}
