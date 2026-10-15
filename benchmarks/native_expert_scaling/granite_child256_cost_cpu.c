/* Synthetic projection/FFN/head operator bundle. No attention, KV, RoPE or generation.
 * Q4_K is used ONLY for expert gate/up/down. I8/A16 remains on Q/K/V/O/head.
 */
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
#include "../phase60/strat01_f32_dot_reference_generic.h"
#define STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2 1
#include "../phase60/strat01_q4k_q8k.h"
#define L 24U
#define D 1024U
#define H 512U
#define C 256U
#define E 32U
#define K 8U
#define V 49155U
#define QB 144U
typedef struct { int16_t q[D]; float s; unsigned n; } Input;
typedef struct { unsigned d,o; int8_t *w; float *s,*y; } I8;
typedef struct { uint8_t *gu_parent,*down_parent,*gu_child,*down_child; } Expert;
typedef struct {
    I8 q,k,v,o; Expert experts[E];
    float an[D],fn[D],router[E*D],selector[E*2*D];
    float ra[D],context[D],xa[D],rf[D],xf[D],residual[D],mixture[D];
    float scores[E],mass[K],selector_logits[K][2],selector_probability[K][2];
    unsigned parents[K],chosen[K];
    float gu[K][2][2*C],down[K][2][D],z[K][2][C];
    Input qa,qo,qf; strat01_q8_k_block qxf[D/C],qz[K][2];
} Layer;
static Layer layers[L]; static I8 head; static Input qhead;
static uint16_t *embedding; static float embedding_row[D],final_norm[D],head_input[D];
static uint8_t *evict; static volatile uint64_t eviction_sink;
static uint64_t allocated,stored_i8,stored_i8_scales,stored_q4_parent,stored_q4_child;
static uint64_t active_i8,active_i8_scales,active_q4,active_router,active_selector,quantized_q8_values;
static uint64_t check_q4_rows; static double check_q4_e2,check_q4_n2;
static double started; static int requested_threads, checking; static unsigned fixture_id;
static void require(int ok,const char *why){if(!ok){fprintf(stderr,"cost_failure:%s\n",why);exit(2);}}
static void *alloc(size_t n,size_t w){void *p=calloc(n,w);require(p!=NULL,"helper_allocation");return p;}
#include "meth388_switch_thread_binding.h"
static double now(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart/f.QuadPart;}
static uint64_t mix(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static float fixture(uint64_t x){return ((int)(mix(x)&511)-256)*(1.f/256.f);}
static void *large(uint64_t n){require(n&&n<=UINT64_C(6)*1024*1024*1024,"allocation_range");void *p=_aligned_malloc((size_t)n,64);require(p!=NULL,"large_allocation");allocated+=n;return p;}
static uint64_t rss(void){PROCESS_MEMORY_COUNTERS p={0};p.cb=sizeof(p);require(GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p)),"rss");return p.PeakWorkingSetSize;}
static void guard(void){require(now()-started<=120&&rss()<=UINT64_C(6)*1024*1024*1024,"process_time_or_memory");}
static int8_t code(uint64_t seed,uint64_t pos){unsigned u=(unsigned)(mix(seed+pos/8)>>(8*(pos%8)))&255;int n=u<128?(int)u:(int)u-256;return (int8_t)(n==-128?-127:n);}
static float scale(uint64_t seed,uint64_t row,unsigned d){return (1.f+(float)(mix(seed+row)&127)/128.f)/(127.f*sqrtf((float)d));}
static float a16_codes(const float *x,int16_t *q,unsigned n){require(fegetround()==FE_TONEAREST,"round_mode");float max=0;for(unsigned i=0;i<n;i++){require(isfinite(x[i]),"finite_activation");if(fabsf(x[i])>max)max=fabsf(x[i]);}float s=max?max/32767.f:1.f;for(unsigned i=0;i<n;i++){long v=lrintf(x[i]/s);q[i]=(int16_t)(v>32767?32767:v< -32767?-32767:v);}return s;}
static void quant(const float *x,unsigned n,Input *q){require(n&&n<=D,"a16_shape");q->n=n;q->s=a16_codes(x,q->q,n);}
static int64_t dot_i64(const int8_t *a,const int16_t *b,unsigned n){worker_bind();__m256i acc=_mm256_setzero_si256();unsigned i=0;for(;i+16<=n;i+=16){__m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+i)));__m256i y=_mm256_loadu_si256((const __m256i*)(b+i));acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x,y));}int32_t lanes[8];_mm256_storeu_si256((__m256i*)lanes,acc);int64_t total=0;for(int j=0;j<8;j++)total+=lanes[j];for(;i<n;i++)total+=(int64_t)a[i]*b[i];return total;}
/* M413 four-row I64-safe I8/A16 tile, retained for Q/K/V/O and full head. */
static void integer_dot4(const int8_t *a,const int16_t *b,int n,int64_t sums[4]){
    worker_bind();require(n>0&&n<=4096,"tile_integer_bound");
    __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};int i=0;
    for(;i+16<=n;i+=16){__m256i y=_mm256_loadu_si256((const __m256i *)(b+i));
        #pragma clang loop unroll(full)
        for(int r=0;r<4;r++){__m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a+(size_t)r*n+i)));acc[r]=_mm256_add_epi32(acc[r],_mm256_madd_epi16(x,y));}}
    for(int r=0;r<4;r++){int32_t values[8];_mm256_storeu_si256((__m256i *)values,acc[r]);int64_t sum=0;for(int j=0;j<8;j++)sum+=(int64_t)values[j];for(int j=i;j<n;j++)sum+=(int64_t)a[(size_t)r*n+j]*(int64_t)b[j];sums[r]=sum;}
}
static int64_t scalar_i64(const int8_t *a,const int16_t *b,unsigned n){int64_t total=0;for(unsigned i=0;i<n;i++)total+=(int64_t)a[i]*b[i];return total;}
static float scalar_f32(const float *a,const float *b,unsigned n){double total=0;for(unsigned i=0;i<n;i++){volatile float p=a[i]*b[i];total+=(double)p;}return (float)total;}
static void make_i8(I8 *m,unsigned d,unsigned o,uint64_t seed){m->d=d;m->o=o;m->w=large((uint64_t)d*o);m->s=large((uint64_t)o*4);m->y=large((uint64_t)o*4);stored_i8+=(uint64_t)d*o;stored_i8_scales+=(uint64_t)o*4;
    #pragma omp parallel for schedule(static)
    for(int64_t r=0;r<o;r++){worker_bind();m->s[r]=scale(seed,(uint64_t)r,d);for(unsigned j=0;j<d;j++)m->w[(size_t)r*d+j]=code(seed,(uint64_t)r*d+j);}
}
static void apply_i8(I8 *m,const Input *x){require(x->n==m->d,"i8_input_shape");active_i8+=(uint64_t)m->d*m->o;active_i8_scales+=(uint64_t)m->o*4;
    #pragma omp parallel for schedule(static)
    for(int64_t r=0;r<(int64_t)(m->o/4);r++){int64_t sums[4];size_t first=(size_t)r*4;integer_dot4(m->w+first*m->d,x->q,(int)m->d,sums);for(unsigned j=0;j<4;j++)m->y[first+j]=(float)(((double)sums[j]*m->s[first+j])*x->s);}
    for(unsigned r=(m->o/4)*4;r<m->o;r++){int64_t v=dot_i64(m->w+(size_t)r*m->d,x->q,m->d);m->y[r]=(float)(((double)v*m->s[r])*x->s);}
}
static void norm(const float *x,const float *w,float *y){double ss=0;for(unsigned j=0;j<D;j++)ss+=(double)x[j]*x[j];float inv=(float)(1./sqrt(ss/D+1e-6));for(unsigned j=0;j<D;j++)y[j]=x[j]*inv*w[j];}
static float silu(float x){return x/(1.f+expf(-x));}
static void q4_block(uint8_t p[QB], uint64_t seed) {
    /* Variable finite positive binary16 d/dmin and valid packed scales. */
    uint64_t s = mix(seed);
    uint16_t d = (uint16_t)(0x2800U + (s & 31U));
    uint16_t dm = (uint16_t)(0x2000U + ((s >> 5) & 31U));
    p[0] = (uint8_t)d; p[1] = (uint8_t)(d >> 8);
    p[2] = (uint8_t)dm; p[3] = (uint8_t)(dm >> 8);
    for (unsigned j = 0; j < 4; j++) {
        p[4 + j] = (uint8_t)(1U + ((s >> (j * 7)) & 15U));
        p[8 + j] = (uint8_t)(1U + ((s >> (j * 9 + 3)) & 15U));
        unsigned low = 1U + ((s >> (j * 5 + 2)) & 7U);
        unsigned high = 1U + ((s >> (j * 6 + 1)) & 7U);
        p[12 + j] = (uint8_t)(low | (high << 4));
    }
    for (unsigned j = 16; j < QB; j++) p[j] = (uint8_t)mix(seed + j);
}
static void make_expert(Expert *e,uint64_t seed){const size_t gb=(size_t)2*H*(D/C)*QB,db=(size_t)D*(H/C)*QB;e->gu_parent=large(gb);e->down_parent=large(db);e->gu_child=large(gb);e->down_child=large(db);stored_q4_parent+=gb+db;stored_q4_child+=gb+db;
    for(size_t b=0;b<gb/QB;b++)q4_block(e->gu_parent+b*QB,seed+b);for(size_t b=0;b<db/QB;b++)q4_block(e->down_parent+b*QB,seed+UINT64_C(999999)+b);
    for(unsigned child=0;child<2;child++){for(unsigned j=0;j<C;j++)for(unsigned branch=0;branch<2;branch++){unsigned old_row=branch*H+child*C+j,new_row=child*(2*C)+branch*C+j;memcpy(e->gu_child+(size_t)new_row*(D/C)*QB,e->gu_parent+(size_t)old_row*(D/C)*QB,(D/C)*QB);}
        for(unsigned r=0;r<D;r++)memcpy(e->down_child+((size_t)child*D+r)*QB,e->down_parent+((size_t)r*2+child)*QB,QB);}
    if(checking){for(unsigned child=0;child<2;child++){for(unsigned j=0;j<C;j++)for(unsigned branch=0;branch<2;branch++){unsigned old_row=branch*H+child*C+j,new_row=child*(2*C)+branch*C+j;require(!memcmp(e->gu_child+(size_t)new_row*(D/C)*QB,e->gu_parent+(size_t)old_row*(D/C)*QB,(D/C)*QB),"q4_gate_up_exact_copy");}for(unsigned r=0;r<D;r++)require(!memcmp(e->down_child+((size_t)child*D+r)*QB,e->down_parent+((size_t)r*2+child)*QB,QB),"q4_down_exact_copy");}}
}
static double decoded(const uint8_t *row,const strat01_q8_k_block *q,unsigned blocks){double v=0;for(unsigned b=0;b<blocks;b++){const uint8_t *p=row+(size_t)b*QB,*s=p+4;for(unsigned j=0;j<C;j++){unsigned g=j/32,off=j%64;uint8_t pack=p[16+(j/64)*32+off%32];unsigned nib=off<32?pack&15U:pack>>4;unsigned sc,mn;if(g<4){sc=s[g]&63U;mn=s[g+4]&63U;}else{sc=(s[g+4]&15U)|((s[g-4]>>6)<<4);mn=(s[g+4]>>4)|((s[g]>>6)<<4);}double w=strat01_q4k_q8k_fp16le(p)*sc*nib-strat01_q4k_q8k_fp16le(p+2)*mn;v+=w*q[b].d*q[b].qs[j];}}return v;}
static float q4_dot(const uint8_t *row,const strat01_q8_k_block *q,unsigned n){float v=strat01_q4k_q8k_dot(row,q,n);require(isfinite(v),"q4_finite_output");return v;}
static void q4_control(const uint8_t *row,const strat01_q8_k_block *q,unsigned n,float actual){if(!checking)return;double ref=decoded(row,q,n/C),e=(double)actual-ref;check_q4_e2+=e*e;check_q4_n2+=ref*ref;check_q4_rows++;}
static void initialize(void){for(unsigned l=0;l<L;l++){Layer *s=&layers[l];uint64_t seed=(uint64_t)(l+1)*UINT64_C(10000000000);for(unsigned j=0;j<D;j++){s->an[j]=1+.1f*fixture(seed+j);s->fn[j]=1+.1f*fixture(seed+3000+j);}for(unsigned r=0;r<E;r++){for(unsigned j=0;j<D;j++){s->router[(size_t)r*D+j]=fixture(seed+100000+(size_t)r*D+j)/sqrtf((float)D);for(unsigned c=0;c<2;c++)s->selector[((size_t)r*2+c)*D+j]=fixture(seed+200000+((size_t)r*2+c)*D+j)/sqrtf((float)D);}make_expert(&s->experts[r],seed+300000+(uint64_t)r*100000);}
        make_i8(&s->q,D,D,seed+1);make_i8(&s->k,D,H,seed+2);make_i8(&s->v,D,H,seed+3);make_i8(&s->o,D,D,seed+4);}
    make_i8(&head,D,V,UINT64_C(1234567890123));embedding=large((uint64_t)V*D*2);for(unsigned r=0;r<V;r++)for(unsigned j=0;j<D;j++){float value=(float)head.w[(size_t)r*D+j]*head.s[r];uint32_t bits;memcpy(&bits,&value,4);embedding[(size_t)r*D+j]=(uint16_t)(bits>>16);}for(unsigned j=0;j<D;j++)final_norm[j]=1+.1f*fixture(700000+j);
    require(stored_q4_parent==UINT64_C(679477248)&&stored_q4_child==stored_q4_parent,"all_expert_blocks_resident");guard();}
static void prepare(unsigned id){fixture_id=id;for(unsigned l=0;l<L;l++){Layer *s=&layers[l];uint64_t seed=(uint64_t)id*UINT64_C(1000000000)+(uint64_t)l*UINT64_C(10000000);for(unsigned j=0;j<D;j++){s->ra[j]=fixture(seed+j);s->context[j]=fixture(seed+20000+j);}}}
static void routes(Layer *s){
    #pragma omp parallel for schedule(static)
    for(int r=0;r<(int)E;r++){worker_bind();s->scores[r]=strat01_f32_dot_reference_generic(s->router+(size_t)r*D,s->xf,D);}active_router+=(uint64_t)E*D*4;unsigned ranked[K];float p[K],sum=0;for(unsigned k=0;k<K;k++){unsigned best=UINT32_MAX;for(unsigned r=0;r<E;r++){int used=0;for(unsigned z=0;z<k;z++)used|=ranked[z]==r;if(!used&&(best==UINT32_MAX||s->scores[r]>s->scores[best]))best=r;}ranked[k]=best;}for(unsigned k=0;k<K;k++){p[k]=expf(s->scores[ranked[k]]-s->scores[ranked[0]]);sum+=p[k];s->parents[k]=ranked[k];s->mass[k]=p[k]/sum;}
    /* Correct the denominator once all top logits have been accumulated. */
    for(unsigned k=0;k<K;k++)s->mass[k]=p[k]/sum;
    for(unsigned i=0;i<K;i++)for(unsigned j=i+1;j<K;j++)if(s->parents[j]<s->parents[i]){unsigned id=s->parents[i];s->parents[i]=s->parents[j];s->parents[j]=id;float m=s->mass[i];s->mass[i]=s->mass[j];s->mass[j]=m;}
}
static void selectors(Layer *s){for(unsigned k=0;k<K;k++){unsigned id=s->parents[k];for(unsigned c=0;c<2;c++)s->selector_logits[k][c]=strat01_f32_dot_reference_generic(s->selector+((size_t)id*2+c)*D,s->xf,D);float a=s->selector_logits[k][0],b=s->selector_logits[k][1],m=a>b?a:b,p0=expf(a-m),p1=expf(b-m),z=p0+p1;s->selector_probability[k][0]=p0/z;s->selector_probability[k][1]=p1/z;s->chosen[k]=b>a?1:0;}active_selector+=(uint64_t)K*2*D*4;}
static void execute(unsigned one_child){active_i8=active_i8_scales=active_q4=active_router=active_selector=quantized_q8_values=0;for(unsigned j=0;j<D;j++){uint32_t bits=(uint32_t)embedding[(size_t)fixture_id*D+j]<<16;float x;memcpy(&x,&bits,4);embedding_row[j]=x*12.f;layers[0].ra[j]=embedding_row[j];}
    for(unsigned l=0;l<L;l++){Layer *s=&layers[l];norm(s->ra,s->an,s->xa);quant(s->xa,D,&s->qa);apply_i8(&s->q,&s->qa);apply_i8(&s->k,&s->qa);apply_i8(&s->v,&s->qa);quant(s->context,D,&s->qo);apply_i8(&s->o,&s->qo);for(unsigned j=0;j<D;j++)s->rf[j]=s->ra[j]+.22f*s->o.y[j];norm(s->rf,s->fn,s->xf);routes(s);selectors(s);require(strat01_quantize_q8_k_row(s->xf,D,s->qxf),"q8_expert_input");quantized_q8_values+=D;
        for(unsigned j=0;j<D;j++)s->mixture[j]=0;
        for(unsigned k=0;k<K;k++){Expert *e=&s->experts[s->parents[k]];for(unsigned c=0;c<2;c++){if(one_child&&c!=s->chosen[k])continue;
                #pragma omp parallel for schedule(static)
                for(int row=0;row<(int)(2*C);row++){worker_bind();s->gu[k][c][row]=q4_dot(e->gu_child+((size_t)c*2*C+row)*(D/C)*QB,s->qxf,D);}
                active_q4+=(uint64_t)2*C*D;
                if(checking)for(unsigned row=0;row<2*C;row++)q4_control(e->gu_child+((size_t)c*2*C+row)*(D/C)*QB,s->qxf,D,s->gu[k][c][row]);
                for(unsigned j=0;j<C;j++)s->z[k][c][j]=silu(s->gu[k][c][j])*s->gu[k][c][C+j];require(strat01_quantize_q8_k_row(s->z[k][c],C,&s->qz[k][c]),"q8_child_down");quantized_q8_values+=C;
                #pragma omp parallel for schedule(static)
                for(int row=0;row<(int)D;row++){worker_bind();s->down[k][c][row]=q4_dot(e->down_child+((size_t)c*D+row)*QB,&s->qz[k][c],C);}
                active_q4+=(uint64_t)D*C;
                if(checking)for(unsigned row=0;row<D;row++)q4_control(e->down_child+((size_t)c*D+row)*QB,&s->qz[k][c],C,s->down[k][c][row]);
                float mass=s->mass[k]*(one_child?2.f*s->selector_probability[k][c]:1.f);for(unsigned row=0;row<D;row++)s->mixture[row]+=mass*s->down[k][c][row];}
        }for(unsigned j=0;j<D;j++)s->residual[j]=s->rf[j]+.22f*s->mixture[j];}
    norm(layers[L-1].residual,final_norm,head_input);quant(head_input,D,&qhead);apply_i8(&head,&qhead);for(unsigned r=0;r<V;r++)head.y[r]/=6.f;
}
static void control_mix(Layer *s,unsigned mode){for(unsigned k=0;k<K;k++)for(unsigned c=0;c<2;c++){float score=scalar_f32(s->selector+((size_t)s->parents[k]*2+c)*D,s->xf,D);require(!memcmp(&score,&s->selector_logits[k][c],4),"selector_scalar_logit");if(mode&&c!=s->chosen[k])continue;for(unsigned j=0;j<C;j++){double a=s->gu[k][c][j],b=s->gu[k][c][C+j],ref=(a/(1.+exp(-a)))*b;require(fabs(s->z[k][c][j]-ref)<=1e-6*fmax(1.,fabs(ref)),"independent_swiglu");}}
    for(unsigned j=0;j<D;j++){float mixed=0;for(unsigned k=0;k<K;k++)for(unsigned c=0;c<2;c++){if(mode&&c!=s->chosen[k])continue;float mass=s->mass[k]*(mode?2.f*s->selector_probability[k][c]:1.f);mixed+=mass*s->down[k][c][j];}float residual=s->rf[j]+.22f*mixed;require(isfinite(mixed)&&isfinite(residual)&&!memcmp(&mixed,s->mixture+j,4)&&!memcmp(&residual,s->residual+j,4),"independent_mixture_residual");}}
static void controls(unsigned mode){require(check_q4_rows>0&&check_q4_n2>0&&sqrt(check_q4_e2/check_q4_n2)<=1e-5,"all_q4_output_scalar_l2");uint64_t sampled=0;for(unsigned l=0;l<L;l++){Layer *s=&layers[l];I8 *ms[]={&s->q,&s->k,&s->v,&s->o};Input *xs[]={&s->qa,&s->qa,&s->qa,&s->qo};for(unsigned r=0;r<E;r++){float v=scalar_f32(s->router+(size_t)r*D,s->xf,D);require(!memcmp(&v,s->scores+r,4),"router_scalar_logit");}unsigned order[E];for(unsigned r=0;r<E;r++)order[r]=r;for(unsigned i=0;i<E;i++)for(unsigned j=i+1;j<E;j++)if(s->scores[order[j]]>s->scores[order[i]]||(s->scores[order[j]]==s->scores[order[i]]&&order[j]<order[i])){unsigned old=order[i];order[i]=order[j];order[j]=old;}float p[K],total=0;for(unsigned k=0;k<K;k++){p[k]=expf(s->scores[order[k]]-s->scores[order[0]]);total+=p[k];}for(unsigned k=0;k<K;k++){unsigned id=order[k],found=0;for(unsigned j=0;j<K;j++)if(s->parents[j]==id){require(fabsf(s->mass[j]-p[k]/total)<1e-7f,"router_source_softmax_mass");found=1;}require(found,"router_top8_ids");}for(unsigned k=1;k<K;k++)require(s->parents[k-1]<s->parents[k],"parent_sorted");float mass=0;for(unsigned k=0;k<K;k++){mass+=s->mass[k];float a=s->selector_logits[k][0],b=s->selector_logits[k][1],m=a>b?a:b,p0=expf(a-m),p1=expf(b-m),z=p0+p1;require(fabsf(s->selector_probability[k][0]-p0/z)<1e-7f&&fabsf(s->selector_probability[k][1]-p1/z)<1e-7f,"selector_softmax");require(s->chosen[k]==(unsigned)(b>a),"selector_id");}require(fabsf(mass-1.f)<1e-5f,"router_mass");for(unsigned t=0;t<4;t++){I8 *mat=ms[t];Input *x=xs[t];for(unsigned n=0;n<8;n++){unsigned row=n*(mat->o-1)/7;int64_t v=scalar_i64(mat->w+(size_t)row*mat->d,x->q,mat->d);require(v==dot_i64(mat->w+(size_t)row*mat->d,x->q,mat->d),"i8_integer_sample");float y=(float)(((double)v*mat->s[row])*x->s);require(!memcmp(&y,mat->y+row,4),"i8_scaled_sample");sampled++;}}control_mix(s,mode);}for(unsigned n=0;n<64;n++){unsigned row=n*(V-1)/63;int64_t v=scalar_i64(head.w+(size_t)row*D,qhead.q,D);require(v==dot_i64(head.w+(size_t)row*D,qhead.q,D),"head_i64_sample");float y=(float)(((double)v*head.s[row])*qhead.s)/6.f;require(!memcmp(&y,head.y+row,4),"head_scaled_sample");sampled++;}printf("{\"event\":\"control\",\"mode\":%u,\"q4_full_rows\":%llu,\"q4_relative_l2\":%.12g,\"i8_sampled_rows\":%llu,\"router_layers\":24,\"selector_layers\":24}\n",mode,(unsigned long long)check_q4_rows,sqrt(check_q4_e2/check_q4_n2),(unsigned long long)sampled);fflush(stdout);}
static void hashed(BCRYPT_HASH_HANDLE h,const void *p,size_t n){require(n<=UINT32_MAX&&BCryptHashData(h,(PUCHAR)p,(ULONG)n,0)>=0,"output_hash_part");}
static void hashed_floats(BCRYPT_HASH_HANDLE h,const float *p,size_t n){for(size_t j=0;j<n;j++)require(isfinite(p[j]),"finite_full_output");hashed(h,p,n*4);}
static void output_hash(unsigned mode,char sha[65]){
    BCRYPT_ALG_HANDLE alg=NULL; BCRYPT_HASH_HANDLE h=NULL; uint8_t bytes[32];
    require(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,NULL,0)>=0&&
            BCryptCreateHash(alg,&h,NULL,0,NULL,0,0)>=0,"hash_start");
    hashed(h,&fixture_id,sizeof(fixture_id)); hashed(h,&mode,sizeof(mode));
    hashed_floats(h,embedding_row,D);
    for(unsigned l=0;l<L;l++){
        Layer *s=&layers[l];
        hashed_floats(h,s->q.y,D); hashed_floats(h,s->k.y,H);
        hashed_floats(h,s->v.y,H); hashed_floats(h,s->o.y,D);
        hashed_floats(h,s->xa,D); hashed_floats(h,s->rf,D);
        hashed_floats(h,s->xf,D); hashed_floats(h,s->scores,E);
        hashed(h,s->parents,K*4); hashed_floats(h,s->mass,K);
        hashed(h,s->chosen,K*4);
        hashed_floats(h,&s->selector_logits[0][0],K*2);
        hashed_floats(h,&s->selector_probability[0][0],K*2);
        hashed_floats(h,s->mixture,D); hashed_floats(h,s->residual,D);
        for(unsigned k=0;k<K;k++)for(unsigned c=0;c<2;c++){
            if(mode&&c!=s->chosen[k])continue;
            hashed_floats(h,s->gu[k][c],2*C);
            hashed_floats(h,s->z[k][c],C);
            hashed_floats(h,s->down[k][c],D);
        }
    }
    hashed_floats(h,head_input,D); hashed_floats(h,head.y,V);
    require(BCryptFinishHash(h,bytes,32,0)>=0,"hash_finish");
    BCryptDestroyHash(h); BCryptCloseAlgorithmProvider(alg,0);
    for(unsigned j=0;j<32;j++)sprintf(sha+2*j,"%02x",bytes[j]);sha[64]=0;
}
static void evict_cache(void){require(evict!=NULL,"eviction_buffer");uint64_t sum=0;for(size_t p=0;p<(size_t)512*1024*1024;p+=64){evict[p]^=(uint8_t)(p>>6);sum+=evict[p];}eviction_sink+=sum;}
int main(int argc,char **argv){require(argc==3,"usage_MODE_PROFILE");int mode=atoi(argv[1]);int profile=atoi(argv[2]);require((mode==0||mode==1||mode==2)&&(profile==0||profile==1),"mode_profile");checking=mode==2;require(fesetround(FE_TONEAREST)==0,"nearest_rounding");started=now();requested_threads=3;worker_setup(3);require(worker_masks[0]==1&&worker_masks[1]==4&&worker_masks[2]==16,"physical_0_2_4_binding");initialize();require(!strat01_q4k_q8k_selftest(),"q4_header_selftest");if(profile&&!checking){evict=large((size_t)512*1024*1024);memset(evict,1,(size_t)512*1024*1024);}printf("{\"event\":\"ready\",\"schema\":\"granite_child256_cost_v1\",\"synthetic_only\":true,\"layers\":24,\"dimension\":1024,\"experts\":32,\"topk\":8,\"parent_width\":512,\"child_width\":256,\"vocab\":49155,\"stored_i8_coefficients\":%llu,\"stored_i8_scale_bytes\":%llu,\"stored_q4_parent_bytes\":%llu,\"stored_q4_child_bytes\":%llu,\"allocated_dynamic_bytes\":%llu",(unsigned long long)stored_i8,(unsigned long long)stored_i8_scales,(unsigned long long)stored_q4_parent,(unsigned long long)stored_q4_child,(unsigned long long)allocated);worker_json();puts("}");fflush(stdout);
    if(checking){for(unsigned m=0;m<2;m++){prepare(0);check_q4_rows=0;check_q4_e2=check_q4_n2=0;execute(m);controls(m);char sha[65];output_hash(m,sha);printf("{\"event\":\"check_output\",\"mode\":%u,\"sha256\":\"%s\"}\n",m,sha);}guard();return 0;}
    /* Exactly one untimed warmup, then eight fixtures x three repetitions. */
    prepare(0);execute((unsigned)mode);for(unsigned rep=0;rep<3;rep++)for(unsigned id=0;id<8;id++){prepare(id);if(profile)evict_cache();double t=now();execute((unsigned)mode);double ms=(now()-t)*1000.;char sha[65];output_hash((unsigned)mode,sha);printf("{\"event\":\"sample\",\"mode\":%d,\"profile\":%d,\"rep\":%u,\"fixture\":%u,\"ms\":%.9f,\"sha256\":\"%s\",\"i8_coefficients\":%llu,\"i8_scale_bytes\":%llu,\"q4_coefficients\":%llu,\"q4_payload_bytes\":%llu,\"router_weight_bytes\":%llu,\"selector_weight_bytes\":%llu,\"q8_quantized_values\":%llu}\n",mode,profile,rep,id,ms,sha,(unsigned long long)active_i8,(unsigned long long)active_i8_scales,(unsigned long long)active_q4,(unsigned long long)(active_q4/C*QB),(unsigned long long)active_router,(unsigned long long)active_selector,(unsigned long long)quantized_q8_values);fflush(stdout);guard();}
    printf("{\"event\":\"finished\",\"peak_rss_bytes\":%llu,\"seconds\":%.6f,\"eviction_sink\":%llu",(unsigned long long)rss(),now()-started,(unsigned long long)eviction_sink);worker_json();puts("}");return 0;}
