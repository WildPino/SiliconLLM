/* Original Qwen2.5-0.5B recipient profile. Source weights only; no learned banks. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <limits.h>
#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <bcrypt.h>
#ifndef RECIPIENT_CATALOG
#error "RECIPIENT_CATALOG must name the generated catalog"
#endif
#include RECIPIENT_CATALOG

/* Catalog contract: RcField {name,dtype,rows,cols,offset,bytes}; codes
   F32=1, BF16=2, W8=3, W4G32=4. All fields are rank-two, including vectors
   as [1,n]. W8 rows are [F32 scale][signed I8 codes]. W4 groups are
   [IEEE binary16 scale][16 packed bytes], low nibble for columns 0..15. */
#define RC_L 24
#define RC_D 896
#define RC_H 4864
#define RC_V 151936
#define RC_NH 14
#define RC_NKV 2
#define RC_HD 64
#define RC_KD 128
#define RC_EPS 1.0e-6
#define RC_THETA 1000000.0
typedef struct {
    const uint8_t *q,*k,*v,*o,*gate,*up,*down;
    const float *qb,*kb,*vb,*in_norm,*post_norm;
} RcLayer;
typedef struct {
    float x[RC_D],norm[RC_D],q[RC_D],k[RC_KD],v[RC_KD],attn[RC_D],tmp[RC_D];
    float active[RC_H],up[RC_H],logits[RC_V],head_norm[RC_D];
    int8_t qa[RC_H];float qscale[RC_H/32];
    float *kc,*vc,*scores;int capacity;
} RcState;
static uint8_t *rc_blob;
static RcLayer rc_layer[RC_L];
static const uint16_t *rc_embed;
static const float *rc_last_norm;
static const uint8_t *rc_head;
static unsigned char rc_used[RC_FIELD_COUNT];
static int rc_threads;
static void require(int ok,const char *what){if(!ok){fprintf(stderr,"recipient: %s\n",what);exit(2);}}
static unsigned char *alloc(size_t bytes,size_t align){(void)align;unsigned char *p=(unsigned char*)malloc(bytes);require(p!=NULL,"worker allocation");return p;}
#include "meth388_switch_thread_binding.h"
static void rc_die(const char *s){require(0,s);}
static float rc_bf16(uint16_t v){uint32_t u=(uint32_t)v<<16;float f;memcpy(&f,&u,4);return f;}
static float rc_fp16(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16,exp=(h>>10)&31u,m=h&1023u,u;
    if(exp==31u)u=sign|0x7f800000u|(m<<13);
    else if(exp){u=sign|((exp+112u)<<23)|(m<<13);}
    else if(!m)u=sign;
    else{int e=-14;while(!(m&1024u)){m<<=1;e--;}u=sign|((uint32_t)(e+127)<<23)|((m&1023u)<<13);}
    float f;memcpy(&f,&u,4);return f;
}
static uint16_t rc_u16(const void *p){uint16_t v;memcpy(&v,p,2);return v;}
static float rc_f32(const void *p){float v;memcpy(&v,p,4);return v;}
static double rc_clock(void){LARGE_INTEGER a,b;require(QueryPerformanceCounter(&a)&&QueryPerformanceFrequency(&b),"clock");return (double)a.QuadPart/(double)b.QuadPart;}
static void rc_sha(const uint8_t *p,uint64_t n,char out[65]){
    BCRYPT_ALG_HANDLE a=NULL;BCRYPT_HASH_HANDLE h=NULL;uint8_t digest[32];
    require(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,NULL,0)>=0,"SHA provider");
    require(BCryptCreateHash(a,&h,NULL,0,NULL,0,0)>=0,"SHA create");
    while(n){ULONG step=(ULONG)(n>0x40000000u?0x40000000u:n);require(BCryptHashData(h,(PUCHAR)p,step,0)>=0,"SHA update");p+=step;n-=step;}
    require(BCryptFinishHash(h,digest,32,0)>=0,"SHA finish");BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);
    for(int i=0;i<32;i++)sprintf(out+2*i,"%02x",digest[i]);out[64]=0;
}
static uint64_t rc_expected(const RcField *f){
    require(f->rows>0&&f->cols>0&&f->rows<=UINT64_MAX/f->cols,"field dimensions");
    uint64_t rows=f->rows,cols=f->cols;
    if(f->dtype==1)return rows*cols*4;
    if(f->dtype==2)return rows*cols*2;
    if(f->dtype==3)return rows*(cols+4);
    if(f->dtype==4){require(cols%32==0,"W4 group alignment");return rows*(cols/32)*18;}
    rc_die("field codec");return 0;
}
static const uint8_t *rc_bind(const char *name,int dtype,uint64_t rows,uint64_t cols){
    for(int i=0;i<RC_FIELD_COUNT;i++)if(!strcmp(rc_fields[i].name,name)){
        const RcField *f=&rc_fields[i];require(!rc_used[i],"duplicate role");
        require(f->dtype==dtype&&f->rows==rows&&f->cols==cols,"role dtype/shape");
        rc_used[i]=1;return rc_blob+f->offset;
    }
    fprintf(stderr,"recipient: missing role %s\n",name);exit(2);
}
static void rc_load(const char *path){
    require(RC_FIELD_COUNT==291,"recipient catalog field count");
    FILE *f=fopen(path,"rb");require(f!=NULL,"archive open");
    require(!_fseeki64(f,0,SEEK_END),"archive seek");
    require((uint64_t)_ftelli64(f)==RC_ARCHIVE_BYTES,"archive size");
    require(!_fseeki64(f,0,SEEK_SET)&&RC_ARCHIVE_BYTES<=SIZE_MAX,"archive seek/size");
    rc_blob=(uint8_t*)malloc((size_t)RC_ARCHIVE_BYTES);require(rc_blob!=NULL,"archive allocation");
    require(fread(rc_blob,1,(size_t)RC_ARCHIVE_BYTES,f)==RC_ARCHIVE_BYTES,"archive read");
    require(fclose(f)==0,"archive close");char hash[65];rc_sha(rc_blob,RC_ARCHIVE_BYTES,hash);
    require(strcmp(hash,RC_ARCHIVE_SHA)==0,"archive SHA");
    uint64_t cursor=0;
    for(int i=0;i<RC_FIELD_COUNT;i++){
        const RcField *x=&rc_fields[i];require(x->name&&x->name[0],"field name");
        for(int j=0;j<i;j++)require(strcmp(x->name,rc_fields[j].name)!=0,"duplicate field name");
        require(x->bytes==rc_expected(x),"field byte count");
        require(x->offset%64==0&&x->offset>=cursor&&x->offset<=RC_ARCHIVE_BYTES&&(i||x->offset==0),"archive field order/alignment");
        for(uint64_t pad=cursor;pad<x->offset;pad++)require(rc_blob[pad]==0,"nonzero archive padding");
        require(x->bytes<=RC_ARCHIVE_BYTES-x->offset,"archive field range");cursor=x->offset+x->bytes;
        const uint8_t *p=rc_blob+x->offset;
        if(x->dtype==1){require(x->offset%4==0,"F32 alignment");for(uint64_t k=0;k<x->bytes;k+=4)require(isfinite(rc_f32(p+k)),"nonfinite F32");}
        if(x->dtype==2){require(x->offset%2==0,"BF16 alignment");for(uint64_t k=0;k<x->bytes;k+=2)require(isfinite(rc_bf16(rc_u16(p+k))),"nonfinite BF16");}
        if(x->dtype==3){uint64_t pitch=x->cols+4;for(uint64_t row=0;row<x->rows;row++){
            const uint8_t *v=p+row*pitch;float scale=rc_f32(v);require(isfinite(scale)&&scale>=0,"W8 scale");
            if(scale==0)for(uint64_t k=0;k<x->cols;k++)require(v[4+k]==0,"zero W8 row codes");
        }}
        if(x->dtype==4){uint64_t groups=x->cols/32;for(uint64_t row=0;row<x->rows;row++)for(uint64_t g=0;g<groups;g++){
            float scale=rc_fp16(rc_u16(p+(row*groups+g)*18));require(isfinite(scale)&&scale>=0,"W4 scale");
            const uint8_t *codes=p+(row*groups+g)*18+2;
            if(scale==0)for(int j=0;j<16;j++)require(codes[j]==0x88u,"zero W4 block codes");
        }}
    }
    require(cursor<=RC_ARCHIVE_BYTES,"archive end");
    for(uint64_t pad=cursor;pad<RC_ARCHIVE_BYTES;pad++)require(rc_blob[pad]==0,"nonzero trailing padding");
    rc_embed=(const uint16_t*)rc_bind("model.embed_tokens.weight",2,RC_V,RC_D);
    rc_last_norm=(const float*)rc_bind("model.norm.weight",1,1,RC_D);
    rc_head=rc_bind("lm_head.weight",3,RC_V,RC_D);
    for(int i=0;i<RC_L;i++){
        RcLayer *l=&rc_layer[i];char name[128];
        #define RC_ROLE(suffix,kind,r,c) (snprintf(name,sizeof name,"model.layers.%d." suffix,i),rc_bind(name,kind,r,c))
        l->q=RC_ROLE("self_attn.q_proj.weight",3,RC_D,RC_D);
        l->k=RC_ROLE("self_attn.k_proj.weight",3,RC_KD,RC_D);
        l->v=RC_ROLE("self_attn.v_proj.weight",3,RC_KD,RC_D);
        l->o=RC_ROLE("self_attn.o_proj.weight",3,RC_D,RC_D);
        l->qb=(const float*)RC_ROLE("self_attn.q_proj.bias",1,1,RC_D);
        l->kb=(const float*)RC_ROLE("self_attn.k_proj.bias",1,1,RC_KD);
        l->vb=(const float*)RC_ROLE("self_attn.v_proj.bias",1,1,RC_KD);
        l->in_norm=(const float*)RC_ROLE("input_layernorm.weight",1,1,RC_D);
        l->post_norm=(const float*)RC_ROLE("post_attention_layernorm.weight",1,1,RC_D);
        l->gate=RC_ROLE("mlp.gate_proj.weight",4,RC_H,RC_D);
        l->up=RC_ROLE("mlp.up_proj.weight",4,RC_H,RC_D);
        l->down=RC_ROLE("mlp.down_proj.weight",4,RC_D,RC_H);
        #undef RC_ROLE
    }
    for(int i=0;i<RC_FIELD_COUNT;i++)require(rc_used[i],"orphan catalog field");
}
static void rc_quant(const float *x,int n,int8_t *codes,float *scales,int group){
    int step=group?32:n;require(n%step==0,"activation quant shape");
    for(int base=0;base<n;base+=step){float maxabs=0;
        for(int j=0;j<step;j++){require(isfinite(x[base+j]),"nonfinite activation");float a=fabsf(x[base+j]);if(a>maxabs)maxabs=a;}
        float scale=maxabs>0?maxabs/127.0f:1.0f;require(isfinite(scale)&&scale>0,"activation scale");scales[base/step]=scale;
        for(int j=0;j<step;j++){float q=nearbyintf(x[base+j]/scale);if(q>127)q=127;if(q< -127)q=-127;codes[base+j]=(int8_t)q;}
    }
}
static int64_t rc_i8dot(const int8_t *w,const int8_t *x,int n){
    __m256i acc=_mm256_setzero_si256();int i=0;
    for(;i+32<=n;i+=32){
        __m256i wa=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w+i)));
        __m256i xa=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x+i)));
        __m256i wb=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w+i+16)));
        __m256i xb=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x+i+16)));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(wa,xa));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(wb,xb));
    }
    int32_t lanes[8];_mm256_storeu_si256((__m256i*)lanes,acc);int64_t sum=0;
    for(int k=0;k<8;k++)sum+=lanes[k];for(;i<n;i++)sum+=(int)w[i]*(int)x[i];return sum;
}
static void rc_w8(const uint8_t *w,const int8_t *x,float as,float *out,int rows,int cols){
    size_t pitch=(size_t)cols+4;
    #pragma omp parallel for schedule(static) if(rc_threads>1)
    for(int row=0;row<rows;row++){worker_bind();const uint8_t *p=w+(size_t)row*pitch;
        float ws=rc_f32(p);float y=(float)((double)rc_i8dot((const int8_t*)(p+4),x,cols)*(double)ws*(double)as);
        require(isfinite(y),"nonfinite W8 output");out[row]=y;
    }
}
static int32_t rc_w4dot(const uint8_t *p,const int8_t *x){
    __m128i packed=_mm_loadu_si128((const __m128i*)(p+2));
    __m128i low=_mm_and_si128(packed,_mm_set1_epi8(15));
    __m128i high=_mm_and_si128(_mm_srli_epi16(packed,4),_mm_set1_epi8(15));
    __m256i weights=_mm256_inserti128_si256(_mm256_castsi128_si256(low),high,1);
    __m256i values=_mm256_loadu_si256((const __m256i*)x);
    /* Unsigned weights <=15: pair sums <=3810, so maddubs cannot saturate.
       Subtracting eight times the signed activation sum recovers q in [-8,7]. */
    __m256i ones16=_mm256_set1_epi16(1);
    __m256i products=_mm256_madd_epi16(_mm256_maddubs_epi16(weights,values),ones16);
    __m256i sums=_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1),values),ones16);
    __m256i adjusted=_mm256_sub_epi32(products,_mm256_slli_epi32(sums,3));
    int32_t lanes[8];_mm256_storeu_si256((__m256i*)lanes,adjusted);int32_t sum=0;
    for(int j=0;j<8;j++)sum+=lanes[j];return sum;
}
static void rc_w4(const uint8_t *w,const int8_t *x,const float *as,float *out,int rows,int cols){
    int groups=cols/32;size_t pitch=(size_t)groups*18;
    #pragma omp parallel for schedule(static) if(rc_threads>1)
    for(int row=0;row<rows;row++){worker_bind();const uint8_t *p=w+(size_t)row*pitch;double y=0;
        for(int g=0;g<groups;g++){const uint8_t *block=p+(size_t)g*18;
            y+=(double)rc_w4dot(block,x+32*g)*(double)_cvtsh_ss(rc_u16(block))*(double)as[g];}
        require(isfinite(y),"nonfinite W4 output");out[row]=(float)y;
    }
}
static void rc_norm(const float *x,const float *w,float *out){
    double sum=0;for(int i=0;i<RC_D;i++){require(isfinite(x[i]),"nonfinite norm input");sum+=(double)x[i]*x[i];}
    double inv=1.0/sqrt(sum/RC_D+RC_EPS);
    for(int i=0;i<RC_D;i++){out[i]=(float)((double)x[i]*inv*(double)w[i]);require(isfinite(out[i]),"nonfinite norm");}
}
static float rc_silu(float x){if(x>=0)return x/(1.0f+expf(-x));float e=expf(x);return x*e/(1.0f+e);}
static void rc_rope(float *x,int heads,const float *co,const float *si){
    for(int h=0;h<heads;h++)for(int j=0;j<32;j++){
        float a=x[h*64+j],b=x[h*64+j+32];x[h*64+j]=a*co[j]-b*si[j];x[h*64+j+32]=b*co[j]+a*si[j];
    }
}
static RcState *rc_state(int capacity){
    RcState *s=(RcState*)calloc(1,sizeof(RcState));require(s!=NULL,"state allocation");s->capacity=capacity;
    size_t cells=(size_t)RC_L*capacity*RC_KD;
    s->kc=(float*)calloc(cells,sizeof(float));s->vc=(float*)calloc(cells,sizeof(float));
    s->scores=(float*)calloc((size_t)RC_NH*capacity,sizeof(float));
    require(s->kc&&s->vc&&s->scores,"KV allocation");return s;
}
static void rc_reset(RcState *s){size_t bytes=(size_t)RC_L*s->capacity*RC_KD*sizeof(float);memset(s->kc,0,bytes);memset(s->vc,0,bytes);}
static void rc_forward(RcState *s,int token,int pos,int head){
    require(token>=0&&token<RC_V&&pos>=0&&pos<s->capacity,"token/context");
    const uint16_t *embedding=rc_embed+(size_t)token*RC_D;
    for(int i=0;i<RC_D;i++)s->x[i]=rc_bf16(embedding[i]);
    float co[32],si[32];for(int j=0;j<32;j++){
        double freq=pow(RC_THETA,-(double)(2*j)/RC_HD),phase=(double)pos*freq;
        co[j]=(float)cos(phase);si[j]=(float)sin(phase);
    }
    for(int li=0;li<RC_L;li++){
        const RcLayer *l=&rc_layer[li];rc_norm(s->x,l->in_norm,s->norm);
        rc_quant(s->norm,RC_D,s->qa,s->qscale,0);
        rc_w8(l->q,s->qa,s->qscale[0],s->q,RC_D,RC_D);
        rc_w8(l->k,s->qa,s->qscale[0],s->k,RC_KD,RC_D);
        rc_w8(l->v,s->qa,s->qscale[0],s->v,RC_KD,RC_D);
        for(int i=0;i<RC_D;i++)s->q[i]+=l->qb[i];
        for(int i=0;i<RC_KD;i++){s->k[i]+=l->kb[i];s->v[i]+=l->vb[i];}
        rc_rope(s->q,RC_NH,co,si);rc_rope(s->k,RC_NKV,co,si);
        memcpy(s->kc+((size_t)li*s->capacity+pos)*RC_KD,s->k,sizeof s->k);
        memcpy(s->vc+((size_t)li*s->capacity+pos)*RC_KD,s->v,sizeof s->v);
        #pragma omp parallel for schedule(static) if(rc_threads>1)
        for(int h=0;h<RC_NH;h++){
            worker_bind();
            float *scores=s->scores+(size_t)h*s->capacity;float maximum=-INFINITY;
            for(int t=0;t<=pos;t++){
                const float *k=s->kc+((size_t)li*s->capacity+t)*RC_KD+(h/7)*64;
                __m256d dot4=_mm256_setzero_pd();for(int j=0;j<64;j+=4){
                    __m256d qv=_mm256_cvtps_pd(_mm_loadu_ps(s->q+h*64+j));
                    __m256d kv=_mm256_cvtps_pd(_mm_loadu_ps(k+j));
                    dot4=_mm256_add_pd(dot4,_mm256_mul_pd(qv,kv));}
                double lanes[4];_mm256_storeu_pd(lanes,dot4);double dot=lanes[0]+lanes[1]+lanes[2]+lanes[3];
                float z=(float)(dot*0.125);require(isfinite(z),"nonfinite attention score");scores[t]=z;if(z>maximum)maximum=z;
            }
            double total=0;for(int t=0;t<=pos;t++){scores[t]=expf(scores[t]-maximum);total+=scores[t];}
            require(isfinite(total)&&total>0,"attention denominator");
            for(int t=0;t<=pos;t++)scores[t]=(float)((double)scores[t]/total);
            __m256d values[16];for(int j=0;j<16;j++)values[j]=_mm256_setzero_pd();
            for(int t=0;t<=pos;t++){
                const float *v=s->vc+((size_t)li*s->capacity+t)*RC_KD+(h/7)*64;
                __m256d score=_mm256_set1_pd((double)scores[t]);
                for(int j=0;j<16;j++)values[j]=_mm256_add_pd(values[j],_mm256_mul_pd(score,_mm256_cvtps_pd(_mm_loadu_ps(v+4*j))));
            }
            for(int j=0;j<16;j++)_mm_storeu_ps(s->attn+h*64+4*j,_mm256_cvtpd_ps(values[j]));
            for(int j=0;j<64;j++)require(isfinite(s->attn[h*64+j]),"attention output");
        }
        rc_quant(s->attn,RC_D,s->qa,s->qscale,0);
        rc_w8(l->o,s->qa,s->qscale[0],s->tmp,RC_D,RC_D);
        for(int i=0;i<RC_D;i++)s->x[i]+=s->tmp[i];
        rc_norm(s->x,l->post_norm,s->norm);
        rc_quant(s->norm,RC_D,s->qa,s->qscale,1);
        rc_w4(l->gate,s->qa,s->qscale,s->active,RC_H,RC_D);
        /* The up projection must see the same quantized post-attention norm. */
        rc_w4(l->up,s->qa,s->qscale,s->up,RC_H,RC_D);
        for(int i=0;i<RC_H;i++){s->active[i]=rc_silu(s->active[i])*s->up[i];require(isfinite(s->active[i]),"FFN activation");}
        rc_quant(s->active,RC_H,s->qa,s->qscale,1);
        rc_w4(l->down,s->qa,s->qscale,s->tmp,RC_D,RC_H);
        for(int i=0;i<RC_D;i++){s->x[i]+=s->tmp[i];require(isfinite(s->x[i]),"residual");}
    }
    rc_norm(s->x,rc_last_norm,s->head_norm);
    if(head){
        rc_quant(s->head_norm,RC_D,s->qa,s->qscale,0);
        rc_w8(rc_head,s->qa,s->qscale[0],s->logits,RC_V,RC_D);
        int top[8];for(int k=0;k<8;k++)top[k]=-1;
        for(int id=0;id<RC_V;id++)for(int k=0;k<8;k++){
            if(top[k]<0||s->logits[id]>s->logits[top[k]]||(s->logits[id]==s->logits[top[k]]&&id<top[k])){
                for(int j=7;j>k;j--)top[j]=top[j-1];top[k]=id;break;
            }
        }
        for(int k=0;k<8;k++){int id=top[k];double z=0;const uint16_t *row=rc_embed+(size_t)id*RC_D;
            for(int j=0;j<RC_D;j++)z+=(double)rc_bf16(row[j])*s->head_norm[j];
            s->logits[id]=(float)z;require(isfinite(s->logits[id]),"refined head");}
    }
}
static int rc_argmax(const float *x){int id=0;for(int i=1;i<RC_V;i++)if(x[i]>x[id])id=i;return id;}
static void rc_write(const void *p,size_t n){require(fwrite(p,1,n,stdout)==n,"output write");}
static void rc_read(void *p,size_t n){require(fread(p,1,n,stdin)==n,"input read");}
static void rc_binding(const char *phase){
    fprintf(stderr,"{\"phase\":\"%s\",\"pid\":%lu,\"worker_masks\":[",phase,(unsigned long)GetCurrentProcessId());
    for(int i=0;i<rc_threads;i++){
        WorkerBinding b=worker_latest[i];require(b.tid&&b.slot==i&&b.group==0&&b.mask==worker_masks[i],"worker identity");
        HANDLE thread=OpenThread(THREAD_QUERY_LIMITED_INFORMATION,FALSE,b.tid);require(thread!=NULL,"worker liveness");
        GROUP_AFFINITY actual;memset(&actual,0,sizeof actual);int ok=GetThreadGroupAffinity(thread,&actual);CloseHandle(thread);
        require(ok&&actual.Group==0&&actual.Mask==b.mask,"worker drift");
        fprintf(stderr,"%s%llu",i?",":"",(unsigned long long)actual.Mask);
    }
    fprintf(stderr,"]}\n");
}
static void rc_probe(RcState *s){
    for(;;){char magic[8];size_t got=fread(magic,1,8,stdin);if(!got&&feof(stdin))break;
        require(got==8&&!memcmp(magic,"QWCP0001",8),"probe magic");uint32_t h[2];rc_read(h,8);
        uint32_t n=h[0],m=h[1];require(n>=1&&n<=(uint32_t)s->capacity&&m>=1&&m<=n,"probe bounds");
        uint32_t *positions=(uint32_t*)malloc((size_t)m*4),*ids=(uint32_t*)malloc((size_t)n*4);
        require(positions&&ids,"probe allocation");rc_read(positions,(size_t)m*4);rc_read(ids,(size_t)n*4);
        for(uint32_t i=0;i<m;i++)require(positions[i]<n&&(!i||positions[i]>positions[i-1]),"probe positions");
        for(uint32_t i=0;i<n;i++)require(ids[i]<RC_V,"probe token");rc_reset(s);
        uint32_t response[4]={RC_V,m,RC_L,RC_D};rc_write("QWCT0001",8);rc_write(response,16);
        uint32_t next=0;for(uint32_t i=0;i<n;i++){
            int chosen=next<m&&positions[next]==i;rc_forward(s,(int)ids[i],(int)i,chosen);
            if(chosen){rc_write(&i,4);rc_write(s->head_norm,sizeof s->head_norm);rc_write(s->logits,sizeof s->logits);next++;}
        }
        require(next==m&&fflush(stdout)==0,"probe flush");free(positions);free(ids);
    }
}
static void rc_stream(RcState *s){
    for(;;){char magic[8];size_t got=fread(magic,1,8,stdin);if(!got&&feof(stdin))break;
        require(got==8&&!memcmp(magic,"QWCR0001",8),"stream magic");uint32_t h[2];rc_read(h,8);
        uint32_t n=h[0],maxnew=h[1];require(n>=1&&n<(uint32_t)s->capacity&&maxnew>=1&&maxnew<=128&&maxnew<=(uint32_t)s->capacity-n,"stream bounds");
        int32_t *ids=(int32_t*)malloc((size_t)n*4);require(ids!=NULL,"stream allocation");rc_read(ids,(size_t)n*4);
        for(uint32_t i=0;i<n;i++)require(ids[i]>=0&&ids[i]<RC_V,"stream token");rc_reset(s);
        double start=rc_clock();for(uint32_t i=0;i<n;i++)rc_forward(s,ids[i],(int)i,i+1==n);
        double prefill=rc_clock()-start;fprintf(stderr,"{\"phase\":\"prefill\",\"context\":%u,\"seconds\":%.9f,\"threads\":%d}\n",n,prefill,rc_threads);
        int32_t generated[128];uint32_t count=0,stop=0;
        for(uint32_t i=0;i<maxnew;i++){
            double step=rc_clock();int id=rc_argmax(s->logits);generated[count++]=id;
            if(id==151645||id==151643){stop=1;}
            else rc_forward(s,id,(int)(n+i),1);
            double seconds=rc_clock()-step;
            fprintf(stderr,"{\"phase\":\"decode\",\"index\":%u,\"generated_id\":%d,\"context\":%u,\"native_forward\":%s,\"seconds\":%.9f,\"threads\":%d}\n",i,id,n+i,stop?"false":"true",seconds,rc_threads);
            if(stop)break;
        }
        double elapsed=rc_clock()-start;uint32_t response[2]={count,stop};
        rc_write("QWCO0001",8);rc_write(response,8);rc_write(&elapsed,8);rc_write(generated,(size_t)count*4);
        require(fflush(stdout)==0,"stream flush");free(ids);
    }
}
int main(int argc,char **argv){
    require(argc==5&&(!strcmp(argv[2],"stream")||!strcmp(argv[2],"probe")),"usage: engine ARCHIVE stream|probe CAPACITY 1|3|6");
    char *end=NULL;long cap=strtol(argv[3],&end,10);require(end&&!*end&&cap>=2&&cap<=4096,"capacity");
    long threads=strtol(argv[4],&end,10);require(end&&!*end&&(threads==1||threads==3||threads==6),"threads");rc_threads=(int)threads;
    require(_setmode(_fileno(stdin),_O_BINARY)>=0&&_setmode(_fileno(stdout),_O_BINARY)>=0,"binary mode");
    require(fesetround(FE_TONEAREST)==0&&fegetround()==FE_TONEAREST,"rounding");
    require(__builtin_cpu_supports("avx2"),"AVX2 required");
    worker_setup(rc_threads);for(int i=0;i<rc_threads;i++)require(worker_masks[i]==((DWORD_PTR)1<<(2*i)),"physical worker indices 0,2,4,6,8,10");
    rc_load(argv[1]);RcState *s=rc_state((int)cap);rc_binding("binding_start");
    if(!strcmp(argv[2],"probe"))rc_probe(s);else rc_stream(s);
    rc_binding("binding_end");
    free(s->kc);free(s->vc);free(s->scores);free(s);free(rc_blob);return 0;
}
