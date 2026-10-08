// Complete compact Qwen profile for engine.c. Requires one actual generated
// catalog. No source dense-FFN fallback; native qualification remains separate.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <bcrypt.h>
#ifndef CHATBOT_COMPACT_CATALOG
#error "CHATBOT_COMPACT_CATALOG must identify the bound exported catalog"
#endif
#include CHATBOT_COMPACT_CATALOG

typedef struct {
    const uint16_t *q,*k,*v,*o,*qb,*kb,*vb,*in_norm,*post_norm;
    const uint16_t *shared_g,*shared_u,*shared_b,*leaf_g,*leaf_u,*leaf_b;
    const float *projection,*parents,*parent_norms;
} CbLayer;
typedef struct {
    float x[CB_D],norm[CB_D],q[CB_D],k[128],v[128],attention[CB_D],tmp[CB_D];
    float logits[CB_V],activation[CB_HS+4*CB_HF],query[CB_Q],mass[4];
    int selected[4];float *kc,*vc,*scores;int capacity;
    int trace_enabled;uint16_t trace_input[CB_L][CB_D];
    float trace_output[CB_L][CB_D],trace_query[CB_L][CB_Q],trace_mass[CB_L][4];
    int32_t trace_ids[CB_L][4];
} CbState;
static uint8_t *cb_blob;
static CbLayer cb_layers[CB_L];
static const uint16_t *cb_embedding,*cb_final_norm;
static const float *cb_inv_freq;
static int cb_used[CB_FIELD_COUNT];

static void cb_die(const char *message){fprintf(stderr,"compact chatbot: %s\n",message);exit(2);}
static float cb_decode(uint16_t value){uint32_t bits=(uint32_t)value<<16;float result;memcpy(&result,&bits,4);return result;}
static uint16_t cb_bits(float value){
    uint32_t bits;memcpy(&bits,&value,4);
    if((bits&0x7f800000u)!=0x7f800000u)bits+=0x7fffu+((bits>>16)&1u);
    return (uint16_t)(bits>>16);
}
static float cb_round(float value){return cb_decode(cb_bits(value));}
static __m256 cb_decode8(const uint16_t *p){
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)),16));
}
static float cb_sum4(__m256 a,__m256 b,__m256 c,__m256 d){
    float lanes[8];_mm256_storeu_ps(lanes,_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d)));
    float sum=0;for(int i=0;i<8;i++)sum+=lanes[i];return sum;
}
static float cb_bfdot(const uint16_t *w,const float *x,int count){
    __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
    for(int i=0;i<count;i+=32){
        a=_mm256_fmadd_ps(cb_decode8(w+i),_mm256_loadu_ps(x+i),a);
        b=_mm256_fmadd_ps(cb_decode8(w+i+8),_mm256_loadu_ps(x+i+8),b);
        c=_mm256_fmadd_ps(cb_decode8(w+i+16),_mm256_loadu_ps(x+i+16),c);
        d=_mm256_fmadd_ps(cb_decode8(w+i+24),_mm256_loadu_ps(x+i+24),d);
    }
    return cb_sum4(a,b,c,d);
}
static float cb_fdot(const float *w,const float *x,int count){
    __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
    for(int i=0;i<count;i+=32){
        a=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i),a);
        b=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+8),_mm256_loadu_ps(x+i+8),b);
        c=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+16),_mm256_loadu_ps(x+i+16),c);
        d=_mm256_fmadd_ps(_mm256_loadu_ps(w+i+24),_mm256_loadu_ps(x+i+24),d);
    }
    return cb_sum4(a,b,c,d);
}
static void cb_sha(const uint8_t *data,uint64_t bytes,char hex[65]){
    BCRYPT_ALG_HANDLE alg=NULL;BCRYPT_HASH_HANDLE hash=NULL;uint8_t result[32];
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,NULL,0)<0 ||
       BCryptCreateHash(alg,&hash,NULL,0,NULL,0,0)<0)cb_die("SHA init");
    while(bytes){ULONG chunk=(ULONG)(bytes>0x40000000u?0x40000000u:bytes);
        if(BCryptHashData(hash,(PUCHAR)data,chunk,0)<0)cb_die("SHA update");data+=chunk;bytes-=chunk;}
    if(BCryptFinishHash(hash,result,32,0)<0)cb_die("SHA finish");
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
    for(int i=0;i<32;i++)sprintf(hex+2*i,"%02x",result[i]);hex[64]=0;
}
static const void *cb_bind(const char *name,int kind,int rank,const uint64_t *shape){
    for(int i=0;i<CB_FIELD_COUNT;i++)if(!strcmp(cb_fields[i].name,name)){
        const CbField *f=&cb_fields[i];if(f->dtype!=kind || f->rank!=rank || cb_used[i])cb_die("field type/rank/duplicate");
        for(int j=0;j<rank;j++)if(f->shape[j]!=shape[j])cb_die("field shape");
        cb_used[i]=1;return cb_blob+f->offset;
    }
    cb_die("missing field");return NULL;
}
static const void *cb_1(const char *n,int k,uint64_t a){return cb_bind(n,k,1,&a);}
static const void *cb_2(const char *n,int k,uint64_t a,uint64_t b){uint64_t s[2]={a,b};return cb_bind(n,k,2,s);}
static const void *cb_3(const char *n,int k,uint64_t a,uint64_t b,uint64_t c){uint64_t s[3]={a,b,c};return cb_bind(n,k,3,s);}
static void cb_load(const char *path){
    FILE *file=fopen(path,"rb");if(!file)cb_die("archive open");
    if(_fseeki64(file,0,SEEK_END) || (uint64_t)_ftelli64(file)!=CB_ARCHIVE_BYTES || _fseeki64(file,0,SEEK_SET))cb_die("archive length");
    cb_blob=malloc((size_t)CB_ARCHIVE_BYTES);if(!cb_blob)cb_die("archive allocation");
    if(fread(cb_blob,1,(size_t)CB_ARCHIVE_BYTES,file)!=(size_t)CB_ARCHIVE_BYTES || fclose(file))cb_die("archive read");
    char hash[65];cb_sha(cb_blob,CB_ARCHIVE_BYTES,hash);if(strcmp(hash,CB_ARCHIVE_SHA))cb_die("archive SHA");
    for(int i=0;i<CB_FIELD_COUNT;i++){
        const CbField *f=&cb_fields[i];if(f->offset>CB_ARCHIVE_BYTES || f->bytes>CB_ARCHIVE_BYTES-f->offset)cb_die("field range");
        if(f->dtype==1){if(f->offset%4 || f->bytes%4)cb_die("F32 alignment");const float *p=(const float*)(cb_blob+f->offset);
            for(uint64_t j=0;j<f->bytes/4;j++)if(!isfinite(p[j]))cb_die("nonfinite F32 field");}
        else if(f->dtype==2){if(f->offset%2 || f->bytes%2)cb_die("BF16 alignment");const uint16_t *p=(const uint16_t*)(cb_blob+f->offset);
            for(uint64_t j=0;j<f->bytes/2;j++)if(!isfinite(cb_decode(p[j])))cb_die("nonfinite BF16 field");}
        else cb_die("unsupported codec");
    }
    cb_embedding=cb_2("model.embed_tokens.weight",2,CB_V,CB_D);cb_final_norm=cb_1("model.norm.weight",2,CB_D);
    cb_inv_freq=cb_1("native.rope.inv_freq",1,32);
    for(int li=0;li<CB_L;li++){
        CbLayer *l=&cb_layers[li];char name[160];
        #define CB_NAME(suffix) snprintf(name,sizeof name,"model.layers.%d." suffix,li)
        #define CB_MATRIX(member,stem,rows,cols) CB_NAME("self_attn." stem ".weight");l->member=cb_2(name,2,rows,cols)
        CB_MATRIX(q,"q_proj",CB_D,CB_D);CB_MATRIX(k,"k_proj",128,CB_D);CB_MATRIX(v,"v_proj",128,CB_D);CB_MATRIX(o,"o_proj",CB_D,CB_D);
        CB_NAME("self_attn.q_proj.bias");l->qb=cb_1(name,2,CB_D);
        CB_NAME("self_attn.k_proj.bias");l->kb=cb_1(name,2,128);CB_NAME("self_attn.v_proj.bias");l->vb=cb_1(name,2,128);
        CB_NAME("input_layernorm.weight");l->in_norm=cb_1(name,2,CB_D);CB_NAME("post_attention_layernorm.weight");l->post_norm=cb_1(name,2,CB_D);
        CB_NAME("mlp.compact.projection");l->projection=cb_2(name,1,CB_Q,CB_D);
        CB_NAME("mlp.compact.parent_centers");l->parents=cb_2(name,1,CB_P,CB_Q);
        CB_NAME("mlp.compact.parent_norms");l->parent_norms=cb_1(name,1,CB_P);
        #define CB_SHARED(member,rows,cols) CB_NAME("mlp.compact." #member);l->member=cb_2(name,2,rows,cols)
        CB_SHARED(shared_g,CB_HS,CB_D);CB_SHARED(shared_u,CB_HS,CB_D);CB_SHARED(shared_b,CB_D,CB_HS);
        #define CB_LEAF(member,rows,cols) CB_NAME("mlp.compact." #member);l->member=cb_3(name,2,CB_P,rows,cols)
        CB_LEAF(leaf_g,CB_HF,CB_D);CB_LEAF(leaf_u,CB_HF,CB_D);CB_LEAF(leaf_b,CB_D,CB_HF);
        #undef CB_NAME
        #undef CB_MATRIX
        #undef CB_SHARED
        #undef CB_LEAF
    }
    for(int i=0;i<CB_FIELD_COUNT;i++)if(!cb_used[i])cb_die("unbound field");
}
static void cb_norm(const float *x,const uint16_t *w,float *out){
    float square=0;for(int i=0;i<CB_D;i++)square+=x[i]*x[i];float inv=1.0f/sqrtf(square/CB_D+1e-6f);
    for(int i=0;i<CB_D;i++)out[i]=cb_round(cb_round(x[i]*inv)*cb_decode(w[i]));
}
static void cb_route(CbState *s,const CbLayer *l){
    for(int j=0;j<CB_Q;j++)s->query[j]=cb_fdot(l->projection+(size_t)j*CB_D,s->norm,CB_D);
    float score[CB_P];for(int p=0;p<CB_P;p++)score[p]=2.0f*cb_fdot(l->parents+(size_t)p*CB_Q,s->query,CB_Q)-l->parent_norms[p];
    for(int k=0;k<4;k++){
        int best=-1;
        for(int p=0;p<CB_P;p++){
            int used=0;for(int j=0;j<k;j++)if(s->selected[j]==p)used=1;
            if(!used && (best<0 || score[p]>score[best]))best=p;
        }
        if(best<0 || !isfinite(score[best]))cb_die("routing score");s->selected[k]=best;
    }
    float total=0;for(int k=0;k<4;k++){s->mass[k]=expf(score[s->selected[k]]-score[s->selected[0]]);total+=s->mass[k];}
    for(int k=0;k<4;k++)s->mass[k]/=total;
    // Torch evaluates the selected unique leaves in ascending numeric ID order.
    for(int i=1;i<4;i++)for(int j=i;j>0 && s->selected[j]<s->selected[j-1];j--){
        int p=s->selected[j];s->selected[j]=s->selected[j-1];s->selected[j-1]=p;
        float m=s->mass[j];s->mass[j]=s->mass[j-1];s->mass[j-1]=m;
    }
}
static float cb_silu(float x){if(x>=0)return x/(1.0f+expf(-x));float e=expf(x);return x*e/(1.0f+e);}
static void cb_compact(CbState *s,const CbLayer *l){
    cb_route(s,l);
    #pragma omp parallel for schedule(static)
    for(int row=0;row<CB_HS+4*CB_HF;row++){
        const uint16_t *g,*u;
        if(row<CB_HS){g=l->shared_g+(size_t)row*CB_D;u=l->shared_u+(size_t)row*CB_D;}
        else{int k=(row-CB_HS)/CB_HF,h=(row-CB_HS)%CB_HF,p=s->selected[k];
            g=l->leaf_g+((size_t)p*CB_HF+h)*CB_D;u=l->leaf_u+((size_t)p*CB_HF+h)*CB_D;}
        s->activation[row]=cb_silu(cb_bfdot(g,s->norm,CB_D))*cb_bfdot(u,s->norm,CB_D);
    }
    #pragma omp parallel for schedule(static)
    for(int row=0;row<CB_D;row++){
        float result=cb_bfdot(l->shared_b+(size_t)row*CB_HS,s->activation,CB_HS);
        for(int k=0;k<4;k++){const uint16_t *b=l->leaf_b+((size_t)s->selected[k]*CB_D+row)*CB_HF;
            float part=cb_bfdot(b,s->activation+CB_HS+k*CB_HF,CB_HF)*s->mass[k];result+=part;}
        s->tmp[row]=result;
    }
}
static void cb_mat(const uint16_t *w,const float *x,float *out,int rows,int cols){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<rows;i++)out[i]=cb_round(cb_bfdot(w+(size_t)i*cols,x,cols));
}
static void cb_qkv(CbState *s,const CbLayer *l){
    #pragma omp parallel for schedule(static)
    for(int row=0;row<CB_D+256;row++){
        int index;const uint16_t *w,*bias;float *out;
        if(row<CB_D){index=row;w=l->q;bias=l->qb;out=s->q;}
        else if(row<CB_D+128){index=row-CB_D;w=l->k;bias=l->kb;out=s->k;}
        else{index=row-CB_D-128;w=l->v;bias=l->vb;out=s->v;}
        out[index]=cb_round(cb_bfdot(w+(size_t)index*CB_D,s->norm,CB_D)+cb_decode(bias[index]));
    }
}
static void cb_rope(float *values,int heads,const float *co,const float *si){
    for(int h=0;h<heads;h++)for(int j=0;j<32;j++){
        float a=values[h*64+j],b=values[h*64+j+32];
        values[h*64+j]=cb_round(cb_round(a*co[j])+cb_round(-b*si[j]));
        values[h*64+j+32]=cb_round(cb_round(b*co[j])+cb_round(a*si[j]));
    }
}
static void cb_forward(CbState *s,int token,int pos,int head){
    if(token<0 || token>=CB_V || pos<0 || pos>=s->capacity)cb_die("token/context");
    for(int i=0;i<CB_D;i++)s->x[i]=cb_decode(cb_embedding[(size_t)token*CB_D+i]);
    float co[32],si[32];for(int j=0;j<32;j++){float phase=(float)pos*cb_inv_freq[j];co[j]=cb_round(cosf(phase));si[j]=cb_round(sinf(phase));}
    for(int li=0;li<CB_L;li++){
        const CbLayer *l=&cb_layers[li];cb_norm(s->x,l->in_norm,s->norm);cb_qkv(s,l);
        cb_rope(s->q,CB_NH,co,si);cb_rope(s->k,CB_NKV,co,si);
        memcpy(s->kc+((size_t)li*s->capacity+pos)*128,s->k,sizeof s->k);
        memcpy(s->vc+((size_t)li*s->capacity+pos)*128,s->v,sizeof s->v);
        for(int h=0;h<CB_NH;h++){
            float *scores=s->scores+(size_t)h*s->capacity;float maximum=-INFINITY;
            for(int t=0;t<=pos;t++){const float *k=s->kc+((size_t)li*s->capacity+t)*128+(h/7)*64;
                float score=cb_round(cb_round(cb_fdot(k,s->q+h*64,64))*0.125f);scores[t]=score;if(score>maximum)maximum=score;}
            float total=0;for(int t=0;t<=pos;t++){scores[t]=expf(scores[t]-maximum);total+=scores[t];}
            for(int t=0;t<=pos;t++)scores[t]=cb_round(scores[t]/total);
            for(int d=0;d<64;d++){float value=0;
                for(int t=0;t<=pos;t++)value+=scores[t]*s->vc[((size_t)li*s->capacity+t)*128+(h/7)*64+d];
                s->attention[h*64+d]=cb_round(value);}
        }
        cb_mat(l->o,s->attention,s->tmp,CB_D,CB_D);
        for(int i=0;i<CB_D;i++)s->x[i]=cb_round(s->x[i]+s->tmp[i]);
        cb_norm(s->x,l->post_norm,s->norm);cb_compact(s,l);
        if(s->trace_enabled){
            for(int i=0;i<CB_D;i++)s->trace_input[li][i]=cb_bits(s->norm[i]);
            memcpy(s->trace_output[li],s->tmp,sizeof s->tmp);
            memcpy(s->trace_query[li],s->query,sizeof s->query);
            memcpy(s->trace_mass[li],s->mass,sizeof s->mass);
            for(int i=0;i<4;i++)s->trace_ids[li][i]=s->selected[i];
        }
        for(int i=0;i<CB_D;i++)s->x[i]=cb_round(s->x[i]+cb_round(s->tmp[i]));
    }
    cb_norm(s->x,cb_final_norm,s->norm);
    if(head){cb_mat(cb_embedding,s->norm,s->logits,CB_V,CB_D);for(int i=0;i<CB_V;i++)if(!isfinite(s->logits[i]))cb_die("nonfinite head");}
}
static void cb_state_init(CbState *s,int capacity){
    memset(s,0,sizeof *s);s->capacity=capacity;
    s->kc=calloc((size_t)CB_L*capacity*128,sizeof(float));s->vc=calloc((size_t)CB_L*capacity*128,sizeof(float));
    s->scores=calloc((size_t)CB_NH*capacity,sizeof(float));if(!s->kc || !s->vc || !s->scores)cb_die("cache allocation");
}
static void cb_write(FILE *stream,const void *p,size_t bytes){if(fwrite(p,1,bytes,stream)!=bytes)cb_die("output write");}
static int cb_argmax(const float *scores){int best=0;for(int i=1;i<CB_V;i++)if(scores[i]>scores[best])best=i;return best;}
static double cb_time(void){LARGE_INTEGER now,freq;if(!QueryPerformanceCounter(&now) || !QueryPerformanceFrequency(&freq))cb_die("clock");return (double)now.QuadPart/freq.QuadPart;}
static void cb_probe(CbState *s){
    uint16_t *bits=malloc((size_t)CB_V*2);if(!bits)cb_die("probe head allocation");
    for(;;){
        char magic[8];size_t got=fread(magic,1,8,stdin);if(got==0 && feof(stdin))break;
        if(got!=8 || memcmp(magic,"QWCP0001",8))cb_die("probe request magic");
        uint32_t header[2];if(fread(header,4,2,stdin)!=2)cb_die("probe request header");
        uint32_t n=header[0],m=header[1],positions[16];
        if(n<1 || n>(uint32_t)s->capacity || m<1 || m>16 || m>n)cb_die("probe limits");
        if(fread(positions,4,m,stdin)!=m)cb_die("probe positions");
        for(uint32_t i=0;i<m;i++)if(positions[i]>=n || (i && positions[i]<=positions[i-1]))cb_die("probe position order/range");
        int32_t *ids=malloc((size_t)n*4);if(!ids || fread(ids,4,n,stdin)!=n)cb_die("probe input IDs");
        for(uint32_t i=0;i<n;i++)if(ids[i]<0 || ids[i]>=CB_V)cb_die("probe ID range");
        memset(s->kc,0,(size_t)CB_L*s->capacity*128*4);memset(s->vc,0,(size_t)CB_L*s->capacity*128*4);
        uint32_t response[4]={CB_V,m,CB_L,CB_D};cb_write(stdout,"QWCT0001",8);cb_write(stdout,response,16);
        uint32_t next=0;
        for(uint32_t i=0;i<n;i++){
            int wanted=next<m && positions[next]==i;s->trace_enabled=wanted;cb_forward(s,ids[i],(int)i,wanted);
            if(wanted){
                for(int j=0;j<CB_V;j++)bits[j]=cb_bits(s->logits[j]);
                cb_write(stdout,&i,4);cb_write(stdout,s->trace_ids,sizeof s->trace_ids);
                cb_write(stdout,s->trace_mass,sizeof s->trace_mass);cb_write(stdout,s->trace_query,sizeof s->trace_query);
                cb_write(stdout,s->trace_input,sizeof s->trace_input);cb_write(stdout,s->trace_output,sizeof s->trace_output);
                cb_write(stdout,bits,(size_t)CB_V*2);next++;
            }
        }
        if(next!=m || fflush(stdout))cb_die("probe response");free(ids);
    }
    s->trace_enabled=0;free(bits);
}
int main(int argc,char **argv){
    if(argc!=4 || (strcmp(argv[2],"stream") && strcmp(argv[2],"probe")))cb_die("usage: engine COMPACT_ARCHIVE stream|probe CONTEXT_CAPACITY");
    int capacity=atoi(argv[3]);if(capacity<2 || capacity>4096)cb_die("context capacity");
    if(_setmode(_fileno(stdin),_O_BINARY)<0 || _setmode(_fileno(stdout),_O_BINARY)<0)cb_die("binary stream mode");
    if(fesetround(FE_TONEAREST) || fegetround()!=FE_TONEAREST || !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))cb_die("rounding/ISA");
    omp_set_dynamic(0);omp_set_num_threads(6);cb_load(argv[1]);
    CbState *s=malloc(sizeof *s);if(!s)cb_die("state allocation");cb_state_init(s,capacity);
    if(!strcmp(argv[2],"probe")){
        cb_probe(s);free(s->kc);free(s->vc);free(s->scores);free(s);free(cb_blob);return 0;
    }
    for(;;){
        char magic[8];size_t got=fread(magic,1,8,stdin);if(got==0 && feof(stdin))break;
        if(got!=8 || memcmp(magic,"QWCR0001",8))cb_die("request magic");
        uint32_t header[2];if(fread(header,4,2,stdin)!=2)cb_die("request header");
        uint32_t n=header[0],maxnew=header[1];if(n<1 || n>=(uint32_t)capacity || maxnew<1 || maxnew>128 || maxnew>(uint32_t)capacity-n)cb_die("request limits");
        int32_t *ids=malloc((size_t)n*4);if(!ids || fread(ids,4,n,stdin)!=n)cb_die("request IDs");
        for(uint32_t i=0;i<n;i++)if(ids[i]<0 || ids[i]>=CB_V)cb_die("request ID range");
        double started=cb_time();memset(s->kc,0,(size_t)CB_L*capacity*128*4);memset(s->vc,0,(size_t)CB_L*capacity*128*4);
        for(uint32_t i=0;i<n;i++)cb_forward(s,ids[i],(int)i,i==n-1);
        int32_t generated[128];uint32_t count=0,stop=0;
        for(uint32_t i=0;i<maxnew;i++){
            int token=cb_argmax(s->logits);generated[count++]=token;
            if(token==151645 || token==151643){stop=1;break;}
            if(i+1<maxnew)cb_forward(s,token,(int)n+(int)i,1);
        }
        double elapsed=cb_time()-started;uint32_t response[2]={count,stop};
        cb_write(stdout,"QWCO0001",8);cb_write(stdout,response,8);cb_write(stdout,&elapsed,8);cb_write(stdout,generated,(size_t)count*4);
        if(fflush(stdout))cb_die("response flush");free(ids);
    }
    free(s->kc);free(s->vc);free(s->scores);free(s);free(cb_blob);return 0;
}
