// Direct complete276 archive execution. Qualification precedes any rate claim.
#include "meth284_source_operator.h"
#include <bcrypt.h>
#include "meth284_archive_catalog.h"

#define CC_V 151936
#define CC_NH 14
#define CC_NKV 2
#define CC_HD 64
typedef struct {
    const uint16_t *q,*k,*v,*o,*a,*b;
    const float *in_norm,*post_norm,*qb,*kb,*vb,*router,*cp,*keys;
    const int32_t *map;
} CcLayer;
static CcLayer cc_layers[L];
static uint8_t *cc_blob;
static const uint16_t *cc_embedding,*cc_head_scales;
static const int8_t *cc_head_codes;
static const float *cc_final_norm;
static int cc_used[CC_FIELD_COUNT];
static int cc_wrong_alias;

static uint16_t cc_bf_bits(float value) {
    uint32_t bits;memcpy(&bits,&value,4);
    if((bits&0x7f800000u)!=0x7f800000u)bits+=0x7fffu+((bits>>16)&1u);
    return (uint16_t)(bits>>16);
}
static float cc_bf(float value){return bf16_to_float(cc_bf_bits(value));}
static void cc_sha(const uint8_t *data,size_t bytes,char hex[65]) {
    BCRYPT_ALG_HANDLE alg=NULL;BCRYPT_HASH_HANDLE hash=NULL;
    uint8_t result[32];
    if(bytes>UINT32_MAX || BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,NULL,0)<0 ||
       BCryptCreateHash(alg,&hash,NULL,0,NULL,0,0)<0 ||
       BCryptHashData(hash,(PUCHAR)data,(ULONG)bytes,0)<0 ||
       BCryptFinishHash(hash,result,32,0)<0)die("archive SHA256 failure");
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
    for(int i=0;i<32;i++)sprintf(hex+2*i,"%02x",result[i]);hex[64]=0;
}
static const void *cc_bind(const char *name,int dtype,int rank,const uint64_t *shape) {
    for(int i=0;i<CC_FIELD_COUNT;i++)if(!strcmp(cc_fields[i].name,name)){
        const CcField *f=&cc_fields[i];
        if(f->dtype!=dtype || f->rank!=rank)die("tensor type/rank binding");
        for(int j=0;j<rank;j++)if(f->shape[j]!=shape[j])die("tensor shape binding");
        if(cc_used[i])die("duplicate tensor binding");cc_used[i]=1;
        return cc_blob+f->offset;
    }
    die("missing archived tensor");return NULL;
}
static const void *cc_1(const char *name,int kind,uint64_t n){return cc_bind(name,kind,1,&n);}
static const void *cc_2(const char *name,int kind,uint64_t a,uint64_t b){uint64_t s[2]={a,b};return cc_bind(name,kind,2,s);}
static const void *cc_3(const char *name,int kind,uint64_t a,uint64_t b,uint64_t c){uint64_t s[3]={a,b,c};return cc_bind(name,kind,3,s);}
static void cc_load(const char *path) {
    size_t size;cc_blob=read_file(path,&size);if(size!=CC_ARCHIVE_BYTES)die("archive length binding");
    char hash[65];cc_sha(cc_blob,size,hash);if(strcmp(hash,CC_ARCHIVE_SHA))die("archive SHA256 binding");
    uint64_t end=0;
    for(int i=0;i<CC_FIELD_COUNT;i++){
        const CcField *f=&cc_fields[i];
        if(f->offset>size || f->bytes>size-f->offset)die("tensor range binding");
        if(f->offset+f->bytes>end)end=f->offset+f->bytes;
        const uint8_t *p=cc_blob+f->offset;
        if(f->dtype==1)for(uint64_t j=0;j<f->bytes/4;j++){float v;memcpy(&v,p+4*j,4);if(!isfinite(v))die("nonfinite F32 tensor");}
        if(f->dtype==2 || f->dtype==6)for(uint64_t j=0;j<f->bytes/2;j++){
            uint16_t v;memcpy(&v,p+2*j,2);
            if(!isfinite(f->dtype==2?bf16_to_float(v):f16_to_float(v)))die("nonfinite 16bit tensor");
        }
    }
    if(end!=size)die("archive coverage binding");
    cc_embedding=cc_2("model.embed_tokens.weight.bf16",2,CC_V,D);
    cc_head_codes=cc_2("model.embed_tokens.weight.q",3,CC_V,D);
    cc_head_scales=cc_1("model.embed_tokens.weight.scale",6,CC_V);
    cc_final_norm=cc_1("model.norm.weight",1,D);
    silu_table=cc_1("ffn.silu_table",1,513);
    for(int li=0;li<L;li++){
        char name[160];CcLayer *l=&cc_layers[li];RowLayer *r=&rowsource[li];
        #define CC_NAME(prefix,suffix) snprintf(name,sizeof name,prefix ".%d." suffix,li)
        #define CC_ORGAN(member,suffix,rows,cols) CC_NAME("model.layers","self_attn." suffix ".weight");l->member=cc_2(name,2,rows,cols)
        CC_ORGAN(q,"q_proj",D,D);CC_ORGAN(k,"k_proj",128,D);
        CC_ORGAN(v,"v_proj",128,D);CC_ORGAN(o,"o_proj",D,D);
        CC_NAME("model.layers","input_layernorm.weight");l->in_norm=cc_1(name,1,D);
        CC_NAME("model.layers","post_attention_layernorm.weight");l->post_norm=cc_1(name,1,D);
        CC_NAME("model.layers","self_attn.q_proj.bias");l->qb=cc_1(name,1,D);
        CC_NAME("model.layers","self_attn.k_proj.bias");l->kb=cc_1(name,1,128);
        CC_NAME("model.layers","self_attn.v_proj.bias");l->vb=cc_1(name,1,128);
        #define CC_ROW(member,nrows,ncols) CC_NAME("ffn",#member ".q");r->member.q=cc_2(name,3,nrows,ncols);CC_NAME("ffn",#member ".scale");r->member.scale=cc_1(name,1,nrows);r->member.rows=nrows;r->member.cols=ncols
        CC_ROW(gate,H,D);CC_ROW(up,H,D);CC_ROW(down,D,H);
        CC_NAME("ffn","down.ids");r->ids=cc_2(name,4,D,32);
        CC_NAME("ffn","down.escape");r->escape=cc_2(name,2,D,32);
        CC_NAME("ffn","bias");r->bias=cc_1(name,1,D);
        CC_NAME("ffn","residual.bias");r->residual_bias=cc_1(name,1,D);
        CC_NAME("ffn","residual.right");r->right=cc_2(name,2,32,H);
        CC_NAME("ffn","residual.left");r->left=cc_2(name,2,D,32);
        CC_NAME("ffn","private.ids");r->private_ids=cc_1(name,4,128);
        CC_NAME("ffn","private.gate");r->private_gate=cc_2(name,2,128,D);
        CC_NAME("ffn","private.up");r->private_up=cc_2(name,2,128,D);
        CC_NAME("bank","router");l->router=cc_1(name,1,58880);
        CC_NAME("bank","child_projection");l->cp=cc_2(name,1,32,D);
        CC_NAME("bank","child_keys");l->keys=cc_3(name,1,128,10,32);
        CC_NAME("bank","a");l->a=cc_3(name,2,128,8,D);
        CC_NAME("bank","b");l->b=cc_3(name,2,cc_function_counts[li],D,8);
        CC_NAME("bank","leaf_map");l->map=cc_1(name,5,1280);
        #undef CC_ROW
        #undef CC_ORGAN
        #undef CC_NAME
        int seen[1280]={0};
        for(int j=0;j<1280;j++){if(l->map[j]<0 || l->map[j]>=cc_function_counts[li])die("alias range");seen[l->map[j]]=1;}
        for(int j=0;j<cc_function_counts[li];j++)if(!seen[j])die("alias surjectivity");
        for(int j=0;j<128;j++){
            if(r->private_ids[j]>=H)die("private ID range");
            for(int k=0;k<j;k++)if(r->private_ids[j]==r->private_ids[k])die("duplicate private ID");
        }
        for(int row=0;row<D;row++)for(int j=0;j<32;j++){
            int id=r->ids[row*32+j];if(id>=H || r->down.q[(size_t)row*H+id])die("escape ID/code binding");
            for(int k=0;k<j;k++)if(r->ids[row*32+k]==id)die("duplicate escape ID");
        }
    }
    for(int i=0;i<CC_FIELD_COUNT;i++)if(!cc_used[i])die("unbound archived tensor");
    fprintf(stderr,"{\"archive_sha256\":\"%s\",\"fields_bound\":725,\"bytes\":%zu,\"fallback\":false}\n",hash,size);
}
static float cc_dot(const float *w,const float *x,int n){float sum=0;for(int i=0;i<n;i++)sum+=w[i]*x[i];return sum;}
static void cc_top4(const float *scores,int n,int *ids){
    int used[16]={0};if(n>16)die("axis width");
    for(int k=0;k<4;k++){int best=-1;for(int i=0;i<n;i++)if(!used[i] && (best<0 || scores[i]>scores[best] || (scores[i]==scores[best] && i<best)))best=i;ids[k]=best;used[best]=1;}
}
static void cc_conditional(int li,const float *x,float *out,int *parents,int *children,int *aliases,float *gates){
    const CcLayer *l=&cc_layers[li];float q[64],as[8],bs[16],scores[16],cq[32];
    int ai[4],bi[4],ids[16],top[4];
    for(int i=0;i<64;i++)q[i]=cc_dot(l->router+(size_t)i*D,x,D);
    for(int i=0;i<8;i++)as[i]=cc_dot(l->router+64*D+i*64,q,64);
    for(int i=0;i<16;i++)bs[i]=cc_dot(l->router+64*D+8*64+i*64,q,64);
    cc_top4(as,8,ai);cc_top4(bs,16,bi);
    for(int i=0;i<4;i++)for(int j=0;j<4;j++){int k=4*i+j;ids[k]=ai[i]*16+bi[j];scores[k]=as[ai[i]]+bs[bi[j]];}
    cc_top4(scores,16,top);float sum=0;
    for(int k=0;k<4;k++){parents[k]=ids[top[k]];gates[k]=expf(scores[top[k]]-scores[top[0]]);sum+=gates[k];}
    for(int k=0;k<4;k++)gates[k]=cc_bf(gates[k]/sum);
    for(int r=0;r<32;r++)cq[r]=cc_dot(l->cp+(size_t)r*D,x,D);
    float hidden[4][8];
    for(int k=0;k<4;k++){
        int best=0;float maximum=cc_dot(l->keys+(size_t)parents[k]*10*32,cq,32);
        for(int c=1;c<10;c++){float value=cc_dot(l->keys+((size_t)parents[k]*10+c)*32,cq,32);if(value>maximum){maximum=value;best=c;}}
        children[k]=parents[k]*10+best;aliases[k]=l->map[children[k]];
        for(int r=0;r<8;r++){
            const uint16_t *w=l->a+((size_t)parents[k]*8+r)*D;
            float value=0;for(int d=0;d<D;d++)value+=bf16_to_float(w[d])*x[d];
            value=cc_bf(value);hidden[k][r]=cc_bf(value/(1.0f+expf(-value)));
        }
    }
    #pragma omp parallel for schedule(static)
    for(int d=0;d<D;d++){
        float total=0;
        for(int k=0;k<4;k++){
            int mapped=cc_wrong_alias?(aliases[k]+1)%cc_function_counts[li]:aliases[k];
            const uint16_t *w=l->b+((size_t)mapped*D+d)*8;
            float part=0;for(int r=0;r<8;r++)part+=hidden[k][r]*bf16_to_float(w[r]);
            total+=cc_bf(cc_bf(part)*gates[k]);
        }
        out[d]=cc_bf(total);if(!isfinite(out[d]))die("nonfinite conditional output");
    }
}

typedef struct {float x[D],norm[D],q[D],k[128],v[128],attention[D],tmp[D],logits[CC_V];float *kc,*vc,*scores;int maxseq;} CcState;
static void cc_norm(const float *x,const float *w,float *out){
    float square=0;for(int i=0;i<D;i++)square+=x[i]*x[i];float inv=1.0f/sqrtf(square/(float)D+1e-6f);
    for(int i=0;i<D;i++)out[i]=cc_bf(cc_bf(x[i]*inv)*cc_bf(w[i]));
}
static void cc_mat(const uint16_t *w,const float *x,const float *bias,float *out,int rows,int cols){
    #pragma omp parallel for schedule(static)
    for(int i=0;i<rows;i++){float value=bf16_dot(w+(size_t)i*cols,x,cols);if(bias)value+=cc_bf(bias[i]);out[i]=cc_bf(value);}
}
static void cc_rope(float *q,int heads,const float *co,const float *si){
    for(int h=0;h<heads;h++)for(int j=0;j<32;j++){
        float a=q[h*64+j],b=q[h*64+j+32];
        q[h*64+j]=cc_bf(cc_bf(a*co[j])+cc_bf(-b*si[j]));
        q[h*64+j+32]=cc_bf(cc_bf(b*co[j])+cc_bf(a*si[j]));
    }
}
static void cc_state_init(CcState *s,int maxseq){
    memset(s,0,sizeof *s);s->maxseq=maxseq;
    s->kc=calloc((size_t)L*maxseq*128,sizeof(float));s->vc=calloc((size_t)L*maxseq*128,sizeof(float));
    s->scores=calloc((size_t)CC_NH*maxseq,sizeof(float));if(!s->kc || !s->vc || !s->scores)die("cache allocation");
}
static void cc_forward(CcState *s,int token,int pos,int experts,int head){
    if(token<0 || token>=CC_V || pos<0 || pos>=s->maxseq)die("token/context binding");
    for(int d=0;d<D;d++)s->x[d]=bf16_to_float(cc_embedding[(size_t)token*D+d]);
    float co[32],si[32];
    for(int j=0;j<32;j++){float frequency=1.0f/powf(1000000.0f,(float)(2*j)/64.0f);float phase=(float)pos*frequency;co[j]=cc_bf(cosf(phase));si[j]=cc_bf(sinf(phase));}
    for(int li=0;li<L;li++){
        const CcLayer *l=&cc_layers[li];cc_norm(s->x,l->in_norm,s->norm);
        cc_mat(l->q,s->norm,l->qb,s->q,D,D);cc_mat(l->k,s->norm,l->kb,s->k,128,D);cc_mat(l->v,s->norm,l->vb,s->v,128,D);
        cc_rope(s->q,CC_NH,co,si);cc_rope(s->k,CC_NKV,co,si);
        memcpy(s->kc+((size_t)li*s->maxseq+pos)*128,s->k,128*4);memcpy(s->vc+((size_t)li*s->maxseq+pos)*128,s->v,128*4);
        #pragma omp parallel for schedule(static)
        for(int h=0;h<CC_NH;h++){
            float *scores=s->scores+(size_t)h*s->maxseq;float maximum=-INFINITY;
            for(int t=0;t<=pos;t++){
                const float *k=s->kc+((size_t)li*s->maxseq+t)*128+(h/7)*64;float value=cc_dot(k,s->q+h*64,64)*0.125f;
                scores[t]=value;if(value>maximum)maximum=value;
            }
            float total=0;for(int t=0;t<=pos;t++){scores[t]=expf(scores[t]-maximum);total+=scores[t];}
            for(int d=0;d<64;d++){
                float value=0;for(int t=0;t<=pos;t++)value+=(scores[t]/total)*s->vc[((size_t)li*s->maxseq+t)*128+(h/7)*64+d];
                s->attention[h*64+d]=cc_bf(value);
            }
        }
        cc_mat(l->o,s->attention,NULL,s->tmp,D,D);for(int d=0;d<D;d++)s->x[d]=s->x[d]+s->tmp[d];
        cc_norm(s->x,l->post_norm,s->norm);uint16_t bits[D];for(int d=0;d<D;d++)bits[d]=cc_bf_bits(s->norm[d]);
        row_forward(li,bits,s->tmp);
        float conditional[D];int parents[4],children[4],aliases[4];float gates[4];
        if(experts)cc_conditional(li,s->norm,conditional,parents,children,aliases,gates);
        for(int d=0;d<D;d++){float ffn=cc_bf(s->tmp[d]);if(experts)ffn=cc_bf(ffn+conditional[d]);s->x[d]=s->x[d]+ffn;}
    }
    cc_norm(s->x,cc_final_norm,s->norm);
    if(head)cc_mat(cc_embedding,s->norm,NULL,s->logits,CC_V,D);
}
static int cc_argmax(const float *scores){int best=0;for(int i=1;i<CC_V;i++)if(scores[i]>scores[best])best=i;return best;}
static void cc_write(FILE *file,const void *data,size_t bytes){if(fwrite(data,1,bytes,file)!=bytes)die("short output write");}
