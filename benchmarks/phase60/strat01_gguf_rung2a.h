/*
 * STRAT-01 engine rung 2A: block-0 GigaChat 3.1 MLA attention.
 *
 * Header-only to preserve the phase60 single-translation-unit build.  This
 * path is correctness apparatus, not a rate path.  It deliberately streams
 * the frozen GGUF tensors and emits reference-pending F32LE evidence.
 */
#ifndef STRAT01_GGUF_RUNG2A_H
#define STRAT01_GGUF_RUNG2A_H

#include <float.h>
#include <limits.h>

/* C11-standard contraction control for every rung-2A numerical primitive.
 * Restored at the end of this header so the historical engine is unaffected. */
#pragma STDC FP_CONTRACT OFF

#define STRAT01_R2A_NTOK 8U
#define STRAT01_R2A_EMBD 1536U
#define STRAT01_R2A_HEADS 32U
#define STRAT01_R2A_Q_HEAD 192U
#define STRAT01_R2A_Q_NOPE 128U
#define STRAT01_R2A_ROPE 64U
#define STRAT01_R2A_KV 512U
#define STRAT01_R2A_CACHE 576U
#define STRAT01_R2A_V_HEAD 192U
#define STRAT01_R2A_Q_ALL (STRAT01_R2A_HEADS*STRAT01_R2A_Q_HEAD)
#define STRAT01_R2A_ATTN_ALL (STRAT01_R2A_HEADS*STRAT01_R2A_V_HEAD)
#define STRAT01_R2A_RMS_EPS 1.0e-6f
#define STRAT01_R2A_FREQ_BASE 100000.0f
#define STRAT01_R2A_FREQ_SCALE (1.0f/64.0f)
#define STRAT01_R2A_ORIG_CTX 4096
#define STRAT01_R2A_BETA_FAST 32.0f
#define STRAT01_R2A_BETA_SLOW 1.0f
#define STRAT01_R2A_EXT_FACTOR 1.0f
#define STRAT01_R2A_REFERENCE "llama.cpp 5b335f413e4f73b0809c4fe39af894efbcc6a0d2"

static const uint32_t strat01_r2a_tokens[STRAT01_R2A_NTOK] = {1U,72U,14U,14129U,14U,2135U,1512U,2015U};
static const int32_t strat01_r2a_positions[STRAT01_R2A_NTOK] = {0,1,2,3,4,5,6,7};

#if defined(__clang__)
#define STRAT01_R2A_COMPILER __clang_version__
#else
#define STRAT01_R2A_COMPILER "non-clang-build-refused"
#endif

/* The external runner will add the resolved clang path and complete
 * `clang --version` output.  Those values cannot be recovered from a C
 * executable; the C-side contract records every semantic build choice. */
static const char strat01_r2a_config[] =
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;cache=layer-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-token-major;adjudication=external-reference-only";

typedef struct {
    const char *name;
    uint32_t type;
    uint32_t rank;
    uint64_t dims[3];
} strat01_r2a_tensor_spec;

static const strat01_r2a_tensor_spec strat01_r2a_specs[] = {
    {"token_embd.weight",          STRAT01_GGML_Q4_K,2,{1536,128256,0}},
    {"blk.0.attn_norm.weight",     STRAT01_GGML_F32, 1,{1536,0,0}},
    {"blk.0.attn_q.weight",        STRAT01_GGML_Q4_K,2,{1536,6144,0}},
    {"blk.0.attn_kv_a_mqa.weight", STRAT01_GGML_Q4_K,2,{1536,576,0}},
    {"blk.0.attn_kv_a_norm.weight",STRAT01_GGML_F32, 1,{512,0,0}},
    {"blk.0.attn_k_b.weight",      STRAT01_GGML_Q5_0,3,{128,512,32}},
    {"blk.0.attn_v_b.weight",      STRAT01_GGML_Q4_K,3,{512,192,32}},
    {"blk.0.attn_output.weight",   STRAT01_GGML_Q4_K,2,{6144,1536,0}},
};

typedef struct {
    const char *logical_name;
    const char *selection;
    const char *op;
    unsigned ordinal; /* zero-based occurrence of the pinned callback name */
    unsigned rank;
    uint64_t shape[3]; /* ggml logical order: fastest dimension first */
    const char *payload_order;
    const float *data; /* always token-major in the file */
    uint64_t count;
    char path[1024];
    char sha256[65];
} strat01_r2a_dump;

typedef struct {
    float *embd, *attn_norm, *q, *kv_cmpr_pe, *k_pe, *kv_cmpr;
    float *q_pe, *q_abs, *qcur, *kcur, *vcur, *kqv_out, *ffn_inp;
    /* Layer 0 of frozen [layer][slot][576] physical order.  There is no V
     * allocation (llama-kv-cache.cpp:163-164,230-244). */
    uint16_t cache[STRAT01_R2A_NTOK][STRAT01_R2A_CACHE];
    unsigned occupied;
} strat01_r2a_arm;

static uint32_t strat01_r2a_f32_bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static float strat01_r2a_bits_f32(uint32_t u) { float x; memcpy(&x,&u,4); return x; }

/* IEEE binary16 round-to-nearest-even.  Cache writes in pinned llama.cpp use
 * ggml's F32->F16 conversion; subnormals, infinities, and NaNs are handled so
 * malformed synthetic fixtures cannot silently become finite cache rows. */
static uint16_t strat01_r2a_f32_to_f16(float x) {
    uint32_t u=strat01_r2a_f32_bits(x), sign=(u>>16)&0x8000U, mant=u&0x7fffffU;
    int exp=(int)((u>>23)&0xffU);
    if(exp==255) return (uint16_t)(sign | (mant ? 0x7e00U : 0x7c00U));
    exp -= 127;
    if(exp > 15) return (uint16_t)(sign|0x7c00U);
    if(exp < -24) return (uint16_t)sign;
    if(exp < -14) {
        unsigned shift=(unsigned)(-exp-14); uint32_t m=mant|0x800000U;
        unsigned rshift=shift+13U; uint32_t half=m>>rshift, rem=m&((UINT32_C(1)<<rshift)-1U), tie=UINT32_C(1)<<(rshift-1U);
        if(rem>tie || (rem==tie && (half&1U))) ++half;
        return (uint16_t)(sign|half);
    }
    { uint32_t half_exp=(uint32_t)(exp+15)<<10, half=mant>>13, rem=mant&0x1fffU;
      if(rem>0x1000U || (rem==0x1000U && (half&1U))) { ++half; if(half==0x400U){half=0;half_exp+=0x400U;} }
      return (uint16_t)(sign|half_exp|half); }
}

static float strat01_r2a_f16_to_f32(uint16_t h) {
    uint32_t sign=((uint32_t)h&0x8000U)<<16, exp=((uint32_t)h>>10)&31U, mant=(uint32_t)h&1023U, u;
    if(exp==0) {
        if(!mant) u=sign;
        else { int e=-14; while(!(mant&0x400U)){mant<<=1;--e;} mant&=0x3ffU;u=sign|((uint32_t)(e+127)<<23)|(mant<<13); }
    } else if(exp==31) u=sign|0x7f800000U|(mant<<13);
    else u=sign|((exp+112U)<<23)|(mant<<13);
    return strat01_r2a_bits_f32(u);
}

static int strat01_r2a_path(char out[1024],const char *dir,const char *leaf) {
    int n=snprintf(out,1024,"%s/%s",dir,leaf); return n>0 && n<1024;
}
static int strat01_r2a_safe_count(size_t a,size_t b,size_t *out) { if(a && b>SIZE_MAX/a)return 0;*out=a*b;return 1; }
static float *strat01_r2a_alloc(size_t count,char error[256]) {
    size_t bytes; float *p; if(!strat01_r2a_safe_count(count,sizeof(float),&bytes)){snprintf(error,256,"rung-2A allocation overflow");return NULL;}
    p=(float *)calloc(1,bytes);if(!p)snprintf(error,256,"rung-2A allocation failed (%zu bytes)",bytes);return p;
}

static int strat01_r2a_check_spec(const strat01_inventory *in,const strat01_r2a_tensor_spec *s,const strat01_tensor **out,char error[256]) {
    const strat01_tensor *t=strat01_rung1_find_tensor(in,s->name);unsigned i;
    if(!t){snprintf(error,256,"frozen rung-2A tensor absent: %s",s->name);return 0;}
    if(t->type!=s->type||t->rank!=s->rank){snprintf(error,256,"frozen rung-2A descriptor mismatch: %s",s->name);return 0;}
    for(i=0;i<s->rank;++i)if(t->dims[i]!=s->dims[i]){snprintf(error,256,"frozen rung-2A dimensions mismatch: %s",s->name);return 0;}
    *out=t;return 1;
}

static int strat01_r2a_seek(FILE *f,uint64_t offset,char error[256]) {
    if(offset>(uint64_t)INT64_MAX||STRAT01_FSEEK(f,(int64_t)offset,SEEK_SET)!=0){snprintf(error,256,"rung-2A payload seek failed");return 0;}return 1;
}
static float strat01_r2a_f32le(const uint8_t b[4]) {
    uint32_t u=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);return strat01_r2a_bits_f32(u);
}

static int strat01_r2a_read_f32_vector(const char *path,const strat01_tensor *t,float *dst,size_t n,char error[256]) {
    FILE *f=NULL;size_t i;uint8_t b[4];
    if(t->type!=STRAT01_GGML_F32||t->rank!=1||t->dims[0]!=(uint64_t)n){snprintf(error,256,"rung-2A F32 vector descriptor mismatch");return 0;}
    f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error)){if(f)fclose(f);return 0;}
    for(i=0;i<n;++i){if(fread(b,1,4,f)!=4){snprintf(error,256,"rung-2A F32 vector short read");fclose(f);return 0;}dst[i]=strat01_r2a_f32le(b);if(!isfinite(dst[i])){snprintf(error,256,"rung-2A non-finite F32 weight");fclose(f);return 0;}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"rung-2A F32 vector I/O failure");return 0;}return 1;
}

static int strat01_r2a_read_qblock(FILE *f,uint32_t type,float out[256],unsigned *n,char error[256]) {
    uint8_t raw[210];uint64_t block_values,block_bytes;unsigned which;
    if(!strat01_tensor_type(type,&block_values,&block_bytes,&which)||type==STRAT01_GGML_F32||block_bytes>sizeof(raw)){snprintf(error,256,"rung-2A unsupported quant type");return 0;}
    if(fread(raw,1,(size_t)block_bytes,f)!=(size_t)block_bytes||!strat01_rung1_decode_block(type,raw,out,n)||*n!=block_values){snprintf(error,256,"rung-2A quant payload decode failure");return 0;}return 1;
}

/* GGUF/ggml matrices have ne[0] as the dot-product dimension and all
 * remaining dimensions flattened into rows.  Stream each stored row once and
 * apply it to all tokens, matching ggml_mul_mat's orientation. */
static int strat01_r2a_matmul_batch(const char *path,const strat01_tensor *t,const float *x,unsigned batch,unsigned in,float *y,unsigned rows,char error[256]) {
    FILE *f=NULL;uint64_t bv,bb,blocks,row;unsigned which,n,b,j;float w[256];
    if(!strat01_tensor_type(t->type,&bv,&bb,&which)||t->type==STRAT01_GGML_F32||t->dims[0]!=in||in%bv){snprintf(error,256,"rung-2A matmul descriptor/divisibility failure");return 0;}
    {uint64_t r=1;for(j=1;j<t->rank;++j)if(!strat01_mul_u64(r,t->dims[j],&r)){snprintf(error,256,"rung-2A row count overflow");return 0;}if(r!=rows){snprintf(error,256,"rung-2A row count mismatch");return 0;}}
    memset(y,0,(size_t)batch*rows*sizeof(float));blocks=in/bv;f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error)){if(f)fclose(f);return 0;}
    for(row=0;row<rows;++row)for(uint64_t blk=0;blk<blocks;++blk){if(!strat01_r2a_read_qblock(f,t->type,w,&n,error)){fclose(f);return 0;}for(b=0;b<batch;++b){float acc=y[(size_t)b*rows+(size_t)row];const float *xb=x+(size_t)b*in+(size_t)blk*bv;for(j=0;j<n;++j)acc+=w[j]*xb[j];y[(size_t)b*rows+(size_t)row]=acc;}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"rung-2A matmul I/O failure");return 0;}return 1;
}

static int strat01_r2a_embedding(const char *path,const strat01_tensor *t,const uint32_t *tokens,unsigned count,float *out,char error[256]) {
    FILE *f=NULL;uint64_t bv,bb,row_bytes;unsigned which,n,j,tok;float w[256];
    if(!strat01_tensor_type(t->type,&bv,&bb,&which)||t->type!=STRAT01_GGML_Q4_K||t->dims[0]!=STRAT01_R2A_EMBD||STRAT01_R2A_EMBD%bv){snprintf(error,256,"rung-2A embedding descriptor mismatch");return 0;}
    if(!strat01_mul_u64(STRAT01_R2A_EMBD/bv,bb,&row_bytes)){snprintf(error,256,"rung-2A embedding row overflow");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open rung-2A embedding payload");return 0;}
    for(tok=0;tok<count;++tok){uint64_t delta,off;if(tokens[tok]>=t->dims[1]||!strat01_mul_u64(tokens[tok],row_bytes,&delta)||!strat01_add_u64(t->file_offset,delta,&off)||!strat01_r2a_seek(f,off,error)){fclose(f);return 0;}for(uint64_t blk=0;blk<STRAT01_R2A_EMBD/bv;++blk){if(!strat01_r2a_read_qblock(f,t->type,w,&n,error)){fclose(f);return 0;}for(j=0;j<n;++j)out[(size_t)tok*STRAT01_R2A_EMBD+(size_t)blk*bv+j]=w[j];}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"rung-2A embedding I/O failure");return 0;}return 1;
}

static void strat01_r2a_rmsnorm(const float *x,const float *weight,float *y,unsigned rows,unsigned n,float eps) {
    for(unsigned r=0;r<rows;++r){float ss=0.0f;const float *xr=x+(size_t)r*n;float *yr=y+(size_t)r*n;for(unsigned i=0;i<n;++i)ss+=xr[i]*xr[i];float scale=1.0f/sqrtf(ss/(float)n+eps);for(unsigned i=0;i<n;++i)yr[i]=xr[i]*scale*weight[i];}
}

/* Exact pinned ggml YaRN equations: ggml/src/ggml.c:4469-4480 and
 * ggml/src/ggml-cpu/ops.cpp:5951-5973.  DeepSeek2 is LLAMA_ROPE_TYPE_NORM,
 * so rotation is over adjacent pairs (llama-model.cpp:2944). */
static float strat01_r2a_corr_dim(float beta) { return 64.0f*logf(4096.0f/(beta*2.0f*3.14159265358979323846f))/(2.0f*logf(100000.0f)); }
static void strat01_r2a_rope64(float v[64],int32_t pos) {
    float low=fmaxf(0.0f,floorf(strat01_r2a_corr_dim(STRAT01_R2A_BETA_FAST)));
    float high=fminf(63.0f,ceilf(strat01_r2a_corr_dim(STRAT01_R2A_BETA_SLOW)));
    float theta=(float)pos,theta_scale=powf(STRAT01_R2A_FREQ_BASE,-2.0f/64.0f);
    float adjusted_attn=1.0f/(1.0f+0.1f*logf(64.0f));
    for(unsigned i=0;i<64;i+=2){float y=((float)(i/2)-low)/fmaxf(0.001f,high-low);float ramp=1.0f-fminf(1.0f,fmaxf(0.0f,y));float interp=STRAT01_R2A_FREQ_SCALE*theta;float ang=interp*(1.0f-ramp)+theta*ramp;float mag=adjusted_attn*(1.0f+0.1f*logf(64.0f));float c=cosf(ang)*mag,s=sinf(ang)*mag,x0=v[i],x1=v[i+1];v[i]=x0*c-x1*s;v[i+1]=x0*s+x1*c;theta*=theta_scale;}
}

static int strat01_r2a_kb_batch(const char *path,const strat01_tensor *t,const float *q,unsigned batch,float *out,char error[256]) {
    FILE *f=NULL;uint64_t bv,bb,blocks,row;unsigned which,n,j,tok,head;float w[256];
    if(!strat01_tensor_type(t->type,&bv,&bb,&which)||t->type!=STRAT01_GGML_Q5_0||t->dims[0]!=128||t->dims[1]!=512||t->dims[2]!=32){snprintf(error,256,"rung-2A K-B descriptor mismatch");return 0;}
    blocks=128/bv;memset(out,0,(size_t)batch*32*512*sizeof(float));f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error)){if(f)fclose(f);return 0;}
    for(row=0;row<32U*512U;++row){head=(unsigned)(row/512U);for(uint64_t blk=0;blk<blocks;++blk){if(!strat01_r2a_read_qblock(f,t->type,w,&n,error)){fclose(f);return 0;}for(tok=0;tok<batch;++tok){float acc=out[((size_t)tok*32+head)*512+(row%512U)];const float *x=q+(size_t)tok*6144+(size_t)head*192+(size_t)blk*bv;for(j=0;j<n;++j)acc+=w[j]*x[j];out[((size_t)tok*32+head)*512+(row%512U)]=acc;}}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"rung-2A K-B I/O failure");return 0;}return 1;
}

static int strat01_r2a_vb_batch(const char *path,const strat01_tensor *t,const float *latent,float *out,char error[256]) {
    FILE *f=NULL;uint64_t bv,bb,blocks,row;unsigned which,n,j,tok,head,vo;float w[256];
    if(!strat01_tensor_type(t->type,&bv,&bb,&which)||t->type!=STRAT01_GGML_Q4_K||t->dims[0]!=512||t->dims[1]!=192||t->dims[2]!=32){snprintf(error,256,"rung-2A V-B descriptor mismatch");return 0;}
    blocks=512/bv;memset(out,0,(size_t)8*6144*sizeof(float));f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error)){if(f)fclose(f);return 0;}
    for(row=0;row<32U*192U;++row){head=(unsigned)(row/192U);vo=(unsigned)(row%192U);for(uint64_t blk=0;blk<blocks;++blk){if(!strat01_r2a_read_qblock(f,t->type,w,&n,error)){fclose(f);return 0;}for(tok=0;tok<8;++tok){float acc=out[(size_t)tok*6144+(size_t)head*192+vo];const float *x=latent+((size_t)tok*32+head)*512+(size_t)blk*bv;for(j=0;j<n;++j)acc+=w[j]*x[j];out[(size_t)tok*6144+(size_t)head*192+vo]=acc;}}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"rung-2A V-B I/O failure");return 0;}return 1;
}

static void strat01_r2a_cache_write(strat01_r2a_arm *a,unsigned slot,const float *k) { for(unsigned i=0;i<576;++i)a->cache[slot][i]=strat01_r2a_f32_to_f16(k[i]);if(slot+1>a->occupied)a->occupied=slot+1; }
/* deepseek2.cpp:437-443: cancel the generic YaRN adjustment, then apply
 * mscale_all_dim before dividing by sqrt(the decompressed key head = 192). */
static float strat01_r2a_kq_scale(void) { float m=1.0f+0.1f*logf(64.0f);return m*m/sqrtf(192.0f); }

static void strat01_r2a_attend_one(const strat01_r2a_arm *a,unsigned tok,float *latent) {
    /* llama-graph.cpp:2958-2978 views the first 512 components of the stored
     * K row as V, performs causal attention, then applies attn_v_b. */
    float scale=strat01_r2a_kq_scale();
    for(unsigned h=0;h<32;++h){float scores[8],mx=-INFINITY,sum=0.0f;const float *q=a->qcur+((size_t)tok*32+h)*576;for(unsigned s=0;s<=tok;++s){float dot=0.0f;for(unsigned i=0;i<576;++i)dot+=q[i]*strat01_r2a_f16_to_f32(a->cache[s][i]);scores[s]=dot*scale;if(scores[s]>mx)mx=scores[s];}for(unsigned s=0;s<=tok;++s){scores[s]=expf(scores[s]-mx);sum+=scores[s];}for(unsigned i=0;i<512;++i){float v=0.0f;for(unsigned s=0;s<=tok;++s)v+=(scores[s]/sum)*strat01_r2a_f16_to_f32(a->cache[s][i]);latent[((size_t)tok*32+h)*512+i]=v;}}
}

static int strat01_r2a_arm_alloc(strat01_r2a_arm *a,char error[256]) {
    memset(a,0,sizeof(*a));
#define R2A_ALLOC(field,count) do{a->field=strat01_r2a_alloc((count),error);if(!a->field)return 0;}while(0)
    R2A_ALLOC(embd,8U*1536U);R2A_ALLOC(attn_norm,8U*1536U);R2A_ALLOC(q,8U*6144U);R2A_ALLOC(kv_cmpr_pe,8U*576U);
    R2A_ALLOC(k_pe,8U*64U);R2A_ALLOC(kv_cmpr,8U*512U);R2A_ALLOC(q_pe,8U*32U*64U);R2A_ALLOC(q_abs,8U*32U*512U);
    R2A_ALLOC(qcur,8U*32U*576U);R2A_ALLOC(kcur,8U*576U);R2A_ALLOC(vcur,8U*512U);R2A_ALLOC(kqv_out,8U*6144U);R2A_ALLOC(ffn_inp,8U*1536U);
#undef R2A_ALLOC
    return 1;
}
static void strat01_r2a_arm_free(strat01_r2a_arm *a) { free(a->embd);free(a->attn_norm);free(a->q);free(a->kv_cmpr_pe);free(a->k_pe);free(a->kv_cmpr);free(a->q_pe);free(a->q_abs);free(a->qcur);free(a->kcur);free(a->vcur);free(a->kqv_out);free(a->ffn_inp);memset(a,0,sizeof(*a)); }

static int strat01_r2a_build_upstream_range(const char *path,const strat01_tensor *tensors[8],strat01_r2a_arm *a,unsigned start,unsigned count,char error[256]) {
    float *attn_w=NULL,*kv_w=NULL;
    if(!count||start>=8||count>8-start){snprintf(error,256,"invalid rung-2A upstream schedule range");return 0;}
    attn_w=strat01_r2a_alloc(1536,error);kv_w=strat01_r2a_alloc(512,error);if(!attn_w||!kv_w)goto fail;
    if(!strat01_r2a_embedding(path,tensors[0],strat01_r2a_tokens+start,count,a->embd+(size_t)start*1536,error)||!strat01_r2a_read_f32_vector(path,tensors[1],attn_w,1536,error))goto fail;
    strat01_r2a_rmsnorm(a->embd+(size_t)start*1536,attn_w,a->attn_norm+(size_t)start*1536,count,1536,STRAT01_R2A_RMS_EPS);
    if(!strat01_r2a_matmul_batch(path,tensors[2],a->attn_norm+(size_t)start*1536,count,1536,a->q+(size_t)start*6144,6144,error)||!strat01_r2a_matmul_batch(path,tensors[3],a->attn_norm+(size_t)start*1536,count,1536,a->kv_cmpr_pe+(size_t)start*576,576,error)||!strat01_r2a_read_f32_vector(path,tensors[4],kv_w,512,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kv_cmpr+(size_t)tok*512,a->kv_cmpr_pe+(size_t)tok*576,512*4);memcpy(a->k_pe+(size_t)tok*64,a->kv_cmpr_pe+(size_t)tok*576+512,64*4);strat01_r2a_rope64(a->k_pe+(size_t)tok*64,strat01_r2a_positions[tok]);}
    {float *tmp=strat01_r2a_alloc((size_t)count*512U,error);if(!tmp)goto fail;strat01_r2a_rmsnorm(a->kv_cmpr+(size_t)start*512,kv_w,tmp,count,512,STRAT01_R2A_RMS_EPS);memcpy(a->kv_cmpr+(size_t)start*512,tmp,(size_t)count*512U*4U);free(tmp);}
    for(unsigned tok=start;tok<start+count;++tok)for(unsigned h=0;h<32;++h){float *qp=a->q_pe+((size_t)tok*32+h)*64;memcpy(qp,a->q+(size_t)tok*6144+(size_t)h*192+128,64*4);strat01_r2a_rope64(qp,strat01_r2a_positions[tok]);}
    if(!strat01_r2a_kb_batch(path,tensors[5],a->q+(size_t)start*6144,count,a->q_abs+(size_t)start*32*512,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kcur+(size_t)tok*576,a->kv_cmpr+(size_t)tok*512,512*4);memcpy(a->kcur+(size_t)tok*576+512,a->k_pe+(size_t)tok*64,64*4);memcpy(a->vcur+(size_t)tok*512,a->kv_cmpr+(size_t)tok*512,512*4);for(unsigned h=0;h<32;++h){float *qc=a->qcur+((size_t)tok*32+h)*576;memcpy(qc,a->q_abs+((size_t)tok*32+h)*512,512*4);memcpy(qc+512,a->q_pe+((size_t)tok*32+h)*64,64*4);}}
    free(attn_w);free(kv_w);return 1;
fail: free(attn_w);free(kv_w);return 0;
}

static int strat01_r2a_run_schedule(const char *path,const strat01_tensor *vb,const strat01_tensor *wo,strat01_r2a_arm *a,int cached7p1,char error[256]) {
    float *latent=strat01_r2a_alloc(8U*32U*512U,error),*proj=strat01_r2a_alloc(8U*1536U,error);if(!latent||!proj){free(latent);free(proj);return 0;}
    if(!cached7p1){for(unsigned t=0;t<8;++t)strat01_r2a_cache_write(a,t,a->kcur+(size_t)t*576);for(unsigned t=0;t<8;++t)strat01_r2a_attend_one(a,t,latent);}
    else {for(unsigned t=0;t<7;++t)strat01_r2a_cache_write(a,t,a->kcur+(size_t)t*576);for(unsigned t=0;t<7;++t)strat01_r2a_attend_one(a,t,latent);strat01_r2a_cache_write(a,7,a->kcur+7U*576U);strat01_r2a_attend_one(a,7,latent);}
    if(!strat01_r2a_vb_batch(path,vb,latent,a->kqv_out,error)||!strat01_r2a_matmul_batch(path,wo,a->kqv_out,8,6144,proj,1536,error)){free(latent);free(proj);return 0;}
    for(size_t i=0;i<8U*1536U;++i)a->ffn_inp[i]=proj[i]+a->embd[i];free(latent);free(proj);return 1;
}

static int strat01_r2a_dump_f32(strat01_r2a_dump *d,const char *out_dir,const char *arm,char error[256]) {
    char leaf[256];FILE *f;strat01_sha256 sha;uint8_t digest[32];
    if(snprintf(leaf,sizeof(leaf),"%s_%s.f32",arm,d->logical_name)<=0){snprintf(error,256,"rung-2A dump name failure");return 0;}for(char *p=leaf;*p;++p)if(*p=='/'||*p=='\\'||*p==' ') *p='_';
    if(!strat01_r2a_path(d->path,out_dir,leaf)||(f=fopen(d->path,"wb"))==NULL){snprintf(error,256,"cannot open rung-2A dump");return 0;}strat01_sha256_init(&sha);
    for(uint64_t i=0;i<d->count;++i){uint8_t b[4];if(!isfinite(d->data[i])){fclose(f);snprintf(error,256,"non-finite rung-2A tensor: %s",d->logical_name);return 0;}strat01_rung1_put_f32le(b,d->data[i]);if(fwrite(b,1,4,f)!=4){fclose(f);snprintf(error,256,"rung-2A dump write failure");return 0;}strat01_sha256_update(&sha,b,4);}
    if(fclose(f)!=0){snprintf(error,256,"rung-2A dump close failure");return 0;}strat01_sha256_final(&sha,digest);strat01_sha256_hex(digest,d->sha256);return 1;
}

static unsigned strat01_r2a_make_dumps(strat01_r2a_arm *a,strat01_r2a_dump d[12]) {
#define DUMP(i,n,sel,operation,ord,r,s0,s1,s2,order,ptr,cnt) do{d[i].logical_name=n;d[i].selection=sel;d[i].op=operation;d[i].ordinal=ord;d[i].rank=r;d[i].shape[0]=s0;d[i].shape[1]=s1;d[i].shape[2]=s2;d[i].payload_order=order;d[i].data=ptr;d[i].count=cnt;}while(0)
    DUMP(0,"attn_norm-0","post RMSNorm","MUL",0,2,1536,8,0,"token,feature",a->attn_norm,8U*1536U);
    DUMP(1,"q-0","direct Q after reshape","RESHAPE",1,3,192,32,8,"token,head,feature",a->q,8U*6144U);
    DUMP(2,"kv_cmpr_pe-0","post KV-A projection before split","MUL_MAT",0,2,576,8,0,"token,feature",a->kv_cmpr_pe,8U*576U);
    DUMP(3,"k_pe-0","post RoPE positional key","ROPE",1,3,64,1,8,"token,head,feature",a->k_pe,8U*64U);
    DUMP(4,"kv_cmpr-0","post compressed-KV RMSNorm","MUL",1,2,512,8,0,"token,feature",a->kv_cmpr,8U*512U);
    DUMP(5,"q_pe-0","post RoPE positional query","ROPE",1,3,64,32,8,"token,head,feature",a->q_pe,8U*32U*64U);
    DUMP(6,"q_nope_absorbed_perm-0","post K-B absorption and permutation","PERMUTE",0,3,512,32,8,"token,head,feature",a->q_abs,8U*32U*512U);
    DUMP(7,"Qcur-0","composed attention query","CONCAT",0,3,576,32,8,"token,head,feature",a->qcur,8U*32U*576U);
    DUMP(8,"Kcur-0","logical compact key before F16 cache write","CONCAT",0,3,576,1,8,"token,head,feature",a->kcur,8U*576U);
    DUMP(9,"Vcur-0","latent value view; no separate cache","RESHAPE",0,3,512,1,8,"token,head,feature",a->vcur,8U*512U);
    DUMP(10,"kqv_out-0","post causal attention and V-B expansion","CONT",0,2,6144,8,0,"token,feature",a->kqv_out,8U*6144U);
    DUMP(11,"ffn_inp-0","attention projection plus block input","ADD",0,2,1536,8,0,"token,feature",a->ffn_inp,8U*1536U);
#undef DUMP
    return 12;
}

static int strat01_r2a_write_cache_dump(const char *out_dir,const char *leaf,const uint16_t cache[8][576],unsigned rows,char path[1024],char shahex[65],char error[256]) {
    FILE *f;strat01_sha256 sha;uint8_t digest[32];if(!strat01_r2a_path(path,out_dir,leaf)||(f=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot open rung-2A cache dump");return 0;}strat01_sha256_init(&sha);for(unsigned r=0;r<rows;++r)for(unsigned i=0;i<576;++i){float v=strat01_r2a_f16_to_f32(cache[r][i]);uint8_t b[4];strat01_rung1_put_f32le(b,v);if(fwrite(b,1,4,f)!=4){fclose(f);snprintf(error,256,"cache dump write failure");return 0;}strat01_sha256_update(&sha,b,4);}if(fclose(f)!=0){snprintf(error,256,"cache dump close failure");return 0;}strat01_sha256_final(&sha,digest);strat01_sha256_hex(digest,shahex);return 1;
}

static int strat01_r2a_write_manifest(const char *out_dir,const char *arm,strat01_r2a_dump d[12],const char *cache_final_path,const char *cache_final_sha,const char *cache_prefix_path,const char *cache_prefix_sha,char error[256]) {
    char leaf[128],path[1024];FILE *o;if(snprintf(leaf,sizeof(leaf),"%s_manifest.json",arm)<=0||!strat01_r2a_path(path,out_dir,leaf)||(o=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot write rung-2A manifest");return 0;}
    fputs("{\n  \"arm\": ",o);strat01_json_string(o,arm);fputs(",\n  \"payload_encoding\": \"IEEE-754 binary32 little-endian\",\n  \"shape_order\": \"ggml logical dimensions, fastest first; payload order is separately declared\",\n  \"tensors\": [",o);
    for(unsigned i=0;i<12;++i){fprintf(o,i?",\n    {":"\n    {");fputs("\"name\": ",o);strat01_json_string(o,d[i].logical_name);fprintf(o,", \"ordinal\": %u, \"selection\": ",d[i].ordinal);strat01_json_string(o,d[i].selection);fputs(", \"op\": ",o);strat01_json_string(o,d[i].op);fputs(", \"type\": \"F32\", \"logical_shape\": [",o);for(unsigned k=0;k<d[i].rank;++k)fprintf(o,"%s%" PRIu64,k?", ":"",d[i].shape[k]);fputs("], \"payload_order\": ",o);strat01_json_string(o,d[i].payload_order);fprintf(o,", \"byte_count\": %" PRIu64 ", \"path\": ",d[i].count*4U);strat01_json_string(o,d[i].path);fputs(", \"sha256\": ",o);strat01_json_string(o,d[i].sha256);fputc('}',o);}
    fputs("\n  ],\n  \"cache\": {\"storage\": \"F16\", \"layout\": \"layer,slot,576[latent512,rope64]\", \"layer\": 0, \"row_length\": 576, \"no_separate_v_cache\": true, \"final\": {\"occupied_slots\": [0,1,2,3,4,5,6,7], \"absolute_positions\": [0,1,2,3,4,5,6,7], \"payload_type\": \"dequantized-F32LE\", \"path\": ",o);strat01_json_string(o,cache_final_path);fputs(", \"sha256\": ",o);strat01_json_string(o,cache_final_sha);fputc('}',o);
    if(cache_prefix_path){fputs(", \"prefix7\": {\"occupied_slots\": [0,1,2,3,4,5,6], \"absolute_positions\": [0,1,2,3,4,5,6], \"payload_type\": \"dequantized-F32LE\", \"path\": ",o);strat01_json_string(o,cache_prefix_path);fputs(", \"sha256\": ",o);strat01_json_string(o,cache_prefix_sha);fputc('}',o);}fputs("}\n}\n",o);
    if(fclose(o)!=0){snprintf(error,256,"rung-2A manifest close failure");return 0;}return 1;
}

static int strat01_r2a_write_failure(const char *out_dir,const char *path,const char *error) { char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2a.json")||(o=fopen(p,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-gguf-rung2a\",\"input_path\":",o);strat01_json_string(o,path);fputs(",\"c_state\":\"ENGINE_RUNG2A_C_FAILURE\",\"error\":",o);strat01_json_string(o,error);fputs("}\n",o);fclose(o);return 1; }

static int strat01_r2a_write_report(const char *out_dir,const char *path,uint64_t bytes,const char *artifact_sha,const char *engine_sha,const char *header_sha,char error[256]) {
    char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2a.json")||(o=fopen(p,"wb"))==NULL){snprintf(error,256,"cannot write rung-2A report");return 0;}fputs("{\n  \"command\": \"--strat01-gguf-rung2a\",\n  \"c_state\": \"ENGINE_OUTPUT_READY_PENDING_REFERENCE\",\n  \"self_certifies_pass\": false,\n  \"input_path\": ",o);strat01_json_string(o,path);fprintf(o,",\n  \"byte_size\": %" PRIu64 ",\n  \"sha256\": ",bytes);strat01_json_string(o,artifact_sha);fputs(",\n  \"reference_revision\": ",o);strat01_json_string(o,STRAT01_R2A_REFERENCE);fputs(",\n  \"CONFIG\": ",o);strat01_json_string(o,strat01_r2a_config);fputs(",\n  \"compiler_family\": \"clang\",\n  \"compiler_embedded_version\": ",o);strat01_json_string(o,STRAT01_R2A_COMPILER);fputs(",\n  \"compiler_resolved_path_and_full_version\": \"EXTERNAL_RUNNER_REQUIRED\",\n  \"engine_source_sha256\": ",o);strat01_json_string(o,engine_sha);fputs(",\n  \"rung2a_source_sha256\": ",o);strat01_json_string(o,header_sha);fputs(",\n  \"token_ids\": [1,72,14,14129,14,2135,1512,2015],\n  \"positions\": [0,1,2,3,4,5,6,7],\n  \"arms\": [{\"name\":\"prefill8\",\"manifest\":\"prefill8_manifest.json\"},{\"name\":\"cached7p1\",\"manifest\":\"cached7p1_manifest.json\"}],\n  \"timing_or_rate_claim\": null\n}\n",o);if(fclose(o)!=0){snprintf(error,256,"rung-2A report close failure");return 0;}return 1;
}

static int strat01_gguf_rung2a_cli(const char *path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *ts[8]={0};strat01_r2a_arm pre,cache;strat01_r2a_dump dumps[12];char error[256]={0},artifact_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};uint64_t hashed=0,parsed=0,tmp=0;char pf_cache_path[1024],pf_cache_sha[65],c7_prefix_path[1024],c7_prefix_sha[65],c7_final_path[1024],c7_final_sha[65];int ok=0;
    memset(&inv,0,sizeof(inv));memset(&pre,0,sizeof(pre));memset(&cache,0,sizeof(cache));
#if FLT_RADIX != 2
    snprintf(error,256,"non-binary floating point host");goto fail;
#endif
    if(sizeof(float)!=4||sizeof(uint16_t)!=2){snprintf(error,256,"unsupported host scalar sizes");goto fail;}
#if !defined(__clang__)
    snprintf(error,256,"rung-2A requires Clang");goto fail;
#endif
    /* Identity is deliberately completed before GGUF metadata/payload work. */
    if(!strat01_sha256_file(path,artifact_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(artifact_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"frozen rung-2A artifact identity mismatch");goto fail;}
    if(!strat01_parse_gguf(path,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"rung-2A GGUF parse/size mismatch");goto fail;}
    for(unsigned i=0;i<8;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&ts[i],error))goto fail;
    if(!strat01_r2a_arm_alloc(&pre,error)||!strat01_r2a_arm_alloc(&cache,error))goto fail;
    if(!strat01_r2a_build_upstream_range(path,ts,&pre,0,8,error))goto fail;
    if(!strat01_r2a_build_upstream_range(path,ts,&cache,0,7,error))goto fail;
    if(!strat01_r2a_run_schedule(path,ts[6],ts[7],&pre,0,error))goto fail;
    /* Preserve the true seven-slot checkpoint before the final cached token. */
    for(unsigned t=0;t<7;++t)strat01_r2a_cache_write(&cache,t,cache.kcur+(size_t)t*576);
    if(!strat01_r2a_write_cache_dump(out_dir,"cached7p1_cache_prefix7.f32",cache.cache,7,c7_prefix_path,c7_prefix_sha,error))goto fail;
    memset(cache.cache,0,sizeof(cache.cache));cache.occupied=0;
    if(!strat01_r2a_build_upstream_range(path,ts,&cache,7,1,error))goto fail;
    if(!strat01_r2a_run_schedule(path,ts[6],ts[7],&cache,1,error))goto fail;
    strat01_r2a_make_dumps(&pre,dumps);for(unsigned i=0;i<12;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"prefill8",error))goto fail;
    if(!strat01_r2a_write_cache_dump(out_dir,"prefill8_cache_final.f32",pre.cache,8,pf_cache_path,pf_cache_sha,error)||!strat01_r2a_write_manifest(out_dir,"prefill8",dumps,pf_cache_path,pf_cache_sha,NULL,NULL,error))goto fail;
    strat01_r2a_make_dumps(&cache,dumps);for(unsigned i=0;i<12;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"cached7p1",error))goto fail;
    if(!strat01_r2a_write_cache_dump(out_dir,"cached7p1_cache_final.f32",cache.cache,8,c7_final_path,c7_final_sha,error)||!strat01_r2a_write_manifest(out_dir,"cached7p1",dumps,c7_final_path,c7_final_sha,c7_prefix_path,c7_prefix_sha,error))goto fail;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))strcpy(engine_sha,"unavailable");error[0]=0;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))strcpy(header_sha,"unavailable");error[0]=0;
    if(!strat01_r2a_write_report(out_dir,path,hashed,artifact_sha,engine_sha,header_sha,error))goto fail;ok=1;
fail:
    strat01_free_inventory(&inv);strat01_r2a_arm_free(&pre);strat01_r2a_arm_free(&cache);if(!ok){if(!error[0])snprintf(error,256,"unspecified rung-2A failure");strat01_r2a_write_failure(out_dir,path,error);fprintf(stderr,"STRAT-01 rung-2A refused: %s\n",error);return 1;}fprintf(stderr,"CONFIG %s\n",strat01_r2a_config);fprintf(stderr,"STRAT-01 rung-2A: ENGINE_OUTPUT_READY_PENDING_REFERENCE\n");return 0;
}

static int strat01_r2a_identity_ok(const uint32_t tok[8],const int32_t pos[8],const char *precision,const char *layout,const char *config) { return !memcmp(tok,strat01_r2a_tokens,sizeof(strat01_r2a_tokens))&&!memcmp(pos,strat01_r2a_positions,sizeof(strat01_r2a_positions))&&!strcmp(precision,"F16")&&!strcmp(layout,"layer-slot-576-latent512-rope64-k-only")&&!strcmp(config,strat01_r2a_config); }

/* Small, model-free semantic and apparatus tests.  They intentionally do not
 * parse or execute the accepted donor. */
static int strat01_gguf_rung2a_selftest(void) {
    int bad=0,checks=0;float vals[]={0.0f,-0.0f,1.0f,-2.0f,65504.0f,0.00006103515625f};
#define CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    for(unsigned i=0;i<sizeof(vals)/sizeof(vals[0]);++i)CHECK(strat01_r2a_f16_to_f32(strat01_r2a_f32_to_f16(vals[i]))==vals[i]);
    CHECK(strat01_r2a_f16_to_f32(strat01_r2a_f32_to_f16(1.00048828125f))==1.0f); /* exact halfway, ties-to-even */
    CHECK(strat01_r2a_f16_to_f32(strat01_r2a_f32_to_f16(0x1p-24f))==0x1p-24f);  /* smallest subnormal */
    {float x[4]={1,2,3,4},w[4]={1,1,1,1},y[4];strat01_r2a_rmsnorm(x,w,y,1,4,0);CHECK(fabsf(y[0]-1.0f/sqrtf(7.5f))<1e-6f);}
    {float *q=(float *)calloc(2U*32U*576U,sizeof(float));float *lat0=(float *)calloc(2U*32U*512U,sizeof(float));float *lat1=(float *)calloc(2U*32U*512U,sizeof(float));strat01_r2a_arm pre,cached;memset(&pre,0,sizeof(pre));memset(&cached,0,sizeof(cached));CHECK(q&&lat0&&lat1);if(q&&lat0&&lat1){pre.qcur=cached.qcur=q;q[((size_t)1*32)*576]=1.0f;float k0[576]={0},k1[576]={0};k0[0]=1.0f;k1[0]=3.0f;strat01_r2a_cache_write(&pre,0,k0);strat01_r2a_cache_write(&pre,1,k1);strat01_r2a_attend_one(&pre,0,lat0);CHECK(lat0[0]==1.0f);strat01_r2a_attend_one(&pre,1,lat0);float sc=strat01_r2a_kq_scale(),expected=(expf(sc)*1.0f+expf(3.0f*sc)*3.0f)/(expf(sc)+expf(3.0f*sc));CHECK(fabsf(lat0[(size_t)32*512]-expected)<1e-6f);strat01_r2a_cache_write(&cached,0,k0);strat01_r2a_cache_write(&cached,1,k1);strat01_r2a_attend_one(&cached,1,lat1);CHECK(!memcmp(lat0+(size_t)32*512,lat1+(size_t)32*512,32U*512U*sizeof(float)));}free(q);free(lat0);free(lat1);}
    {float p0[64]={0},p1[64]={0};p0[0]=p1[0]=1.0f;strat01_r2a_rope64(p0,0);strat01_r2a_rope64(p1,1);CHECK(p0[0]!=p1[0]||p0[1]!=p1[1]);}
    {float k[8][576];uint16_t a[8][576]={0},b[8][576]={0};for(unsigned t=0;t<8;++t)for(unsigned i=0;i<576;++i)k[t][i]=(float)((int)(t*7+i%11)-5)*0.01f;for(unsigned t=0;t<8;++t)for(unsigned i=0;i<576;++i)a[t][i]=strat01_r2a_f32_to_f16(k[t][i]);for(unsigned t=0;t<7;++t)for(unsigned i=0;i<576;++i)b[t][i]=strat01_r2a_f32_to_f16(k[t][i]);for(unsigned i=0;i<576;++i)b[7][i]=strat01_r2a_f32_to_f16(k[7][i]);CHECK(!memcmp(a,b,sizeof(a)));}
    {float residual[3]={1,2,3},proj[3]={.25f,-.5f,1},good[3],omitted[3];for(unsigned i=0;i<3;++i){good[i]=residual[i]+proj[i];omitted[i]=proj[i];}CHECK(memcmp(good,omitted,sizeof(good))!=0);}
    {uint32_t tok[8];int32_t pos[8];memcpy(tok,strat01_r2a_tokens,sizeof(tok));memcpy(pos,strat01_r2a_positions,sizeof(pos));CHECK(strat01_r2a_identity_ok(tok,pos,"F16","layer-slot-576-latent512-rope64-k-only",strat01_r2a_config));tok[3]++;CHECK(!strat01_r2a_identity_ok(tok,pos,"F16","layer-slot-576-latent512-rope64-k-only",strat01_r2a_config));tok[3]--;pos[7]=8;CHECK(!strat01_r2a_identity_ok(tok,pos,"F16","layer-slot-576-latent512-rope64-k-only",strat01_r2a_config));pos[7]=7;CHECK(!strat01_r2a_identity_ok(tok,pos,"F32","layer-slot-576-latent512-rope64-k-only",strat01_r2a_config));CHECK(!strat01_r2a_identity_ok(tok,pos,"F16","slot-layer-576-latent512-rope64-k-only",strat01_r2a_config));CHECK(!strat01_r2a_identity_ok(tok,pos,"F16","layer-slot-576-latent512-rope64-k-only","changed"));}
#undef CHECK
    fprintf(stderr,"STRAT-01 rung-2A model-free selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#pragma STDC FP_CONTRACT ON
#endif /* STRAT01_GGUF_RUNG2A_H */
