/*
 * STRAT-01 engine Rung 2C: accepted dense block 0 followed by one complete
 * block-1 MLA + routed/shared MoE layer. Correctness apparatus only.
 */
#ifndef STRAT01_GGUF_RUNG2C_H
#define STRAT01_GGUF_RUNG2C_H

#pragma STDC FP_CONTRACT OFF

#define STRAT01_R2C_EXPERTS 64U
#define STRAT01_R2C_TOPK 4U
#define STRAT01_R2C_FFN 1280U
#define STRAT01_R2C_FLOAT_DUMPS 31U
#define STRAT01_R2C_DUMPS 32U

static const char strat01_r2c_config[] =
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;f16_dot=pinned-generic-f64;cache=layers0-1-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "rms_accum=double;kb=q5_0xq8_0;q6kq8k=reference-generic-noavx-noavx2-nofma-noinline;"
    "block0=dense-swiglu-sse2-nofma4;block1=mla-sigmoid-bias-select-top4-unbiased-normalized-q4k-q6k-shared-residual;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-or-i32le-token-major;adjudication=external-reference-only";

static const strat01_r2a_tensor_spec strat01_r2c_attn_specs[] = {
    {"blk.1.attn_norm.weight",      STRAT01_GGML_F32, 1,{1536,0,0}},
    {"blk.1.attn_q.weight",         STRAT01_GGML_Q4_K,2,{1536,6144,0}},
    {"blk.1.attn_kv_a_mqa.weight",  STRAT01_GGML_Q4_K,2,{1536,576,0}},
    {"blk.1.attn_kv_a_norm.weight", STRAT01_GGML_F32, 1,{512,0,0}},
    {"blk.1.attn_k_b.weight",       STRAT01_GGML_Q5_0,3,{128,512,32}},
    {"blk.1.attn_v_b.weight",       STRAT01_GGML_Q4_K,3,{512,192,32}},
    {"blk.1.attn_output.weight",    STRAT01_GGML_Q4_K,2,{6144,1536,0}},
};

static const strat01_r2a_tensor_spec strat01_r2c_moe_specs[] = {
    {"blk.1.ffn_norm.weight",       STRAT01_GGML_F32, 1,{1536,0,0}},
    {"blk.1.ffn_gate_inp.weight",   STRAT01_GGML_F32, 2,{1536,64,0}},
    {"blk.1.exp_probs_b.bias",      STRAT01_GGML_F32, 1,{64,0,0}},
    {"blk.1.ffn_up_exps.weight",    STRAT01_GGML_Q4_K,3,{1536,1280,64}},
    {"blk.1.ffn_gate_exps.weight",  STRAT01_GGML_Q4_K,3,{1536,1280,64}},
    {"blk.1.ffn_down_exps.weight",  STRAT01_GGML_Q6_K,3,{1280,1536,64}},
    {"blk.1.ffn_up_shexp.weight",   STRAT01_GGML_Q4_K,2,{1536,1280,0}},
    {"blk.1.ffn_gate_shexp.weight", STRAT01_GGML_Q4_K,2,{1536,1280,0}},
    {"blk.1.ffn_down_shexp.weight", STRAT01_GGML_Q6_K,2,{1280,1536,0}},
};

typedef struct {
    float *norm,*logits,*probs,*biased,*weights,*weights_norm;
    int32_t *topk;
    float *moe_up,*moe_gate,*moe_swiglu,*moe_down,*moe_weighted,*moe_out;
    float *up,*gate,*swiglu,*shexp,*out,*l_out;
} strat01_r2c_moe_arm;

typedef struct {
    const char *logical_name,*selection,*op,*type,*payload_order;
    unsigned ordinal,rank;uint64_t shape[3],count;const float *f32;const int32_t *i32;
    char path[1024],sha256[65];
} strat01_r2c_dump;

static int strat01_r2c_moe_alloc(strat01_r2c_moe_arm *a,char error[256]) {
    memset(a,0,sizeof(*a));
#define R2C_ALLOC(field,count) do{a->field=strat01_r2a_alloc((count),error);if(!a->field)return 0;}while(0)
    R2C_ALLOC(norm,8U*1536U);R2C_ALLOC(logits,8U*64U);R2C_ALLOC(probs,8U*64U);R2C_ALLOC(biased,8U*64U);
    a->topk=(int32_t *)calloc(8U*STRAT01_R2C_TOPK,sizeof(*a->topk));if(!a->topk){snprintf(error,256,"Rung-2C top-k allocation failed");return 0;}
    R2C_ALLOC(weights,8U*4U);R2C_ALLOC(weights_norm,8U*4U);
    R2C_ALLOC(moe_up,8U*4U*1280U);R2C_ALLOC(moe_gate,8U*4U*1280U);R2C_ALLOC(moe_swiglu,8U*4U*1280U);
    R2C_ALLOC(moe_down,8U*4U*1536U);R2C_ALLOC(moe_weighted,8U*4U*1536U);R2C_ALLOC(moe_out,8U*1536U);
    R2C_ALLOC(up,8U*1280U);R2C_ALLOC(gate,8U*1280U);R2C_ALLOC(swiglu,8U*1280U);R2C_ALLOC(shexp,8U*1536U);
    R2C_ALLOC(out,8U*1536U);R2C_ALLOC(l_out,8U*1536U);
#undef R2C_ALLOC
    return 1;
}

static void strat01_r2c_moe_free(strat01_r2c_moe_arm *a) {
    free(a->norm);free(a->logits);free(a->probs);free(a->biased);free(a->topk);free(a->weights);free(a->weights_norm);
    free(a->moe_up);free(a->moe_gate);free(a->moe_swiglu);free(a->moe_down);free(a->moe_weighted);free(a->moe_out);
    free(a->up);free(a->gate);free(a->swiglu);free(a->shexp);free(a->out);free(a->l_out);memset(a,0,sizeof(*a));
}

/* Layer-1 orchestration reuses every accepted Rung-2A attention primitive. */
static int strat01_r2c_build_attention_range(const char *path,const strat01_tensor *t[7],strat01_r2a_arm *a,unsigned start,unsigned count,char error[256]) {
    float *attn_w=NULL,*kv_w=NULL;
    if(!count||start>=8U||count>8U-start){snprintf(error,256,"invalid Rung-2C attention schedule range");return 0;}
    attn_w=strat01_r2a_alloc(1536U,error);kv_w=strat01_r2a_alloc(512U,error);if(!attn_w||!kv_w)goto fail;
    if(!strat01_r2a_read_f32_vector(path,t[0],attn_w,1536U,error))goto fail;
    strat01_r2a_rmsnorm_pinned(a->embd+(size_t)start*1536U,attn_w,a->attn_norm+(size_t)start*1536U,count,1536U,STRAT01_R2A_RMS_EPS);
    if(!strat01_r2a_matmul_batch(path,t[1],a->attn_norm+(size_t)start*1536U,count,1536U,a->q+(size_t)start*6144U,6144U,error)||
       !strat01_r2a_matmul_batch(path,t[2],a->attn_norm+(size_t)start*1536U,count,1536U,a->kv_cmpr_pe+(size_t)start*576U,576U,error)||
       !strat01_r2a_read_f32_vector(path,t[3],kv_w,512U,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kv_cmpr+(size_t)tok*512U,a->kv_cmpr_pe+(size_t)tok*576U,512U*4U);memcpy(a->k_pe+(size_t)tok*64U,a->kv_cmpr_pe+(size_t)tok*576U+512U,64U*4U);strat01_r2a_rope64(a->k_pe+(size_t)tok*64U,strat01_r2a_positions[tok]);}
    {float *tmp=strat01_r2a_alloc((size_t)count*512U,error);if(!tmp)goto fail;strat01_r2a_rmsnorm_pinned(a->kv_cmpr+(size_t)start*512U,kv_w,tmp,count,512U,STRAT01_R2A_RMS_EPS);memcpy(a->kv_cmpr+(size_t)start*512U,tmp,(size_t)count*512U*4U);free(tmp);}
    for(unsigned tok=start;tok<start+count;++tok)for(unsigned h=0;h<32U;++h){float *qp=a->q_pe+((size_t)tok*32U+h)*64U;memcpy(qp,a->q+(size_t)tok*6144U+(size_t)h*192U+128U,64U*4U);strat01_r2a_rope64(qp,strat01_r2a_positions[tok]);}
    if(!strat01_r2a_kb_batch(path,t[4],a->q+(size_t)start*6144U,count,a->q_abs+(size_t)start*32U*512U,error))goto fail;
    for(unsigned tok=start;tok<start+count;++tok){memcpy(a->kcur+(size_t)tok*576U,a->kv_cmpr+(size_t)tok*512U,512U*4U);memcpy(a->kcur+(size_t)tok*576U+512U,a->k_pe+(size_t)tok*64U,64U*4U);memcpy(a->vcur+(size_t)tok*512U,a->kv_cmpr+(size_t)tok*512U,512U*4U);for(unsigned h=0;h<32U;++h){float *qc=a->qcur+((size_t)tok*32U+h)*576U;memcpy(qc,a->q_abs+((size_t)tok*32U+h)*512U,512U*4U);memcpy(qc+512U,a->q_pe+((size_t)tok*32U+h)*64U,64U*4U);}}
    free(attn_w);free(kv_w);return 1;
fail:free(attn_w);free(kv_w);return 0;
}

static int strat01_r2c_f32_matmul_batch(const char *path,const strat01_tensor *t,const float *x,unsigned batch,unsigned in,float *y,unsigned rows,char error[256]) {
    FILE *f=NULL;float *w=NULL;uint8_t raw[4];
    if(!t||t->type!=STRAT01_GGML_F32||t->rank!=2||t->dims[0]!=in||t->dims[1]!=rows||!batch){snprintf(error,256,"Rung-2C F32 matrix descriptor mismatch");return 0;}
    w=strat01_r2a_alloc(in,error);if(!w)return 0;f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error))goto fail;
    for(unsigned row=0;row<rows;++row){for(unsigned i=0;i<in;++i){if(fread(raw,1,4,f)!=4){snprintf(error,256,"Rung-2C F32 matrix short read");goto fail;}w[i]=strat01_r2a_f32le(raw);}for(unsigned b=0;b<batch;++b){float sum=0.0f;const float *xb=x+(size_t)b*in;for(unsigned i=0;i<in;++i)sum+=w[i]*xb[i];y[(size_t)b*rows+row]=sum;}}
    if(ferror(f)||fclose(f)!=0){f=NULL;snprintf(error,256,"Rung-2C F32 matrix I/O failure");goto fail;}free(w);return 1;
fail:if(f)fclose(f);free(w);return 0;
}

static uint64_t strat01_r2c_quant_row_bytes(uint32_t type,unsigned in) {
    uint64_t bv=0,bb=0;unsigned which=0;if(!strat01_tensor_type(type,&bv,&bb,&which)||!bv||in%bv)return 0;return (uint64_t)(in/bv)*bb;
}

static int strat01_r2c_expert_view(const strat01_tensor *src,unsigned expert,unsigned in,unsigned rows,strat01_tensor *view,char error[256]) {
    uint64_t row_bytes=strat01_r2c_quant_row_bytes(src?src->type:0,in),slice=0,delta=0,off=0;
    if(!src||src->rank!=3||src->dims[0]!=in||src->dims[1]!=rows||src->dims[2]!=64U||expert>=64U||!row_bytes||
       !strat01_mul_u64(row_bytes,rows,&slice)||!strat01_mul_u64(slice,expert,&delta)||!strat01_add_u64(src->file_offset,delta,&off)){snprintf(error,256,"Rung-2C expert slice descriptor mismatch");return 0;}
    *view=*src;view->rank=2;view->dims[0]=in;view->dims[1]=rows;view->dims[2]=0;view->file_offset=off;return 1;
}

static int strat01_r2c_q6_matmul_batch(const char *path,const strat01_tensor *t,const float *x,unsigned batch,unsigned in,float *y,char error[256]) {
    FILE *f=NULL;strat01_q8_k_block *q8=NULL;uint8_t *raw=NULL;size_t q8_count,row_bytes;unsigned blocks;
    if(!t||t->type!=STRAT01_GGML_Q6_K||t->rank!=2||t->dims[0]!=in||t->dims[1]!=1536U||!batch||in%STRAT01_QK_K){snprintf(error,256,"Rung-2C Q6_K descriptor mismatch");return 0;}
    blocks=in/STRAT01_QK_K;if(!strat01_r2a_safe_count(batch,blocks,&q8_count)||!strat01_r2a_safe_count(blocks,STRAT01_R2B_Q6_BLOCK_BYTES,&row_bytes)){snprintf(error,256,"Rung-2C Q6_K allocation overflow");return 0;}
    q8=(strat01_q8_k_block *)malloc(q8_count*sizeof(*q8));raw=(uint8_t *)malloc(row_bytes);if(!q8||!raw){snprintf(error,256,"Rung-2C Q6_K allocation failed");goto fail;}
    for(unsigned b=0;b<batch;++b)if(!strat01_quantize_q8_k_row(x+(size_t)b*in,in,q8+(size_t)b*blocks)){snprintf(error,256,"Rung-2C Q8_K activation quantization failed");goto fail;}
    f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error))goto fail;
    for(unsigned row=0;row<1536U;++row){if(fread(raw,1,row_bytes,f)!=row_bytes){snprintf(error,256,"Rung-2C Q6_K row short read");goto fail;}for(unsigned b=0;b<batch;++b){float v=strat01_q6k_q8k_dot(raw,q8+(size_t)b*blocks,in);if(!isfinite(v)){snprintf(error,256,"Rung-2C Q6_K non-finite output");goto fail;}y[(size_t)b*1536U+row]=v;}}
    if(ferror(f)||fclose(f)!=0){f=NULL;snprintf(error,256,"Rung-2C Q6_K I/O failure");goto fail;}free(q8);free(raw);return 1;
fail:if(f)fclose(f);free(q8);free(raw);return 0;
}

static void strat01_r2c_top4(const float score[64],int32_t ids[4]) {
    int32_t all[64];for(int32_t i=0;i<64;++i)all[i]=i;
    for(unsigned i=0;i<4U;++i){unsigned best=i;for(unsigned j=i+1U;j<64U;++j)if(score[all[j]]>score[all[best]])best=j;{int32_t tmp=all[i];all[i]=all[best];all[best]=tmp;}ids[i]=all[i];}
}

static int strat01_r2c_run_moe(const char *path,const strat01_tensor *t[9],const strat01_r2a_arm *attn,strat01_r2c_moe_arm *a,char error[256]) {
    float *norm_w=strat01_r2a_alloc(1536U,error),*bias=strat01_r2a_alloc(64U,error);if(!norm_w||!bias){free(norm_w);free(bias);return 0;}
    if(!strat01_r2a_read_f32_vector(path,t[0],norm_w,1536U,error)||!strat01_r2a_read_f32_vector(path,t[2],bias,64U,error))goto fail;
    strat01_r2a_rmsnorm_pinned(attn->ffn_inp,norm_w,a->norm,8U,1536U,STRAT01_R2A_RMS_EPS);
    if(!strat01_r2c_f32_matmul_batch(path,t[1],a->norm,8U,1536U,a->logits,64U,error))goto fail;
    for(unsigned tok=0;tok<8U;++tok){float sum=0.0f;for(unsigned e=0;e<64U;++e){float p=1.0f/(1.0f+expf(-a->logits[(size_t)tok*64U+e]));a->probs[(size_t)tok*64U+e]=p;a->biased[(size_t)tok*64U+e]=p+bias[e];}strat01_r2c_top4(a->biased+(size_t)tok*64U,a->topk+(size_t)tok*4U);for(unsigned s=0;s<4U;++s){float w=a->probs[(size_t)tok*64U+(unsigned)a->topk[(size_t)tok*4U+s]];a->weights[(size_t)tok*4U+s]=w;sum+=w;}sum=fmaxf(sum,6.103515625e-5f);for(unsigned s=0;s<4U;++s)a->weights_norm[(size_t)tok*4U+s]=a->weights[(size_t)tok*4U+s]/sum;}
    for(unsigned tok=0;tok<8U;++tok)for(unsigned s=0;s<4U;++s){unsigned expert=(unsigned)a->topk[(size_t)tok*4U+s];strat01_tensor up,gate,down;float *u=a->moe_up+((size_t)tok*4U+s)*1280U,*g=a->moe_gate+((size_t)tok*4U+s)*1280U,*sw=a->moe_swiglu+((size_t)tok*4U+s)*1280U,*dn=a->moe_down+((size_t)tok*4U+s)*1536U,*wt=a->moe_weighted+((size_t)tok*4U+s)*1536U;
        if(!strat01_r2c_expert_view(t[3],expert,1536U,1280U,&up,error)||!strat01_r2c_expert_view(t[4],expert,1536U,1280U,&gate,error)||!strat01_r2c_expert_view(t[5],expert,1280U,1536U,&down,error)||
           !strat01_r2a_matmul_batch(path,&up,a->norm+(size_t)tok*1536U,1U,1536U,u,1280U,error)||!strat01_r2a_matmul_batch(path,&gate,a->norm+(size_t)tok*1536U,1U,1536U,g,1280U,error))goto fail;
        for(unsigned i=0;i<1280U;++i)sw[i]=(g[i]/(1.0f+expf(-g[i])))*u[i];if(!strat01_r2c_q6_matmul_batch(path,&down,sw,1U,1280U,dn,error))goto fail;for(unsigned i=0;i<1536U;++i)wt[i]=dn[i]*a->weights_norm[(size_t)tok*4U+s];}
    for(unsigned tok=0;tok<8U;++tok)for(unsigned i=0;i<1536U;++i){float sum=0.0f;for(unsigned s=0;s<4U;++s)sum+=a->moe_weighted[((size_t)tok*4U+s)*1536U+i];a->moe_out[(size_t)tok*1536U+i]=sum;}
    if(!strat01_r2a_matmul_batch(path,t[6],a->norm,8U,1536U,a->up,1280U,error)||!strat01_r2a_matmul_batch(path,t[7],a->norm,8U,1536U,a->gate,1280U,error))goto fail;
    for(size_t i=0;i<8U*1280U;++i)a->swiglu[i]=(a->gate[i]/(1.0f+expf(-a->gate[i])))*a->up[i];if(!strat01_r2c_q6_matmul_batch(path,t[8],a->swiglu,8U,1280U,a->shexp,error))goto fail;
    for(size_t i=0;i<8U*1536U;++i){a->out[i]=a->moe_out[i]+a->shexp[i];a->l_out[i]=a->out[i]+attn->ffn_inp[i];}
    free(norm_w);free(bias);return 1;
fail:free(norm_w);free(bias);return 0;
}

static void strat01_r2c_set_dump(strat01_r2c_dump *d,const char *name,const char *selection,const char *op,const char *type,unsigned rank,uint64_t s0,uint64_t s1,uint64_t s2,const char *order,const float *f32,const int32_t *i32,uint64_t count) {
    memset(d,0,sizeof(*d));d->logical_name=name;d->selection=selection;d->op=op;d->type=type;d->rank=rank;d->shape[0]=s0;d->shape[1]=s1;d->shape[2]=s2;d->payload_order=order;d->f32=f32;d->i32=i32;d->count=count;
}

static unsigned strat01_r2c_make_dumps(const strat01_r2b_arm *block0,const strat01_r2a_arm *attn,const strat01_r2c_moe_arm *moe,strat01_r2c_dump d[STRAT01_R2C_DUMPS]) {
    unsigned n=0;
#define F(name,sel,op,rank,s0,s1,s2,order,ptr,count) strat01_r2c_set_dump(&d[n++],name,sel,op,"F32",rank,s0,s1,s2,order,ptr,NULL,count)
    F("l_out-0","accepted block-0 terminal start state","ADD",2,1536,8,0,"token,feature",block0->l_out,8U*1536U);
    F("attn_norm-1","post RMSNorm","MUL",2,1536,8,0,"token,feature",attn->attn_norm,8U*1536U);
    F("q-1","direct Q after reshape","RESHAPE",3,192,32,8,"token,head,feature",attn->q,8U*6144U);
    F("kv_cmpr_pe-1","post KV-A projection before split","MUL_MAT",2,576,8,0,"token,feature",attn->kv_cmpr_pe,8U*576U);
    F("k_pe-1","post RoPE positional key","ROPE",3,64,1,8,"token,head,feature",attn->k_pe,8U*64U);
    F("kv_cmpr-1","post compressed-KV RMSNorm","MUL",2,512,8,0,"token,feature",attn->kv_cmpr,8U*512U);
    F("q_pe-1","post RoPE positional query","ROPE",3,64,32,8,"token,head,feature",attn->q_pe,8U*32U*64U);
    F("q_nope_absorbed_perm-1","post K-B absorption and permutation","PERMUTE",3,512,32,8,"token,head,feature",attn->q_abs,8U*32U*512U);
    F("Qcur-1","composed attention query","CONCAT",3,576,32,8,"token,head,feature",attn->qcur,8U*32U*576U);
    F("Kcur-1","logical compact key before F16 cache write","CONCAT",3,576,1,8,"token,head,feature",attn->kcur,8U*576U);
    F("Vcur-1","latent value view; no separate cache","RESHAPE",3,512,1,8,"token,head,feature",attn->vcur,8U*512U);
    F("kqv_out-1","post causal attention and V-B expansion","CONT",2,6144,8,0,"token,feature",attn->kqv_out,8U*6144U);
    F("ffn_inp-1","attention projection plus block input","ADD",2,1536,8,0,"token,feature",attn->ffn_inp,8U*1536U);
    F("ffn_norm-1","post FFN RMSNorm","MUL",2,1536,8,0,"token,feature",moe->norm,8U*1536U);
    F("ffn_moe_logits-1","F32 router projection","MUL_MAT",2,64,8,0,"token,expert",moe->logits,8U*64U);
    F("ffn_moe_probs-1","unbiased sigmoid probabilities","UNARY",2,64,8,0,"token,expert",moe->probs,8U*64U);
    F("ffn_moe_probs_biased-1","selection-only bias add","ADD",2,64,8,0,"token,expert",moe->biased,8U*64U);
    strat01_r2c_set_dump(&d[n++],"ffn_moe_topk-1","ordered biased-score top four","VIEW","I32",2,4,8,0,"token,slot",NULL,moe->topk,8U*4U);
    F("ffn_moe_weights-1","selected unbiased sigmoid weights","GET_ROWS",2,4,8,0,"token,slot",moe->weights,8U*4U);
    F("ffn_moe_weights_norm-1","clamped-sum normalized route weights","DIV",2,4,8,0,"token,slot",moe->weights_norm,8U*4U);
    F("ffn_moe_up-1","selected expert up projections","MUL_MAT_ID",3,1280,4,8,"token,slot,feature",moe->moe_up,8U*4U*1280U);
    F("ffn_moe_gate-1","selected expert gate projections","MUL_MAT_ID",3,1280,4,8,"token,slot,feature",moe->moe_gate,8U*4U*1280U);
    F("ffn_moe_swiglu-1","selected expert SwiGLU","GLU",3,1280,4,8,"token,slot,feature",moe->moe_swiglu,8U*4U*1280U);
    F("ffn_moe_down-1","selected expert down projections","MUL_MAT_ID",3,1536,4,8,"token,slot,feature",moe->moe_down,8U*4U*1536U);
    F("ffn_moe_weighted-1","normalized per-slot route weighting","MUL",3,1536,4,8,"token,slot,feature",moe->moe_weighted,8U*4U*1536U);
    F("ffn_moe_out-1","ordered routed-slot sum","ADD",2,1536,8,0,"token,feature",moe->moe_out,8U*1536U);
    F("ffn_up-1","shared expert up projection","MUL_MAT",2,1280,8,0,"token,feature",moe->up,8U*1280U);
    F("ffn_gate-1","shared expert gate projection","MUL_MAT",2,1280,8,0,"token,feature",moe->gate,8U*1280U);
    F("ffn_swiglu-1","shared expert SwiGLU","GLU",2,1280,8,0,"token,feature",moe->swiglu,8U*1280U);
    F("ffn_shexp-1","shared expert down projection","MUL_MAT",2,1536,8,0,"token,feature",moe->shexp,8U*1536U);
    F("ffn_out-1","routed plus shared output","ADD",2,1536,8,0,"token,feature",moe->out,8U*1536U);
    F("l_out-1","MoE output plus attention residual","ADD",2,1536,8,0,"token,feature",moe->l_out,8U*1536U);
#undef F
    return n;
}

static int strat01_r2c_dump_payload(strat01_r2c_dump *d,const char *out_dir,const char *arm,char error[256]) {
    char leaf[256];FILE *f=NULL;strat01_sha256 sha;uint8_t digest[32];
    if(snprintf(leaf,sizeof(leaf),"%s_%s.%s",arm,d->logical_name,!strcmp(d->type,"I32")?"i32":"f32")<=0){snprintf(error,256,"Rung-2C dump name failure");return 0;}for(char *p=leaf;*p;++p)if(*p=='/'||*p=='\\'||*p==' ')*p='_';
    if(!strat01_r2a_path(d->path,out_dir,leaf)||(f=fopen(d->path,"wb"))==NULL){snprintf(error,256,"cannot open Rung-2C dump");return 0;}strat01_sha256_init(&sha);
    for(uint64_t i=0;i<d->count;++i){uint8_t b[4];if(!strcmp(d->type,"I32")){uint32_t u=(uint32_t)d->i32[i];b[0]=(uint8_t)u;b[1]=(uint8_t)(u>>8);b[2]=(uint8_t)(u>>16);b[3]=(uint8_t)(u>>24);}else{if(!isfinite(d->f32[i])){snprintf(error,256,"non-finite Rung-2C tensor: %s",d->logical_name);fclose(f);return 0;}strat01_rung1_put_f32le(b,d->f32[i]);}if(fwrite(b,1,4,f)!=4){snprintf(error,256,"Rung-2C dump write failure");fclose(f);return 0;}strat01_sha256_update(&sha,b,4);}
    if(fclose(f)!=0){snprintf(error,256,"Rung-2C dump close failure");return 0;}strat01_sha256_final(&sha,digest);strat01_sha256_hex(digest,d->sha256);return 1;
}

static int strat01_r2c_write_manifest(const char *out_dir,const char *arm,strat01_r2c_dump d[STRAT01_R2C_DUMPS],const char *cache_paths[4],const char *cache_shas[4],int cached,char error[256]) {
    char leaf[128],path[1024];FILE *o=NULL;if(snprintf(leaf,sizeof(leaf),"%s_manifest.json",arm)<=0||!strat01_r2a_path(path,out_dir,leaf)||(o=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot write Rung-2C manifest");return 0;}
    fputs("{\n  \"arm\": ",o);strat01_json_string(o,arm);fputs(",\n  \"payload_encoding\": {\"F32\":\"IEEE-754 binary32 little-endian\",\"I32\":\"signed int32 little-endian\"},\n  \"shape_order\": \"ggml logical dimensions, fastest first; payload order declared per tensor\",\n  \"tensors\": [",o);
    for(unsigned i=0;i<STRAT01_R2C_DUMPS;++i){fprintf(o,i?",\n    {":"\n    {");fputs("\"name\":",o);strat01_json_string(o,d[i].logical_name);fprintf(o,",\"ordinal\":%u,\"selection\":",d[i].ordinal);strat01_json_string(o,d[i].selection);fputs(",\"op\":",o);strat01_json_string(o,d[i].op);fputs(",\"type\":",o);strat01_json_string(o,d[i].type);fputs(",\"logical_shape\":[",o);for(unsigned k=0;k<d[i].rank;++k)fprintf(o,"%s%" PRIu64,k?",":"",d[i].shape[k]);fputs("],\"payload_order\":",o);strat01_json_string(o,d[i].payload_order);fprintf(o,",\"byte_count\":%" PRIu64 ",\"path\":",d[i].count*4U);strat01_json_string(o,d[i].path);fputs(",\"sha256\":",o);strat01_json_string(o,d[i].sha256);fputc('}',o);}
    fputs("\n  ],\n  \"caches\": [",o);for(unsigned layer=0;layer<2U;++layer){if(layer)fputc(',',o);fprintf(o,"{\"layer\":%u,\"storage\":\"F16\",\"row_length\":576,\"no_separate_v_cache\":true,\"final\":{\"occupied_slots\":[0,1,2,3,4,5,6,7],\"absolute_positions\":[0,1,2,3,4,5,6,7],\"payload_type\":\"dequantized-F32LE\",\"path\":",layer);strat01_json_string(o,cache_paths[layer*2U]);fputs(",\"sha256\":",o);strat01_json_string(o,cache_shas[layer*2U]);fputc('}',o);if(cached){fputs(",\"prefix7\":{\"occupied_slots\":[0,1,2,3,4,5,6],\"absolute_positions\":[0,1,2,3,4,5,6],\"payload_type\":\"dequantized-F32LE\",\"path\":",o);strat01_json_string(o,cache_paths[layer*2U+1U]);fputs(",\"sha256\":",o);strat01_json_string(o,cache_shas[layer*2U+1U]);fputc('}',o);}fputc('}',o);}fputs("]\n}\n",o);
    if(fclose(o)!=0){snprintf(error,256,"Rung-2C manifest close failure");return 0;}return 1;
}

static int strat01_r2c_write_report(const char *out_dir,const char *path,uint64_t bytes,const char *artifact_sha,const char *engine_sha,const char *header_sha,char error[256]) {
    char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2c.json")||(o=fopen(p,"wb"))==NULL){snprintf(error,256,"cannot write Rung-2C report");return 0;}
    fputs("{\n\"command\":\"--strat01-gguf-rung2c\",\n\"c_state\":\"ENGINE_RUNG2C_OUTPUT_READY_PENDING_REFERENCE\",\n\"self_certifies_pass\":false,\n\"input_path\":",o);strat01_json_string(o,path);fprintf(o,",\n\"byte_size\":%" PRIu64 ",\n\"sha256\":",bytes);strat01_json_string(o,artifact_sha);fputs(",\n\"reference_revision\":",o);strat01_json_string(o,STRAT01_R2A_REFERENCE);fputs(",\n\"CONFIG\":",o);strat01_json_string(o,strat01_r2c_config);fputs(",\n\"compiler_family\":\"clang\",\n\"compiler_embedded_version\":",o);strat01_json_string(o,STRAT01_R2A_COMPILER);fputs(",\n\"compiler_resolved_path_and_full_version\":\"EXTERNAL_RUNNER_REQUIRED\",\n\"engine_source_sha256\":",o);strat01_json_string(o,engine_sha);fputs(",\n\"rung2c_source_sha256\":",o);strat01_json_string(o,header_sha);fputs(",\n\"token_ids\":[1,72,14,14129,14,2135,1512,2015],\n\"positions\":[0,1,2,3,4,5,6,7],\n\"arms\":[{\"name\":\"prefill8\",\"manifest\":\"prefill8_manifest.json\"},{\"name\":\"cached7p1\",\"manifest\":\"cached7p1_manifest.json\"}],\n\"timing_or_rate_claim\":null\n}\n",o);if(fclose(o)!=0){snprintf(error,256,"Rung-2C report close failure");return 0;}return 1;
}

static int strat01_r2c_write_failure(const char *out_dir,const char *path,const char *error) {char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2c.json")||(o=fopen(p,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-gguf-rung2c\",\"input_path\":",o);strat01_json_string(o,path);fputs(",\"c_state\":\"ENGINE_RUNG2C_C_FAILURE\",\"error\":",o);strat01_json_string(o,error);fputs("}\n",o);fclose(o);return 1;}

static int strat01_r2c_dump_arm(const char *out_dir,const char *arm,const strat01_r2b_arm *b0,const strat01_r2a_arm *a0,const strat01_r2a_arm *a1,const strat01_r2c_moe_arm *moe,int cached,char error[256]) {
    strat01_r2c_dump d[STRAT01_R2C_DUMPS];const char *paths[4]={0};const char *shas[4]={0};char cache_path[4][1024],cache_sha[4][65];char leaf[4][96];
    if(strat01_r2c_make_dumps(b0,a1,moe,d)!=STRAT01_R2C_DUMPS){snprintf(error,256,"Rung-2C dump count mismatch");return 0;}for(unsigned i=0;i<STRAT01_R2C_DUMPS;++i)if(!strat01_r2c_dump_payload(&d[i],out_dir,arm,error))return 0;
    for(unsigned layer=0;layer<2U;++layer){const strat01_r2a_arm *src=layer?a1:a0;snprintf(leaf[layer*2U],sizeof(leaf[0]),"%s_layer%u_cache_final.f32",arm,layer);if(!strat01_r2a_write_cache_dump(out_dir,leaf[layer*2U],src->cache,8U,cache_path[layer*2U],cache_sha[layer*2U],error))return 0;paths[layer*2U]=cache_path[layer*2U];shas[layer*2U]=cache_sha[layer*2U];if(cached){snprintf(leaf[layer*2U+1U],sizeof(leaf[0]),"%s_layer%u_cache_prefix7.f32",arm,layer);if(!strat01_r2a_write_cache_dump(out_dir,leaf[layer*2U+1U],src->cache,7U,cache_path[layer*2U+1U],cache_sha[layer*2U+1U],error))return 0;paths[layer*2U+1U]=cache_path[layer*2U+1U];shas[layer*2U+1U]=cache_sha[layer*2U+1U];}}
    return strat01_r2c_write_manifest(out_dir,arm,d,paths,shas,cached,error);
}

static int strat01_gguf_rung2c_cli(const char *path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *base[8]={0},*b0ffn[4]={0},*attn[7]={0},*moe[9]={0};strat01_r2a_arm p0a,c0a,p1a,c1a;strat01_r2b_arm p0b,c0b;strat01_r2c_moe_arm pm,cm;char error[256]={0},artifact_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};uint64_t hashed=0,parsed=0,tmp=0;int ok=0;
    memset(&inv,0,sizeof(inv));memset(&p0a,0,sizeof(p0a));memset(&c0a,0,sizeof(c0a));memset(&p1a,0,sizeof(p1a));memset(&c1a,0,sizeof(c1a));memset(&p0b,0,sizeof(p0b));memset(&c0b,0,sizeof(c0b));memset(&pm,0,sizeof(pm));memset(&cm,0,sizeof(cm));strat01_f16vec_reset_counts();
#if !defined(__clang__)
    snprintf(error,256,"Rung-2C requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(path,artifact_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(artifact_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"frozen Rung-2C artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(path,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"Rung-2C GGUF parse/size mismatch");goto finish;}
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&base[i],error))goto finish;for(unsigned i=0;i<4U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[i],&b0ffn[i],error))goto finish;for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&attn[i],error))goto finish;for(unsigned i=0;i<9U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[i],&moe[i],error))goto finish;
    if(!strat01_r2a_arm_alloc(&p0a,error)||!strat01_r2a_arm_alloc(&c0a,error)||!strat01_r2a_arm_alloc(&p1a,error)||!strat01_r2a_arm_alloc(&c1a,error)||!strat01_r2b_arm_alloc(&p0b,error)||!strat01_r2b_arm_alloc(&c0b,error)||!strat01_r2c_moe_alloc(&pm,error)||!strat01_r2c_moe_alloc(&cm,error))goto finish;
    if(!strat01_r2a_build_upstream_range(path,base,&p0a,0,8,error)||!strat01_r2a_run_schedule(path,base[6],base[7],&p0a,0,error)||!strat01_r2b_run(path,b0ffn[0],b0ffn[1],b0ffn[2],b0ffn[3],&p0a,&p0b,error))goto finish;
    if(!strat01_r2a_build_upstream_range(path,base,&c0a,0,7,error)||!strat01_r2a_build_upstream_range(path,base,&c0a,7,1,error)||!strat01_r2a_run_schedule(path,base[6],base[7],&c0a,1,error)||!strat01_r2b_run(path,b0ffn[0],b0ffn[1],b0ffn[2],b0ffn[3],&c0a,&c0b,error))goto finish;
    memcpy(p1a.embd,p0b.l_out,8U*1536U*4U);memcpy(c1a.embd,c0b.l_out,8U*1536U*4U);
    if(!strat01_r2c_build_attention_range(path,attn,&p1a,0,8,error)||!strat01_r2a_run_schedule(path,attn[5],attn[6],&p1a,0,error)||!strat01_r2c_run_moe(path,moe,&p1a,&pm,error))goto finish;
    fprintf(stderr,"STRAT01_RUNG2C_GRAPH_COMPLETE arm=prefill8\n");
    if(!strat01_r2c_build_attention_range(path,attn,&c1a,0,7,error)||!strat01_r2c_build_attention_range(path,attn,&c1a,7,1,error)||!strat01_r2a_run_schedule(path,attn[5],attn[6],&c1a,1,error)||!strat01_r2c_run_moe(path,moe,&c1a,&cm,error))goto finish;
    fprintf(stderr,"STRAT01_RUNG2C_GRAPH_COMPLETE arm=cached7p1\n");
    if(!strat01_r2c_dump_arm(out_dir,"prefill8",&p0b,&p0a,&p1a,&pm,0,error)||!strat01_r2c_dump_arm(out_dir,"cached7p1",&c0b,&c0a,&c1a,&cm,1,error)||!strat01_f16v_write_counts(out_dir,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))strcpy(engine_sha,"unavailable");error[0]=0;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))strcpy(header_sha,"unavailable");error[0]=0;if(!strat01_r2c_write_report(out_dir,path,hashed,artifact_sha,engine_sha,header_sha,error))goto finish;ok=1;
finish:strat01_free_inventory(&inv);strat01_r2a_arm_free(&p0a);strat01_r2a_arm_free(&c0a);strat01_r2a_arm_free(&p1a);strat01_r2a_arm_free(&c1a);strat01_r2b_arm_free(&p0b);strat01_r2b_arm_free(&c0b);strat01_r2c_moe_free(&pm);strat01_r2c_moe_free(&cm);if(!ok){if(!error[0])snprintf(error,256,"unspecified Rung-2C failure");strat01_r2c_write_failure(out_dir,path,error);fprintf(stderr,"STRAT-01 Rung-2C refused: %s\n",error);return 1;}fprintf(stderr,"CONFIG %s\n",strat01_r2c_config);fprintf(stderr,"STRAT-01 Rung-2C: ENGINE_RUNG2C_OUTPUT_READY_PENDING_REFERENCE\n");return 0;
}

static int strat01_gguf_rung2c_selftest(void) {
    int bad=0,checks=0;
#define R2C_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    R2C_CHECK(strat01_gguf_rung2b_selftest()==0);
    {float score[64]={0};int32_t ids[4];score[3]=4;score[7]=3;score[9]=2;score[11]=1;strat01_r2c_top4(score,ids);R2C_CHECK(ids[0]==3&&ids[1]==7&&ids[2]==9&&ids[3]==11);score[5]=5;strat01_r2c_top4(score,ids);R2C_CHECK(ids[0]==5);}
    {float p[4]={.2f,.3f,.4f,.5f},bias[4]={1,0,0,0},unbiased=p[1]/(p[1]+p[2]),wrong=(p[1]+bias[1])/((p[1]+bias[1])+(p[2]+bias[2]));R2C_CHECK(unbiased==wrong);bias[1]=.5f;wrong=(p[1]+bias[1])/((p[1]+bias[1])+(p[2]+bias[2]));R2C_CHECK(unbiased!=wrong);}
    {float x[4]={1,2,3,4},sum=10,norm=0;for(unsigned i=0;i<4;++i)norm+=x[i]/sum;R2C_CHECK(fabsf(norm-1.0f)<1e-6f&&sum!=1.0f);}
    {float score[64]={0};int32_t ids[4];score[1]=4;score[2]=3;score[3]=2;score[4]=1;strat01_r2c_top4(score,ids);int32_t saved=ids[0];ids[0]=ids[1];R2C_CHECK(ids[0]!=saved);}
    {strat01_tensor t={0},v0,v1;t.type=STRAT01_GGML_Q4_K;t.rank=3;t.dims[0]=1536;t.dims[1]=1280;t.dims[2]=64;t.file_offset=1000;char e[256]={0};R2C_CHECK(strat01_r2c_expert_view(&t,0,1536,1280,&v0,e)&&strat01_r2c_expert_view(&t,1,1536,1280,&v1,e)&&v1.file_offset>v0.file_offset);}
    {float gate=-1,up=2,good=(gate/(1+expf(-gate)))*up,swap=(up/(1+expf(-up)))*gate;R2C_CHECK(good!=swap);}
    {float routed[2]={1,2},shared[2]={3,4},out[2]={routed[0]+shared[0],routed[1]+shared[1]};R2C_CHECK(memcmp(out,routed,sizeof(out))!=0);float noadd[2]={shared[0],shared[1]};R2C_CHECK(memcmp(out,noadd,sizeof(out))!=0);}
    {float ffn[2]={.5f,-.5f},res[2]={1,2},out[2]={ffn[0]+res[0],ffn[1]+res[1]};R2C_CHECK(memcmp(out,ffn,sizeof(out))!=0);}
    {const char *hash="258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44";char mutated[65];strcpy(mutated,hash);mutated[0]='0';R2C_CHECK(strcmp(hash,mutated)!=0);unsigned layer=1,slot=7;R2C_CHECK(layer==1&&slot==7&&!(layer==0&&slot==7));}
#undef R2C_CHECK
    fprintf(stderr,"STRAT-01 Rung-2C model-free selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#pragma STDC FP_CONTRACT DEFAULT
#endif /* STRAT01_GGUF_RUNG2C_H */
