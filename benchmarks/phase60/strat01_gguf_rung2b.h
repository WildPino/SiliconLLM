/*
 * STRAT-01 engine rung 2B: block-0 dense SwiGLU on the accepted GigaChat
 * artifact. This extends the independently closed rung-2A boundary and is
 * correctness apparatus, not a throughput path.
 */
#ifndef STRAT01_GGUF_RUNG2B_H
#define STRAT01_GGUF_RUNG2B_H

#define STRAT01_R2B_FFN 8960U
#define STRAT01_R2B_Q6_BLOCK_BYTES 210U

static const char strat01_r2b_config[] =
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;cache=layer-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "ffn=block0-rmsnorm-q4kq8k-gate-up-silu-q6kq8k-down-residual;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-token-major;adjudication=external-reference-only";

static const strat01_r2a_tensor_spec strat01_r2b_specs[] = {
    {"blk.0.ffn_norm.weight", STRAT01_GGML_F32, 1, {1536,0,0}},
    {"blk.0.ffn_gate.weight", STRAT01_GGML_Q4_K,2, {1536,8960,0}},
    {"blk.0.ffn_up.weight",   STRAT01_GGML_Q4_K,2, {1536,8960,0}},
    {"blk.0.ffn_down.weight", STRAT01_GGML_Q6_K,2, {8960,1536,0}},
};

typedef struct {
    float *norm,*up,*gate,*swiglu,*out,*l_out;
} strat01_r2b_arm;

static int strat01_r2b_arm_alloc(strat01_r2b_arm *a,char error[256]) {
    memset(a,0,sizeof(*a));
#define R2B_ALLOC(field,count) do { a->field=strat01_r2a_alloc((count),error); if(!a->field)return 0; } while(0)
    R2B_ALLOC(norm,8U*1536U);R2B_ALLOC(up,8U*STRAT01_R2B_FFN);
    R2B_ALLOC(gate,8U*STRAT01_R2B_FFN);R2B_ALLOC(swiglu,8U*STRAT01_R2B_FFN);
    R2B_ALLOC(out,8U*1536U);R2B_ALLOC(l_out,8U*1536U);
#undef R2B_ALLOC
    return 1;
}

static void strat01_r2b_arm_free(strat01_r2b_arm *a) {
    free(a->norm);free(a->up);free(a->gate);free(a->swiglu);free(a->out);free(a->l_out);memset(a,0,sizeof(*a));
}

/* Scalar form of pinned ggml_vec_dot_q6_K_q8_K_generic. */
static float strat01_q6k_q8k_dot(const uint8_t *q6_blocks,const strat01_q8_k_block *q8_blocks,unsigned count) {
    float sums[8]={0,0,0,0,0,0,0,0},result=0.0f;
    if(!q6_blocks||!q8_blocks||!count||count%STRAT01_QK_K)return NAN;
    for(unsigned block=0;block<count/STRAT01_QK_K;++block){
        const uint8_t *raw=q6_blocks+(size_t)block*STRAT01_R2B_Q6_BLOCK_BYTES;
        const uint8_t *q4=raw,*qh=raw+128U;const int8_t *scales=(const int8_t *)(raw+192U);
        const int8_t *q8=q8_blocks[block].qs;int8_t unpacked[256];int32_t lanes[8]={0,0,0,0,0,0,0,0};int8_t *a=unpacked;
        for(unsigned j=0;j<256U;j+=128U){
            for(unsigned l=0;l<32U;++l){
                a[l+0]=(int8_t)((q4[l+0]&15U)|(((qh[l]>>0)&3U)<<4))-32;
                a[l+32]=(int8_t)((q4[l+32]&15U)|(((qh[l]>>2)&3U)<<4))-32;
                a[l+64]=(int8_t)((q4[l+0]>>4)|(((qh[l]>>4)&3U)<<4))-32;
                a[l+96]=(int8_t)((q4[l+32]>>4)|(((qh[l]>>6)&3U)<<4))-32;
            }
            a+=128;q4+=64;qh+=32;
        }
        a=unpacked;
        for(unsigned group=0;group<16U;++group){
            int scale=scales[group];
            for(unsigned lane=0;lane<8U;++lane)lanes[lane]+=scale*(int16_t)q8[lane]*(int16_t)a[lane];
            q8+=8;a+=8;
            for(unsigned lane=0;lane<8U;++lane)lanes[lane]+=scale*(int16_t)q8[lane]*(int16_t)a[lane];
            q8+=8;a+=8;
        }
        {float d=strat01_q4k_q8k_fp16le(raw+208U)*q8_blocks[block].d;for(unsigned lane=0;lane<8U;++lane)sums[lane]+=d*(float)lanes[lane];}
    }
    for(unsigned lane=0;lane<8U;++lane)result+=sums[lane];return result;
}

static int strat01_r2b_q6_matmul_batch(const char *path,const strat01_tensor *t,const float *x,unsigned batch,float *y,char error[256]) {
    FILE *f=NULL;strat01_q8_k_block *q8=NULL;uint8_t *raw=NULL;size_t q8_count,row_bytes;const unsigned blocks=STRAT01_R2B_FFN/STRAT01_QK_K;
    if(!t||t->type!=STRAT01_GGML_Q6_K||t->rank!=2||t->dims[0]!=STRAT01_R2B_FFN||t->dims[1]!=1536U||!batch){snprintf(error,256,"rung-2B Q6_K descriptor mismatch");return 0;}
    if(!strat01_r2a_safe_count(batch,blocks,&q8_count)||!strat01_r2a_safe_count(blocks,STRAT01_R2B_Q6_BLOCK_BYTES,&row_bytes)){snprintf(error,256,"rung-2B Q6_K allocation overflow");return 0;}
    q8=(strat01_q8_k_block *)malloc(q8_count*sizeof(*q8));raw=(uint8_t *)malloc(row_bytes);
    if(!q8||!raw){snprintf(error,256,"rung-2B Q6_K allocation failed");goto fail;}
    for(unsigned b=0;b<batch;++b)if(!strat01_quantize_q8_k_row(x+(size_t)b*STRAT01_R2B_FFN,STRAT01_R2B_FFN,q8+(size_t)b*blocks)){snprintf(error,256,"rung-2B Q6_K Q8_K activation quantization failed");goto fail;}
    f=fopen(path,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error))goto fail;
    for(unsigned row=0;row<1536U;++row){
        if(fread(raw,1,row_bytes,f)!=row_bytes){snprintf(error,256,"rung-2B Q6_K row short read");goto fail;}
        for(unsigned b=0;b<batch;++b){float v=strat01_q6k_q8k_dot(raw,q8+(size_t)b*blocks,STRAT01_R2B_FFN);if(!isfinite(v)){snprintf(error,256,"rung-2B Q6_K/Q8_K non-finite output");goto fail;}y[(size_t)b*1536U+row]=v;}
    }
    if(ferror(f)){snprintf(error,256,"rung-2B Q6_K I/O failure");goto fail;}
    if(fclose(f)!=0){f=NULL;snprintf(error,256,"rung-2B Q6_K close failure");goto fail;}
    f=NULL;free(q8);free(raw);return 1;
fail:
    if(f)fclose(f);free(q8);free(raw);return 0;
}

static int strat01_r2b_run(const char *path,const strat01_tensor *norm_w,const strat01_tensor *gate_w,const strat01_tensor *up_w,const strat01_tensor *down_w,const strat01_r2a_arm *input,strat01_r2b_arm *out,char error[256]) {
    float *weight=strat01_r2a_alloc(1536U,error);if(!weight)return 0;
    if(!strat01_r2a_read_f32_vector(path,norm_w,weight,1536U,error)){free(weight);return 0;}
    strat01_r2a_rmsnorm(input->ffn_inp,weight,out->norm,8U,1536U,STRAT01_R2A_RMS_EPS);free(weight);
    if(!strat01_r2a_matmul_batch(path,up_w,out->norm,8U,1536U,out->up,STRAT01_R2B_FFN,error)||
       !strat01_r2a_matmul_batch(path,gate_w,out->norm,8U,1536U,out->gate,STRAT01_R2B_FFN,error))return 0;
    for(size_t i=0;i<8U*STRAT01_R2B_FFN;++i)out->swiglu[i]=(out->gate[i]/(1.0f+expf(-out->gate[i])))*out->up[i];
    if(!strat01_r2b_q6_matmul_batch(path,down_w,out->swiglu,8U,out->out,error))return 0;
    for(size_t i=0;i<8U*1536U;++i)out->l_out[i]=out->out[i]+input->ffn_inp[i];return 1;
}

static unsigned strat01_r2b_make_dumps(strat01_r2b_arm *a,strat01_r2a_dump d[6]) {
#define R2B_DUMP(i,n,sel,operation,s0,ptr,cnt) do{memset(&d[i],0,sizeof(d[i]));d[i].logical_name=n;d[i].selection=sel;d[i].op=operation;d[i].ordinal=0;d[i].rank=2;d[i].shape[0]=s0;d[i].shape[1]=8;d[i].payload_order="token,feature";d[i].data=ptr;d[i].count=cnt;}while(0)
    R2B_DUMP(0,"ffn_norm-0","post FFN RMSNorm","MUL",1536,a->norm,8U*1536U);
    R2B_DUMP(1,"ffn_up-0","parallel Q4_K up projection","MUL_MAT",8960,a->up,8U*8960U);
    R2B_DUMP(2,"ffn_gate-0","parallel Q4_K gate projection","MUL_MAT",8960,a->gate,8U*8960U);
    R2B_DUMP(3,"ffn_swiglu-0","SiLU(gate) times up","GLU",8960,a->swiglu,8U*8960U);
    R2B_DUMP(4,"ffn_out-0","Q6_K down projection before residual","MUL_MAT",1536,a->out,8U*1536U);
    R2B_DUMP(5,"l_out-0","dense FFN output plus ffn_inp residual","ADD",1536,a->l_out,8U*1536U);
#undef R2B_DUMP
    return 6;
}

static int strat01_r2b_write_manifest(const char *out_dir,const char *arm,strat01_r2a_dump d[6],char error[256]) {
    char leaf[128],path[1024];FILE *o;
    if(snprintf(leaf,sizeof(leaf),"%s_manifest.json",arm)<=0||!strat01_r2a_path(path,out_dir,leaf)||(o=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot write rung-2B manifest");return 0;}
    fputs("{\n  \"arm\": ",o);strat01_json_string(o,arm);fputs(",\n  \"payload_encoding\": \"IEEE-754 binary32 little-endian\",\n  \"shape_order\": \"ggml logical dimensions, fastest first\",\n  \"tensors\": [",o);
    for(unsigned i=0;i<6;++i){fprintf(o,i?",\n    {":"\n    {");fputs("\"name\": ",o);strat01_json_string(o,d[i].logical_name);fprintf(o,", \"ordinal\": %u, \"selection\": ",d[i].ordinal);strat01_json_string(o,d[i].selection);fputs(", \"op\": ",o);strat01_json_string(o,d[i].op);fprintf(o,", \"type\": \"F32\", \"logical_shape\": [%" PRIu64 ", %" PRIu64 "], \"payload_order\": \"token,feature\", \"byte_count\": %" PRIu64 ", \"path\": ",d[i].shape[0],d[i].shape[1],d[i].count*4U);strat01_json_string(o,d[i].path);fputs(", \"sha256\": ",o);strat01_json_string(o,d[i].sha256);fputc('}',o);}
    fputs("\n  ]\n}\n",o);if(fclose(o)!=0){snprintf(error,256,"rung-2B manifest close failure");return 0;}return 1;
}

static int strat01_r2b_write_report(const char *out_dir,const char *path,uint64_t bytes,const char *artifact_sha,const char *engine_sha,const char *header_sha,char error[256]) {
    char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2b.json")||(o=fopen(p,"wb"))==NULL){snprintf(error,256,"cannot write rung-2B report");return 0;}
    fputs("{\n  \"command\": \"--strat01-gguf-rung2b\",\n  \"c_state\": \"ENGINE_RUNG2B_OUTPUT_READY_PENDING_REFERENCE\",\n  \"self_certifies_pass\": false,\n  \"input_path\": ",o);strat01_json_string(o,path);fprintf(o,",\n  \"byte_size\": %" PRIu64 ",\n  \"sha256\": ",bytes);strat01_json_string(o,artifact_sha);fputs(",\n  \"reference_revision\": ",o);strat01_json_string(o,STRAT01_R2A_REFERENCE);fputs(",\n  \"CONFIG\": ",o);strat01_json_string(o,strat01_r2b_config);fputs(",\n  \"compiler_family\": \"clang\",\n  \"compiler_embedded_version\": ",o);strat01_json_string(o,STRAT01_R2A_COMPILER);fputs(",\n  \"compiler_resolved_path_and_full_version\": \"EXTERNAL_RUNNER_REQUIRED\",\n  \"engine_source_sha256\": ",o);strat01_json_string(o,engine_sha);fputs(",\n  \"rung2b_source_sha256\": ",o);strat01_json_string(o,header_sha);fputs(",\n  \"token_ids\": [1,72,14,14129,14,2135,1512,2015],\n  \"positions\": [0,1,2,3,4,5,6,7],\n  \"arms\": [{\"name\":\"prefill8\",\"manifest\":\"prefill8_manifest.json\"},{\"name\":\"cached7p1\",\"manifest\":\"cached7p1_manifest.json\"}],\n  \"timing_or_rate_claim\": null\n}\n",o);
    if(fclose(o)!=0){snprintf(error,256,"rung-2B report close failure");return 0;}return 1;
}

static int strat01_r2b_write_failure(const char *out_dir,const char *path,const char *error) {
    char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2b.json")||(o=fopen(p,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-gguf-rung2b\",\"input_path\":",o);strat01_json_string(o,path);fputs(",\"c_state\":\"ENGINE_RUNG2B_C_FAILURE\",\"error\":",o);strat01_json_string(o,error);fputs("}\n",o);fclose(o);return 1;
}

static int strat01_gguf_rung2b_cli(const char *path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *base[8]={0},*ffn[4]={0};strat01_r2a_arm pre_in,cache_in;strat01_r2b_arm pre,cache;strat01_r2a_dump dumps[6];
    char error[256]={0},artifact_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};uint64_t hashed=0,parsed=0,tmp=0;int ok=0;
    memset(&inv,0,sizeof(inv));memset(&pre_in,0,sizeof(pre_in));memset(&cache_in,0,sizeof(cache_in));memset(&pre,0,sizeof(pre));memset(&cache,0,sizeof(cache));
#if !defined(__clang__)
    snprintf(error,256,"rung-2B requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(path,artifact_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(artifact_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"frozen rung-2B artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(path,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"rung-2B GGUF parse/size mismatch");goto finish;}
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&base[i],error))goto finish;
    for(unsigned i=0;i<4U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[i],&ffn[i],error))goto finish;
    if(!strat01_r2a_arm_alloc(&pre_in,error)||!strat01_r2a_arm_alloc(&cache_in,error)||!strat01_r2b_arm_alloc(&pre,error)||!strat01_r2b_arm_alloc(&cache,error))goto finish;
    if(!strat01_r2a_build_upstream_range(path,base,&pre_in,0,8,error)||!strat01_r2a_run_schedule(path,base[6],base[7],&pre_in,0,error))goto finish;
    if(!strat01_r2a_build_upstream_range(path,base,&cache_in,0,7,error)||!strat01_r2a_build_upstream_range(path,base,&cache_in,7,1,error)||!strat01_r2a_run_schedule(path,base[6],base[7],&cache_in,1,error))goto finish;
    if(!strat01_r2b_run(path,ffn[0],ffn[1],ffn[2],ffn[3],&pre_in,&pre,error)||!strat01_r2b_run(path,ffn[0],ffn[1],ffn[2],ffn[3],&cache_in,&cache,error))goto finish;
    strat01_r2b_make_dumps(&pre,dumps);for(unsigned i=0;i<6U;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"prefill8",error))goto finish;if(!strat01_r2b_write_manifest(out_dir,"prefill8",dumps,error))goto finish;
    strat01_r2b_make_dumps(&cache,dumps);for(unsigned i=0;i<6U;++i)if(!strat01_r2a_dump_f32(&dumps[i],out_dir,"cached7p1",error))goto finish;if(!strat01_r2b_write_manifest(out_dir,"cached7p1",dumps,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))strcpy(engine_sha,"unavailable");error[0]=0;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))strcpy(header_sha,"unavailable");error[0]=0;
    if(!strat01_r2b_write_report(out_dir,path,hashed,artifact_sha,engine_sha,header_sha,error))goto finish;ok=1;
finish:
    strat01_free_inventory(&inv);strat01_r2a_arm_free(&pre_in);strat01_r2a_arm_free(&cache_in);strat01_r2b_arm_free(&pre);strat01_r2b_arm_free(&cache);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified rung-2B failure");strat01_r2b_write_failure(out_dir,path,error);fprintf(stderr,"STRAT-01 rung-2B refused: %s\n",error);return 1;}
    fprintf(stderr,"CONFIG %s\n",strat01_r2b_config);fprintf(stderr,"STRAT-01 rung-2B: ENGINE_RUNG2B_OUTPUT_READY_PENDING_REFERENCE\n");return 0;
}

static int strat01_gguf_rung2b_selftest(void) {
    int bad=0,checks=0;
#define R2B_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    R2B_CHECK(strat01_gguf_rung2a_selftest()==0);
    {float gate[4]={-2,-.5f,.5f,2},up[4]={1,2,3,4},good[4],swapped[4],linear[4];for(unsigned i=0;i<4;++i){good[i]=(gate[i]/(1+expf(-gate[i])))*up[i];swapped[i]=(up[i]/(1+expf(-up[i])))*gate[i];linear[i]=gate[i]*up[i];}R2B_CHECK(memcmp(good,swapped,sizeof(good))!=0);R2B_CHECK(memcmp(good,linear,sizeof(good))!=0);}
    {uint8_t q6[STRAT01_R2B_Q6_BLOCK_BYTES]={0};float x[256]={0};strat01_q8_k_block q8;q6[0]=1;q6[192]=1;q6[208]=0;q6[209]=60;x[0]=1;R2B_CHECK(strat01_quantize_q8_k_block(x,&q8));float a=strat01_q6k_q8k_dot(q6,&q8,256);q6[0]^=1;float b=strat01_q6k_q8k_dot(q6,&q8,256);R2B_CHECK(isfinite(a)&&isfinite(b)&&a!=b);}
    {float ffn[3]={.25f,-.5f,1},res[3]={1,2,3},with_res[3];for(unsigned i=0;i<3;++i)with_res[i]=ffn[i]+res[i];R2B_CHECK(memcmp(with_res,ffn,sizeof(ffn))!=0);}
#undef R2B_CHECK
    fprintf(stderr,"STRAT-01 rung-2B model-free selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif /* STRAT01_GGUF_RUNG2B_H */
