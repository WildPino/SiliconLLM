/*
 * STRAT-01 engine Rung 2D: accepted blocks 0-1 followed by complete block 2.
 * Correctness apparatus only; no timing or generation claim.
 */
#ifndef STRAT01_GGUF_RUNG2D_H
#define STRAT01_GGUF_RUNG2D_H

#pragma STDC FP_CONTRACT OFF

static const char strat01_r2d_config[] =
    "reference=llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2;cpu_threads=1;"
    "n_ctx=8;n_batch=8;n_ubatch=8;flash_attn=false;offload_kqv=false;"
    "type_k=f16;type_v=f16-no-allocation-mla;f16_dot=pinned-generic-f64;cache=layers0-2-slot-576-f16-k-only-latent512-rope64;"
    "tokens=1,72,14,14129,14,2135,1512,2015;positions=0,1,2,3,4,5,6,7;"
    "rms_eps=1e-6;rope=deepseek2-normal-yarn;rope_base=100000;rope_factor=64;"
    "rope_orig_ctx=4096;beta_fast=32;beta_slow=1;mscale=1;mscale_all_dim=1;"
    "rms_accum=double;kb=q5_0xq8_0;"
    "block0=dense-swiglu-sse2-nofma4;block1=accepted-rung2c-unchanged;"
    "block2=mla-sigmoid-bias-select-top4-unbiased-normalized-q4k-q6k-shared-residual;"
    "build=clang-c11-O3-mavx2-mfma-no-fast-math;fp_contract=off-c11-pragma;"
    "payload=f32le-or-i32le-token-major;adjudication=external-reference-only";

static const strat01_r2a_tensor_spec strat01_r2d_attn_specs[] = {
    {"blk.2.attn_norm.weight",      STRAT01_GGML_F32, 1,{1536,0,0}},
    {"blk.2.attn_q.weight",         STRAT01_GGML_Q4_K,2,{1536,6144,0}},
    {"blk.2.attn_kv_a_mqa.weight",  STRAT01_GGML_Q4_K,2,{1536,576,0}},
    {"blk.2.attn_kv_a_norm.weight", STRAT01_GGML_F32, 1,{512,0,0}},
    {"blk.2.attn_k_b.weight",       STRAT01_GGML_Q5_0,3,{128,512,32}},
    {"blk.2.attn_v_b.weight",       STRAT01_GGML_Q4_K,3,{512,192,32}},
    {"blk.2.attn_output.weight",    STRAT01_GGML_Q4_K,2,{6144,1536,0}},
};

static const strat01_r2a_tensor_spec strat01_r2d_moe_specs[] = {
    {"blk.2.ffn_norm.weight",       STRAT01_GGML_F32, 1,{1536,0,0}},
    {"blk.2.ffn_gate_inp.weight",   STRAT01_GGML_F32, 2,{1536,64,0}},
    {"blk.2.exp_probs_b.bias",      STRAT01_GGML_F32, 1,{64,0,0}},
    {"blk.2.ffn_up_exps.weight",    STRAT01_GGML_Q4_K,3,{1536,1280,64}},
    {"blk.2.ffn_gate_exps.weight",  STRAT01_GGML_Q4_K,3,{1536,1280,64}},
    {"blk.2.ffn_down_exps.weight",  STRAT01_GGML_Q6_K,3,{1280,1536,64}},
    {"blk.2.ffn_up_shexp.weight",   STRAT01_GGML_Q4_K,2,{1536,1280,0}},
    {"blk.2.ffn_gate_shexp.weight", STRAT01_GGML_Q4_K,2,{1536,1280,0}},
    {"blk.2.ffn_down_shexp.weight", STRAT01_GGML_Q6_K,2,{1280,1536,0}},
};

static unsigned strat01_r2d_make_dumps(const float *start,const strat01_r2a_arm *attn,const strat01_r2c_moe_arm *moe,strat01_r2c_dump d[STRAT01_R2C_DUMPS]) {
    unsigned n=0;
#define F(name,sel,op,rank,s0,s1,s2,order,ptr,count) strat01_r2c_set_dump(&d[n++],name,sel,op,"F32",rank,s0,s1,s2,order,ptr,NULL,count)
    F("l_out-1","accepted layer-1 terminal start state","ADD",2,1536,8,0,"token,feature",start,8U*1536U);
    F("attn_norm-2","post RMSNorm","MUL",2,1536,8,0,"token,feature",attn->attn_norm,8U*1536U);
    F("q-2","direct Q after reshape","RESHAPE",3,192,32,8,"token,head,feature",attn->q,8U*6144U);
    F("kv_cmpr_pe-2","post KV-A projection before split","MUL_MAT",2,576,8,0,"token,feature",attn->kv_cmpr_pe,8U*576U);
    F("k_pe-2","post RoPE positional key","ROPE",3,64,1,8,"token,head,feature",attn->k_pe,8U*64U);
    F("kv_cmpr-2","post compressed-KV RMSNorm","MUL",2,512,8,0,"token,feature",attn->kv_cmpr,8U*512U);
    F("q_pe-2","post RoPE positional query","ROPE",3,64,32,8,"token,head,feature",attn->q_pe,8U*32U*64U);
    F("q_nope_absorbed_perm-2","post K-B absorption and permutation","PERMUTE",3,512,32,8,"token,head,feature",attn->q_abs,8U*32U*512U);
    F("Qcur-2","composed attention query","CONCAT",3,576,32,8,"token,head,feature",attn->qcur,8U*32U*576U);
    F("Kcur-2","logical compact key before F16 cache write","CONCAT",3,576,1,8,"token,head,feature",attn->kcur,8U*576U);
    F("Vcur-2","latent value view; no separate cache","RESHAPE",3,512,1,8,"token,head,feature",attn->vcur,8U*512U);
    F("kqv_out-2","post causal attention and V-B expansion","CONT",2,6144,8,0,"token,feature",attn->kqv_out,8U*6144U);
    F("ffn_inp-2","attention projection plus block input","ADD",2,1536,8,0,"token,feature",attn->ffn_inp,8U*1536U);
    F("ffn_norm-2","post FFN RMSNorm","MUL",2,1536,8,0,"token,feature",moe->norm,8U*1536U);
    F("ffn_moe_logits-2","F32 router projection","MUL_MAT",2,64,8,0,"token,expert",moe->logits,8U*64U);
    F("ffn_moe_probs-2","unbiased sigmoid probabilities","UNARY",2,64,8,0,"token,expert",moe->probs,8U*64U);
    F("ffn_moe_probs_biased-2","selection-only bias add","ADD",2,64,8,0,"token,expert",moe->biased,8U*64U);
    strat01_r2c_set_dump(&d[n++],"ffn_moe_topk-2","ordered biased-score top four","VIEW","I32",2,4,8,0,"token,slot",NULL,moe->topk,8U*4U);
    F("ffn_moe_weights-2","selected unbiased sigmoid weights","GET_ROWS",2,4,8,0,"token,slot",moe->weights,8U*4U);
    F("ffn_moe_weights_norm-2","clamped-sum normalized route weights","DIV",2,4,8,0,"token,slot",moe->weights_norm,8U*4U);
    F("ffn_moe_up-2","selected expert up projections","MUL_MAT_ID",3,1280,4,8,"token,slot,feature",moe->moe_up,8U*4U*1280U);
    F("ffn_moe_gate-2","selected expert gate projections","MUL_MAT_ID",3,1280,4,8,"token,slot,feature",moe->moe_gate,8U*4U*1280U);
    F("ffn_moe_swiglu-2","selected expert SwiGLU","GLU",3,1280,4,8,"token,slot,feature",moe->moe_swiglu,8U*4U*1280U);
    F("ffn_moe_down-2","selected expert down projections","MUL_MAT_ID",3,1536,4,8,"token,slot,feature",moe->moe_down,8U*4U*1536U);
    F("ffn_moe_weighted-2","normalized per-slot route weighting","MUL",3,1536,4,8,"token,slot,feature",moe->moe_weighted,8U*4U*1536U);
    F("ffn_moe_out-2","ordered routed-slot sum","ADD",2,1536,8,0,"token,feature",moe->moe_out,8U*1536U);
    F("ffn_up-2","shared expert up projection","MUL_MAT",2,1280,8,0,"token,feature",moe->up,8U*1280U);
    F("ffn_gate-2","shared expert gate projection","MUL_MAT",2,1280,8,0,"token,feature",moe->gate,8U*1280U);
    F("ffn_swiglu-2","shared expert SwiGLU","GLU",2,1280,8,0,"token,feature",moe->swiglu,8U*1280U);
    F("ffn_shexp-2","shared expert down projection","MUL_MAT",2,1536,8,0,"token,feature",moe->shexp,8U*1536U);
    F("ffn_out-2","routed plus shared output","ADD",2,1536,8,0,"token,feature",moe->out,8U*1536U);
    F("l_out-2","MoE output plus attention residual","ADD",2,1536,8,0,"token,feature",moe->l_out,8U*1536U);
#undef F
    return n;
}

static int strat01_r2d_write_manifest(const char *out_dir,const char *arm,strat01_r2c_dump d[STRAT01_R2C_DUMPS],const char *cache_paths[6],const char *cache_shas[6],int cached,char error[256]) {
    char leaf[128],path[1024];FILE *o=NULL;if(snprintf(leaf,sizeof(leaf),"%s_manifest.json",arm)<=0||!strat01_r2a_path(path,out_dir,leaf)||(o=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot write Rung-2D manifest");return 0;}
    fputs("{\n  \"arm\": ",o);strat01_json_string(o,arm);fputs(",\n  \"payload_encoding\": {\"F32\":\"IEEE-754 binary32 little-endian\",\"I32\":\"signed int32 little-endian\"},\n  \"shape_order\": \"ggml logical dimensions, fastest first; payload order declared per tensor\",\n  \"tensors\": [",o);
    for(unsigned i=0;i<STRAT01_R2C_DUMPS;++i){fprintf(o,i?",\n    {":"\n    {");fputs("\"name\":",o);strat01_json_string(o,d[i].logical_name);fprintf(o,",\"ordinal\":%u,\"selection\":",d[i].ordinal);strat01_json_string(o,d[i].selection);fputs(",\"op\":",o);strat01_json_string(o,d[i].op);fputs(",\"type\":",o);strat01_json_string(o,d[i].type);fputs(",\"logical_shape\":[",o);for(unsigned k=0;k<d[i].rank;++k)fprintf(o,"%s%" PRIu64,k?",":"",d[i].shape[k]);fputs("],\"payload_order\":",o);strat01_json_string(o,d[i].payload_order);fprintf(o,",\"byte_count\":%" PRIu64 ",\"path\":",d[i].count*4U);strat01_json_string(o,d[i].path);fputs(",\"sha256\":",o);strat01_json_string(o,d[i].sha256);fputc('}',o);}
    fputs("\n  ],\n  \"caches\": [",o);for(unsigned layer=0;layer<3U;++layer){if(layer)fputc(',',o);fprintf(o,"{\"layer\":%u,\"storage\":\"F16\",\"row_length\":576,\"no_separate_v_cache\":true,\"final\":{\"occupied_slots\":[0,1,2,3,4,5,6,7],\"absolute_positions\":[0,1,2,3,4,5,6,7],\"payload_type\":\"dequantized-F32LE\",\"path\":",layer);strat01_json_string(o,cache_paths[layer*2U]);fputs(",\"sha256\":",o);strat01_json_string(o,cache_shas[layer*2U]);fputc('}',o);if(cached){fputs(",\"prefix7\":{\"occupied_slots\":[0,1,2,3,4,5,6],\"absolute_positions\":[0,1,2,3,4,5,6],\"payload_type\":\"dequantized-F32LE\",\"path\":",o);strat01_json_string(o,cache_paths[layer*2U+1U]);fputs(",\"sha256\":",o);strat01_json_string(o,cache_shas[layer*2U+1U]);fputc('}',o);}fputc('}',o);}fputs("]\n}\n",o);
    if(fclose(o)!=0){snprintf(error,256,"Rung-2D manifest close failure");return 0;}return 1;
}

static int strat01_r2d_dump_arm(const char *out_dir,const char *arm,const strat01_r2a_arm *layers[3],const strat01_r2c_moe_arm *layer1,const strat01_r2c_moe_arm *layer2,int cached,char error[256]) {
    strat01_r2c_dump d[STRAT01_R2C_DUMPS];const char *paths[6]={0};const char *shas[6]={0};char cache_path[6][1024],cache_sha[6][65],leaf[6][96];
    if(strat01_r2d_make_dumps(layer1->l_out,layers[2],layer2,d)!=STRAT01_R2C_DUMPS){snprintf(error,256,"Rung-2D dump count mismatch");return 0;}for(unsigned i=0;i<STRAT01_R2C_DUMPS;++i)if(!strat01_r2c_dump_payload(&d[i],out_dir,arm,error))return 0;
    for(unsigned layer=0;layer<3U;++layer){snprintf(leaf[layer*2U],sizeof(leaf[0]),"%s_layer%u_cache_final.f32",arm,layer);if(!strat01_r2a_write_cache_dump(out_dir,leaf[layer*2U],layers[layer]->cache,8U,cache_path[layer*2U],cache_sha[layer*2U],error))return 0;paths[layer*2U]=cache_path[layer*2U];shas[layer*2U]=cache_sha[layer*2U];if(cached){snprintf(leaf[layer*2U+1U],sizeof(leaf[0]),"%s_layer%u_cache_prefix7.f32",arm,layer);if(!strat01_r2a_write_cache_dump(out_dir,leaf[layer*2U+1U],layers[layer]->cache,7U,cache_path[layer*2U+1U],cache_sha[layer*2U+1U],error))return 0;paths[layer*2U+1U]=cache_path[layer*2U+1U];shas[layer*2U+1U]=cache_sha[layer*2U+1U];}}
    return strat01_r2d_write_manifest(out_dir,arm,d,paths,shas,cached,error);
}

static int strat01_r2d_write_report(const char *out_dir,const char *path,uint64_t bytes,const char *artifact_sha,const char *engine_sha,const char *header_sha,char error[256]) {
    char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2d.json")||(o=fopen(p,"wb"))==NULL){snprintf(error,256,"cannot write Rung-2D report");return 0;}
    fputs("{\n\"command\":\"--strat01-gguf-rung2d\",\n\"c_state\":\"ENGINE_RUNG2D_OUTPUT_READY_PENDING_REFERENCE\",\n\"self_certifies_pass\":false,\n\"input_path\":",o);strat01_json_string(o,path);fprintf(o,",\n\"byte_size\":%" PRIu64 ",\n\"sha256\":",bytes);strat01_json_string(o,artifact_sha);fputs(",\n\"reference_revision\":",o);strat01_json_string(o,STRAT01_R2A_REFERENCE);fputs(",\n\"CONFIG\":",o);strat01_json_string(o,strat01_r2d_config);fputs(",\n\"compiler_family\":\"clang\",\n\"compiler_embedded_version\":",o);strat01_json_string(o,STRAT01_R2A_COMPILER);fputs(",\n\"compiler_resolved_path_and_full_version\":\"EXTERNAL_RUNNER_REQUIRED\",\n\"engine_source_sha256\":",o);strat01_json_string(o,engine_sha);fputs(",\n\"rung2d_source_sha256\":",o);strat01_json_string(o,header_sha);fputs(",\n\"token_ids\":[1,72,14,14129,14,2135,1512,2015],\n\"positions\":[0,1,2,3,4,5,6,7],\n\"arms\":[{\"name\":\"prefill8\",\"manifest\":\"prefill8_manifest.json\"},{\"name\":\"cached7p1\",\"manifest\":\"cached7p1_manifest.json\"}],\n\"timing_or_rate_claim\":null\n}\n",o);if(fclose(o)!=0){snprintf(error,256,"Rung-2D report close failure");return 0;}return 1;
}

static int strat01_r2d_write_failure(const char *out_dir,const char *path,const char *error){char p[1024];FILE *o;if(!strat01_r2a_path(p,out_dir,"strat01_rung2d.json")||(o=fopen(p,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-gguf-rung2d\",\"input_path\":",o);strat01_json_string(o,path);fputs(",\"c_state\":\"ENGINE_RUNG2D_C_FAILURE\",\"error\":",o);strat01_json_string(o,error);fputs("}\n",o);fclose(o);return 1;}

static int strat01_gguf_rung2d_cli(const char *path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *base[8]={0},*b0ffn[4]={0},*a1[7]={0},*m1[9]={0},*a2[7]={0},*m2[9]={0};
    strat01_r2a_arm pa[3],ca[3];strat01_r2b_arm pb0,cb0;strat01_r2c_moe_arm pm[2],cm[2];char error[256]={0},artifact_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};uint64_t hashed=0,parsed=0,tmp=0;int ok=0;
    memset(&inv,0,sizeof(inv));memset(pa,0,sizeof(pa));memset(ca,0,sizeof(ca));memset(&pb0,0,sizeof(pb0));memset(&cb0,0,sizeof(cb0));memset(pm,0,sizeof(pm));memset(cm,0,sizeof(cm));strat01_f16vec_reset_counts();
#if !defined(__clang__)
    snprintf(error,256,"Rung-2D requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(path,artifact_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(artifact_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"frozen Rung-2D artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(path,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"Rung-2D GGUF parse/size mismatch");goto finish;}
    for(unsigned i=0;i<8U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2a_specs[i],&base[i],error))goto finish;for(unsigned i=0;i<4U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2b_specs[i],&b0ffn[i],error))goto finish;
    for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&a1[i],error)||!strat01_r2a_check_spec(&inv,&strat01_r2d_attn_specs[i],&a2[i],error))goto finish;
    for(unsigned i=0;i<9U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[i],&m1[i],error)||!strat01_r2a_check_spec(&inv,&strat01_r2d_moe_specs[i],&m2[i],error))goto finish;
    for(unsigned l=0;l<3U;++l)if(!strat01_r2a_arm_alloc(&pa[l],error)||!strat01_r2a_arm_alloc(&ca[l],error))goto finish;if(!strat01_r2b_arm_alloc(&pb0,error)||!strat01_r2b_arm_alloc(&cb0,error))goto finish;for(unsigned l=0;l<2U;++l)if(!strat01_r2c_moe_alloc(&pm[l],error)||!strat01_r2c_moe_alloc(&cm[l],error))goto finish;
    if(!strat01_r2a_build_upstream_range(path,base,&pa[0],0,8,error)||!strat01_r2a_run_schedule(path,base[6],base[7],&pa[0],0,error)||!strat01_r2b_run(path,b0ffn[0],b0ffn[1],b0ffn[2],b0ffn[3],&pa[0],&pb0,error))goto finish;
    if(!strat01_r2a_build_upstream_range(path,base,&ca[0],0,7,error)||!strat01_r2a_build_upstream_range(path,base,&ca[0],7,1,error)||!strat01_r2a_run_schedule(path,base[6],base[7],&ca[0],1,error)||!strat01_r2b_run(path,b0ffn[0],b0ffn[1],b0ffn[2],b0ffn[3],&ca[0],&cb0,error))goto finish;
    memcpy(pa[1].embd,pb0.l_out,8U*1536U*4U);memcpy(ca[1].embd,cb0.l_out,8U*1536U*4U);
    if(!strat01_r2c_build_attention_range(path,a1,&pa[1],0,8,error)||!strat01_r2a_run_schedule(path,a1[5],a1[6],&pa[1],0,error)||!strat01_r2c_run_moe(path,m1,&pa[1],&pm[0],error))goto finish;
    if(!strat01_r2c_build_attention_range(path,a1,&ca[1],0,7,error)||!strat01_r2c_build_attention_range(path,a1,&ca[1],7,1,error)||!strat01_r2a_run_schedule(path,a1[5],a1[6],&ca[1],1,error)||!strat01_r2c_run_moe(path,m1,&ca[1],&cm[0],error))goto finish;
    memcpy(pa[2].embd,pm[0].l_out,8U*1536U*4U);memcpy(ca[2].embd,cm[0].l_out,8U*1536U*4U);
    if(!strat01_r2c_build_attention_range(path,a2,&pa[2],0,8,error)||!strat01_r2a_run_schedule(path,a2[5],a2[6],&pa[2],0,error)||!strat01_r2c_run_moe(path,m2,&pa[2],&pm[1],error))goto finish;fprintf(stderr,"STRAT01_RUNG2D_GRAPH_COMPLETE arm=prefill8\n");
    if(!strat01_r2c_build_attention_range(path,a2,&ca[2],0,7,error)||!strat01_r2c_build_attention_range(path,a2,&ca[2],7,1,error)||!strat01_r2a_run_schedule(path,a2[5],a2[6],&ca[2],1,error)||!strat01_r2c_run_moe(path,m2,&ca[2],&cm[1],error))goto finish;fprintf(stderr,"STRAT01_RUNG2D_GRAPH_COMPLETE arm=cached7p1\n");
    {const strat01_r2a_arm *pls[3]={&pa[0],&pa[1],&pa[2]},*cls[3]={&ca[0],&ca[1],&ca[2]};if(!strat01_r2d_dump_arm(out_dir,"prefill8",pls,&pm[0],&pm[1],0,error)||!strat01_r2d_dump_arm(out_dir,"cached7p1",cls,&cm[0],&cm[1],1,error)||!strat01_f16v_write_counts(out_dir,error))goto finish;}
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))strcpy(engine_sha,"unavailable");error[0]=0;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))strcpy(header_sha,"unavailable");error[0]=0;if(!strat01_r2d_write_report(out_dir,path,hashed,artifact_sha,engine_sha,header_sha,error))goto finish;ok=1;
finish:for(unsigned l=0;l<3U;++l){strat01_r2a_arm_free(&pa[l]);strat01_r2a_arm_free(&ca[l]);}strat01_r2b_arm_free(&pb0);strat01_r2b_arm_free(&cb0);for(unsigned l=0;l<2U;++l){strat01_r2c_moe_free(&pm[l]);strat01_r2c_moe_free(&cm[l]);}strat01_free_inventory(&inv);if(!ok){if(!error[0])snprintf(error,256,"unspecified Rung-2D failure");strat01_r2d_write_failure(out_dir,path,error);fprintf(stderr,"STRAT-01 Rung-2D refused: %s\n",error);return 1;}fprintf(stderr,"CONFIG %s\n",strat01_r2d_config);fprintf(stderr,"STRAT-01 Rung-2D: ENGINE_RUNG2D_OUTPUT_READY_PENDING_REFERENCE\n");return 0;
}

static int strat01_gguf_rung2d_selftest(void){int bad=0,checks=0;
#define CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    CHECK(strat01_gguf_rung2c_selftest()==0);CHECK(sizeof(strat01_r2d_attn_specs)/sizeof(strat01_r2d_attn_specs[0])==7U);CHECK(sizeof(strat01_r2d_moe_specs)/sizeof(strat01_r2d_moe_specs[0])==9U);CHECK(!strcmp(strat01_r2d_attn_specs[0].name,"blk.2.attn_norm.weight"));CHECK(!strcmp(strat01_r2d_moe_specs[8].name,"blk.2.ffn_down_shexp.weight"));CHECK(strcmp(strat01_r2d_config,strat01_r2c_config)!=0);CHECK(3U*2304U==6912U);CHECK(3U*262144U==786432U);
#undef CHECK
    fprintf(stderr,"STRAT-01 Rung-2D model-free selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;}

#pragma STDC FP_CONTRACT DEFAULT
#endif /* STRAT01_GGUF_RUNG2D_H */
