/* Frozen-input layer-2 KV RMSNorm attribution diagnostic. */
#ifndef STRAT01_GGUF_LAYER2_KV_RMSNORM_CROSS_INPUT_H
#define STRAT01_GGUF_LAYER2_KV_RMSNORM_CROSS_INPUT_H

#define STRAT01_L2RN_REF_Q_SHA "46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea"
#define STRAT01_L2RN_REF_K_SHA "b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b"
#define STRAT01_L2RN_REF_PRE_SHA "81c439e40a9dfd06221a86b7cdc9b6019ea8742e6cf11371d1d2c50546346c61"
#define STRAT01_L2RN_REF_NORM_SHA "0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91"
#define STRAT01_L2RN_C_PRE_SHA "aae2f54a6391b84c156c3b61f3206be1170afb7e09995cc891f1fd9e7ebfc444"
#define STRAT01_L2RN_C_NORM_SHA "e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9"

typedef struct {const char *name,*kind,*source;int input_control,weight_control;} strat01_l2rn_arm_spec;
static const strat01_l2rn_arm_spec strat01_l2rn_arms[6]={
    {"captured_ref_norm","captured","reference",0,0},
    {"captured_c_norm","captured","c",0,0},
    {"computed_ref_input","computed","reference",0,0},
    {"computed_c_input","computed","c",0,0},
    {"control_ref_input_token7_negated","computed","reference",1,0},
    {"control_norm_weight_negated","computed","reference",0,1}
};

static int strat01_l2rn_norm_descriptor_ok(const strat01_tensor *t){return t&&!strcmp(t->name,"blk.2.attn_kv_a_norm.weight")&&t->type==STRAT01_GGML_F32&&t->rank==1&&t->dims[0]==512U&&t->span==2048U;}
static int strat01_l2rn_vb_descriptor_ok(const strat01_tensor *t){return strat01_l2kpx_descriptor_ok(t);}

static void strat01_l2rn_compute_prefix(const float *pre,const float *weight,float *prefix,int input_control,int weight_control){
    float input[8U*512U],w[512U];
    for(unsigned tok=0;tok<8U;++tok)memcpy(input+(size_t)tok*512U,pre+(size_t)tok*576U,512U*sizeof(float));
    memcpy(w,weight,sizeof(w));
    if(input_control)for(unsigned i=0;i<512U;++i)input[7U*512U+i]=-input[7U*512U+i];
    if(weight_control)for(unsigned i=0;i<512U;++i)w[i]=-w[i];
    strat01_r2a_rmsnorm_pinned(input,w,prefix,8U,512U,STRAT01_R2A_RMS_EPS);
}

static void strat01_l2rn_compose_k(float *k,const float *ref_k,const float *prefix){
    memcpy(k,ref_k,STRAT01_R2C_CROSS_K_BYTES);
    for(unsigned tok=0;tok<8U;++tok)memcpy(k+(size_t)tok*576U,prefix+(size_t)tok*512U,512U*sizeof(float));
}

static int strat01_l2rn_write_failure(const char *out_dir,const char *error){char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_layer2_kv_rmsnorm_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-layer2-kv-rmsnorm-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;}

static int strat01_l2rn_cli(const char *model,const char *ref_q_path,const char *ref_k_path,const char *ref_pre_path,const char *ref_norm_path,const char *c_pre_path,const char *c_norm_path,const char *out_dir,const char *engine_source_path){
    strat01_inventory inv;const strat01_tensor *nt=NULL,*vb=NULL;float *rq=NULL,*rk=NULL,*rpre=NULL,*rnorm=NULL,*cpre=NULL,*cnorm=NULL,*weight=NULL,*prefix=NULL,*k=NULL,*out=NULL;FILE *report=NULL;int ok=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};char input_sha[6][65]={{0}},prefix_sha[6][65]={{0}},output_sha[6][65]={{0}},prefix_path[6][1024],output_path[6][1024],report_path[1024];uint64_t hashed=0,parsed=0,tmp=0;memset(&inv,0,sizeof(inv));
#if !defined(__clang__)
    snprintf(error,256,"layer-2 KV RMSNorm diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"layer-2 KV RMSNorm artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"layer-2 KV RMSNorm GGUF parse/size mismatch");goto finish;}
    nt=strat01_rung1_find_tensor(&inv,"blk.2.attn_kv_a_norm.weight");vb=strat01_rung1_find_tensor(&inv,"blk.2.attn_v_b.weight");if(!strat01_l2rn_norm_descriptor_ok(nt)||!strat01_l2rn_vb_descriptor_ok(vb)){snprintf(error,256,"layer-2 KV RMSNorm tensor descriptor mismatch");goto finish;}
    rq=strat01_r2a_alloc(STRAT01_R2C_CROSS_Q_COUNT,error);rk=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);rpre=strat01_r2a_alloc(8U*576U,error);rnorm=strat01_r2a_alloc(8U*512U,error);cpre=strat01_r2a_alloc(8U*576U,error);cnorm=strat01_r2a_alloc(8U*512U,error);weight=strat01_r2a_alloc(512U,error);prefix=strat01_r2a_alloc(8U*512U,error);k=strat01_r2a_alloc(8U*576U,error);out=strat01_r2a_alloc(STRAT01_R2C_CROSS_OUT_COUNT,error);if(!rq||!rk||!rpre||!rnorm||!cpre||!cnorm||!weight||!prefix||!k||!out)goto finish;
    if(!strat01_r2c_cross_read_f32(ref_q_path,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_L2RN_REF_Q_SHA,rq,input_sha[0],error)||!strat01_r2c_cross_read_f32(ref_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2RN_REF_K_SHA,rk,input_sha[1],error)||!strat01_r2c_cross_read_f32(ref_pre_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2RN_REF_PRE_SHA,rpre,input_sha[2],error)||!strat01_r2c_cross_read_f32(ref_norm_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_L2RN_REF_NORM_SHA,rnorm,input_sha[3],error)||!strat01_r2c_cross_read_f32(c_pre_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2RN_C_PRE_SHA,cpre,input_sha[4],error)||!strat01_r2c_cross_read_f32(c_norm_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_L2RN_C_NORM_SHA,cnorm,input_sha[5],error)||!strat01_r2a_read_f32_vector(model,nt,weight,512U,error))goto finish;
    if(!strat01_r2c_cross_v_matches_k(rk,rnorm)){snprintf(error,256,"reference normalized prefix/K mismatch");goto finish;}
    for(unsigned arm=0;arm<6U;++arm){const strat01_l2rn_arm_spec *s=&strat01_l2rn_arms[arm];if(!strcmp(s->kind,"captured"))memcpy(prefix,!strcmp(s->source,"reference")?rnorm:cnorm,STRAT01_R2C_CROSS_V_BYTES);else strat01_l2rn_compute_prefix(!strcmp(s->source,"reference")?rpre:cpre,weight,prefix,s->input_control,s->weight_control);strat01_l2rn_compose_k(k,rk,prefix);
        {char leaf[160];snprintf(leaf,sizeof(leaf),"%s.prefix.f32le",s->name);if(!strat01_r2a_path(prefix_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(prefix_path[arm],prefix,8U*512U,prefix_sha[arm],error))goto finish;}
        if(!strat01_r2c_cross_run_arm(model,vb,rq,k,0,0,out,error))goto finish;
        {char leaf[160];snprintf(leaf,sizeof(leaf),"%s.f32le",s->name);if(!strat01_r2a_path(output_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(output_path[arm],out,STRAT01_R2C_CROSS_OUT_COUNT,output_sha[arm],error))goto finish;}
    }
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_layer2_kv_rmsnorm_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot create layer-2 KV RMSNorm report");goto finish;}
    fputs("{\n  \"command\":\"--strat01-layer2-kv-rmsnorm-cross-input\",\n  \"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\":false,\n  \"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n  \"inputs\":{",report);
    {const char *keys[6]={"ref_q","ref_k","ref_pre","ref_norm","c_pre","c_norm"};const char *paths[6]={ref_q_path,ref_k_path,ref_pre_path,ref_norm_path,c_pre_path,c_norm_path};const uint64_t sizes[6]={589824,18432,18432,16384,18432,16384};for(unsigned i=0;i<6U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",keys[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",sizes[i]);strat01_json_string(report,input_sha[i]);fputc('}',report);}}
    fprintf(report,"},\n  \"split\":512,\n  \"matrices\":{\"norm\":{\"name\":\"blk.2.attn_kv_a_norm.weight\",\"type\":\"F32\",\"shape\":[512],\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\"v_b\":{\"name\":\"blk.2.attn_v_b.weight\",\"type\":\"Q4_K\",\"shape\":[512,192,32],\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}},\n  \"outputs\":{",nt->offset,nt->file_offset,nt->span,vb->offset,vb->file_offset,vb->span);
    for(unsigned arm=0;arm<6U;++arm){const strat01_l2rn_arm_spec *s=&strat01_l2rn_arms[arm];fprintf(report,"%s\"%s\":{\"kind\":",arm?",":"",s->name);strat01_json_string(report,s->kind);fputs(",\"source\":",report);strat01_json_string(report,s->source);fprintf(report,",\"input_control\":%s,\"weight_control\":%s,\"prefix\":{\"path\":",s->input_control?"true":"false",s->weight_control?"true":"false");strat01_json_string(report,prefix_path[arm]);fputs(",\"bytes\":16384,\"sha256\":",report);strat01_json_string(report,prefix_sha[arm]);fputs("},\"downstream\":{\"path\":",report);strat01_json_string(report,output_path[arm]);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,output_sha[arm]);fputs("}}",report);}
    fputs("},\n  \"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n  \"compiler_family\":\"clang\",\n  \"donor_graph_executions\":0,\n  \"timing_or_rate_claim\":null\n}\n",report);if(fclose(report)!=0){report=NULL;snprintf(error,256,"layer-2 KV RMSNorm report close failure");goto finish;}report=NULL;ok=1;
finish: if(report)fclose(report);free(rq);free(rk);free(rpre);free(rnorm);free(cpre);free(cnorm);free(weight);free(prefix);free(k);free(out);strat01_free_inventory(&inv);if(!ok){if(!error[0])snprintf(error,256,"unspecified layer-2 KV RMSNorm failure");strat01_l2rn_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 layer-2 KV RMSNorm refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 layer-2 KV RMSNorm: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_l2rn_selftest(void){int bad=0,checks=0;strat01_tensor n,v;float pre[8U*576U]={0},w[512U],p[8U*512U],k[8U*576U]={0};
#define L2RN_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    memset(&n,0,sizeof(n));n.name=(char *)"blk.2.attn_kv_a_norm.weight";n.type=STRAT01_GGML_F32;n.rank=1;n.dims[0]=512;n.span=2048;L2RN_CHECK(strat01_l2rn_norm_descriptor_ok(&n));n.name=(char *)"blk.1.attn_kv_a_norm.weight";L2RN_CHECK(!strat01_l2rn_norm_descriptor_ok(&n));
    memset(&v,0,sizeof(v));v.name=(char *)"blk.2.attn_v_b.weight";v.type=STRAT01_GGML_Q4_K;v.rank=3;v.dims[0]=512;v.dims[1]=192;v.dims[2]=32;v.span=1769472;L2RN_CHECK(strat01_l2rn_vb_descriptor_ok(&v));
    for(unsigned i=0;i<512U;++i){pre[i]=(float)(i+1);w[i]=1.0f;}strat01_l2rn_compute_prefix(pre,w,p,0,0);L2RN_CHECK(isfinite(p[0])&&p[0]>0.0f);strat01_l2rn_compute_prefix(pre,w,p,1,0);L2RN_CHECK(p[7U*512U]==0.0f);strat01_l2rn_compute_prefix(pre,w,p,0,1);L2RN_CHECK(p[0]<0.0f);for(unsigned i=512;i<576U;++i)k[i]=3.0f;strat01_l2rn_compose_k(k,k,p);L2RN_CHECK(k[0]==p[0]&&k[512]==3.0f);
#undef L2RN_CHECK
    fprintf(stderr,"STRAT-01 layer-2 KV RMSNorm selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;}

#endif
