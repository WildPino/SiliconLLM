#ifndef STRAT01_GGUF_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_H
#define STRAT01_GGUF_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_H

#define STRAT01_L1TC_REF_INP_SHA "99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8"
#define STRAT01_L1TC_C_INP_SHA "c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2"
#define STRAT01_L1TC_REF_OUT_SHA "e8cdbbf3154447c90fa2dbddff5a454bd37200091fa2c5aaf7021104e76c104e"
#define STRAT01_L1TC_C_OUT_SHA "2932f1d3b23fc439ebdbee791a031a93e0681724f62935967e324b19a09196f5"
#define STRAT01_L1TC_REF_SUM_SHA STRAT01_L2AN_REF_INPUT_SHA
#define STRAT01_L1TC_C_SUM_SHA STRAT01_L2AN_C_INPUT_SHA

typedef struct {
    const char *name,*kind,*inp_origin,*out_origin;
    int output_control;
} strat01_l1tc_arm_spec;

static const strat01_l1tc_arm_spec strat01_l1tc_arms[7]={
    {"captured_ref_l_out","captured","reference","reference",0},
    {"captured_c_l_out","captured","c","c",0},
    {"computed_ref_inp_ref_out","computed","reference","reference",0},
    {"computed_c_inp_c_out","computed","c","c",0},
    {"cross_ref_inp_c_out","computed","reference","c",0},
    {"cross_c_inp_ref_out","computed","c","reference",0},
    {"control_ref_out_token7_negated","computed","reference","reference",1}
};

static void strat01_l1tc_add(const float *ffn_inp,const float *ffn_out,float *l_out){
    for(size_t i=0;i<8U*1536U;++i)l_out[i]=ffn_out[i]+ffn_inp[i];
}

static void strat01_l1tc_control(float *ffn_out,int enabled){
    if(enabled)for(unsigned i=0;i<1536U;++i)ffn_out[7U*1536U+i]=-ffn_out[7U*1536U+i];
}

static int strat01_l1tc_write_failure(const char *dir,const char *error){
    char path[1024];FILE *f;
    if(!strat01_r2a_path(path,dir,"strat01_layer1_terminal_component_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-layer1-terminal-component-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);
    strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_l1tc_cli(
    const char *model,const char *ref_q_path,const char *ref_k_path,
    const char *ref_inp_path,const char *ref_out_path,const char *ref_sum_path,
    const char *c_inp_path,const char *c_out_path,const char *c_sum_path,
    const char *out_dir,const char *engine_source_path){
    strat01_inventory inv;const strat01_tensor *at=NULL,*pt=NULL,*nt=NULL,*vb=NULL;
    float *rq=NULL,*rk=NULL,*ri=NULL,*ro=NULL,*rs=NULL,*ci=NULL,*co=NULL,*cs=NULL;
    float *work_out=NULL,*sum=NULL,*aw=NULL,*norm=NULL,*proj=NULL,*kvw=NULL,*prefix=NULL,*k=NULL,*out=NULL;
    FILE *report=NULL;int ok=0;uint64_t hashed=0,parsed=0,tmp=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};
    char input_sha[8][65]={{0}},sum_sha[7][65]={{0}},norm_sha[7][65]={{0}},proj_sha[7][65]={{0}},prefix_sha[7][65]={{0}},out_sha[7][65]={{0}};
    char sum_path[7][1024],norm_path[7][1024],proj_path[7][1024],prefix_path[7][1024],out_path[7][1024],report_path[1024];
    const char *input_paths[8]={ref_q_path,ref_k_path,ref_inp_path,ref_out_path,ref_sum_path,c_inp_path,c_out_path,c_sum_path};
    const char *input_names[8]={"ref_q","ref_k","ref_ffn_inp","ref_ffn_out","ref_l_out","c_ffn_inp","c_ffn_out","c_l_out"};
    const char *input_expected[8]={STRAT01_L2RN_REF_Q_SHA,STRAT01_L2RN_REF_K_SHA,STRAT01_L1TC_REF_INP_SHA,STRAT01_L1TC_REF_OUT_SHA,STRAT01_L1TC_REF_SUM_SHA,STRAT01_L1TC_C_INP_SHA,STRAT01_L1TC_C_OUT_SHA,STRAT01_L1TC_C_SUM_SHA};
    const uint64_t input_sizes[8]={589824U,18432U,49152U,49152U,49152U,49152U,49152U,49152U};
    float *input_values[8]={NULL};
    memset(&inv,0,sizeof(inv));
#if !defined(__clang__)
    snprintf(error,256,"layer-1 terminal-component diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"terminal-component artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed)goto finish;
    at=strat01_rung1_find_tensor(&inv,"blk.2.attn_norm.weight");pt=strat01_rung1_find_tensor(&inv,"blk.2.attn_kv_a_mqa.weight");nt=strat01_rung1_find_tensor(&inv,"blk.2.attn_kv_a_norm.weight");vb=strat01_rung1_find_tensor(&inv,"blk.2.attn_v_b.weight");
    if(!strat01_l2an_norm_descriptor_ok(at)||!strat01_l2kva_projection_descriptor_ok(pt)||!strat01_l2rn_norm_descriptor_ok(nt)||!strat01_l2rn_vb_descriptor_ok(vb)){snprintf(error,256,"terminal-component tensor descriptor mismatch");goto finish;}
#define L1TC_ALLOC(p,n) do{p=strat01_r2a_alloc((n),error);if(!p)goto finish;}while(0)
    L1TC_ALLOC(rq,8U*32U*576U);L1TC_ALLOC(rk,8U*576U);L1TC_ALLOC(ri,8U*1536U);L1TC_ALLOC(ro,8U*1536U);L1TC_ALLOC(rs,8U*1536U);
    L1TC_ALLOC(ci,8U*1536U);L1TC_ALLOC(co,8U*1536U);L1TC_ALLOC(cs,8U*1536U);L1TC_ALLOC(work_out,8U*1536U);L1TC_ALLOC(sum,8U*1536U);
    L1TC_ALLOC(aw,1536U);L1TC_ALLOC(norm,8U*1536U);L1TC_ALLOC(proj,8U*576U);L1TC_ALLOC(kvw,512U);L1TC_ALLOC(prefix,8U*512U);L1TC_ALLOC(k,8U*576U);L1TC_ALLOC(out,8U*6144U);
#undef L1TC_ALLOC
    input_values[0]=rq;input_values[1]=rk;input_values[2]=ri;input_values[3]=ro;input_values[4]=rs;input_values[5]=ci;input_values[6]=co;input_values[7]=cs;
    for(unsigned i=0;i<8U;++i)if(!strat01_r2c_cross_read_f32(input_paths[i],input_sizes[i],input_expected[i],input_values[i],input_sha[i],error))goto finish;
    if(!strat01_r2a_read_f32_vector(model,at,aw,1536U,error)||!strat01_r2a_read_f32_vector(model,nt,kvw,512U,error))goto finish;
    for(unsigned arm=0;arm<7U;++arm){
        const strat01_l1tc_arm_spec *s=&strat01_l1tc_arms[arm];
        if(!strcmp(s->kind,"captured"))memcpy(sum,!strcmp(s->inp_origin,"reference")?rs:cs,49152U);
        else{const float *inp=!strcmp(s->inp_origin,"reference")?ri:ci;const float *ffn_out=!strcmp(s->out_origin,"reference")?ro:co;memcpy(work_out,ffn_out,49152U);strat01_l1tc_control(work_out,s->output_control);strat01_l1tc_add(inp,work_out,sum);}
        strat01_r2a_rmsnorm_pinned(sum,aw,norm,8U,1536U,STRAT01_R2A_RMS_EPS);
        if(!strat01_r2a_matmul_batch(model,pt,norm,8U,1536U,proj,576U,error))goto finish;
        strat01_l2rn_compute_prefix(proj,kvw,prefix,0,0);strat01_l2rn_compose_k(k,rk,prefix);
        if(!strat01_r2c_cross_run_arm(model,vb,rq,k,0,0,out,error))goto finish;
        {char leaf[160];
            snprintf(leaf,sizeof(leaf),"%s.l_out.f32le",s->name);if(!strat01_r2a_path(sum_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(sum_path[arm],sum,8U*1536U,sum_sha[arm],error))goto finish;
            snprintf(leaf,sizeof(leaf),"%s.norm.f32le",s->name);if(!strat01_r2a_path(norm_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(norm_path[arm],norm,8U*1536U,norm_sha[arm],error))goto finish;
            snprintf(leaf,sizeof(leaf),"%s.projection.f32le",s->name);if(!strat01_r2a_path(proj_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(proj_path[arm],proj,8U*576U,proj_sha[arm],error))goto finish;
            snprintf(leaf,sizeof(leaf),"%s.prefix.f32le",s->name);if(!strat01_r2a_path(prefix_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(prefix_path[arm],prefix,8U*512U,prefix_sha[arm],error))goto finish;
            snprintf(leaf,sizeof(leaf),"%s.f32le",s->name);if(!strat01_r2a_path(out_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(out_path[arm],out,8U*6144U,out_sha[arm],error))goto finish;
        }
    }
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_layer1_terminal_component_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot create terminal-component report");goto finish;}
    fputs("{\n  \"command\":\"--strat01-layer1-terminal-component-cross-input\",\n  \"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\":false,\n  \"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n  \"inputs\":{",report);
    for(unsigned i=0;i<8U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",input_names[i]);strat01_json_string(report,input_paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",input_sizes[i]);strat01_json_string(report,input_sha[i]);fputc('}',report);}
    fprintf(report,"},\n  \"matrices\":{\"attn_norm\":{\"name\":\"blk.2.attn_norm.weight\",\"type\":\"F32\",\"shape\":[1536],\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\"projection\":{\"name\":\"blk.2.attn_kv_a_mqa.weight\",\"type\":\"Q4_K\",\"shape\":[1536,576]},\"kv_norm\":{\"name\":\"blk.2.attn_kv_a_norm.weight\",\"type\":\"F32\",\"shape\":[512]},\"v_b\":{\"name\":\"blk.2.attn_v_b.weight\",\"type\":\"Q4_K\",\"shape\":[512,192,32]}},\n  \"outputs\":{",at->offset,at->file_offset,at->span);
    for(unsigned arm=0;arm<7U;++arm){const strat01_l1tc_arm_spec *s=&strat01_l1tc_arms[arm];fprintf(report,"%s\"%s\":{\"kind\":",arm?",":"",s->name);strat01_json_string(report,s->kind);fputs(",\"ffn_inp_origin\":",report);strat01_json_string(report,s->inp_origin);fputs(",\"ffn_out_origin\":",report);strat01_json_string(report,s->out_origin);fprintf(report,",\"output_control\":%s,\"l_out\":{\"path\":",s->output_control?"true":"false");strat01_json_string(report,sum_path[arm]);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,sum_sha[arm]);fputs("},\"norm\":{\"path\":",report);strat01_json_string(report,norm_path[arm]);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,norm_sha[arm]);fputs("},\"projection\":{\"path\":",report);strat01_json_string(report,proj_path[arm]);fputs(",\"bytes\":18432,\"sha256\":",report);strat01_json_string(report,proj_sha[arm]);fputs("},\"prefix\":{\"path\":",report);strat01_json_string(report,prefix_path[arm]);fputs(",\"bytes\":16384,\"sha256\":",report);strat01_json_string(report,prefix_sha[arm]);fputs("},\"downstream\":{\"path\":",report);strat01_json_string(report,out_path[arm]);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,out_sha[arm]);fputs("}}",report);}
    fputs("},\n  \"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\"compiler_family\":\"clang\",\"donor_graph_executions\":0,\"reference_graph_executions\":0,\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"terminal-component report close failed");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);strat01_free_inventory(&inv);free(rq);free(rk);free(ri);free(ro);free(rs);free(ci);free(co);free(cs);free(work_out);free(sum);free(aw);free(norm);free(proj);free(kvw);free(prefix);free(k);free(out);
    if(!ok){strat01_l1tc_write_failure(out_dir,error[0]?error:"terminal-component diagnostic failed");fprintf(stderr,"STRAT-01 layer-1 terminal component: %s\n",error[0]?error:"failed");return 2;}
    fprintf(stderr,"STRAT-01 layer-1 terminal component: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_l1tc_selftest(void){
    int bad=0,checks=0;float inp[8U*1536U]={0},out[8U*1536U]={0},sum[8U*1536U]={0};
#define L1TC_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    inp[0]=1.0f;out[0]=2.0f;inp[1]=-2.0f;out[1]=0.5f;out[7U*1536U]=3.0f;
    strat01_l1tc_add(inp,out,sum);L1TC_CHECK(sum[0]==3.0f&&sum[1]==-1.5f);
    strat01_l1tc_control(out,1);L1TC_CHECK(out[7U*1536U]==-3.0f);
    L1TC_CHECK(strlen(STRAT01_L1TC_REF_INP_SHA)==64U&&strlen(STRAT01_L1TC_C_INP_SHA)==64U);
    L1TC_CHECK(strlen(STRAT01_L1TC_REF_OUT_SHA)==64U&&strlen(STRAT01_L1TC_C_OUT_SHA)==64U);
    L1TC_CHECK(!strcmp(strat01_l1tc_arms[4].inp_origin,"reference")&&!strcmp(strat01_l1tc_arms[4].out_origin,"c"));
    L1TC_CHECK(strat01_l1tc_arms[6].output_control==1);
#undef L1TC_CHECK
    fprintf(stderr,"STRAT-01 layer-1 terminal-component selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif
