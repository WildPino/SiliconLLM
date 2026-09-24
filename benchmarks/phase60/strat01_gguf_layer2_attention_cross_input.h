/*
 * Frozen-input layer-2 attention/V-B diagnostic for STRAT-01.
 * Reuses the accepted Rung-2C boundary operator and executes no donor graph.
 */
#ifndef STRAT01_GGUF_LAYER2_ATTENTION_CROSS_INPUT_H
#define STRAT01_GGUF_LAYER2_ATTENTION_CROSS_INPUT_H

#define STRAT01_L2AX_REF_Q_SHA "46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea"
#define STRAT01_L2AX_REF_K_SHA "b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b"
#define STRAT01_L2AX_REF_V_SHA "0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91"
#define STRAT01_L2AX_C_Q_SHA   "a1bcd7529bae496e48db28964123f8f28bfc26fd2559c54ada51fb9ae49464d5"
#define STRAT01_L2AX_C_K_SHA   "163c0eb7f03892c0ff86cfd1ad455382aa9f59a245c2f849918a92d5537a41f9"
#define STRAT01_L2AX_C_V_SHA   "e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9"

static int strat01_l2ax_descriptor_ok(const strat01_tensor *t) {
    return t&&!strcmp(t->name,"blk.2.attn_v_b.weight")&&t->type==STRAT01_GGML_Q4_K&&t->rank==3&&
           t->dims[0]==512U&&t->dims[1]==192U&&t->dims[2]==32U&&t->span==UINT64_C(1769472);
}

static int strat01_l2ax_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_layer2_attention_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-layer2-attention-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_l2ax_cli(const char *model,const char *ref_q_path,const char *ref_k_path,const char *ref_v_path,const char *c_q_path,const char *c_k_path,const char *c_v_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *vb=NULL;float *rq=NULL,*rk=NULL,*rv=NULL,*cq=NULL,*ck=NULL,*cv=NULL,*out=NULL;FILE *report=NULL;int ok=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};char input_sha[6][65]={{0}},output_sha[6][65]={{0}},output_path[6][1024],report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;memset(&inv,0,sizeof(inv));
#if !defined(__clang__)
    snprintf(error,256,"layer-2 attention cross-input diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4){snprintf(error,256,"unsupported layer-2 attention cross-input host layout");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"layer-2 attention artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"layer-2 attention GGUF parse/size mismatch");goto finish;}
    vb=strat01_rung1_find_tensor(&inv,"blk.2.attn_v_b.weight");if(!strat01_l2ax_descriptor_ok(vb)){snprintf(error,256,"layer-2 attention V-B descriptor mismatch");goto finish;}
    rq=strat01_r2a_alloc(STRAT01_R2C_CROSS_Q_COUNT,error);rk=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);rv=strat01_r2a_alloc(STRAT01_R2C_CROSS_V_COUNT,error);
    cq=strat01_r2a_alloc(STRAT01_R2C_CROSS_Q_COUNT,error);ck=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);cv=strat01_r2a_alloc(STRAT01_R2C_CROSS_V_COUNT,error);out=strat01_r2a_alloc(STRAT01_R2C_CROSS_OUT_COUNT,error);
    if(!rq||!rk||!rv||!cq||!ck||!cv||!out)goto finish;
    if(!strat01_r2c_cross_read_f32(ref_q_path,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_L2AX_REF_Q_SHA,rq,input_sha[0],error)||
       !strat01_r2c_cross_read_f32(ref_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2AX_REF_K_SHA,rk,input_sha[1],error)||
       !strat01_r2c_cross_read_f32(ref_v_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_L2AX_REF_V_SHA,rv,input_sha[2],error)||
       !strat01_r2c_cross_read_f32(c_q_path,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_L2AX_C_Q_SHA,cq,input_sha[3],error)||
       !strat01_r2c_cross_read_f32(c_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2AX_C_K_SHA,ck,input_sha[4],error)||
       !strat01_r2c_cross_read_f32(c_v_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_L2AX_C_V_SHA,cv,input_sha[5],error))goto finish;
    if(!strat01_r2c_cross_v_matches_k(rk,rv)||!strat01_r2c_cross_v_matches_k(ck,cv)){snprintf(error,256,"layer-2 attention V/K-prefix mismatch");goto finish;}
    for(unsigned arm=0;arm<6U;++arm){const float *q=NULL,*k=NULL;strat01_r2c_cross_select((strat01_r2c_cross_arm_id)arm,rq,rk,cq,ck,&q,&k);
        if(!strat01_r2c_cross_run_arm(model,vb,q,k,strat01_r2c_cross_arms[arm].query_control,strat01_r2c_cross_arms[arm].kv_control,out,error))goto finish;
        {char leaf[160];snprintf(leaf,sizeof(leaf),"%s.f32le",strat01_r2c_cross_arms[arm].name);if(!strat01_r2a_path(output_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(output_path[arm],out,STRAT01_R2C_CROSS_OUT_COUNT,output_sha[arm],error))goto finish;}
    }
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_layer2_attention_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot create layer-2 attention cross-input report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-layer2-attention-cross-input\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n",report);
    fputs("  \"inputs\": {",report);
    {const char *keys[6]={"ref_q","ref_k","ref_v","c_q","c_k","c_v"};const char *paths[6]={ref_q_path,ref_k_path,ref_v_path,c_q_path,c_k_path,c_v_path};const uint64_t sizes[6]={STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_V_BYTES,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_V_BYTES};for(unsigned i=0;i<6U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",keys[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",sizes[i]);strat01_json_string(report,input_sha[i]);fputc('}',report);}}
    fprintf(report,"},\n  \"v_k_prefix_equal\": {\"reference\":true,\"c\":true},\n  \"matrix\": {\"name\":\"blk.2.attn_v_b.weight\",\"type\":\"Q4_K\",\"shape\":[512,192,32],\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\n  \"outputs\": {",vb->offset,vb->file_offset,vb->span);
    for(unsigned arm=0;arm<6U;++arm){const strat01_r2c_cross_arm_spec *s=&strat01_r2c_cross_arms[arm];fprintf(report,"%s\"%s\":{\"path\":",arm?",":"",s->name);strat01_json_string(report,output_path[arm]);fprintf(report,",\"bytes\":196608,\"sha256\":");strat01_json_string(report,output_sha[arm]);fputs(",\"q_source\":",report);strat01_json_string(report,s->q_source);fputs(",\"kv_source\":",report);strat01_json_string(report,s->kv_source);fprintf(report,",\"query_control\":%s,\"kv_control\":%s}",s->query_control?"true":"false",s->kv_control?"true":"false");}
    fputs("},\n  \"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n  \"compiler_family\":\"clang\",\n  \"donor_graph_executions\":0,\n  \"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"layer-2 attention report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(rq);free(rk);free(rv);free(cq);free(ck);free(cv);free(out);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified layer-2 attention cross-input failure");strat01_l2ax_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 layer-2 attention cross-input refused: %s\n",error);return 1;}
    fprintf(stderr,"STRAT-01 layer-2 attention cross-input: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_l2ax_selftest(void) {
    int bad=0,checks=0;strat01_tensor t;float k[8U*576U]={0},v[8U*512U]={0};
#define L2AX_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    memset(&t,0,sizeof(t));t.name=(char *)"blk.2.attn_v_b.weight";t.type=STRAT01_GGML_Q4_K;t.rank=3;t.dims[0]=512;t.dims[1]=192;t.dims[2]=32;t.span=1769472;
    L2AX_CHECK(strat01_l2ax_descriptor_ok(&t));t.name=(char *)"blk.1.attn_v_b.weight";L2AX_CHECK(!strat01_l2ax_descriptor_ok(&t));t.name=(char *)"blk.2.attn_v_b.weight";t.dims[2]=31;L2AX_CHECK(!strat01_l2ax_descriptor_ok(&t));
    for(unsigned tok=0;tok<8U;++tok)for(unsigned i=0;i<512U;++i)k[(size_t)tok*576U+i]=v[(size_t)tok*512U+i]=(float)(tok+i);
    L2AX_CHECK(strat01_r2c_cross_v_matches_k(k,v));v[700]+=1.0f;L2AX_CHECK(!strat01_r2c_cross_v_matches_k(k,v));
#undef L2AX_CHECK
    fprintf(stderr,"STRAT-01 layer-2 attention cross-input selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif /* STRAT01_GGUF_LAYER2_ATTENTION_CROSS_INPUT_H */
