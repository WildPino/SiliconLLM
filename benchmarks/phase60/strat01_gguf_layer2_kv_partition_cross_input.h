/*
 * Frozen-input layer-2 compact-KV partition diagnostic for STRAT-01.
 * Reference Q is fixed; the latent/value prefix and positional tail are
 * independently composed before the accepted cache/attention/V-B operator.
 */
#ifndef STRAT01_GGUF_LAYER2_KV_PARTITION_CROSS_INPUT_H
#define STRAT01_GGUF_LAYER2_KV_PARTITION_CROSS_INPUT_H

#define STRAT01_L2KPX_REF_Q_SHA "46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea"
#define STRAT01_L2KPX_REF_K_SHA "b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b"
#define STRAT01_L2KPX_REF_V_SHA "0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91"
#define STRAT01_L2KPX_C_K_SHA   "163c0eb7f03892c0ff86cfd1ad455382aa9f59a245c2f849918a92d5537a41f9"
#define STRAT01_L2KPX_C_V_SHA   "e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9"
#define STRAT01_L2KPX_PREFIX 512U
#define STRAT01_L2KPX_TAIL 64U

typedef struct {
    const char *name;
    const char *latent_source;
    const char *positional_source;
    int latent_control;
    int positional_control;
} strat01_l2kpx_arm_spec;

static const strat01_l2kpx_arm_spec strat01_l2kpx_arms[6]={
    {"ref_latent__ref_positional","reference","reference",0,0},
    {"c_latent__c_positional","c","c",0,0},
    {"c_latent__ref_positional","c","reference",0,0},
    {"ref_latent__c_positional","reference","c",0,0},
    {"control_ref_latent_token7_negated","reference","reference",1,0},
    {"control_ref_positional_rows0_7_swapped","reference","reference",0,1}
};

static int strat01_l2kpx_descriptor_ok(const strat01_tensor *t) {
    return t&&!strcmp(t->name,"blk.2.attn_v_b.weight")&&t->type==STRAT01_GGML_Q4_K&&t->rank==3&&
           t->dims[0]==512U&&t->dims[1]==192U&&t->dims[2]==32U&&t->span==UINT64_C(1769472);
}

static void strat01_l2kpx_compose(float *dst,const float *rk,const float *ck,const strat01_l2kpx_arm_spec *s) {
    const float *latent=!strcmp(s->latent_source,"reference")?rk:ck;
    const float *positional=!strcmp(s->positional_source,"reference")?rk:ck;
    for(unsigned tok=0;tok<8U;++tok){
        memcpy(dst+(size_t)tok*576U,latent+(size_t)tok*576U,STRAT01_L2KPX_PREFIX*sizeof(float));
        memcpy(dst+(size_t)tok*576U+STRAT01_L2KPX_PREFIX,positional+(size_t)tok*576U+STRAT01_L2KPX_PREFIX,STRAT01_L2KPX_TAIL*sizeof(float));
    }
    if(s->latent_control)for(unsigned i=0;i<STRAT01_L2KPX_PREFIX;++i)dst[(size_t)7U*576U+i]=-dst[(size_t)7U*576U+i];
    if(s->positional_control)for(unsigned i=0;i<STRAT01_L2KPX_TAIL;++i){
        size_t a=STRAT01_L2KPX_PREFIX+i,b=(size_t)7U*576U+STRAT01_L2KPX_PREFIX+i;float tmp=dst[a];dst[a]=dst[b];dst[b]=tmp;
    }
}

static int strat01_l2kpx_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_layer2_kv_partition_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-layer2-kv-partition-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_l2kpx_cli(const char *model,const char *ref_q_path,const char *ref_k_path,const char *ref_v_path,const char *c_k_path,const char *c_v_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *vb=NULL;float *rq=NULL,*rk=NULL,*rv=NULL,*ck=NULL,*cv=NULL,*mixed=NULL,*out=NULL;FILE *report=NULL;int ok=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};char input_sha[5][65]={{0}},output_sha[6][65]={{0}},output_path[6][1024],report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;memset(&inv,0,sizeof(inv));
#if !defined(__clang__)
    snprintf(error,256,"layer-2 KV-partition diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4||STRAT01_L2KPX_PREFIX+STRAT01_L2KPX_TAIL!=576U){snprintf(error,256,"invalid layer-2 KV-partition host/layout contract");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"layer-2 KV-partition artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"layer-2 KV-partition GGUF parse/size mismatch");goto finish;}
    vb=strat01_rung1_find_tensor(&inv,"blk.2.attn_v_b.weight");if(!strat01_l2kpx_descriptor_ok(vb)){snprintf(error,256,"layer-2 KV-partition V-B descriptor mismatch");goto finish;}
    rq=strat01_r2a_alloc(STRAT01_R2C_CROSS_Q_COUNT,error);rk=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);rv=strat01_r2a_alloc(STRAT01_R2C_CROSS_V_COUNT,error);
    ck=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);cv=strat01_r2a_alloc(STRAT01_R2C_CROSS_V_COUNT,error);mixed=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);out=strat01_r2a_alloc(STRAT01_R2C_CROSS_OUT_COUNT,error);
    if(!rq||!rk||!rv||!ck||!cv||!mixed||!out)goto finish;
    if(!strat01_r2c_cross_read_f32(ref_q_path,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_L2KPX_REF_Q_SHA,rq,input_sha[0],error)||
       !strat01_r2c_cross_read_f32(ref_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2KPX_REF_K_SHA,rk,input_sha[1],error)||
       !strat01_r2c_cross_read_f32(ref_v_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_L2KPX_REF_V_SHA,rv,input_sha[2],error)||
       !strat01_r2c_cross_read_f32(c_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_L2KPX_C_K_SHA,ck,input_sha[3],error)||
       !strat01_r2c_cross_read_f32(c_v_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_L2KPX_C_V_SHA,cv,input_sha[4],error))goto finish;
    if(!strat01_r2c_cross_v_matches_k(rk,rv)||!strat01_r2c_cross_v_matches_k(ck,cv)){snprintf(error,256,"layer-2 KV-partition V/K-prefix mismatch");goto finish;}
    for(unsigned arm=0;arm<6U;++arm){
        strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[arm]);
        if(!strat01_r2c_cross_run_arm(model,vb,rq,mixed,0,0,out,error))goto finish;
        {char leaf[160];snprintf(leaf,sizeof(leaf),"%s.f32le",strat01_l2kpx_arms[arm].name);if(!strat01_r2a_path(output_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(output_path[arm],out,STRAT01_R2C_CROSS_OUT_COUNT,output_sha[arm],error))goto finish;}
    }
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_layer2_kv_partition_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot create layer-2 KV-partition report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-layer2-kv-partition-cross-input\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n",report);
    fputs("  \"inputs\": {",report);
    {const char *keys[5]={"ref_q","ref_k","ref_v","c_k","c_v"};const char *paths[5]={ref_q_path,ref_k_path,ref_v_path,c_k_path,c_v_path};const uint64_t sizes[5]={STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_V_BYTES,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_V_BYTES};for(unsigned i=0;i<5U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",keys[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",sizes[i]);strat01_json_string(report,input_sha[i]);fputc('}',report);}}
    fprintf(report,"},\n  \"v_k_prefix_equal\": {\"reference\":true,\"c\":true},\n  \"partition\": {\"latent_value\":[0,512],\"positional\":[512,576]},\n  \"matrix\": {\"name\":\"blk.2.attn_v_b.weight\",\"type\":\"Q4_K\",\"shape\":[512,192,32],\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\n  \"outputs\": {",vb->offset,vb->file_offset,vb->span);
    for(unsigned arm=0;arm<6U;++arm){const strat01_l2kpx_arm_spec *s=&strat01_l2kpx_arms[arm];fprintf(report,"%s\"%s\":{\"path\":",arm?",":"",s->name);strat01_json_string(report,output_path[arm]);fprintf(report,",\"bytes\":196608,\"sha256\":");strat01_json_string(report,output_sha[arm]);fputs(",\"q_source\":\"reference\",\"latent_source\":",report);strat01_json_string(report,s->latent_source);fputs(",\"positional_source\":",report);strat01_json_string(report,s->positional_source);fprintf(report,",\"latent_control\":%s,\"positional_control\":%s}",s->latent_control?"true":"false",s->positional_control?"true":"false");}
    fputs("},\n  \"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n  \"compiler_family\":\"clang\",\n  \"donor_graph_executions\":0,\n  \"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"layer-2 KV-partition report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(rq);free(rk);free(rv);free(ck);free(cv);free(mixed);free(out);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified layer-2 KV-partition failure");strat01_l2kpx_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 layer-2 KV-partition refused: %s\n",error);return 1;}
    fprintf(stderr,"STRAT-01 layer-2 KV-partition: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_l2kpx_selftest(void) {
    int bad=0,checks=0;strat01_tensor t;float rk[8U*576U],ck[8U*576U],mixed[8U*576U];
#define L2KPX_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    memset(&t,0,sizeof(t));t.name=(char *)"blk.2.attn_v_b.weight";t.type=STRAT01_GGML_Q4_K;t.rank=3;t.dims[0]=512;t.dims[1]=192;t.dims[2]=32;t.span=1769472;
    L2KPX_CHECK(strat01_l2kpx_descriptor_ok(&t));t.name=(char *)"blk.1.attn_v_b.weight";L2KPX_CHECK(!strat01_l2kpx_descriptor_ok(&t));t.name=(char *)"blk.2.attn_v_b.weight";t.dims[0]=511;L2KPX_CHECK(!strat01_l2kpx_descriptor_ok(&t));
    for(unsigned tok=0;tok<8U;++tok)for(unsigned i=0;i<576U;++i){rk[(size_t)tok*576U+i]=(float)(1000U+tok*576U+i);ck[(size_t)tok*576U+i]=(float)(2000U+tok*576U+i);}
    strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[0]);L2KPX_CHECK(!memcmp(mixed,rk,sizeof(rk)));
    strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[1]);L2KPX_CHECK(!memcmp(mixed,ck,sizeof(ck)));
    strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[2]);L2KPX_CHECK(mixed[0]==ck[0]&&mixed[511]==ck[511]&&mixed[512]==rk[512]&&mixed[575]==rk[575]);
    strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[3]);L2KPX_CHECK(mixed[0]==rk[0]&&mixed[511]==rk[511]&&mixed[512]==ck[512]&&mixed[575]==ck[575]);
    strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[4]);L2KPX_CHECK(mixed[7U*576U]==-rk[7U*576U]&&mixed[7U*576U+511U]==-rk[7U*576U+511U]&&mixed[7U*576U+512U]==rk[7U*576U+512U]);
    strat01_l2kpx_compose(mixed,rk,ck,&strat01_l2kpx_arms[5]);L2KPX_CHECK(mixed[0]==rk[0]&&mixed[512]==rk[7U*576U+512U]&&mixed[7U*576U+512U]==rk[512]);
    L2KPX_CHECK(STRAT01_L2KPX_PREFIX==512U&&STRAT01_L2KPX_PREFIX+STRAT01_L2KPX_TAIL==576U);
#undef L2KPX_CHECK
    fprintf(stderr,"STRAT-01 layer-2 KV-partition selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif /* STRAT01_GGUF_LAYER2_KV_PARTITION_CROSS_INPUT_H */
