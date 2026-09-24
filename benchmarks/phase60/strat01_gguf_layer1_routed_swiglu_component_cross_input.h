/* Layer-1 routed SwiGLU component split through current Q6 and closed downstream. */
#ifndef STRAT01_GGUF_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_H
#define STRAT01_GGUF_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_H

#define STRAT01_L1SW_COUNT (8U*4U*1280U)
#define STRAT01_L1SW_REF_GATE_SHA "341d79877218695fe5d26ab66fb6fb8a25737107c5199e8177eca466deed1a23"
#define STRAT01_L1SW_REF_UP_SHA "e14f712c83cfd96404dd832ac851d06b6ab81d882567af8cb6590cc858182c9b"
#define STRAT01_L1SW_C_GATE_SHA "2261171cad3033adc659e1f70348d1c5ea9ac8d1adb57f40aba679f853db1293"
#define STRAT01_L1SW_C_UP_SHA "7c2f80efe17e22ac2ef5bbf3059d67dcb45ff593cb36e9f4b54c639817a51859"
#define STRAT01_L1SW_REF_SW_SHA "5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8"
#define STRAT01_L1SW_C_SW_SHA "68fbf31260eacf458be0032e63fffa4f5ea33280ecdb81a53c88a815b2326a8c"

typedef struct {const char *name,*kind;float *sw;char sw_path[1024],sw_sha[65],down_path[1024],down_sha[65],moe_path[1024],moe_sha[65],out_path[1024],out_sha[65];} strat01_l1sw_arm;

static const char *strat01_l1sw_names[9]={
    "captured_reference_swiglu","captured_production_c_swiglu",
    "scalar_reference_gate_reference_up","scalar_c_gate_c_up",
    "sse2_reference_gate_reference_up","sse2_c_gate_reference_up",
    "sse2_reference_gate_c_up","sse2_c_gate_c_up",
    "control_reference_token7_negated"};
static const char *strat01_l1sw_kinds[9]={"capture","capture","scalar","scalar","sse2","sse2","sse2","sse2","control"};

static int strat01_l1sw_write_failure(const char *dir,const char *error){char path[1024];FILE *f;if(!strat01_r2a_path(path,dir,"strat01_layer1_routed_swiglu_component_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-layer1-routed-swiglu-component-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0,\"reference_graph_executions\":0}\n",f);fclose(f);return 1;}

static int strat01_l1sw_write_payload(const char *dir,const char *arm,const char *suffix,const float *values,uint64_t count,char path[1024],char sha[65],char error[256]){char leaf[192];snprintf(leaf,sizeof(leaf),"%s.%s.f32le",arm,suffix);return strat01_r2a_path(path,dir,leaf)&&strat01_q4q8_write_output(path,values,count,sha,error);}

static int strat01_l1sw_cli(const char *model,const char *topk_path,const char *rg_path,const char *ru_path,const char *cg_path,const char *cu_path,const char *rsw_path,const char *csw_path,const char *rq_path,const char *rk_path,const char *ri_path,const char *rshared_path,const char *rw_path,const char *out_dir,const char *engine_source_path){
    strat01_inventory inv;const strat01_tensor *experts=NULL,*at=NULL,*pt=NULL,*nt=NULL,*vb=NULL;int32_t ids[STRAT01_L1Q6_TOPK_COUNT];FILE *report=NULL;int ok=0;uint64_t hashed=0,parsed=0,tmp=0;
    float *rg=NULL,*ru=NULL,*cg=NULL,*cu=NULL,*rsw=NULL,*csw=NULL,*rq=NULL,*rk=NULL,*ri=NULL,*rshared=NULL,*rw=NULL,*down=NULL,*weighted=NULL,*moe=NULL,*ffn=NULL,*lout=NULL,*aw=NULL,*norm=NULL,*proj=NULL,*kvw=NULL,*prefix=NULL,*k=NULL,*out=NULL;
    strat01_l1sw_arm arms[9]={{0}};char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},topk_sha[65]={0},input_sha[12][65]={{0}},report_path[1024];memset(&inv,0,sizeof(inv));
    const char *paths[12]={rg_path,ru_path,cg_path,cu_path,rsw_path,csw_path,rq_path,rk_path,ri_path,rshared_path,rw_path,topk_path};
    const char *names[12]={"reference_gate","reference_up","c_gate","c_up","reference_swiglu","c_swiglu","ref_q","ref_k","ref_ffn_inp","ref_shared_out","ref_weights","topk"};
    const char *expected[11]={STRAT01_L1SW_REF_GATE_SHA,STRAT01_L1SW_REF_UP_SHA,STRAT01_L1SW_C_GATE_SHA,STRAT01_L1SW_C_UP_SHA,STRAT01_L1SW_REF_SW_SHA,STRAT01_L1SW_C_SW_SHA,STRAT01_L2RN_REF_Q_SHA,STRAT01_L2RN_REF_K_SHA,STRAT01_L1TC_REF_INP_SHA,STRAT01_L1FO_REF_SHARED_SHA,STRAT01_L1RM_REF_WEIGHTS_SHA};
    const uint64_t sizes[11]={163840U,163840U,163840U,163840U,163840U,163840U,589824U,18432U,49152U,49152U,128U};float *values[11]={NULL};
#if !defined(__clang__)
    snprintf(error,256,"layer-1 routed-SwiGLU component diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"layer-1 routed-SwiGLU artifact mismatch");goto finish;}if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed)goto finish;
    if(!strat01_r2a_check_spec(&inv,&strat01_r2c_moe_specs[5],&experts,error))goto finish;at=strat01_rung1_find_tensor(&inv,"blk.2.attn_norm.weight");pt=strat01_rung1_find_tensor(&inv,"blk.2.attn_kv_a_mqa.weight");nt=strat01_rung1_find_tensor(&inv,"blk.2.attn_kv_a_norm.weight");vb=strat01_rung1_find_tensor(&inv,"blk.2.attn_v_b.weight");if(!strat01_l2an_norm_descriptor_ok(at)||!strat01_l2kva_projection_descriptor_ok(pt)||!strat01_l2rn_norm_descriptor_ok(nt)||!strat01_l2rn_vb_descriptor_ok(vb)){snprintf(error,256,"layer-1 routed-SwiGLU downstream descriptor mismatch");goto finish;}
#define L1SW_ALLOC(p,n) do{p=strat01_r2a_alloc((n),error);if(!p)goto finish;}while(0)
    L1SW_ALLOC(rg,STRAT01_L1SW_COUNT);L1SW_ALLOC(ru,STRAT01_L1SW_COUNT);L1SW_ALLOC(cg,STRAT01_L1SW_COUNT);L1SW_ALLOC(cu,STRAT01_L1SW_COUNT);L1SW_ALLOC(rsw,STRAT01_L1SW_COUNT);L1SW_ALLOC(csw,STRAT01_L1SW_COUNT);L1SW_ALLOC(rq,8U*32U*576U);L1SW_ALLOC(rk,8U*576U);L1SW_ALLOC(ri,8U*1536U);L1SW_ALLOC(rshared,8U*1536U);L1SW_ALLOC(rw,8U*4U);L1SW_ALLOC(down,8U*4U*1536U);L1SW_ALLOC(weighted,8U*4U*1536U);L1SW_ALLOC(moe,8U*1536U);L1SW_ALLOC(ffn,8U*1536U);L1SW_ALLOC(lout,8U*1536U);L1SW_ALLOC(aw,1536U);L1SW_ALLOC(norm,8U*1536U);L1SW_ALLOC(proj,8U*576U);L1SW_ALLOC(kvw,512U);L1SW_ALLOC(prefix,8U*512U);L1SW_ALLOC(k,8U*576U);L1SW_ALLOC(out,8U*6144U);for(unsigned i=0;i<9U;++i){arms[i].name=strat01_l1sw_names[i];arms[i].kind=strat01_l1sw_kinds[i];L1SW_ALLOC(arms[i].sw,STRAT01_L1SW_COUNT);}
#undef L1SW_ALLOC
    values[0]=rg;values[1]=ru;values[2]=cg;values[3]=cu;values[4]=rsw;values[5]=csw;values[6]=rq;values[7]=rk;values[8]=ri;values[9]=rshared;values[10]=rw;for(unsigned i=0;i<11U;++i)if(!strat01_r2c_cross_read_f32(paths[i],sizes[i],expected[i],values[i],input_sha[i],error))goto finish;if(!strat01_l1q6_read_topk(topk_path,ids,topk_sha,error))goto finish;if(!strat01_r2a_read_f32_vector(model,at,aw,1536U,error)||!strat01_r2a_read_f32_vector(model,nt,kvw,512U,error))goto finish;
    memcpy(arms[0].sw,rsw,163840U);memcpy(arms[1].sw,csw,163840U);if(!strat01_sw_compute(rg,ru,arms[2].sw,STRAT01_L1SW_COUNT,error)||!strat01_sw_compute(cg,cu,arms[3].sw,STRAT01_L1SW_COUNT,error)||!strat01_sse2_swiglu_compute(rg,ru,arms[4].sw,STRAT01_L1SW_COUNT,error)||!strat01_sse2_swiglu_compute(cg,ru,arms[5].sw,STRAT01_L1SW_COUNT,error)||!strat01_sse2_swiglu_compute(rg,cu,arms[6].sw,STRAT01_L1SW_COUNT,error)||!strat01_sse2_swiglu_compute(cg,cu,arms[7].sw,STRAT01_L1SW_COUNT,error))goto finish;memcpy(arms[8].sw,rsw,163840U);for(size_t i=7U*4U*1280U;i<8U*4U*1280U;++i)arms[8].sw[i]=-arms[8].sw[i];
    if(memcmp(arms[2].sw,rsw,163840U)||memcmp(arms[3].sw,csw,163840U)){snprintf(error,256,"layer-1 routed-SwiGLU scalar capture replay mismatch");goto finish;}
    for(unsigned a=0;a<9U;++a){if(!strat01_l1q6_routed(model,experts,ids,arms[a].sw,down,error))goto finish;strat01_l1rm_reduce(down,rw,weighted,moe);strat01_l1fo_add(moe,rshared,ffn);strat01_l1tc_add(ri,ffn,lout);strat01_r2a_rmsnorm_pinned(lout,aw,norm,8U,1536U,STRAT01_R2A_RMS_EPS);if(!strat01_r2a_matmul_batch(model,pt,norm,8U,1536U,proj,576U,error))goto finish;strat01_l2rn_compute_prefix(proj,kvw,prefix,0,0);strat01_l2rn_compose_k(k,rk,prefix);if(!strat01_r2c_cross_run_arm(model,vb,rq,k,0,0,out,error))goto finish;if(!strat01_l1sw_write_payload(out_dir,arms[a].name,"swiglu",arms[a].sw,STRAT01_L1SW_COUNT,arms[a].sw_path,arms[a].sw_sha,error)||!strat01_l1sw_write_payload(out_dir,arms[a].name,"down",down,8U*4U*1536U,arms[a].down_path,arms[a].down_sha,error)||!strat01_l1sw_write_payload(out_dir,arms[a].name,"moe_out",moe,8U*1536U,arms[a].moe_path,arms[a].moe_sha,error)||!strat01_l1sw_write_payload(out_dir,arms[a].name,"downstream",out,8U*6144U,arms[a].out_path,arms[a].out_sha,error))goto finish;}
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;if(!strat01_r2a_path(report_path,out_dir,"strat01_layer1_routed_swiglu_component_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write layer-1 routed-SwiGLU report");goto finish;}
    fputs("{\n\"command\":\"--strat01-layer1-routed-swiglu-component-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"inputs\":{",report);for(unsigned i=0;i<11U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",names[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",sizes[i]);strat01_json_string(report,input_sha[i]);fputc('}',report);}fputs(",\"topk\":{\"path\":",report);strat01_json_string(report,topk_path);fputs(",\"bytes\":128,\"sha256\":",report);strat01_json_string(report,topk_sha);fputs("}},\n\"outputs\":{",report);
    for(unsigned a=0;a<9U;++a){fprintf(report,"%s\"%s\":{\"kind\":",a?",":"",arms[a].name);strat01_json_string(report,arms[a].kind);fputs(",\"swiglu\":{\"path\":",report);strat01_json_string(report,arms[a].sw_path);fputs(",\"bytes\":163840,\"sha256\":",report);strat01_json_string(report,arms[a].sw_sha);fputs("},\"down\":{\"path\":",report);strat01_json_string(report,arms[a].down_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,arms[a].down_sha);fputs("},\"moe_out\":{\"path\":",report);strat01_json_string(report,arms[a].moe_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,arms[a].moe_sha);fputs("},\"downstream\":{\"path\":",report);strat01_json_string(report,arms[a].out_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,arms[a].out_sha);fputs("}}",report);}
    fputs("},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\"compiler_family\":\"clang\",\"q6_arms\":9,\"donor_graph_executions\":0,\"reference_graph_executions\":0,\"timing_or_rate_claim\":null\n}\n",report);if(fclose(report)!=0){report=NULL;snprintf(error,256,"layer-1 routed-SwiGLU report close failed");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);for(unsigned i=0;i<9U;++i)free(arms[i].sw);free(rg);free(ru);free(cg);free(cu);free(rsw);free(csw);free(rq);free(rk);free(ri);free(rshared);free(rw);free(down);free(weighted);free(moe);free(ffn);free(lout);free(aw);free(norm);free(proj);free(kvw);free(prefix);free(k);free(out);strat01_free_inventory(&inv);if(!ok){strat01_l1sw_write_failure(out_dir,error[0]?error:"layer-1 routed-SwiGLU failure");fprintf(stderr,"STRAT-01 layer-1 routed-SwiGLU component: %s\n",error[0]?error:"failed");return 2;}fprintf(stderr,"STRAT-01 layer-1 routed-SwiGLU component: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_l1sw_selftest(void){int bad=0,checks=0;float g[8]={-2,-1,0,1,2,3,4,5},u[8]={1,1,1,1,1,1,1,1},a[8],b[8];char error[256]={0};
#define L1SW_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    L1SW_CHECK(sizeof(strat01_l1sw_names)/sizeof(strat01_l1sw_names[0])==9U);L1SW_CHECK(STRAT01_L1SW_COUNT==40960U);L1SW_CHECK(strlen(STRAT01_L1SW_REF_SW_SHA)==64U);L1SW_CHECK(strat01_sw_compute(g,u,a,8U,error));L1SW_CHECK(strat01_sse2_swiglu_compute(g,u,b,8U,error));L1SW_CHECK(memcmp(a,b,sizeof(a))!=0);L1SW_CHECK(!strcmp(strat01_l1sw_kinds[8],"control"));
#undef L1SW_CHECK
    fprintf(stderr,"STRAT-01 layer-1 routed-SwiGLU component selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;}

#endif
