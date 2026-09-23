/* Frozen l_out-0 -> layer-1 attention propagation diagnostic. */
#ifndef STRAT01_GGUF_RUNG2C_LAYER1_START_CROSS_INPUT_H
#define STRAT01_GGUF_RUNG2C_LAYER1_START_CROSS_INPUT_H

#define STRAT01_R2C_L1START_INPUT_COUNT (8U*1536U)
#define STRAT01_R2C_L1START_INPUT_BYTES UINT64_C(49152)
#define STRAT01_R2C_L1START_REF_SHA "385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa"
#define STRAT01_R2C_L1START_C_SHA   "258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44"

typedef struct { const char *name;float *data;uint64_t count;char path[1024];char sha[65]; } strat01_r2c_l1start_dump;

static int strat01_r2c_l1start_read(const char *path,const char *expected,float *out,char actual[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];
    if(!strat01_sha256_file(path,actual,&bytes,error)||bytes!=STRAT01_R2C_L1START_INPUT_BYTES||strcmp(actual,expected)){if(!error[0])snprintf(error,256,"layer1-start input identity mismatch");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open layer1-start input");return 0;}for(size_t i=0;i<STRAT01_R2C_L1START_INPUT_COUNT;++i){if(fread(raw,1,4,f)!=4){fclose(f);snprintf(error,256,"layer1-start input short read");return 0;}out[i]=strat01_r2a_f32le(raw);if(!isfinite(out[i])){fclose(f);snprintf(error,256,"layer1-start input non-finite");return 0;}}if(ferror(f)||fclose(f)!=0){snprintf(error,256,"layer1-start input I/O failure");return 0;}return 1;
}

static void strat01_r2c_l1start_make_dumps(strat01_r2a_arm *a,strat01_r2c_l1start_dump d[11]) {
#define L1D(i,n,p,c) do{d[i].name=n;d[i].data=p;d[i].count=c;d[i].path[0]=0;d[i].sha[0]=0;}while(0)
    L1D(0,"attn_norm-1",a->attn_norm,8U*1536U);L1D(1,"q-1",a->q,8U*6144U);L1D(2,"kv_cmpr_pe-1",a->kv_cmpr_pe,8U*576U);
    L1D(3,"k_pe-1",a->k_pe,8U*64U);L1D(4,"kv_cmpr-1",a->kv_cmpr,8U*512U);L1D(5,"q_pe-1",a->q_pe,8U*32U*64U);
    L1D(6,"q_nope_absorbed_perm-1",a->q_abs,8U*32U*512U);L1D(7,"Qcur-1",a->qcur,8U*32U*576U);L1D(8,"Kcur-1",a->kcur,8U*576U);
    L1D(9,"Vcur-1",a->vcur,8U*512U);L1D(10,"kqv_out-1",a->kqv_out,8U*6144U);
#undef L1D
}

static int strat01_r2c_l1start_run(const char *model,const strat01_tensor *t[7],const float *input,int negate7,int swap07,strat01_r2a_arm *a,char error[256]) {
    memcpy(a->embd,input,STRAT01_R2C_L1START_INPUT_BYTES);
    if(negate7)for(size_t i=(size_t)7U*1536U;i<STRAT01_R2C_L1START_INPUT_COUNT;++i)a->embd[i]=-a->embd[i];
    if(swap07)for(unsigned i=0;i<1536U;++i){float x=a->embd[i];a->embd[i]=a->embd[7U*1536U+i];a->embd[7U*1536U+i]=x;}
    return strat01_r2c_build_attention_range(model,t,a,0,8,error)&&strat01_r2a_run_schedule(model,t[5],t[6],a,0,error);
}

static int strat01_r2c_l1start_write_arm(const char *out_dir,const char *arm,strat01_r2c_l1start_dump d[11],char error[256]) {
    for(unsigned i=0;i<11U;++i){char leaf[160];snprintf(leaf,sizeof(leaf),"%s_%s.f32le",arm,d[i].name);if(!strat01_r2a_path(d[i].path,out_dir,leaf)||!strat01_q4q8_write_output(d[i].path,d[i].data,(size_t)d[i].count,d[i].sha,error))return 0;}return 1;
}

static int strat01_r2c_l1start_write_failure(const char *out_dir,const char *error) { char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_rung2c_layer1_start_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;fputs("{\"command\":\"--strat01-rung2c-layer1-start-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1; }

static int strat01_r2c_l1start_cli(const char *model,const char *ref_path,const char *c_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *t[7]={0};strat01_r2a_arm rr,cc,n7,s7;strat01_r2c_l1start_dump rd[11],cd[11];float *ri=NULL,*ci=NULL;FILE *report=NULL;int ok=0;
    char error[256]={0},model_sha[65]={0},ref_sha[65]={0},c_sha[65]={0},engine_sha[65]={0},header_sha[65]={0},n7_path[1024],s7_path[1024],n7_sha[65]={0},s7_sha[65]={0},report_path[1024];uint64_t hashed=0,parsed=0,tmp=0;
    memset(&inv,0,sizeof(inv));memset(&rr,0,sizeof(rr));memset(&cc,0,sizeof(cc));memset(&n7,0,sizeof(n7));memset(&s7,0,sizeof(s7));
#if !defined(__clang__)
    snprintf(error,256,"layer1-start diagnostic requires Clang");goto finish;
#endif
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"layer1-start artifact mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"layer1-start GGUF parse mismatch");goto finish;}
    for(unsigned i=0;i<7U;++i)if(!strat01_r2a_check_spec(&inv,&strat01_r2c_attn_specs[i],&t[i],error))goto finish;
    ri=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);ci=strat01_r2a_alloc(STRAT01_R2C_L1START_INPUT_COUNT,error);if(!ri||!ci)goto finish;
    if(!strat01_r2c_l1start_read(ref_path,STRAT01_R2C_L1START_REF_SHA,ri,ref_sha,error)||!strat01_r2c_l1start_read(c_path,STRAT01_R2C_L1START_C_SHA,ci,c_sha,error))goto finish;
    if(!strat01_r2a_arm_alloc(&rr,error)||!strat01_r2a_arm_alloc(&cc,error)||!strat01_r2a_arm_alloc(&n7,error)||!strat01_r2a_arm_alloc(&s7,error))goto finish;
    if(!strat01_r2c_l1start_run(model,t,ri,0,0,&rr,error)||!strat01_r2c_l1start_run(model,t,ci,0,0,&cc,error)||!strat01_r2c_l1start_run(model,t,ri,1,0,&n7,error)||!strat01_r2c_l1start_run(model,t,ri,0,1,&s7,error))goto finish;
    strat01_r2c_l1start_make_dumps(&rr,rd);strat01_r2c_l1start_make_dumps(&cc,cd);if(!strat01_r2c_l1start_write_arm(out_dir,"reference_input",rd,error)||!strat01_r2c_l1start_write_arm(out_dir,"c_input",cd,error))goto finish;
    if(!strat01_r2a_path(n7_path,out_dir,"control_reference_token7_negated_kqv_out-1.f32le")||!strat01_r2a_path(s7_path,out_dir,"control_reference_rows0_7_swapped_kqv_out-1.f32le")||!strat01_q4q8_write_output(n7_path,n7.kqv_out,8U*6144U,n7_sha,error)||!strat01_q4q8_write_output(s7_path,s7.kqv_out,8U*6144U,s7_sha,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_rung2c_layer1_start_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write layer1-start report");goto finish;}
    fputs("{\n\"command\":\"--strat01-rung2c-layer1-start-cross-input\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"model\":{\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n\"inputs\":{\"reference\":{\"path\":",report);strat01_json_string(report,ref_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,ref_sha);fputs("},\"c\":{\"path\":",report);strat01_json_string(report,c_path);fputs(",\"bytes\":49152,\"sha256\":",report);strat01_json_string(report,c_sha);fputs("}},\n\"tensors\":[",report);
    for(unsigned i=0;i<7U;++i){fprintf(report,"%s{\"name\":",i?",":"");strat01_json_string(report,t[i]->name);fprintf(report,",\"type\":%u,\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "}",t[i]->type,t[i]->offset,t[i]->file_offset,t[i]->span);}fputs("],\n\"outputs\":{\"reference_input\":{",report);
    for(unsigned i=0;i<11U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",rd[i].name);strat01_json_string(report,rd[i].path);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",rd[i].count*4U);strat01_json_string(report,rd[i].sha);fputc('}',report);}fputs("},\"c_input\":{",report);
    for(unsigned i=0;i<11U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",cd[i].name);strat01_json_string(report,cd[i].path);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",cd[i].count*4U);strat01_json_string(report,cd[i].sha);fputc('}',report);}fputs("}},\n\"controls\":{\"reference_token7_negated\":{\"path\":",report);strat01_json_string(report,n7_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,n7_sha);fputs("},\"reference_rows0_7_swapped\":{\"path\":",report);strat01_json_string(report,s7_path);fputs(",\"bytes\":196608,\"sha256\":",report);strat01_json_string(report,s7_sha);fputs("}},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"layer1-start report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(ri);free(ci);strat01_r2a_arm_free(&rr);strat01_r2a_arm_free(&cc);strat01_r2a_arm_free(&n7);strat01_r2a_arm_free(&s7);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified layer1-start failure");strat01_r2c_l1start_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 layer1-start refused: %s\n",error);return 1;}fprintf(stderr,"STRAT-01 layer1-start: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_r2c_l1start_selftest(void) { int bad=0,checks=0;float x[8U*1536U]={0};strat01_r2c_l1start_dump d[11];strat01_r2a_arm a;memset(&a,0,sizeof(a));
#define L1S_CHECK(z) do{++checks;if(!(z))++bad;}while(0)
    L1S_CHECK(strlen(STRAT01_R2C_L1START_REF_SHA)==64U);L1S_CHECK(strlen(STRAT01_R2C_L1START_C_SHA)==64U);x[7U*1536U]=2.0f;for(unsigned i=0;i<1536U;++i){float t=x[i];x[i]=x[7U*1536U+i];x[7U*1536U+i]=t;}L1S_CHECK(x[0]==2.0f&&x[7U*1536U]==0.0f);
    L1S_CHECK(strat01_r2a_arm_alloc(&a,(char[256]){0}));if(a.attn_norm){strat01_r2c_l1start_make_dumps(&a,d);L1S_CHECK(!strcmp(d[0].name,"attn_norm-1"));L1S_CHECK(!strcmp(d[10].name,"kqv_out-1"));L1S_CHECK(d[7].count==8U*32U*576U);L1S_CHECK(d[10].count==8U*6144U);}strat01_r2a_arm_free(&a);
#undef L1S_CHECK
    fprintf(stderr,"STRAT-01 layer1-start selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0; }

#endif
