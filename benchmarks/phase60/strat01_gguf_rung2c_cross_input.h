/*
 * Frozen-input layer-1 attention/V-B diagnostic for STRAT-01 Rung 2C.
 * It executes six boundary-only arms and never executes a donor graph.
 */
#ifndef STRAT01_GGUF_RUNG2C_CROSS_INPUT_H
#define STRAT01_GGUF_RUNG2C_CROSS_INPUT_H

#define STRAT01_R2C_CROSS_Q_COUNT (8U*32U*576U)
#define STRAT01_R2C_CROSS_K_COUNT (8U*576U)
#define STRAT01_R2C_CROSS_V_COUNT (8U*512U)
#define STRAT01_R2C_CROSS_OUT_COUNT (8U*6144U)
#define STRAT01_R2C_CROSS_Q_BYTES UINT64_C(589824)
#define STRAT01_R2C_CROSS_K_BYTES UINT64_C(18432)
#define STRAT01_R2C_CROSS_V_BYTES UINT64_C(16384)
#define STRAT01_R2C_CROSS_OUT_BYTES UINT64_C(196608)

#define STRAT01_R2C_CROSS_REF_Q_SHA "9cb50fb41e0d33549eb53bd4ae19605705331562d36f975bb2ac82cc4c41cb16"
#define STRAT01_R2C_CROSS_REF_K_SHA "73febde321c2ea45a56a5d27a3605b820e08f8b038420a379afc4b9c3f0c72b4"
#define STRAT01_R2C_CROSS_REF_V_SHA "3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387"
#define STRAT01_R2C_CROSS_C_Q_SHA   "3faa0a37833c9d9b9cf92d3bf985d13bf6f09b9c278f81e8813258e86e7d2b2f"
#define STRAT01_R2C_CROSS_C_K_SHA   "febfcaa2ca7192a71b7ffca1af26252c9e81b010068478928ed6d56fa78fb0d4"
#define STRAT01_R2C_CROSS_C_V_SHA   "58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57"

typedef enum {
    STRAT01_R2C_CROSS_CC=0,
    STRAT01_R2C_CROSS_RR=1,
    STRAT01_R2C_CROSS_CR=2,
    STRAT01_R2C_CROSS_RC=3,
    STRAT01_R2C_CROSS_Q_CONTROL=4,
    STRAT01_R2C_CROSS_KV_CONTROL=5
} strat01_r2c_cross_arm_id;

typedef struct {
    const char *name;
    const char *q_source;
    const char *kv_source;
    int query_control;
    int kv_control;
} strat01_r2c_cross_arm_spec;

static const strat01_r2c_cross_arm_spec strat01_r2c_cross_arms[6]={
    {"c_q__c_kv","c","c",0,0},
    {"ref_q__ref_kv","reference","reference",0,0},
    {"c_q__ref_kv","c","reference",0,0},
    {"ref_q__c_kv","reference","c",0,0},
    {"control_ref_q_token7_negated","reference","reference",1,0},
    {"control_ref_kv_rows0_7_swapped","reference","reference",0,1}
};

static int strat01_r2c_cross_descriptor_ok(const strat01_tensor *t) {
    return t&&!strcmp(t->name,"blk.1.attn_v_b.weight")&&t->type==STRAT01_GGML_Q4_K&&t->rank==3&&
           t->dims[0]==512U&&t->dims[1]==192U&&t->dims[2]==32U&&t->span==UINT64_C(1769472);
}

static int strat01_r2c_cross_read_f32(const char *path,uint64_t expected_bytes,const char *expected_sha,float *out,char actual_sha[65],char error[256]) {
    FILE *f=NULL;uint64_t bytes=0;uint8_t raw[4];size_t count=(size_t)(expected_bytes/4U);
    if(!strat01_sha256_file(path,actual_sha,&bytes,error)||bytes!=expected_bytes||strcmp(actual_sha,expected_sha)){
        if(!error[0])snprintf(error,256,"Rung-2C cross-input payload identity mismatch");return 0;
    }
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open Rung-2C cross-input payload");return 0;}
    for(size_t i=0;i<count;++i){
        if(fread(raw,1,4,f)!=4){snprintf(error,256,"Rung-2C cross-input payload short read");fclose(f);return 0;}
        out[i]=strat01_r2a_f32le(raw);if(!isfinite(out[i])){snprintf(error,256,"Rung-2C cross-input payload non-finite");fclose(f);return 0;}
    }
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"Rung-2C cross-input payload I/O failure");return 0;}return 1;
}

static int strat01_r2c_cross_v_matches_k(const float *k,const float *v) {
    for(unsigned tok=0;tok<8U;++tok)
        if(memcmp(k+(size_t)tok*576U,v+(size_t)tok*512U,512U*sizeof(float)))return 0;
    return 1;
}

static void strat01_r2c_cross_apply_controls(float *q,float *k,int query_control,int kv_control) {
    if(query_control)for(size_t i=(size_t)7U*32U*576U;i<STRAT01_R2C_CROSS_Q_COUNT;++i)q[i]=-q[i];
    if(kv_control)for(unsigned i=0;i<576U;++i){float tmp=k[i];k[i]=k[(size_t)7U*576U+i];k[(size_t)7U*576U+i]=tmp;}
}

static void strat01_r2c_cross_select(strat01_r2c_cross_arm_id id,const float *rq,const float *rk,const float *cq,const float *ck,const float **q,const float **k) {
    const strat01_r2c_cross_arm_spec *s=&strat01_r2c_cross_arms[(unsigned)id];
    *q=!strcmp(s->q_source,"reference")?rq:cq;*k=!strcmp(s->kv_source,"reference")?rk:ck;
}

static int strat01_r2c_cross_run_arm(const char *model,const strat01_tensor *vb,const float *q_source,const float *k_source,int query_control,int kv_control,float *out,char error[256]) {
    strat01_r2a_arm a;float *latent=NULL;int ok=0;memset(&a,0,sizeof(a));
    if(!strat01_r2a_arm_alloc(&a,error))goto finish;
    memcpy(a.qcur,q_source,STRAT01_R2C_CROSS_Q_BYTES);memcpy(a.kcur,k_source,STRAT01_R2C_CROSS_K_BYTES);
    strat01_r2c_cross_apply_controls(a.qcur,a.kcur,query_control,kv_control);
    latent=strat01_r2a_alloc(8U*32U*512U,error);if(!latent)goto finish;
    for(unsigned tok=0;tok<8U;++tok)strat01_r2a_cache_write(&a,tok,a.kcur+(size_t)tok*576U);
    for(unsigned tok=0;tok<8U;++tok)strat01_r2a_attend_one(&a,tok,latent);
    if(!strat01_r2a_vb_batch(model,vb,latent,a.kqv_out,error))goto finish;
    memcpy(out,a.kqv_out,STRAT01_R2C_CROSS_OUT_BYTES);ok=1;
finish: free(latent);strat01_r2a_arm_free(&a);return ok;
}

static int strat01_r2c_cross_write_failure(const char *out_dir,const char *error) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_rung2c_cross_input.json")||(f=fopen(path,"wb"))==NULL)return 0;
    fputs("{\"command\":\"--strat01-rung2c-cross-input\",\"state\":\"DIAGNOSTIC_FAILURE\",\"error\":",f);strat01_json_string(f,error);fputs(",\"donor_graph_executions\":0}\n",f);fclose(f);return 1;
}

static int strat01_r2c_cross_input_cli(const char *model,const char *ref_q_path,const char *ref_k_path,const char *ref_v_path,const char *c_q_path,const char *c_k_path,const char *c_v_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *vb=NULL;float *rq=NULL,*rk=NULL,*rv=NULL,*cq=NULL,*ck=NULL,*cv=NULL,*out=NULL;FILE *report=NULL;int ok=0;
    char error[256]={0},model_sha[65]={0},engine_sha[65]={0},header_sha[65]={0};char input_sha[6][65]={{0}},output_sha[6][65]={{0}},output_path[6][1024],report_path[1024];
    uint64_t hashed=0,parsed=0,tmp=0;memset(&inv,0,sizeof(inv));
#if !defined(__clang__)
    snprintf(error,256,"Rung-2C cross-input diagnostic requires Clang");goto finish;
#endif
    if(sizeof(float)!=4){snprintf(error,256,"unsupported Rung-2C cross-input host layout");goto finish;}
    if(!strat01_sha256_file(model,model_sha,&hashed,error)||hashed!=STRAT01_EXPECTED_SIZE||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"Rung-2C cross-input artifact identity mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=hashed){if(!error[0])snprintf(error,256,"Rung-2C cross-input GGUF parse/size mismatch");goto finish;}
    vb=strat01_rung1_find_tensor(&inv,"blk.1.attn_v_b.weight");if(!strat01_r2c_cross_descriptor_ok(vb)){snprintf(error,256,"Rung-2C cross-input V-B descriptor mismatch");goto finish;}
    rq=strat01_r2a_alloc(STRAT01_R2C_CROSS_Q_COUNT,error);rk=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);rv=strat01_r2a_alloc(STRAT01_R2C_CROSS_V_COUNT,error);
    cq=strat01_r2a_alloc(STRAT01_R2C_CROSS_Q_COUNT,error);ck=strat01_r2a_alloc(STRAT01_R2C_CROSS_K_COUNT,error);cv=strat01_r2a_alloc(STRAT01_R2C_CROSS_V_COUNT,error);out=strat01_r2a_alloc(STRAT01_R2C_CROSS_OUT_COUNT,error);
    if(!rq||!rk||!rv||!cq||!ck||!cv||!out)goto finish;
    if(!strat01_r2c_cross_read_f32(ref_q_path,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_REF_Q_SHA,rq,input_sha[0],error)||
       !strat01_r2c_cross_read_f32(ref_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_REF_K_SHA,rk,input_sha[1],error)||
       !strat01_r2c_cross_read_f32(ref_v_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_R2C_CROSS_REF_V_SHA,rv,input_sha[2],error)||
       !strat01_r2c_cross_read_f32(c_q_path,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_C_Q_SHA,cq,input_sha[3],error)||
       !strat01_r2c_cross_read_f32(c_k_path,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_C_K_SHA,ck,input_sha[4],error)||
       !strat01_r2c_cross_read_f32(c_v_path,STRAT01_R2C_CROSS_V_BYTES,STRAT01_R2C_CROSS_C_V_SHA,cv,input_sha[5],error))goto finish;
    if(!strat01_r2c_cross_v_matches_k(rk,rv)||!strat01_r2c_cross_v_matches_k(ck,cv)){snprintf(error,256,"Rung-2C cross-input V/K-prefix mismatch");goto finish;}
    for(unsigned arm=0;arm<6U;++arm){const float *q=NULL,*k=NULL;strat01_r2c_cross_select((strat01_r2c_cross_arm_id)arm,rq,rk,cq,ck,&q,&k);
        if(!strat01_r2c_cross_run_arm(model,vb,q,k,strat01_r2c_cross_arms[arm].query_control,strat01_r2c_cross_arms[arm].kv_control,out,error))goto finish;
        {char leaf[160];snprintf(leaf,sizeof(leaf),"%s.f32le",strat01_r2c_cross_arms[arm].name);if(!strat01_r2a_path(output_path[arm],out_dir,leaf)||!strat01_q4q8_write_output(output_path[arm],out,STRAT01_R2C_CROSS_OUT_COUNT,output_sha[arm],error))goto finish;}
    }
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error)||!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_rung2c_cross_input.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot create Rung-2C cross-input report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-rung2c-cross-input\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\":",report);strat01_json_string(report,model);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",hashed);strat01_json_string(report,model_sha);fputs("},\n",report);
    fputs("  \"inputs\": {",report);
    {const char *keys[6]={"ref_q","ref_k","ref_v","c_q","c_k","c_v"};const char *paths[6]={ref_q_path,ref_k_path,ref_v_path,c_q_path,c_k_path,c_v_path};const uint64_t sizes[6]={STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_V_BYTES,STRAT01_R2C_CROSS_Q_BYTES,STRAT01_R2C_CROSS_K_BYTES,STRAT01_R2C_CROSS_V_BYTES};for(unsigned i=0;i<6U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",keys[i]);strat01_json_string(report,paths[i]);fprintf(report,",\"bytes\":%" PRIu64 ",\"sha256\":",sizes[i]);strat01_json_string(report,input_sha[i]);fputc('}',report);}}
    fprintf(report,"},\n  \"v_k_prefix_equal\": {\"reference\":true,\"c\":true},\n  \"matrix\": {\"name\":\"blk.1.attn_v_b.weight\",\"type\":\"Q4_K\",\"shape\":[512,192,32],\"offset\":%" PRIu64 ",\"file_offset\":%" PRIu64 ",\"span\":%" PRIu64 "},\n  \"outputs\": {",vb->offset,vb->file_offset,vb->span);
    for(unsigned arm=0;arm<6U;++arm){const strat01_r2c_cross_arm_spec *s=&strat01_r2c_cross_arms[arm];fprintf(report,"%s\"%s\":{\"path\":",arm?",":"",s->name);strat01_json_string(report,output_path[arm]);fprintf(report,",\"bytes\":196608,\"sha256\":");strat01_json_string(report,output_sha[arm]);fputs(",\"q_source\":",report);strat01_json_string(report,s->q_source);fputs(",\"kv_source\":",report);strat01_json_string(report,s->kv_source);fprintf(report,",\"query_control\":%s,\"kv_control\":%s}",s->query_control?"true":"false",s->kv_control?"true":"false");}
    fputs("},\n  \"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n  \"compiler_family\":\"clang\",\n  \"donor_graph_executions\":0,\n  \"timing_or_rate_claim\":null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"Rung-2C cross-input report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(rq);free(rk);free(rv);free(cq);free(ck);free(cv);free(out);strat01_free_inventory(&inv);
    if(!ok){if(!error[0])snprintf(error,256,"unspecified Rung-2C cross-input failure");strat01_r2c_cross_write_failure(out_dir,error);fprintf(stderr,"STRAT-01 Rung-2C cross-input refused: %s\n",error);return 1;}
    fprintf(stderr,"STRAT-01 Rung-2C cross-input: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

static int strat01_r2c_cross_input_selftest(void) {
    int bad=0,checks=0;strat01_tensor t;float k[8U*576U]={0},v[8U*512U]={0},q[8U*32U*576U]={0};const float *sq=NULL,*sk=NULL;
#define R2C_CROSS_CHECK(x) do{++checks;if(!(x))++bad;}while(0)
    memset(&t,0,sizeof(t));t.name=(char *)"blk.1.attn_v_b.weight";t.type=STRAT01_GGML_Q4_K;t.rank=3;t.dims[0]=512;t.dims[1]=192;t.dims[2]=32;t.span=1769472;
    R2C_CROSS_CHECK(strat01_r2c_cross_descriptor_ok(&t));t.name=(char *)"blk.0.attn_v_b.weight";R2C_CROSS_CHECK(!strat01_r2c_cross_descriptor_ok(&t));t.name=(char *)"blk.1.attn_v_b.weight";t.dims[2]=31;R2C_CROSS_CHECK(!strat01_r2c_cross_descriptor_ok(&t));
    for(unsigned tok=0;tok<8U;++tok)for(unsigned i=0;i<512U;++i)k[(size_t)tok*576U+i]=v[(size_t)tok*512U+i]=(float)(tok+i);
    R2C_CROSS_CHECK(strat01_r2c_cross_v_matches_k(k,v));v[700]+=1.0f;R2C_CROSS_CHECK(!strat01_r2c_cross_v_matches_k(k,v));v[700]-=1.0f;
    q[(size_t)7U*32U*576U]=2.0f;k[0]=1.0f;k[7U*576U]=9.0f;strat01_r2c_cross_apply_controls(q,k,1,1);R2C_CROSS_CHECK(q[(size_t)7U*32U*576U]==-2.0f&&k[0]==9.0f&&k[7U*576U]==1.0f);
    {float rq=1,rk=2,cq=3,ck=4;strat01_r2c_cross_select(STRAT01_R2C_CROSS_CC,&rq,&rk,&cq,&ck,&sq,&sk);R2C_CROSS_CHECK(sq==&cq&&sk==&ck);strat01_r2c_cross_select(STRAT01_R2C_CROSS_RR,&rq,&rk,&cq,&ck,&sq,&sk);R2C_CROSS_CHECK(sq==&rq&&sk==&rk);strat01_r2c_cross_select(STRAT01_R2C_CROSS_CR,&rq,&rk,&cq,&ck,&sq,&sk);R2C_CROSS_CHECK(sq==&cq&&sk==&rk);strat01_r2c_cross_select(STRAT01_R2C_CROSS_RC,&rq,&rk,&cq,&ck,&sq,&sk);R2C_CROSS_CHECK(sq==&rq&&sk==&ck);}
    R2C_CROSS_CHECK(strat01_r2a_f32_to_f16(1.0001f)==strat01_r2a_f32_to_f16(1.0f));
#undef R2C_CROSS_CHECK
    fprintf(stderr,"STRAT-01 Rung-2C cross-input selftest: %s (%d checks)\n",bad?"FAIL":"PASS",checks);return bad?1:0;
}

#endif /* STRAT01_GGUF_RUNG2C_CROSS_INPUT_H */
