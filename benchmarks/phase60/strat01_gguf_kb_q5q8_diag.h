/* STRAT-01 diagnostic-only K-B Q5_0 x Q8_0 operator probe. */
#ifndef STRAT01_GGUF_KB_Q5Q8_DIAG_H
#define STRAT01_GGUF_KB_Q5Q8_DIAG_H

#include <float.h>
#include <limits.h>

#define STRAT01_KB_Q5Q8_INPUT_COUNT (8U*32U*192U)
#define STRAT01_KB_Q5Q8_OUTPUT_COUNT (8U*32U*512U)
#define STRAT01_KB_Q5Q8_Q8_BLOCKS (8U*32U*4U)
#define STRAT01_KB_Q5Q8_Q5_BLOCK_BYTES 22U
#define STRAT01_KB_Q5Q8_Q8_BLOCK_BYTES 34U
#define STRAT01_KB_Q5Q8_REF_Q_SHA "4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b"
#define STRAT01_KB_Q5Q8_DOUBLE_Q_SHA "6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366"
#define STRAT01_KB_Q5Q8_FLOAT_Q_SHA "cbc263ed903c9a7d992ebd2f14795ea60f60b999efc3e5365be6b7d84fe1252b"

static int strat01_kb_q5q8_descriptor_ok(const strat01_tensor *t) {
    return t && !strcmp(t->name,"blk.0.attn_k_b.weight") &&
           t->type==STRAT01_GGML_Q5_0 && t->rank==3 &&
           t->dims[0]==128U && t->dims[1]==512U && t->dims[2]==32U &&
           t->offset==UINT64_C(272421888) && t->span==UINT64_C(1441792) &&
           t->file_offset==UINT64_C(278524800);
}

static int strat01_kb_read_input(const char *path,const char *expected,float *values,char error[256]) {
    char sha[65];uint64_t bytes=0;FILE *f=NULL;
    if(!strat01_sha256_file(path,sha,&bytes,error)||bytes!=STRAT01_KB_Q5Q8_INPUT_COUNT*4U||strcmp(sha,expected)){if(!error[0])snprintf(error,256,"K-B input identity mismatch");return 0;}
    f=fopen(path,"rb");if(!f){snprintf(error,256,"cannot open K-B input");return 0;}
    for(size_t i=0;i<STRAT01_KB_Q5Q8_INPUT_COUNT;++i){uint8_t b[4];if(fread(b,1,4,f)!=4){snprintf(error,256,"K-B input short read");fclose(f);return 0;}values[i]=strat01_r2a_f32le(b);if(!isfinite(values[i])){snprintf(error,256,"K-B input contains non-finite value");fclose(f);return 0;}}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"K-B input I/O failure");return 0;}return 1;
}

static int strat01_kb_write_payload(const char *dir,const char *leaf,const void *data,size_t bytes,char path[1024],char sha[65],char error[256]) {
    FILE *f;uint64_t measured=0;if(!strat01_r2a_path(path,dir,leaf)||(f=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot create K-B payload");return 0;}
    if(bytes&&fwrite(data,1,bytes,f)!=bytes){fclose(f);snprintf(error,256,"K-B payload short write");return 0;}
    if(fclose(f)!=0||!strat01_sha256_file(path,sha,&measured,error)||measured!=bytes){if(!error[0])snprintf(error,256,"K-B payload hash/size failure");return 0;}return 1;
}

static int strat01_kb_load_weights(const char *model,const strat01_tensor *t,uint8_t *weights,char error[256]) {
    FILE *f=fopen(model,"rb");if(!f||!strat01_r2a_seek(f,t->file_offset,error)){if(f)fclose(f);return 0;}
    if(fread(weights,1,(size_t)t->span,f)!=(size_t)t->span){snprintf(error,256,"K-B tensor short read");fclose(f);return 0;}
    if(ferror(f)||fclose(f)!=0){snprintf(error,256,"K-B tensor I/O failure");return 0;}return 1;
}

static int strat01_kb_pinned_batch(const uint8_t *weights,const float *q,float *out,strat01_kb_q8_0_block *serialized,unsigned head_stride,int corrupt_high_bits,char error[256]) {
    if(head_stride<128U){snprintf(error,256,"K-B head stride too small");return 0;}
    for(unsigned tok=0;tok<8U;++tok)for(unsigned head=0;head<32U;++head){
        const float *input=q+(size_t)tok*6144U+(size_t)head*head_stride;
        strat01_kb_q8_0_block local[4];strat01_kb_quantize_q8_0(input,local);
        if(serialized)memcpy(serialized+((size_t)tok*32U+head)*4U,local,sizeof(local));
        for(unsigned feature=0;feature<512U;++feature){const uint8_t *row=weights+((size_t)head*512U+feature)*88U;float value=strat01_kb_q5q8_dot(row,local,corrupt_high_bits);if(!isfinite(value)){snprintf(error,256,"K-B Q5_0/Q8_0 non-finite output");return 0;}out[((size_t)tok*32U+head)*512U+feature]=value;}
    }
    return 1;
}

static int strat01_kb_q5q8_diag_selftest(void) {
    int bad=0;float x[128],decoded[32];strat01_kb_q8_0_block q8[4];uint8_t q5[88]={0};
    for(unsigned i=0;i<128U;++i)x[i]=sinf((float)i*0.071f);strat01_kb_quantize_q8_0(x,q8);
    if(sizeof(strat01_kb_q8_0_block)!=34U)++bad;
    for(unsigned b=0;b<4U;++b){if(!q8[b].d)++bad;for(unsigned j=0;j<32U;++j)if(q8[b].qs[j]<-127||q8[b].qs[j]>127)++bad;}
    q5[0]=0x00;q5[1]=0x3c; /* d=1 */
    for(unsigned j=0;j<16U;++j)q5[6+j]=0x88U;
    if(!isfinite(strat01_kb_q5q8_dot(q5,q8,0)))++bad;
    (void)decoded;
    fprintf(stderr,"STRAT-01 K-B Q5_0/Q8_0 selftest: %s\n",bad?"FAIL":"PASS");return bad?1:0;
}

typedef struct { char current_path[1024],current_sha[65],pinned_path[1024],pinned_sha[65],q8_path[1024],q8_sha[65]; } strat01_kb_arm_files;

static int strat01_kb_emit_arm(const char *dir,const char *label,const char *model,const strat01_tensor *t,const uint8_t *weights,const float *q,strat01_kb_arm_files *files,char error[256]) {
    float *current=strat01_r2a_alloc(STRAT01_KB_Q5Q8_OUTPUT_COUNT,error),*pinned=strat01_r2a_alloc(STRAT01_KB_Q5Q8_OUTPUT_COUNT,error);strat01_kb_q8_0_block *q8=(strat01_kb_q8_0_block *)malloc(STRAT01_KB_Q5Q8_Q8_BLOCKS*sizeof(*q8));char leaf[128];int ok=0;
    if(!current||!pinned||!q8){if(!error[0])snprintf(error,256,"K-B arm allocation failed");goto finish;}
    if(!strat01_r2a_kb_batch(model,t,q,8U,current,error)||!strat01_kb_pinned_batch(weights,q,pinned,q8,192U,0,error))goto finish;
    snprintf(leaf,sizeof(leaf),"%s_current.f32le",label);if(!strat01_kb_write_payload(dir,leaf,current,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,files->current_path,files->current_sha,error))goto finish;
    snprintf(leaf,sizeof(leaf),"%s_q5q8.f32le",label);if(!strat01_kb_write_payload(dir,leaf,pinned,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,files->pinned_path,files->pinned_sha,error))goto finish;
    snprintf(leaf,sizeof(leaf),"%s_q8.bin",label);if(!strat01_kb_write_payload(dir,leaf,q8,STRAT01_KB_Q5Q8_Q8_BLOCKS*sizeof(*q8),files->q8_path,files->q8_sha,error))goto finish;
    ok=1;
finish: free(current);free(pinned);free(q8);return ok;
}

static int strat01_kb_emit_current(const char *dir,const char *label,const char *model,const strat01_tensor *t,const float *q,strat01_kb_arm_files *files,char error[256]) {
    float *current=strat01_r2a_alloc(STRAT01_KB_Q5Q8_OUTPUT_COUNT,error);strat01_kb_q8_0_block *q8=(strat01_kb_q8_0_block *)malloc(STRAT01_KB_Q5Q8_Q8_BLOCKS*sizeof(*q8));char leaf[128];int ok=0;
    if(!current||!q8){if(!error[0])snprintf(error,256,"K-B control allocation failed");goto finish;}if(!strat01_r2a_kb_batch(model,t,q,8U,current,error))goto finish;
    for(unsigned tok=0;tok<8U;++tok)for(unsigned head=0;head<32U;++head)strat01_kb_quantize_q8_0(q+(size_t)tok*6144U+(size_t)head*192U,q8+((size_t)tok*32U+head)*4U);
    snprintf(leaf,sizeof(leaf),"%s_current.f32le",label);if(!strat01_kb_write_payload(dir,leaf,current,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,files->current_path,files->current_sha,error))goto finish;ok=1;
    snprintf(leaf,sizeof(leaf),"%s_q8.bin",label);if(!strat01_kb_write_payload(dir,leaf,q8,STRAT01_KB_Q5Q8_Q8_BLOCKS*sizeof(*q8),files->q8_path,files->q8_sha,error))ok=0;
finish: free(current);free(q8);return ok;
}

static int strat01_kb_q5q8_diag_cli(const char *model,const char *ref_path,const char *double_path,const char *float_path,const char *out_dir,const char *engine_source_path) {
    strat01_inventory inv;const strat01_tensor *tensor=NULL;char error[256]={0},model_sha[65],engine_sha[65],header_sha[65];uint64_t parsed=0,bytes=0,tmp=0;float *ref=NULL,*dbl=NULL,*flt=NULL,*control=NULL;uint8_t *weights=NULL;strat01_kb_arm_files arms[3];char one_path[1024],one_sha[65],stride_path[1024],stride_sha[65],corrupt_path[1024],corrupt_sha[65],report_path[1024];FILE *report=NULL;int ok=0;
    memset(&inv,0,sizeof(inv));memset(arms,0,sizeof(arms));
    if(sizeof(float)!=4||sizeof(strat01_kb_q8_0_block)!=34U){snprintf(error,256,"K-B host layout mismatch");goto finish;}
    if(!strat01_parse_gguf(model,&inv,&parsed,error)||parsed!=STRAT01_EXPECTED_SIZE||!strat01_sha256_file(model,model_sha,&bytes,error)||bytes!=parsed||strcmp(model_sha,STRAT01_EXPECTED_SHA256)){if(!error[0])snprintf(error,256,"K-B artifact identity mismatch");goto finish;}
    tensor=strat01_rung1_find_tensor(&inv,"blk.0.attn_k_b.weight");if(!strat01_kb_q5q8_descriptor_ok(tensor)){snprintf(error,256,"K-B frozen descriptor mismatch");goto finish;}
    ref=strat01_r2a_alloc(STRAT01_KB_Q5Q8_INPUT_COUNT,error);dbl=strat01_r2a_alloc(STRAT01_KB_Q5Q8_INPUT_COUNT,error);flt=strat01_r2a_alloc(STRAT01_KB_Q5Q8_INPUT_COUNT,error);control=strat01_r2a_alloc(STRAT01_KB_Q5Q8_OUTPUT_COUNT,error);weights=(uint8_t *)malloc((size_t)tensor->span);
    if(!ref||!dbl||!flt||!control||!weights){if(!error[0])snprintf(error,256,"K-B diagnostic allocation failed");goto finish;}
    if(!strat01_kb_read_input(ref_path,STRAT01_KB_Q5Q8_REF_Q_SHA,ref,error)||!strat01_kb_read_input(double_path,STRAT01_KB_Q5Q8_DOUBLE_Q_SHA,dbl,error)||!strat01_kb_read_input(float_path,STRAT01_KB_Q5Q8_FLOAT_Q_SHA,flt,error)||!strat01_kb_load_weights(model,tensor,weights,error))goto finish;
    if(!strat01_kb_emit_arm(out_dir,"reference",model,tensor,weights,ref,&arms[0],error)||!strat01_kb_emit_arm(out_dir,"upstream_double",model,tensor,weights,dbl,&arms[1],error)||!strat01_kb_emit_current(out_dir,"accepted_float",model,tensor,flt,&arms[2],error))goto finish;
    {float *mut=(float *)malloc(STRAT01_KB_Q5Q8_INPUT_COUNT*4U);if(!mut){snprintf(error,256,"K-B mutation allocation failed");goto finish;}memcpy(mut,ref,STRAT01_KB_Q5Q8_INPUT_COUNT*4U);for(size_t i=0;i<STRAT01_KB_Q5Q8_INPUT_COUNT;++i)if(mut[i]!=0.0f){uint32_t bits=strat01_r2a_f32_bits(mut[i])^UINT32_C(0x80000000);mut[i]=strat01_r2a_bits_f32(bits);break;}if(!strat01_kb_pinned_batch(weights,mut,control,NULL,192U,0,error)){free(mut);goto finish;}free(mut);}
    if(!strat01_kb_write_payload(out_dir,"control_one_byte_input_mutation.f32le",control,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,one_path,one_sha,error))goto finish;
    if(!strat01_kb_pinned_batch(weights,ref,control,NULL,128U,0,error)||!strat01_kb_write_payload(out_dir,"control_wrong_head_stride.f32le",control,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,stride_path,stride_sha,error))goto finish;
    if(!strat01_kb_pinned_batch(weights,ref,control,NULL,192U,1,error)||!strat01_kb_write_payload(out_dir,"control_corrupt_qh.f32le",control,STRAT01_KB_Q5Q8_OUTPUT_COUNT*4U,corrupt_path,corrupt_sha,error))goto finish;
    if(!strat01_sha256_file(engine_source_path,engine_sha,&tmp,error))goto finish;if(!strat01_sha256_file(__FILE__,header_sha,&tmp,error))goto finish;
    if(!strat01_r2a_path(report_path,out_dir,"strat01_kb_q5q8_diag.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot create K-B report");goto finish;}
    fputs("{\n  \"command\": \"--strat01-kb-q5q8-diagnostic\",\n  \"state\": \"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n  \"self_certifies_pass\": false,\n  \"model\": {\"path\": ",report);strat01_json_string(report,model);fprintf(report,", \"bytes\": %" PRIu64 ", \"sha256\": ",bytes);strat01_json_string(report,model_sha);
    fputs("},\n  \"tensor\": {\"name\": \"blk.0.attn_k_b.weight\", \"type\": \"Q5_0\", \"dims\": [128,512,32], \"offset\": 272421888, \"span\": 1441792, \"file_offset\": 278524800},\n  \"engine_source_sha256\": ",report);strat01_json_string(report,engine_sha);fputs(",\n  \"diagnostic_source_sha256\": ",report);strat01_json_string(report,header_sha);
    fputs(",\n  \"arms\": {",report);for(unsigned i=0;i<2U;++i){const char *name=i==0?"reference":"upstream_double";fprintf(report,"%s\n    \"%s\": {\"current\": {\"path\": ",i?",":"",name);strat01_json_string(report,arms[i].current_path);fputs(", \"sha256\": ",report);strat01_json_string(report,arms[i].current_sha);fputs("}, \"q5q8\": {\"path\": ",report);strat01_json_string(report,arms[i].pinned_path);fputs(", \"sha256\": ",report);strat01_json_string(report,arms[i].pinned_sha);fputs("}, \"q8\": {\"path\": ",report);strat01_json_string(report,arms[i].q8_path);fputs(", \"sha256\": ",report);strat01_json_string(report,arms[i].q8_sha);fputs("}}",report);}fputs(",\n    \"accepted_float\": {\"current\": {\"path\": ",report);strat01_json_string(report,arms[2].current_path);fputs(", \"sha256\": ",report);strat01_json_string(report,arms[2].current_sha);fputs("}, \"q8\": {\"path\": ",report);strat01_json_string(report,arms[2].q8_path);fputs(", \"sha256\": ",report);strat01_json_string(report,arms[2].q8_sha);fputs("}}\n  },\n  \"controls\": {\"one_byte_input_mutation\": ",report);strat01_json_string(report,one_path);fputs(", \"wrong_head_stride\": ",report);strat01_json_string(report,stride_path);fputs(", \"corrupt_qh\": ",report);strat01_json_string(report,corrupt_path);fputs("},\n  \"donor_graph_executions\": 0,\n  \"timing_or_rate_claim\": null\n}\n",report);
    if(fclose(report)!=0){report=NULL;snprintf(error,256,"K-B report close failure");goto finish;}report=NULL;ok=1;
finish:
    if(report)fclose(report);free(ref);free(dbl);free(flt);free(control);free(weights);strat01_free_inventory(&inv);
    if(!ok){fprintf(stderr,"STRAT-01 K-B Q5_0/Q8_0 diagnostic refused: %s\n",error[0]?error:"unspecified failure");return 2;}fprintf(stderr,"STRAT-01 K-B Q5_0/Q8_0 diagnostic: OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\n");return 0;
}

#endif
