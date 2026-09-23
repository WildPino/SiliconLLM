/* Model-free identity gate for the exact pinned F16 vector reduction. */
#ifndef STRAT01_GGUF_F16_VECTOR_PARITY_H
#define STRAT01_GGUF_F16_VECTOR_PARITY_H

#define STRAT01_F16V_Q_COUNT (8U*32U*576U)
#define STRAT01_F16V_K_COUNT (8U*576U)
#define STRAT01_F16V_P_COUNT (8U*32U*256U)
#define STRAT01_F16V_QK_COUNT (8U*32U*8U)
#define STRAT01_F16V_VALUE_COUNT (8U*32U*512U)

static int strat01_f16v_read(const char *path,size_t count,float **out,char error[256]) {
    FILE *f=fopen(path,"rb");float *v;if(!f){snprintf(error,256,"cannot open F16-vector input");return 0;}v=(float *)malloc(count*4U);if(!v){fclose(f);snprintf(error,256,"F16-vector allocation failure");return 0;}if(fread(v,4,count,f)!=count||fgetc(f)!=EOF){free(v);fclose(f);snprintf(error,256,"F16-vector input size mismatch");return 0;}if(fclose(f)!=0){free(v);snprintf(error,256,"F16-vector input close failure");return 0;}for(size_t i=0;i<count;++i)if(!isfinite(v[i])){free(v);snprintf(error,256,"non-finite F16-vector input");return 0;}*out=v;return 1;
}

static int strat01_f16v_write_halves(const char *path,const uint16_t *parts[4],const size_t counts[4],char sha[65],char error[256]) {
    FILE *f=fopen(path,"wb");strat01_sha256 h;uint8_t digest[32];if(!f){snprintf(error,256,"cannot open F16 conversion output");return 0;}strat01_sha256_init(&h);
    for(unsigned p=0;p<4U;++p)for(size_t i=0;i<counts[p];++i){uint8_t b[2]={(uint8_t)parts[p][i],(uint8_t)(parts[p][i]>>8)};if(fwrite(b,1,2,f)!=2){fclose(f);snprintf(error,256,"F16 conversion output write failure");return 0;}strat01_sha256_update(&h,b,2);}
    if(fclose(f)!=0){snprintf(error,256,"F16 conversion output close failure");return 0;}strat01_sha256_final(&h,digest);strat01_sha256_hex(digest,sha);return 1;
}

static int strat01_f16v_write_counts(const char *out_dir,char error[256]) {
    char path[1024];FILE *f;if(!strat01_r2a_path(path,out_dir,"strat01_f16_vector_counts.json")||(f=fopen(path,"wb"))==NULL){snprintf(error,256,"cannot write F16-vector count report");return 0;}
    fprintf(f,"{\"mode\":\"pinned-generic-f64\",\"qk_invocations\":%" PRIu64 ",\"value_invocations\":%" PRIu64 "}\n",strat01_f16vec_qk_invocations,strat01_f16vec_value_invocations);if(fclose(f)!=0){snprintf(error,256,"F16-vector count report close failure");return 0;}return 1;
}

static int strat01_f16v_cli(const char *q_path,const char *k_path,const char *p_path,const char *out_dir,const char *engine_source_path) {
    float *q=0,*k=0,*p=0,*qkv=0,*qks=0,*qkm=0,*vv=0,*vs=0,*vm=0;uint16_t *qh=0,*kh=0,*ph=0;char error[256]={0},path[7][1024],sha[7][65],engine_sha[65]={0},header_sha[65]={0};uint64_t ignored=0;int ok=0;
    if(!strat01_f16v_read(q_path,STRAT01_F16V_Q_COUNT,&q,error)||!strat01_f16v_read(k_path,STRAT01_F16V_K_COUNT,&k,error)||!strat01_f16v_read(p_path,STRAT01_F16V_P_COUNT,&p,error))goto finish;
    qh=(uint16_t *)malloc(STRAT01_F16V_Q_COUNT*2U);kh=(uint16_t *)malloc(STRAT01_F16V_K_COUNT*2U);ph=(uint16_t *)malloc(STRAT01_F16V_P_COUNT*2U);qkv=(float *)malloc(STRAT01_F16V_QK_COUNT*4U);qks=(float *)malloc(STRAT01_F16V_QK_COUNT*4U);qkm=(float *)malloc(STRAT01_F16V_QK_COUNT*4U);vv=(float *)malloc(STRAT01_F16V_VALUE_COUNT*4U);vs=(float *)malloc(STRAT01_F16V_VALUE_COUNT*4U);vm=(float *)malloc(STRAT01_F16V_VALUE_COUNT*4U);if(!qh||!kh||!ph||!qkv||!qks||!qkm||!vv||!vs||!vm){snprintf(error,256,"F16-vector product allocation failure");goto finish;}
    for(size_t i=0;i<STRAT01_F16V_Q_COUNT;++i)qh[i]=strat01_r2a_f32_to_f16(q[i]);for(size_t i=0;i<STRAT01_F16V_K_COUNT;++i)kh[i]=strat01_r2a_f32_to_f16(k[i]);for(size_t i=0;i<STRAT01_F16V_P_COUNT;++i)ph[i]=strat01_r2a_f32_to_f16(p[i]);
    for(unsigned t=0;t<8U;++t)for(unsigned h=0;h<32U;++h){const uint16_t *qr=qh+((size_t)t*32U+h)*576U;for(unsigned s=0;s<8U;++s){size_t o=((size_t)t*32U+h)*8U+s;const uint16_t *kr=kh+(size_t)s*576U;qkv[o]=strat01_f16vec_dot(576U,kr,qr);qks[o]=strat01_f16vec_scalar_dot(576U,kr,qr);if(t==7U&&h==31U){uint16_t mq[576];memcpy(mq,qr,sizeof(mq));mq[0]^=0x0100U;qkm[o]=strat01_f16vec_dot(576U,kr,mq);}else qkm[o]=qkv[o];}
        {const uint16_t *pr=ph+((size_t)t*32U+h)*256U;for(unsigned f=0;f<512U;++f){uint16_t values[256]={0},mp[256];size_t o=((size_t)t*32U+h)*512U+f;for(unsigned s=0;s<8U;++s)values[s]=kh[(size_t)s*576U+f];vv[o]=strat01_f16vec_dot(256U,values,pr);vs[o]=strat01_f16vec_scalar_dot(256U,values,pr);if(t==7U&&h==31U){memcpy(mp,pr,sizeof(mp));mp[0]^=0x0100U;vm[o]=strat01_f16vec_dot(256U,values,mp);}else vm[o]=vv[o];}}}
    {const char *leaves[6]={"qk_vector.f32le","qk_scalar.f32le","qk_mutated.f32le","value_vector.f32le","value_scalar.f32le","value_mutated.f32le"};float *values[6]={qkv,qks,qkm,vv,vs,vm};size_t counts[6]={STRAT01_F16V_QK_COUNT,STRAT01_F16V_QK_COUNT,STRAT01_F16V_QK_COUNT,STRAT01_F16V_VALUE_COUNT,STRAT01_F16V_VALUE_COUNT,STRAT01_F16V_VALUE_COUNT};for(unsigned i=0;i<6U;++i)if(!strat01_r2a_path(path[i],out_dir,leaves[i])||!strat01_q4q8_write_output(path[i],values[i],counts[i],sha[i],error))goto finish;}
    if(!strat01_r2a_path(path[6],out_dir,"conversion_stream.f16le")){snprintf(error,256,"F16 conversion output path overflow");goto finish;}{
        static const uint32_t boundary_bits[20]={
            0x00000000U,0x80000000U,0x00000001U,0x80000001U,
            0x33000000U,0x33800000U,0x38000000U,0x387fc000U,
            0x38800000U,0x3f800000U,0xbf800000U,0x477fe000U,
            0x477ff000U,0x477fffffU,0x47800000U,0x7f800000U,
            0xff800000U,0x7fc00000U,0x7fa00001U,0xffc12345U
        };
        uint16_t boundary[20];
        for(unsigned i=0;i<20U;++i){float x;memcpy(&x,&boundary_bits[i],4U);boundary[i]=strat01_r2a_f32_to_f16(x);}
        {const uint16_t *parts[4]={kh,qh,ph,boundary};const size_t counts[4]={STRAT01_F16V_K_COUNT,STRAT01_F16V_Q_COUNT,STRAT01_F16V_P_COUNT,20U};if(!strat01_f16v_write_halves(path[6],parts,counts,sha[6],error))goto finish;}
    }
    if(!strat01_sha256_file(engine_source_path,engine_sha,&ignored,error)||!strat01_sha256_file(__FILE__,header_sha,&ignored,error)){if(!error[0])snprintf(error,256,"F16-vector source hash failure");goto finish;}
    {char report_path[1024];FILE *report;if(!strat01_r2a_path(report_path,out_dir,"strat01_f16_vector_parity.json")||(report=fopen(report_path,"wb"))==NULL){snprintf(error,256,"cannot write F16-vector report");goto finish;}fputs("{\n\"command\":\"--strat01-f16-vector-parity\",\n\"state\":\"OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION\",\n\"self_certifies_pass\":false,\n\"outputs\":{",report);const char *names[7]={"qk_vector","qk_scalar","qk_mutated","value_vector","value_scalar","value_mutated","conversion_stream"};const size_t bytes[7]={8192U,8192U,8192U,524288U,524288U,524288U,435240U};for(unsigned i=0;i<7U;++i){fprintf(report,"%s\"%s\":{\"path\":",i?",":"",names[i]);strat01_json_string(report,path[i]);fprintf(report,",\"bytes\":%zu,\"sha256\":",bytes[i]);strat01_json_string(report,sha[i]);fputc('}',report);}fputs("},\n\"engine_source_sha256\":",report);strat01_json_string(report,engine_sha);fputs(",\n\"diagnostic_source_sha256\":",report);strat01_json_string(report,header_sha);fputs(",\n\"compiler_family\":\"clang\",\n\"donor_graph_executions\":0,\n\"reference_graph_executions\":0,\n\"timing_or_rate_claim\":null\n}\n",report);if(fclose(report)!=0){snprintf(error,256,"F16-vector report close failure");goto finish;}}
    ok=1;
finish:free(q);free(k);free(p);free(qh);free(kh);free(ph);free(qkv);free(qks);free(qkm);free(vv);free(vs);free(vm);if(!ok)fprintf(stderr,"STRAT-01 F16 vector parity refused: %s\n",error[0]?error:"unspecified failure");return ok?0:1;
}

static int strat01_f16v_selftest(void) {
    uint16_t a[32],b[32];int bad=0;for(unsigned i=0;i<32U;++i){a[i]=strat01_r2a_f32_to_f16((float)((int)i-16)*0.03125f);b[i]=strat01_r2a_f32_to_f16((float)((int)(i%7)-3)*0.0625f);}float v=strat01_f16vec_dot(32U,a,b),s=strat01_f16vec_scalar_dot(32U,a,b);if(!isfinite(v)||!isfinite(s))bad=1;if(strat01_f16vec_half_to_float(0x3c00U)!=1.0f||strat01_f16vec_half_to_float(0xbc00U)!=-1.0f)bad=1;fprintf(stderr,"STRAT-01 F16 vector reduction selftest: %s (4 checks)\n",bad?"FAIL":"PASS");return bad?1:0;
}

#endif
