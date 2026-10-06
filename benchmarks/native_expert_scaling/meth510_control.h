/* One retained-state paired head inquiry; no encode/decode/model observations. */
static void head_setup510(void){
    require(fegetround()==FE_TONEAREST&&SetProcessAffinityMask(GetCurrentProcess(),21),"510_control_rounding_pool");
    worker_setup(3);LARGE_INTEGER frequency;require(QueryPerformanceFrequency(&frequency),"performance_frequency");
    cost_frequency=(double)frequency.QuadPart;
}
static int contract510(const char *mode){
    head_setup510();int8_t q[8]={0};float source[8]={0},scales[8],x[1]={1.f},y[8];
    for(int i=0;i<8;i++)scales[i]=1.f;Matrix w={NULL,scales,q,8,1,0};
    if(strcmp(mode,"positive")){
        if(!strcmp(mode,"rows"))w.rows=7;
        else if(!strcmp(mode,"cols"))w.cols=4097;
        else if(!strcmp(mode,"dtype"))w.q=NULL;
        else if(!strcmp(mode,"source"))w.f=source;
        else if(!strcmp(mode,"rounding"))require(fesetround(FE_DOWNWARD)==0,"510_rounding_set");
        else require(0,"510_unknown_negative");
        hybrid_head510(w,!strcmp(mode,"source")?NULL:source,x,y,w.rows,w.cols);
        require(0,"510_negative_was_accepted");
    }
    int8_t extremes[4096];int16_t codes[4096];int64_t totals[3];
    for(int mode=0;mode<3;mode++){
        int64_t total=0;for(int j=0;j<4096;j++){
            extremes[j]=mode==0?-128:mode==1?127:(j%2?-128:127);
            codes[j]=mode==0?32767:mode==1?-32767:(j%2?32767:-32767);
            total+=(int64_t)extremes[j]*codes[j];
        }
        totals[mode]=head_integer_dot(extremes,codes,4096);require(totals[mode]==total,"510_I64_extremes");
    }
    float zero[768]={0};require(head_activation_codes(zero,codes,768)==1.f,"510_zero_scale");
    for(int i=0;i<768;i++)require(codes[i]==0,"510_zero_codes");
    float ties[7]={-2.5f,-1.5f,-.5f,.5f,1.5f,2.5f,32767.f};int16_t expected[7]={-2,-2,0,0,2,2,32767};
    require(head_activation_codes(ties,codes,7)==1.f&&!memcmp(codes,expected,sizeof(expected)),"510_even_ties");
    /* Tie ordering, replacement into original IDs, and unselected tail winner. */
    int8_t qt[10]={0};float wt[10]={0},st[10];for(int i=0;i<10;i++)st[i]=1.f;
    Matrix ten={NULL,st,qt,10,1,0};float yt[10];head_mv(ten,x,yt,10,1);hybrid_refine510(ten,wt,x,yt,10,1);
    for(int i=0;i<8;i++)require(hybrid_ids510[i]==i&&yt[i]==0.f,"510_lowest_ID_tie");
    for(int i=0;i<8;i++)wt[i]=-1.f;head_mv(ten,x,yt,10,1);hybrid_refine510(ten,wt,x,yt,10,1);
    int winner=0;for(int i=1;i<10;i++)if(yt[i]>yt[winner])winner=i;require(winner==8,"510_tail_winner");
    printf("{\"contract\":true,\"I64_totals\":[%lld,%lld,%lld],\"zero\":true,\"even_ties\":true,\"lowest_ID_top8\":true,\"tail_winner\":%d",
        (long long)totals[0],(long long)totals[1],(long long)totals[2],winner);worker_json();puts("}");return 0;
}
static int retained_head510(const char *manifest,const char *input,const char *output){
    head_setup510();double initial=cost_clock();Model *m=load(manifest);double loaded=cost_clock();
    require(m->c.d==768&&m->c.vocab==32128&&m->head.q&&m->embed,"510_source_artifact");
    FILE *f=fopen(input,"rb");require(f!=NULL,"510_input_open");char magic[8];
    require(fread(magic,1,8,f)==8&&!memcmp(magic,"SW510X01",8),"510_input_magic");
    uint32_t count=u32(f),d=u32(f);require(count==996&&d==768,"510_input_dimensions");
    uint32_t *cases=alloc(count,4),*steps=alloc(count,4);float *states=alloc((size_t)count*d,4);
    for(uint32_t i=0;i<count;i++){cases[i]=u32(f);steps[i]=u32(f);
        require(cases[i]<96&&steps[i]<64&&fread(states+(size_t)i*d,4,d,f)==d,"510_input_state");}
    require(fgetc(f)==EOF&&fclose(f)==0,"510_input_close");
    float scale=(float)(1./sqrt((double)d));for(size_t i=0;i<(size_t)count*d;i++){
        require(isfinite(states[i]),"510_input_finite");states[i]*=scale;}
    float *old=alloc(m->c.vocab,4),*hybrid=alloc(m->c.vocab,4);
    float *gold_old=alloc((size_t)count*m->c.vocab,4),*gold_hybrid=alloc((size_t)count*m->c.vocab,4);
    int *gold_ids=alloc((size_t)count*8,4);double *gold_dots=alloc((size_t)count*8,8);
    FILE *out=fopen(output,"wb");require(out!=NULL,"510_output_open");
    fwrite("SW510H01",1,8,out);uint32_t hdr[]={count,m->c.vocab,d,8};fwrite(hdr,4,4,out);
    /* Two complete warmups; three observed passes. Pair order AB,BA,AB,BA,AB. */
    cost_enabled=1;cost_profile=1;cost_phase=1;cost_kind=3;
    for(int pass=-2;pass<3;pass++)for(uint32_t i=0;i<count;i++){
        double times[2],matrix[2];HybridCounter510 hc={0};
        for(int order=0;order<2;order++){
            int arm=(pass==-1||pass==1)?1-order:order;
            memset(cost_counts,0,sizeof(cost_counts));memset(&hybrid_counts510,0,sizeof(hybrid_counts510));double before=cost_clock();
            if(arm)hybrid_head510(m->head,m->embed,states+(size_t)i*d,hybrid,m->c.vocab,d);
            else head_mv(m->head,states+(size_t)i*d,old,m->c.vocab,d);
            times[arm]=cost_clock()-before;matrix[arm]=cost_counts[1][3].matrix_seconds;
            require(cost_counts[1][3].calls==1&&cost_counts[1][3].codes==(uint64_t)32128*768&&cost_counts[1][3].scales==32128*4,"510_I8_counter");
            if(arm){hc=hybrid_counts510;require(hc.calls==1&&hc.source_rows==8&&hc.source_MACs==8*768&&hc.f32_bytes==8*768*4&&hc.visited==32128&&cost_counts[1][3].f32==hc.f32_bytes,"510_source_counter");}
            else require(cost_counts[1][3].f32==0,"510_baseline_counter");
        }
        /* Every warmup and measured call checks complete cells/IDs/dots BYTE. */
        for(int j=0;j<m->c.vocab;j++)require(isfinite(old[j])&&isfinite(hybrid[j]),"510_output_finite");
        size_t offset=(size_t)i*m->c.vocab,bytes=(size_t)m->c.vocab*4;
        if(pass==-2){memcpy(gold_old+offset,old,bytes);memcpy(gold_hybrid+offset,hybrid,bytes);
            memcpy(gold_ids+(size_t)i*8,hybrid_ids510,32);memcpy(gold_dots+(size_t)i*8,hybrid_dots510,64);}
        else require(!memcmp(gold_old+offset,old,bytes)&&!memcmp(gold_hybrid+offset,hybrid,bytes)&&
            !memcmp(gold_ids+(size_t)i*8,hybrid_ids510,32)&&!memcmp(gold_dots+(size_t)i*8,hybrid_dots510,64),"510_repeated_complete_BYTE");
        if(pass==0){fwrite(old,4,m->c.vocab,out);fwrite(hybrid,4,m->c.vocab,out);fwrite(hybrid_ids510,4,8,out);fwrite(hybrid_dots510,8,8,out);}
        printf("{\"pass\":%d,\"position\":%u,\"case\":%u,\"step\":%u,\"order\":\"%s\",\"old_seconds\":%.9f,\"hybrid_seconds\":%.9f,\"old_matrix_seconds\":%.9f,\"hybrid_matrix_seconds\":%.9f,\"scan_seconds\":%.9f,\"refine_seconds\":%.9f,\"comparisons\":%llu}\n",
            pass,i,cases[i],steps[i],(pass==-1||pass==1)?"BA":"AB",times[0],times[1],matrix[0],matrix[1],hc.scan_seconds,hc.refine_seconds,(unsigned long long)hc.comparisons);fflush(stdout);
    }
    require(fclose(out)==0,"510_output_close");cost_enabled=0;
    printf("{\"terminal\":true,\"positions\":996,\"warmups\":2,\"repetitions\":3,\"I8_calls\":9960,\"hybrid_calls\":4980,\"source_rows\":39840,\"source_MACs\":30597120,\"extra_f32_logical_bytes\":122388480,\"load_seconds\":%.9f",loaded-initial);worker_json();puts("}");
    free(old);free(hybrid);free(gold_old);free(gold_hybrid);free(gold_ids);free(gold_dots);free(cases);free(steps);free(states);unload(m);return 0;
}
