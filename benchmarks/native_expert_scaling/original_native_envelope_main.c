/* Consumer only. Appended to a fresh copy of the original packed wrapper.
   The20 original computational bodies remain byte-identical to extraction. */
static uint32_t consume(float *lg){
    uint32_t best=0;
    for(int v=0;v<V;v++){if(!isfinite(lg[v]))fail("nonfinite logit");if(lg[v]>lg[best])best=(uint32_t)v;}
    for(int l=0;l<L;l++){double sum=0;
        for(int k=0;k<KTOP;k++){if(actual_ids[l][k]<0||actual_ids[l][k]>=E||!isfinite(actual_mass[l][k])||actual_mass[l][k]<0)fail("route support");sum+=actual_mass[l][k];}
        if(fabs(sum-1)>1e-6)fail("route mass");}
    return best;
}
static void component(FILE *r,Tacc *t){
    fprintf(r,"\"components\":{\"scan\":%.17g,\"scan_other\":%.17g,\"swa\":%.17g,\"mlp_including_router\":%.17g,\"router\":%.17g,\"head\":%.17g,\"norm\":%.17g}",
        t->scan,t->scan_other,t->swa,t->mlp,t->router,t->head,t->norm);
}
int main(int argc,char **argv){
    if(argc!=5)fail("usage: model requests parity|speed|profile output_prefix");
    int parity=!strcmp(argv[3],"parity"),profile=!strcmp(argv[3],"profile");
    if(!parity&&!profile&&strcmp(argv[3],"speed"))fail("mode");
    if(!SetProcessAffinityMask(GetCurrentProcess(),(DWORD_PTR)1))fail("affinity");
    if(fesetround(FE_TONEAREST))fail("round mode");
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    double started=now_s();load_model(argv[1]);double load_seconds=now_s()-started;
    FILE *q=fopen(argv[2],"rb");uint32_t requests;
    if(!q||fread(&requests,4,1,q)!=1||requests!=3)fail("request header");
    uint32_t lengths[3],*ids[3];
    for(int i=0;i<3;i++){if(fread(lengths+i,4,1,q)!=1||lengths[i]<1||lengths[i]>16384)fail("request length");
        ids[i]=malloc((size_t)lengths[i]*4);if(!ids[i]||fread(ids[i],4,lengths[i],q)!=lengths[i])fail("request IDs");
        for(uint32_t j=0;j<lengths[i];j++)if(ids[i][j]>=(uint32_t)V)fail("input ID");}
    if(fgetc(q)!=EOF)fail("request extent");fclose(q);
    char path[4096];FILE *logits_out=NULL,*routes_out=NULL;
    if(parity){snprintf(path,sizeof(path),"%s.f32",argv[4]);logits_out=output(path);
        snprintf(path,sizeof(path),"%s.routes",argv[4]);routes_out=output(path);
        snprintf(path,sizeof(path),"%s.witness",argv[4]);witnesses(path);}
    snprintf(path,sizeof(path),"%s.native.json",argv[4]);FILE *r=output(path);
    float *logits=malloc((size_t)V*4);uint32_t sink=0; if(!logits)fail("logits allocation");
    volatile double timer_sink=0;double calibration=now_s();
    for(int i=0;i<500000;i++)timer_sink+=now_s();
    calibration=now_s()-calibration;
    fprintf(r,"{\"DN\":%d,\"DTR\":%d,\"n\":%d,\"V\":%d,\"mode\":\"%s\",\"affinity_mask\":1,\"threads\":1,\"load_seconds\":%.17g,\"timer500000_seconds\":%.17g,\"blob_bytes\":%llu,\"code_bytes\":%llu,\"f32_bytes\":%llu,\"state_router_bytes\":%llu,\"heap_allocated_bytes\":%llu,\"expert_reference_bytes\":0,\"requests\":[",
        DN,DTR,E,V,argv[3],load_seconds,calibration,(unsigned long long)blob_bytes,(unsigned long long)code_bytes,
        (unsigned long long)f32_bytes,(unsigned long long)state_bytes,(unsigned long long)heap_bytes);
    for(int i=0;i<3;i++){
        state_reset();Tacc t={0};double seconds=0,start=now_s();
        for(uint32_t j=0;j<lengths[i];j++){
            double token_start=parity?now_s():0;
            forward_token(ids[i][j],logits,1,0,1,profile?&t:NULL);sink^=consume(logits);
            if(parity){seconds+=now_s()-token_start;put(logits_out,logits,(size_t)V*4);
                for(int l=0;l<L;l++){put(routes_out,actual_ids[l],KTOP*4);put(routes_out,actual_mass[l],KTOP*4);}}
        }
        if(!parity)seconds=now_s()-start;
        if(i)fputc(',',r);
        fprintf(r,"{\"index\":%d,\"tokens\":%u,\"batch1_seconds\":%.17g,\"request_wall_seconds\":%.17g,\"raw_IDs_per_second\":%.17g,",i,lengths[i],seconds,now_s()-start,lengths[i]/seconds);
        component(r,&t);fputc('}',r);
    }
    fprintf(r,"],\"sink\":%u,\"timer_sink_finite\":%s,\"process_seconds\":%.17g}\n",sink,isfinite(timer_sink)?"true":"false",now_s()-started);
    if(parity){fclose(logits_out);fclose(routes_out);}fclose(r);return 0;
}
