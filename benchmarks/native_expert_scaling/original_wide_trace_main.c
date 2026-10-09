/* Diagnostic consumer only; math remains in the observed original bodies. */
int main(int argc,char **argv){
    if(argc!=5)fail("usage: model three_requests trace heads");
    if(!SetProcessAffinityMask(GetCurrentProcess(),(DWORD_PTR)1))fail("affinity");
    if(fesetround(FE_TONEAREST))fail("round mode");
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    load_model(argv[1]);FILE *q=fopen(argv[2],"rb");uint32_t nr,n=0,*ids=NULL;
    if(!q||fread(&nr,4,1,q)!=1||nr!=3)fail("request header");
    for(int i=0;i<3;i++){
        uint32_t count;if(fread(&count,4,1,q)!=1||!count||count>16384)fail("length");
        uint32_t *row=malloc(count*4);if(!row||fread(row,4,count,q)!=count)fail("IDs");
        if(i==1){ids=row;n=count;}else free(row);
    }
    if(fgetc(q)!=EOF||n!=1507)fail("request extent");fclose(q);
    FILE *trace=output(argv[3]),*heads=output(argv[4]);float *lg=malloc(V*4);
    if(!lg)fail("head allocation");state_reset();
    for(uint32_t t=0;t<n;t++){
        if(ids[t]>=(uint32_t)V)fail("ID bounds");memset(tr,0,sizeof(tr));
        forward_token(ids[t],lg,1,0,1,NULL);
        for(int v=0;v<V;v++)if(!isfinite(lg[v]))fail("nonfinite head");
        put(trace,tr,sizeof(tr));put(heads,lg,V*4);
    }
    if(fclose(trace)||fclose(heads))fail("close outputs");return 0;
}
