/* Diagnostic only: frozen454 reference plus separately timed equivalent stages. */
#define main meth455_retained454_main
#include "meth454_switch_sparse_wo_cpu.c"
#undef main
#define K455 10
typedef struct {int enabled;double last,seconds[K455];} Profile455;
enum {BASIS455,WIQ455,BUILD455,WI455,RELU455,WOQ455,SCAN455,COLUMN455,SCALE455,ORIGINALWO455};
static void begin455(Profile455 *p){if(p->enabled)p->last=cost_clock();}
static void stamp455(Profile455 *p,int k){if(p->enabled){double now=cost_clock();p->seconds[k]=now-p->last;p->last=now;}}

/* Same original integer-dot/scales/parallel loop; quantizer separated explicitly.
   Its stack placement differs from mv. Mode1 measures that separation bias. */
static void original455(Model *model,const Trace454 *t,Work454 *w,Profile455 *p,float *raw){
    Block *b=model->dec+11;Matrix wi=b->expertwi[t->id],wo=b->expertwo[t->id];
    require(wi.q&&wo.q&&wi.rows==M454&&wi.cols==D454&&wo.rows==D454&&wo.cols==M454,"split_original_dimensions");
    begin455(p);
    w->wi_scale=head_activation_codes(t->input,w->wi_codes,D454);stamp455(p,WIQ455);
    #pragma omp parallel for schedule(static) if(M454>64)
    for(int row=0;row<M454;row++){int64_t value=head_integer_dot(wi.q+(size_t)row*D454,w->wi_codes,D454);w->up[row]=(float)(((double)value*(double)wi.scales[row])*(double)w->wi_scale);}
    stamp455(p,WI455);
    if(raw){memcpy(raw,w->up,M454*4);memcpy(w->transformed,t->input,D454*4);}
    for(int j=0;j<M454;j++)if(w->up[j]<0.f)w->up[j]=0.f;stamp455(p,RELU455);
    w->wo_scale=head_activation_codes(w->up,w->wo_codes,M454);stamp455(p,WOQ455);
    #pragma omp parallel for schedule(static) if(D454>64)
    for(int row=0;row<D454;row++){int64_t value=head_integer_dot(wo.q+(size_t)row*M454,w->wo_codes,M454);w->down[row]=(float)(((double)value*(double)wo.scales[row])*(double)w->wo_scale);}
    stamp455(p,ORIGINALWO455);
    if(raw){w->nonzero=0;for(int j=0;j<M454;j++)w->nonzero+=w->wo_codes[j]!=0;}
}
/* Exact454 LUT application with build removed; same ascending block order. */
static void lookup455(Pair454 pair,Work454 *w){
    memset(w->weighted,0,sizeof(w->weighted));
    for(int b=0;b<D454/64;b++)for(int r=0;r<M454;r++){
        const uint8_t *q=pair.wi+(size_t)r*PAIRS454+b*32;int32_t part=0;
        for(int p=0;p<32;p++)part+=w->table[(b*32+p)*256+q[p]];
        w->weighted[r]+=(double)part*(double)pair.ws[r*(D454/64)+b];
    }
    for(int r=0;r<M454;r++)w->up[r]=(float)(w->weighted[r]*(double)w->wi_scale);
}
/* Exact454 sparse integer accumulation; scan and final scale are separate stages. */
static void column455(Pair454 pair,Work454 *w){
    memset(w->total,0,sizeof(w->total));
    for(uint32_t start=0;start<w->nonzero;start+=CHUNK454){
        uint32_t end=start+CHUNK454;if(end>w->nonzero)end=w->nonzero;memset(w->partial,0,sizeof(w->partial));
        for(uint32_t p=start;p<end;p++){
            uint32_t j=w->indices[p];const int8_t *q=pair.wo+(size_t)j*D454;
            __m256i c=_mm256_set1_epi32((int32_t)w->wo_codes[j]);
            for(int r=0;r<D454;r+=8){
                __m256i v=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q+r)));
                __m256i partial=_mm256_loadu_si256((const __m256i *)(w->partial+r));
                _mm256_storeu_si256((__m256i *)(w->partial+r),_mm256_add_epi32(partial,_mm256_mullo_epi32(v,c)));
            }
        }
        for(int r=0;r<D454;r+=4){
            __m256i part=_mm256_cvtepi32_epi64(_mm_loadu_si128((const __m128i *)(w->partial+r)));
            __m256i total=_mm256_loadu_si256((const __m256i *)(w->total+r));
            _mm256_storeu_si256((__m256i *)(w->total+r),_mm256_add_epi64(total,part));
        }
    }
}
static void candidate455(const Bank454 *bank,const Trace454 *t,uint32_t id,Work454 *w,int lut,Profile455 *p,float *raw){
    Pair454 pair=bank->expert[id];begin455(p);
    basis454(t->input,bank->signs,w);stamp455(p,BASIS455);
    w->wi_scale=head_activation_codes(w->transformed,w->wi_codes,D454);stamp455(p,WIQ455);
    if(lut){build454(w);stamp455(p,BUILD455);lookup455(pair,w);}else wi454(pair,w,0);
    stamp455(p,WI455);
    if(raw)memcpy(raw,w->up,M454*4);
    for(int j=0;j<M454;j++)if(w->up[j]<0.f)w->up[j]=0.f;stamp455(p,RELU455);
    w->wo_scale=head_activation_codes(w->up,w->wo_codes,M454);stamp455(p,WOQ455);
    w->nonzero=0;for(int j=0;j<M454;j++)if(w->wo_codes[j]!=0)w->indices[w->nonzero++]=(uint16_t)j;stamp455(p,SCAN455);
    column455(pair,w);stamp455(p,COLUMN455);
    for(int r=0;r<D454;r++)w->down[r]=(float)(((double)w->total[r]*(double)pair.os[r])*(double)w->wo_scale);stamp455(p,SCALE455);
}
static void dispatch455(int arm,int mode,Model *model,const Bank454 *bank,const Trace454 *t,Work454 *w,Profile455 *p,float *raw){
    if(mode==0){if(arm==0)original454(model,t,w,raw);else candidate454(bank,t,t->id,w,arm==2,raw);}
    else if(arm==0)original455(model,t,w,p,raw);else candidate455(bank,t,t->id,w,arm==2,p,raw);
}
static void tiny455(Work454 *w){
    tiny454(w);int8_t *columns=alloc((size_t)M454*D454,1);float scales[D454];Pair454 pair={NULL,NULL,columns,scales};
    for(int j=0;j<M454;j++)for(int r=0;r<D454;r++)columns[(size_t)j*D454+r]=r%3==0?127:r%3==1?-127:(j%2?-127:127);
    for(int r=0;r<D454;r++)scales[r]=r%3==0?.5f:r%3==1?2.f:.25f;
    for(int mode=0;mode<5;mode++){
        w->nonzero=0;for(int j=0;j<M454;j++){w->wo_codes[j]=mode==0?32767:mode==1?-32767:mode==2?(j%2?-32767:32767):mode==3?0:(j==M454-1?-32767:0);if(w->wo_codes[j])w->indices[w->nonzero++]=(uint16_t)j;}
        column455(pair,w);
        for(int r=0;r<D454;r++){int64_t value=0;for(int j=0;j<M454;j++)value+=(int64_t)columns[(size_t)j*D454+r]*(int64_t)w->wo_codes[j];require(value==w->total[r],"split_sparse_full_I64_extrema");}
    }free(columns);
}
int main(int argc,char **argv){
    require(argc==7&&(!strcmp(argv[1],"--qualify")||!strcmp(argv[1],"--profile")),"455mode manifest bank trace gold output");
    require(fegetround()==FE_TONEAREST,"rounding");worker_setup(1);cost_enabled=cost_profile=0;
    LARGE_INTEGER freq;require(QueryPerformanceFrequency(&freq),"frequency");cost_frequency=(double)freq.QuadPart;double start=cost_clock();
    Model *model=load(argv[2]);require(model->c.d==D454&&model->c.ff==M454&&model->c.n==E454&&model->dec[11].sparse,"original_shape");
    Bank454 bank=bank454(argv[3]);Trace454 *trace=trace454(argv[4]);Work454 *w=alloc(1,sizeof(Work454));int qualify=!strcmp(argv[1],"--qualify");
    double empty[3]={0.};
    if(qualify){
        tiny455(w);FILE *f=fopen(argv[6],"wb");require(f!=NULL,"455primal_output");uint32_t h[5]={N454,D454,M454,10*N454,26144};
        require(fwrite("MC455Q01",1,8,f)==8&&fwrite(h,4,5,f)==5,"455primal_header");float raw[M454];
        for(uint32_t mode=1;mode<=2;mode++)for(int arm=0;arm<3;arm++)for(int wrong=0;wrong<(arm==0?1:2);wrong++)for(int i=0;i<N454;i++){
            uint32_t id=(trace[i].id+wrong)%E454;Profile455 p={0};p.enabled=mode==2;
            if(arm==0)original455(model,trace+i,w,&p,raw);else candidate455(&bank,trace+i,id,w,arm==2,&p,raw);
            require(fwrite(&mode,4,1,f)==1,"455primal_mode");save454(f,i,arm,wrong,id,w,raw);
        }require(fclose(f)==0,"455primal_close");
    }else{
        float *gold=gold454(argv[5],trace);FILE *f=fopen(argv[6],"w");require(f!=NULL,"455profile_output");
        for(int arm=0;arm<3;arm++){int stages=arm==0?5:arm==1?8:9;
            for(int repeat=0;repeat<10000;repeat++){Profile455 p={0};p.enabled=1;double before=cost_clock();begin455(&p);for(int k=0;k<stages;k++)stamp455(&p,k);empty[arm]+=cost_clock()-before;for(int k=0;k<stages;k++)checksum454+=p.seconds[k];}empty[arm]/=10000.;
        }
        for(int rep=-2;rep<5;rep++)for(int slot=0;slot<9;slot++){
            int combination=((rep+2)%9+slot)%9,arm=combination/3,mode=combination%3;
            for(int i=0;i<N454;i++){
                Profile455 p={0};p.enabled=mode==2;
                double before=cost_clock();dispatch455(arm,mode,model,&bank,trace+i,w,&p,NULL);double elapsed=cost_clock()-before;
                require(isfinite(elapsed)&&elapsed>0.,"455elapsed");require(!memcmp(w->down,gold+((size_t)arm*N454+i)*D454,D454*4),"455timed_output_BYTE_exact454");checksum454+=(double)w->down[i%D454];
                double sum=0.;for(int k=0;k<K455;k++){require(isfinite(p.seconds[k])&&p.seconds[k]>=0.,"455component_time");sum+=p.seconds[k];}require(sum<=elapsed+1e-9,"455partition_within_whole");
                require(fprintf(f,"{\"repetition\":%d,\"arm\":%d,\"mode\":%d,\"query\":%d,\"book\":%d,\"id\":%u,\"nonzero\":%d,\"seconds\":%.12f,\"components\":[",rep,arm,mode,i,trace[i].book,trace[i].id,arm==0?-1:(int)w->nonzero,elapsed)>0,"455profile_write");
                for(int k=0;k<K455;k++)require(fprintf(f,"%s%.12f",k?",":"",p.seconds[k])>0,"455component_write");require(fprintf(f,"]}\n")>0,"455profile_newline");
            }require(cost_clock()-start<=600.,"455native600_stop");
        }require(fclose(f)==0,"455profile_close");free(gold);
    }
    PROCESS_MEMORY_COUNTERS memory={0};memory.cb=sizeof(memory);require(GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))&&memory.PeakWorkingSetSize<=((SIZE_T)2<<30),"455native_peak2GiB");
    printf("{\"mode\":\"%s\",\"native_seconds\":%.9f,\"native_peak_bytes\":%llu,\"workspace_bytes\":%llu,\"timer_frequency\":%.0f,\"empty_boundary_sequence_mean_seconds\":[%.12f,%.12f,%.12f],\"tiny_qualified\":%s,\"checksum\":%.17g",argv[1],cost_clock()-start,(unsigned long long)memory.PeakWorkingSetSize,(unsigned long long)sizeof(Work454),cost_frequency,empty[0],empty[1],empty[2],qualify?"true":"false",(double)checksum454);worker_json();printf("}\n");
    UnmapViewOfFile(bank.mapped.data);CloseHandle(bank.mapped.map);CloseHandle(bank.mapped.file);free(trace);free(w);unload(model);return 0;
}
