/* One exact physical permutation. Retained original/direct references unchanged. */
#define main meth456_retained454_main
#include "meth454_switch_sparse_wo_cpu.c"
#undef main
#define TILE456 16
#define BLOCKS456 (D454/64)
#define TILES456 (M454/TILE456)
_Static_assert(64*7*32767==14679616,"tile_block_signed_bound");
_Static_assert(M454%TILE456==0,"tile_no_padding");
typedef struct {__m256i first,second;} Dot456;
static Dot456 dot456(const uint8_t *q,const int16_t *x){
    Dot456 result={_mm256_setzero_si256(),_mm256_setzero_si256()};
    __m128i mask=_mm_set1_epi8(15),sign=_mm_set1_epi8(8);
    for(int p=0;p<32;p++){
        __m128i bytes=_mm_loadu_si128((const __m128i *)(q+p*TILE456));
        __m128i low=_mm_and_si128(bytes,mask),high=_mm_and_si128(_mm_srli_epi16(bytes,4),mask);
        low=_mm_sub_epi8(_mm_xor_si128(low,sign),sign);high=_mm_sub_epi8(_mm_xor_si128(high,sign),sign);
        __m256i a=_mm256_cvtepi8_epi16(_mm_unpacklo_epi8(low,high));
        __m256i b=_mm256_cvtepi8_epi16(_mm_unpackhi_epi8(low,high));
        uint32_t bits=(uint32_t)(uint16_t)x[2*p]|((uint32_t)(uint16_t)x[2*p+1]<<16);int32_t pair;memcpy(&pair,&bits,4);
        __m256i activation=_mm256_set1_epi32(pair);
        result.first=_mm256_add_epi32(result.first,_mm256_madd_epi16(a,activation));
        result.second=_mm256_add_epi32(result.second,_mm256_madd_epi16(b,activation));
    }return result;
}
static void wi456(Pair454 pair,Work454 *w){
    __m256d alpha=_mm256_set1_pd((double)w->wi_scale);
    for(int tile=0;tile<TILES456;tile++){
        __m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd(),v2=_mm256_setzero_pd(),v3=_mm256_setzero_pd();
        for(int block=0;block<BLOCKS456;block++){
            Dot456 part=dot456(pair.wi+((size_t)tile*BLOCKS456+block)*32*TILE456,w->wi_codes+block*64);
            const float *s=pair.ws+((size_t)tile*BLOCKS456+block)*TILE456;
            v0=_mm256_add_pd(v0,_mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(part.first)),_mm256_cvtps_pd(_mm_loadu_ps(s))));
            v1=_mm256_add_pd(v1,_mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(part.first,1)),_mm256_cvtps_pd(_mm_loadu_ps(s+4))));
            v2=_mm256_add_pd(v2,_mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(part.second)),_mm256_cvtps_pd(_mm_loadu_ps(s+8))));
            v3=_mm256_add_pd(v3,_mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(part.second,1)),_mm256_cvtps_pd(_mm_loadu_ps(s+12))));
        }
        _mm_storeu_ps(w->up+tile*TILE456,_mm256_cvtpd_ps(_mm256_mul_pd(v0,alpha)));
        _mm_storeu_ps(w->up+tile*TILE456+4,_mm256_cvtpd_ps(_mm256_mul_pd(v1,alpha)));
        _mm_storeu_ps(w->up+tile*TILE456+8,_mm256_cvtpd_ps(_mm256_mul_pd(v2,alpha)));
        _mm_storeu_ps(w->up+tile*TILE456+12,_mm256_cvtpd_ps(_mm256_mul_pd(v3,alpha)));
    }
}
static Bank454 bank456(const char *path){
    Bank454 bank={0};bank.mapped=mapping454(path);unsigned char *p=bank.mapped.data;uint32_t h[5];memcpy(h,p+8,20);
    require(!memcmp(p,"MC456B01",8)&&h[0]==D454&&h[1]==M454&&h[2]==E454&&h[3]==HEADER454&&h[4]==TILE456,"tiled_header");
    require(bank.mapped.bytes==HEADER454+D454+STRIDE454*E454,"tiled_exact_size");bank.signs=(const int8_t *)(p+HEADER454);
    for(int i=0;i<D454;i++)require(bank.signs[i]==1||bank.signs[i]==-1,"tiled_signs");
    for(int e=0;e<E454;e++){
        const unsigned char *q=p+HEADER454+D454+STRIDE454*e;Pair454 *pair=bank.expert+e;
        pair->wi=q;pair->ws=(const float *)(q+1179648);pair->wo=(const int8_t *)(q+1327104);pair->os=(const float *)(q+3686400);
        require((uintptr_t)pair->wi%64==0&&(uintptr_t)pair->ws%64==0&&(uintptr_t)pair->wo%64==0&&(uintptr_t)pair->os%64==0,"tiled64_alignment");
        for(size_t i=0;i<1179648;i++)require(valid454(pair->wi[i]),"tiled_minus8_nibble");
        for(size_t i=0;i<2359296;i++)require(pair->wo[i]!=-128,"tiled_minus128");
        for(int i=0;i<M454*BLOCKS456;i++)require(isfinite(pair->ws[i])&&pair->ws[i]>0.f,"tiled_WI_scale");
        for(int i=0;i<D454;i++)require(isfinite(pair->os[i])&&pair->os[i]>0.f,"tiled_WO_scale");
    }return bank;
}
static void candidate456(const Bank454 *bank,const Trace454 *t,uint32_t id,Work454 *w,float *raw){
    basis454(t->input,bank->signs,w);w->wi_scale=head_activation_codes(w->transformed,w->wi_codes,D454);
    wi456(bank->expert[id],w);if(raw)memcpy(raw,w->up,M454*4);
    for(int j=0;j<M454;j++)if(w->up[j]<0.f)w->up[j]=0.f;
    w->wo_scale=head_activation_codes(w->up,w->wo_codes,M454);sparse454(bank->expert[id],w);
}
static void tiny456(Work454 *w){
    tiny454(w);uint8_t q[32*TILE456];int16_t x[64];int32_t actual[TILE456];int levels[]={-32767,0,32767};
    for(int a=-7;a<=7;a++)for(int b=-7;b<=7;b++)for(int ix=0;ix<3;ix++)for(int iy=0;iy<3;iy++){
        for(int p=0;p<32;p++){
            x[2*p]=(int16_t)levels[ix];x[2*p+1]=(int16_t)levels[iy];
            for(int r=0;r<TILE456;r++){int c=(a+7+r)%15-7,d=(b+7+2*r)%15-7;q[p*TILE456+r]=(uint8_t)((c&15)|((d&15)<<4));}
        }
        Dot456 part=dot456(q,x);_mm256_storeu_si256((__m256i *)actual,part.first);_mm256_storeu_si256((__m256i *)(actual+8),part.second);
        for(int r=0;r<TILE456;r++){int64_t c=(a+7+r)%15-7,d=(b+7+2*r)%15-7,expected=32*(c*levels[ix]+d*levels[iy]);require(actual[r]==expected,"tiny_tiled2025_each16_lane");}
    }
    for(int p=0;p<32;p++){x[2*p]=(int16_t)levels[p%3];x[2*p+1]=(int16_t)levels[(p+1)%3];for(int r=0;r<TILE456;r++){int a=(p*3+r*7)%15-7,b=(p*11+r*2)%15-7;q[p*TILE456+r]=(uint8_t)((a&15)|((b&15)<<4));}}
    Dot456 part=dot456(q,x);_mm256_storeu_si256((__m256i *)actual,part.first);_mm256_storeu_si256((__m256i *)(actual+8),part.second);
    for(int r=0;r<TILE456;r++){int64_t expected=0;for(int p=0;p<32;p++)expected+=(int64_t)((p*3+r*7)%15-7)*x[2*p]+(int64_t)((p*11+r*2)%15-7)*x[2*p+1];require(actual[r]==expected,"tiny_tiled_mixed_scalar_I64");}
}
static void close456(Bank454 *bank){UnmapViewOfFile(bank->mapped.data);CloseHandle(bank->mapped.map);CloseHandle(bank->mapped.file);}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--bad-nibble")){require(valid454(0x08),"tiled_minus8_nibble");return 0;}
    require(argc==8&&(!strcmp(argv[1],"--qualify")||!strcmp(argv[1],"--bench")),"456mode manifest originalbank tilebank trace gold output");
    require(fegetround()==FE_TONEAREST,"rounding");worker_setup(1);cost_enabled=cost_profile=0;
    LARGE_INTEGER freq;require(QueryPerformanceFrequency(&freq),"frequency");cost_frequency=(double)freq.QuadPart;double start=cost_clock();
    Model *model=load(argv[2]);require(model->c.d==D454&&model->c.ff==M454&&model->c.n==E454&&model->dec[11].sparse,"original_shape");
    Bank454 original=bank454(argv[3]),tiled=bank456(argv[4]);require(!memcmp(original.signs,tiled.signs,D454),"tiled_signs_exact_original");
    Trace454 *trace=trace454(argv[5]);Work454 *w=alloc(1,sizeof(Work454));int qualify=!strcmp(argv[1],"--qualify");
    if(qualify){
        tiny456(w);FILE *f=fopen(argv[7],"wb");require(f!=NULL,"456primal_output");uint32_t h[4]={N454,D454,M454,5*N454};
        require(fwrite("MC456Q01",1,8,f)==8&&fwrite(h,4,4,f)==4,"456primal_header");float raw[M454];
        for(int arm=0;arm<3;arm++)for(int wrong=0;wrong<(arm==0?1:2);wrong++)for(int i=0;i<N454;i++){
            uint32_t id=(trace[i].id+wrong)%E454;
            if(arm==0)original454(model,trace+i,w,raw);else if(arm==1)candidate454(&original,trace+i,id,w,0,raw);else candidate456(&tiled,trace+i,id,w,raw);
            save454(f,i,arm,wrong,id,w,raw);
        }require(fclose(f)==0,"456primal_close");
    }else{
        float *gold=gold454(argv[6],trace);FILE *f=fopen(argv[7],"w");require(f!=NULL,"456timing_output");
        for(int rep=-2;rep<5;rep++)for(int slot=0;slot<3;slot++){
            int arm=((rep+2)%3+slot)%3;
            for(int i=0;i<N454;i++){
                double before=cost_clock();if(arm==0)original454(model,trace+i,w,NULL);else if(arm==1)candidate454(&original,trace+i,trace[i].id,w,0,NULL);else candidate456(&tiled,trace+i,trace[i].id,w,NULL);
                double elapsed=cost_clock()-before;require(isfinite(elapsed)&&elapsed>0.,"456query_elapsed");
                require(!memcmp(w->down,gold+((size_t)arm*N454+i)*D454,D454*4),"456timed_output_BYTE_exact454");checksum454+=(double)w->down[i%D454];
                require(fprintf(f,"{\"repetition\":%d,\"arm\":%d,\"query\":%d,\"book\":%d,\"id\":%u,\"seconds\":%.12f}\n",rep,arm,i,trace[i].book,trace[i].id,elapsed)>0,"456timing_write");
            }require(cost_clock()-start<=600.,"456native600_stop");
        }require(fclose(f)==0,"456timing_close");free(gold);
    }
    PROCESS_MEMORY_COUNTERS memory={0};memory.cb=sizeof(memory);require(GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))&&memory.PeakWorkingSetSize<=((SIZE_T)2<<30),"456native_peak2GiB");
    printf("{\"mode\":\"%s\",\"native_seconds\":%.9f,\"native_peak_bytes\":%llu,\"workspace_bytes\":%llu,\"original_bank_mapped_bytes\":%llu,\"tiled_bank_mapped_bytes\":%llu,\"tile_width\":16,\"LUT_entries_built_per_timed_query\":0,\"tiny_qualified\":%s,\"checksum\":%.17g",argv[1],cost_clock()-start,(unsigned long long)memory.PeakWorkingSetSize,(unsigned long long)sizeof(Work454),(unsigned long long)original.mapped.bytes,(unsigned long long)tiled.mapped.bytes,qualify?"true":"false",(double)checksum454);worker_json();printf("}\n");
    close456(&original);close456(&tiled);free(trace);free(w);unload(model);return 0;
}
