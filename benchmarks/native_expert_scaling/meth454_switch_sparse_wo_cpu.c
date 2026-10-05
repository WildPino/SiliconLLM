/* New candidate kernels; original baseline included unchanged, not reimplemented. */
#include "meth374_switch_physical_workers.c"
#include <psapi.h>
#define D454 768
#define M454 3072
#define E454 128
#define N454 336
#define STRIDE454 3689472ULL
#define HEADER454 4096
#define PAIRS454 (D454/2)
#define CHUNK454 512
_Static_assert(CHUNK454*127*32767==2130641408,"signed_partial_bound");
typedef struct {const uint8_t *wi;const float *ws;const int8_t *wo;const float *os;} Pair454;
typedef struct {Mapping mapped;const int8_t *signs;Pair454 expert[E454];} Bank454;
typedef struct {float input[D454];uint32_t id;int32_t book;} Trace454;
_Static_assert(sizeof(Trace454)==3080,"trace_wire_size");
typedef struct {
    double basis[D454],weighted[M454];float transformed[D454],up[M454],down[D454];
    int16_t wi_codes[D454],wo_codes[M454];uint16_t indices[M454];
    int32_t partial[D454],table[PAIRS454*256];int64_t total[D454];
    float wi_scale,wo_scale;uint32_t nonzero;
} Work454;
static volatile double checksum454;

static int valid454(uint8_t q){return (q&15)!=8&&(q>>4)!=8;}
static int nibble454(int n){return n>=8?n-16:n;}
static int32_t direct64_454(const uint8_t *q,const int16_t *x){
    __m256i acc=_mm256_setzero_si256();__m128i mask=_mm_set1_epi8(15),sign=_mm_set1_epi8(8);
    for(int p=0;p<32;p+=16){
        __m128i bytes=_mm_loadu_si128((const __m128i *)(q+p));
        __m128i low=_mm_and_si128(bytes,mask),high=_mm_and_si128(_mm_srli_epi16(bytes,4),mask);
        low=_mm_sub_epi8(_mm_xor_si128(low,sign),sign);high=_mm_sub_epi8(_mm_xor_si128(high,sign),sign);
        __m256i a=_mm256_cvtepi8_epi16(_mm_unpacklo_epi8(low,high));
        __m256i b=_mm256_cvtepi8_epi16(_mm_unpackhi_epi8(low,high));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(a,_mm256_loadu_si256((const __m256i *)(x+2*p))));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(b,_mm256_loadu_si256((const __m256i *)(x+2*p+16))));
    }
    int32_t values[8];_mm256_storeu_si256((__m256i *)values,acc);int32_t total=0;
    for(int r=0;r<8;r++)total+=values[r];return total;
}
static void build454(Work454 *w){
    for(int p=0;p<PAIRS454;p++)for(int q=0;q<256;q++)
        w->table[p*256+q]=valid454((uint8_t)q)?nibble454(q&15)*(int32_t)w->wi_codes[2*p]+nibble454(q>>4)*(int32_t)w->wi_codes[2*p+1]:INT32_MIN;
}
static void wi454(Pair454 pair,Work454 *w,int lut){
    if(!lut){
        for(int r=0;r<M454;r++){
            double value=0.;for(int b=0;b<D454/64;b++){
                int32_t part=direct64_454(pair.wi+(size_t)r*PAIRS454+b*32,w->wi_codes+b*64);
                value+=(double)part*(double)pair.ws[r*(D454/64)+b];
            }w->up[r]=(float)(value*(double)w->wi_scale);
        }
    }else{
        build454(w);memset(w->weighted,0,sizeof(w->weighted));
        /* A complete block's tables fit in32768B; residency is not assumed. */
        for(int b=0;b<D454/64;b++)for(int r=0;r<M454;r++){
            const uint8_t *q=pair.wi+(size_t)r*PAIRS454+b*32;int32_t part=0;
            for(int p=0;p<32;p++)part+=w->table[(b*32+p)*256+q[p]];
            w->weighted[r]+=(double)part*(double)pair.ws[r*(D454/64)+b];
        }
        for(int r=0;r<M454;r++)w->up[r]=(float)(w->weighted[r]*(double)w->wi_scale);
    }
}
static void basis454(const float *x,const int8_t *signs,Work454 *w){
    for(int i=0;i<D454;i++)w->basis[i]=(double)x[i]*(double)signs[i];
    for(int start=0;start<D454;start+=256)for(int step=1;step<256;step*=2)
        for(int base=start;base<start+256;base+=2*step)for(int j=0;j<step;j++){
            double a=w->basis[base+j],b=w->basis[base+j+step];
            w->basis[base+j]=a+b;w->basis[base+j+step]=a-b;
        }
    for(int i=0;i<D454;i++)w->transformed[i]=(float)(w->basis[i]/16.);
}
static void sparse454(Pair454 pair,Work454 *w){
    w->nonzero=0;for(int j=0;j<M454;j++)if(w->wo_codes[j]!=0)w->indices[w->nonzero++]=(uint16_t)j;
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
    for(int r=0;r<D454;r++)w->down[r]=(float)(((double)w->total[r]*(double)pair.os[r])*(double)w->wo_scale);
}
static void original454(Model *m,const Trace454 *t,Work454 *w,float *raw){
    Block *b=m->dec+11;mv(b->expertwi[t->id],t->input,w->up,M454,D454);
    if(raw){memcpy(raw,w->up,M454*4);memcpy(w->transformed,t->input,D454*4);w->wi_scale=head_activation_codes(t->input,w->wi_codes,D454);}
    for(int j=0;j<M454;j++)if(w->up[j]<0.f)w->up[j]=0.f;
    mv(b->expertwo[t->id],w->up,w->down,D454,M454);
    if(raw){w->wo_scale=head_activation_codes(w->up,w->wo_codes,M454);w->nonzero=0;for(int j=0;j<M454;j++)w->nonzero+=w->wo_codes[j]!=0;}
}
static void candidate454(const Bank454 *bank,const Trace454 *t,uint32_t id,Work454 *w,int lut,float *raw){
    basis454(t->input,bank->signs,w);w->wi_scale=head_activation_codes(w->transformed,w->wi_codes,D454);
    wi454(bank->expert[id],w,lut);if(raw)memcpy(raw,w->up,M454*4);
    for(int j=0;j<M454;j++)if(w->up[j]<0.f)w->up[j]=0.f;
    w->wo_scale=head_activation_codes(w->up,w->wo_codes,M454);sparse454(bank->expert[id],w);
}
static Mapping mapping454(const char *path){
    Mapping m={0};m.file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    require(m.file!=INVALID_HANDLE_VALUE,"candidate_file");LARGE_INTEGER size;require(GetFileSizeEx(m.file,&size)&&size.QuadPart>0,"candidate_size");m.bytes=size.QuadPart;
    m.map=CreateFileMappingA(m.file,NULL,PAGE_READONLY,0,0,NULL);require(m.map!=NULL,"candidate_map");
    m.data=MapViewOfFile(m.map,FILE_MAP_READ,0,0,0);require(m.data!=NULL,"candidate_view");return m;
}
static Bank454 bank454(const char *path){
    Bank454 bank={0};bank.mapped=mapping454(path);unsigned char *p=bank.mapped.data;uint32_t h[4];memcpy(h,p+8,16);
    require(!memcmp(p,"MC454B01",8)&&h[0]==D454&&h[1]==M454&&h[2]==E454&&h[3]==HEADER454,"candidate_header");
    require(bank.mapped.bytes==HEADER454+D454+STRIDE454*E454,"candidate_exact_size");bank.signs=(const int8_t *)(p+HEADER454);
    for(int i=0;i<D454;i++)require(bank.signs[i]==1||bank.signs[i]==-1,"candidate_signs");
    for(int e=0;e<E454;e++){
        const unsigned char *q=p+HEADER454+D454+STRIDE454*e;Pair454 *pair=bank.expert+e;
        pair->wi=q;pair->ws=(const float *)(q+1179648);pair->wo=(const int8_t *)(q+1327104);pair->os=(const float *)(q+3686400);
        require((uintptr_t)pair->wi%64==0&&(uintptr_t)pair->ws%64==0&&(uintptr_t)pair->wo%64==0&&(uintptr_t)pair->os%64==0,"candidate64_alignment");
        for(size_t i=0;i<1179648;i++)require(valid454(pair->wi[i]),"candidate_minus8_nibble");
        for(size_t i=0;i<2359296;i++)require(pair->wo[i]!=-128,"candidate_minus128");
        for(int i=0;i<M454*(D454/64);i++)require(isfinite(pair->ws[i])&&pair->ws[i]>0.f,"candidate_WI_scale");
        for(int i=0;i<D454;i++)require(isfinite(pair->os[i])&&pair->os[i]>0.f,"candidate_WO_scale");
    }return bank;
}
static Trace454 *trace454(const char *path){
    FILE *f=fopen(path,"rb");require(f!=NULL,"trace_open");char magic[8];uint32_t h[5];
    require(fread(magic,1,8,f)==8&&!memcmp(magic,"MC454T01",8)&&fread(h,4,5,f)==5,"trace_header");
    require(h[0]==D454&&h[1]==M454&&h[2]==E454&&h[3]==N454&&h[4]==11,"trace_shape");
    Trace454 *t=alloc(N454,sizeof(Trace454));require(fread(t,sizeof(Trace454),N454,f)==N454&&fgetc(f)==EOF&&fclose(f)==0,"trace_payload");
    for(int i=0;i<N454;i++){require(t[i].id<E454&&t[i].book>=18&&t[i].book<=23,"trace_ids_books");for(int j=0;j<D454;j++)require(isfinite(t[i].input[j]),"trace_input_finite");}return t;
}
static void tiny454(Work454 *w){
    uint8_t packed[32];int16_t activation[64];int levels[]={-32767,0,32767};
    for(int a=-7;a<=7;a++)for(int b=-7;b<=7;b++)for(int x=0;x<3;x++)for(int y=0;y<3;y++){
        uint8_t q=(uint8_t)((a&15)|((b&15)<<4));for(int p=0;p<32;p++){packed[p]=q;activation[2*p]=levels[x];activation[2*p+1]=levels[y];}
        require(direct64_454(packed,activation)==32*(a*levels[x]+b*levels[y]),"tiny_direct2025");
    }
    for(int p=0;p<PAIRS454;p++){w->wi_codes[2*p]=levels[p%3];w->wi_codes[2*p+1]=levels[(p+1)%3];}build454(w);
    for(int p=0;p<PAIRS454;p++)for(int q=0;q<256;q++){
        if(!valid454((uint8_t)q))require(w->table[p*256+q]==INT32_MIN,"tiny_unused_sentinel");
        else{int64_t a=(q&15)>=8?(q&15)-16:(q&15),b=(q>>4)>=8?(q>>4)-16:(q>>4);
            require(w->table[p*256+q]==a*w->wi_codes[2*p]+b*w->wi_codes[2*p+1],"tiny_pair_table98304");}
    }
    require(!valid454(0x08)&&!valid454(0x80)&&!valid454(0x88),"tiny_minus8_fault");
    float zero[D454]={0};require(head_activation_codes(zero,w->wi_codes,D454)==1.f,"tiny_zero_scale");
    int8_t *columns=alloc((size_t)M454*D454,1);float scales[D454];Pair454 pair={NULL,NULL,columns,scales};
    for(int j=0;j<M454;j++)for(int r=0;r<D454;r++)columns[(size_t)j*D454+r]=r%3==0?127:r%3==1?-127:(j%2?-127:127);
    for(int r=0;r<D454;r++)scales[r]=r%3==0?.5f:r%3==1?2.f:.25f;
    for(int mode=0;mode<5;mode++){
        for(int j=0;j<M454;j++)w->wo_codes[j]=mode==0?32767:mode==1?-32767:mode==2?(j%2?-32767:32767):mode==3?0:(j==M454-1?-32767:0);
        w->wo_scale=.25f;sparse454(pair,w);
        for(int r=0;r<D454;r++){int64_t value=0;for(int j=0;j<M454;j++)value+=(int64_t)columns[(size_t)j*D454+r]*(int64_t)w->wo_codes[j];
            float expected=(float)(((double)value*(double)scales[r])*.25);require(value==w->total[r]&&!memcmp(&expected,w->down+r,4),"tiny_sparse_full_I64_scale");}
        require(w->nonzero==(mode==3?0:mode==4?1:M454),"tiny_active_counts");
    }free(columns);
    for(int i=0;i<D454;i++)zero[i]=(float)i*.0009765625f;int8_t signs[D454];for(int i=0;i<D454;i++)signs[i]=i%3?-1:1;
    basis454(zero,signs,w);
    for(int start=0;start<D454;start+=256)for(int r=0;r<256;r++){
        double sum=0.;for(int j=0;j<256;j++){unsigned k=(unsigned)(r&j),parity=0;while(k){parity^=k&1;k>>=1;}sum+=(parity?-1.:1.)*(double)zero[start+j]*(double)signs[start+j];}
        float expected=(float)(sum/16.);require(!memcmp(&expected,w->transformed+start+r,4),"tiny_explicit_Walsh");
    }
}
static void save454(FILE *f,uint32_t query,uint32_t arm,uint32_t wrong,uint32_t id,Work454 *w,const float *raw){
    uint32_t meta[5]={query,arm,wrong,id,w->nonzero};float scales[2]={w->wi_scale,w->wo_scale};
    require(fwrite(meta,4,5,f)==5&&fwrite(scales,4,2,f)==2&&fwrite(w->transformed,4,D454,f)==D454&&fwrite(raw,4,M454,f)==M454&&fwrite(w->down,4,D454,f)==D454&&fwrite(w->wi_codes,2,D454,f)==D454&&fwrite(w->wo_codes,2,M454,f)==M454,"primal_record_write");
}
static float *gold454(const char *path,const Trace454 *trace){
    FILE *f=fopen(path,"rb");require(f!=NULL,"gold_open");char magic[8];uint32_t h[4];
    require(fread(magic,1,8,f)==8&&!memcmp(magic,"MC454Q01",8)&&fread(h,4,4,f)==4&&h[0]==N454&&h[1]==D454&&h[2]==M454&&h[3]==5*N454,"gold_header");
    float *gold=alloc((size_t)3*N454*D454,4);int seen[3][N454]={0};
    for(uint32_t n=0;n<h[3];n++){
        uint32_t m[5];float scales[2];require(fread(m,4,5,f)==5&&fread(scales,4,2,f)==2&&m[0]<N454&&m[1]<3&&m[2]<2,"gold_metadata");
        require(m[3]==(trace[m[0]].id+m[2])%E454&&!(m[1]==0&&m[2]==1),"gold_identity");
        require(fseek(f,(D454+M454)*4,SEEK_CUR)==0,"gold_skip_basis_raw");
        if(!m[2]){require(!seen[m[1]][m[0]]++,"gold_unique");require(fread(gold+((size_t)m[1]*N454+m[0])*D454,4,D454,f)==D454,"gold_down");}
        else require(fseek(f,D454*4,SEEK_CUR)==0,"gold_skip_wrong_down");
        require(fseek(f,(D454+M454)*2,SEEK_CUR)==0,"gold_skip_codes");
    }
    for(int a=0;a<3;a++)for(int i=0;i<N454;i++)require(seen[a][i]==1,"gold_complete");require(fgetc(f)==EOF&&fclose(f)==0,"gold_close");return gold;
}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--bad-nibble")){require(valid454(0x08),"candidate_minus8_nibble");return 0;}
    require(argc==7&&(!strcmp(argv[1],"--qualify")||!strcmp(argv[1],"--bench")),"mode manifest bank trace primal output");
    require(fegetround()==FE_TONEAREST,"rounding");worker_setup(1);cost_enabled=cost_profile=0;
    LARGE_INTEGER freq;require(QueryPerformanceFrequency(&freq),"frequency");cost_frequency=(double)freq.QuadPart;double start=cost_clock();
    Model *model=load(argv[2]);require(model->c.d==D454&&model->c.ff==M454&&model->c.n==E454&&model->dec[11].sparse,"original_shape");
    Bank454 bank=bank454(argv[3]);Trace454 *trace=trace454(argv[4]);Work454 *w=alloc(1,sizeof(Work454));int qualify=!strcmp(argv[1],"--qualify");
    if(qualify){
        tiny454(w);FILE *f=fopen(argv[6],"wb");require(f!=NULL,"primal_output");uint32_t h[4]={N454,D454,M454,5*N454};
        require(fwrite("MC454Q01",1,8,f)==8&&fwrite(h,4,4,f)==4,"primal_header_write");float raw[M454];
        for(int arm=0;arm<3;arm++)for(int wrong=0;wrong<(arm==0?1:2);wrong++)for(int i=0;i<N454;i++){
            uint32_t id=(trace[i].id+wrong)%E454;
            if(arm==0)original454(model,trace+i,w,raw);else candidate454(&bank,trace+i,id,w,arm==2,raw);
            save454(f,i,arm,wrong,id,w,raw);
        }require(fclose(f)==0,"primal_output_close");
    }else{
        float *gold=gold454(argv[5],trace);FILE *f=fopen(argv[6],"w");require(f!=NULL,"timing_output");
        for(int rep=-2;rep<5;rep++)for(int slot=0;slot<3;slot++){
            int arm=((rep+2)%3+slot)%3;
            for(int i=0;i<N454;i++){
                double before=cost_clock();if(arm==0)original454(model,trace+i,w,NULL);else candidate454(&bank,trace+i,trace[i].id,w,arm==2,NULL);
                double elapsed=cost_clock()-before;require(isfinite(elapsed)&&elapsed>0.,"query_elapsed");
                require(!memcmp(w->down,gold+((size_t)arm*N454+i)*D454,D454*4),"timed_output_byte_exact_qualified");checksum454+=(double)w->down[i%D454];
                require(fprintf(f,"{\"repetition\":%d,\"arm\":%d,\"query\":%d,\"book\":%d,\"id\":%u,\"seconds\":%.12f}\n",rep,arm,i,trace[i].book,trace[i].id,elapsed)>0,"timing_write");
            }require(cost_clock()-start<=600.,"native600_stop");
        }require(fclose(f)==0,"timing_output_close");free(gold);
    }
    PROCESS_MEMORY_COUNTERS memory;memset(&memory,0,sizeof(memory));memory.cb=sizeof(memory);
    require(GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))&&memory.PeakWorkingSetSize<=((SIZE_T)2<<30),"native_peak2GiB");
    printf("{\"mode\":\"%s\",\"native_seconds\":%.9f,\"native_peak_bytes\":%llu,\"workspace_bytes\":%llu,\"candidate_mapped_bytes\":%llu,\"table_entries_built_per_LUT_query\":98304,\"table_bytes\":393216,\"lookup_count_per_LUT_query\":1179648,\"column_alignment\":64,\"tiny_qualified\":%s,\"checksum\":%.17g",argv[1],cost_clock()-start,(unsigned long long)memory.PeakWorkingSetSize,(unsigned long long)sizeof(Work454),(unsigned long long)bank.mapped.bytes,qualify?"true":"false",(double)checksum454);worker_json();printf("}\n");
    UnmapViewOfFile(bank.mapped.data);CloseHandle(bank.mapped.map);CloseHandle(bank.mapped.file);free(trace);free(w);unload(model);return 0;
}
