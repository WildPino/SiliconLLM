/* New physical composition: unchanged source WI/A16 and zero-only column WO. */
#include "meth374_switch_physical_workers.c"
#include <psapi.h>
#include <io.h>
#include <fcntl.h>
#define D505 768
#define H505 3072
#define E505 128
#define N505 17540
#define HEADER505 4096ULL
#define STRIDE505 4733952ULL
#define CHUNK505 512
_Static_assert(CHUNK505*128*32767==2147418112,"signed_partial_bound");
typedef struct {Mapping mapped; Matrix wi[E505];const int8_t *wo[E505];const float *os[E505];} Bank505;
typedef struct {uint32_t e,accepted;float alpha,x[D505];int16_t q[D505];float p;} Input505;
typedef struct {uint32_t uid,e,nonzero;float hidden[H505];int16_t codes[H505];float alpha,down[D505];} Wire505;
typedef struct {Wire505 wire;uint16_t indices[H505];int32_t partial[D505];int64_t total[D505];} Work505;
_Static_assert(sizeof(Input505)==4624,"input_wire");
_Static_assert(sizeof(Wire505)==21520,"output_wire");
static volatile double checksum505;
static Mapping map505(const char *path){
    Mapping m={0};m.file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    require(m.file!=INVALID_HANDLE_VALUE,"505_file");LARGE_INTEGER size;require(GetFileSizeEx(m.file,&size)&&size.QuadPart>=24,"505_size");m.bytes=size.QuadPart;
    m.map=CreateFileMappingA(m.file,NULL,PAGE_READONLY,0,0,NULL);require(m.map!=NULL,"505_map");
    m.data=MapViewOfFile(m.map,FILE_MAP_READ,0,0,0);require(m.data!=NULL,"505_view");return m;
}
static void close505(Mapping m){require(UnmapViewOfFile(m.data)&&CloseHandle(m.map)&&CloseHandle(m.file),"505_unmap");}
static void header505(Mapping m,const char *magic,uint32_t stride,uint32_t res,uint64_t n){
    uint32_t h[2];uint64_t count;memcpy(h,m.data+8,8);memcpy(&count,m.data+16,8);
    require(!memcmp(m.data,magic,8)&&h[0]==stride&&h[1]==res&&count==n&&m.bytes==24+n*stride,"505_wire_header_size");
}
static Bank505 bank505(const char *path){
    Bank505 b={0};b.mapped=map505(path);unsigned char *p=b.mapped.data;uint32_t h[4];uint64_t stride;
    memcpy(h,p+8,16);memcpy(&stride,p+24,8);
    require(!memcmp(p,"M505BNK1",8)&&h[0]==D505&&h[1]==H505&&h[2]==E505&&h[3]==HEADER505&&stride==STRIDE505&&b.mapped.bytes==HEADER505+E505*STRIDE505,"505_bank_header");
    for(int j=32;j<HEADER505;j++)require(p[j]==0,"505_header_padding");
    for(int e=0;e<E505;e++){
        const unsigned char *q=p+HEADER505+e*STRIDE505;
        b.wi[e]=(Matrix){NULL,(const float *)(q+2359296),(const int8_t *)q,H505,D505};
        b.wo[e]=(const int8_t *)(q+2371584);b.os[e]=(const float *)(q+4730880);
        require((uintptr_t)b.wi[e].q%64==0&&(uintptr_t)b.wi[e].scales%64==0&&(uintptr_t)b.wo[e]%64==0&&(uintptr_t)b.os[e]%64==0,"505_alignment");
        for(int j=0;j<H505;j++)require(isfinite(b.wi[e].scales[j])&&b.wi[e].scales[j]>0,"505_wi_scale");
        for(int r=0;r<D505;r++)require(isfinite(b.os[e][r])&&b.os[e][r]>0,"505_wo_scale");
    }return b;
}
/* Same accumulation/block/order as454, with conservative -128 bound. */
static void sparse505(const int8_t *columns,const float *scales,Work505 *w){
    w->wire.nonzero=0;for(int j=0;j<H505;j++)if(w->wire.codes[j]!=0)w->indices[w->wire.nonzero++]=(uint16_t)j;
    memset(w->total,0,sizeof(w->total));
    for(uint32_t start=0;start<w->wire.nonzero;start+=CHUNK505){
        uint32_t end=start+CHUNK505;if(end>w->wire.nonzero)end=w->wire.nonzero;memset(w->partial,0,sizeof(w->partial));
        for(uint32_t p=start;p<end;p++){
            uint32_t j=w->indices[p];const int8_t *q=columns+(size_t)j*D505;
            __m256i c=_mm256_set1_epi32((int32_t)w->wire.codes[j]);
            for(int r=0;r<D505;r+=8){
                __m256i v=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q+r)));
                __m256i partial=_mm256_loadu_si256((const __m256i *)(w->partial+r));
                _mm256_storeu_si256((__m256i *)(w->partial+r),_mm256_add_epi32(partial,_mm256_mullo_epi32(v,c)));
            }
        }
        for(int r=0;r<D505;r+=4){
            __m256i part=_mm256_cvtepi32_epi64(_mm_loadu_si128((const __m128i *)(w->partial+r)));
            __m256i total=_mm256_loadu_si256((const __m256i *)(w->total+r));
            _mm256_storeu_si256((__m256i *)(w->total+r),_mm256_add_epi64(total,part));
        }
    }
    for(int r=0;r<D505;r++)w->wire.down[r]=(float)(((double)w->total[r]*(double)scales[r])*(double)w->wire.alpha);
}
static void candidate505(const Bank505 *b,const Input505 *t,Work505 *w){
    mv(b->wi[t->e],t->x,w->wire.hidden,H505,D505);
    for(int j=0;j<H505;j++)if(w->wire.hidden[j]<0.f)w->wire.hidden[j]=0.f;
    w->wire.alpha=head_activation_codes(w->wire.hidden,w->wire.codes,H505);
    sparse505(b->wo[t->e],b->os[t->e],w);
}
static void baseline505(Model *m,const Input505 *t,Work505 *w){
    Block *b=m->dec+11;mv(b->expertwi[t->e],t->x,w->wire.hidden,H505,D505);
    for(int j=0;j<H505;j++)if(w->wire.hidden[j]<0.f)w->wire.hidden[j]=0.f;
    mv(b->expertwo[t->e],w->wire.hidden,w->wire.down,D505,H505);
}
static void controls505(Work505 *w){
    int8_t *columns=alloc((size_t)H505*D505,1);float s[D505];int8_t a[4096];int16_t x[4096];
    for(int r=0;r<D505;r++)s[r]=r%3==0?.5f:r%3==1?2.f:.25f;
    for(int j=0;j<H505;j++)for(int r=0;r<D505;r++)columns[(size_t)j*D505+r]=r%3==0?-128:r%3==1?127:(j%2?-128:127);
    for(int mode=0;mode<6;mode++){
        for(int j=0;j<H505;j++)w->wire.codes[j]=mode==0?32767:mode==1?-32767:mode==2?(j%2?-32767:32767):mode==3?0:mode==4?(j==H505-1?-32767:0):(j%3==0?32767:0);
        w->wire.alpha=.25f;sparse505(columns,s,w);
        for(int r=0;r<D505;r++){
            int64_t sum=0;for(int j=0;j<H505;j++)sum+=(int64_t)columns[(size_t)j*D505+r]*w->wire.codes[j];
            float gold=(float)(((double)sum*(double)s[r])*.25);
            require(sum==w->total[r]&&!memcmp(&gold,w->wire.down+r,4),"505_scalar_sparse_extreme");
        }
        require(w->wire.nonzero==(mode==3?0:mode==4?1:mode==5?1024:H505),"505_active_controls");
        for(uint32_t k=1;k<w->wire.nonzero;k++)require(w->indices[k]>w->indices[k-1],"505_index_order");
    }
    for(int mode=0;mode<3;mode++){
        int64_t sum=0;for(int j=0;j<4096;j++){a[j]=mode==0?-128:mode==1?127:(j%2?-128:127);x[j]=mode==0?32767:mode==1?-32767:(j%2?32767:-32767);sum+=(int64_t)a[j]*x[j];}
        require(head_integer_dot(a,x,4096)==sum,"505_original_WI_I64_extreme");
    }
    float zero[D505]={0};require(head_activation_codes(zero,x,D505)==1.f,"505_zero_quant");for(int j=0;j<D505;j++)require(x[j]==0,"505_zero_codes");
    free(columns);
}
static void outputheader505(FILE *f,const char *magic,uint32_t stride,uint32_t res,uint64_t count){
    require(fwrite(magic,1,8,f)==8&&fwrite(&stride,4,1,f)==1&&fwrite(&res,4,1,f)==1&&fwrite(&count,8,1,f)==1,"505_output_header");
}
int main(int argc,char **argv){
    require(argc==7&&(!strcmp(argv[1],"--primal")||!strcmp(argv[1],"--bench")),"505 mode manifest bank inputs targets output");
    require(fegetround()==FE_TONEAREST,"505_rounding");worker_setup(1);cost_enabled=cost_profile=0;
    LARGE_INTEGER freq;require(QueryPerformanceFrequency(&freq),"505_frequency");cost_frequency=(double)freq.QuadPart;double start=cost_clock();
    Bank505 b=bank505(argv[3]);Mapping input=map505(argv[4]),gold=map505(argv[5]);
    header505(input,"M499INP1",4624,D505,N505);header505(gold,"M499F001",3072,D505,N505);
    const Input505 *t=(const Input505 *)(input.data+24);const float *target=(const float *)(gold.data+24);
    for(int i=0;i<N505;i++){
        require(t[i].e<E505&&t[i].accepted==1&&isfinite(t[i].alpha)&&t[i].alpha>0&&isfinite(t[i].p)&&t[i].p>0&&t[i].p<=1,"505_input_metadata");
        int16_t q[D505];float alpha=head_activation_codes(t[i].x,q,D505);
        require(!memcmp(&alpha,&t[i].alpha,4)&&!memcmp(q,t[i].q,sizeof(q)),"505_input_quant");
    }
    Work505 *w=alloc(1,sizeof(Work505));HANDLE output=CreateFileA(argv[6],GENERIC_WRITE,0,NULL,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL);
    require(output!=INVALID_HANDLE_VALUE,"505_output_exclusive");int fd=_open_osfhandle((intptr_t)output,_O_BINARY|_O_WRONLY);require(fd>=0,"505_output_fd");
    FILE *f=_fdopen(fd,"wb");require(f!=NULL,"505_output_stream");
    int primal=!strcmp(argv[1],"--primal");Model *model=NULL;
    if(primal){
        controls505(w);outputheader505(f,"M505OUT1",sizeof(Wire505),D505,N505);
        for(int i=0;i<N505;i++){
            candidate505(&b,t+i,w);w->wire.uid=i;w->wire.e=t[i].e;
            require(!memcmp(w->wire.down,target+(size_t)i*D505,D505*4),"505_primal_gold_BYTE");
            require(fwrite(&w->wire,sizeof(Wire505),1,f)==1,"505_primal_record");
        }
    }else{
        model=load(argv[2]);require(model->c.d==D505&&model->c.ff==H505&&model->c.n==E505&&model->dec[11].sparse,"505_source_shape");
        double *times=alloc((size_t)3*2*N505,8);outputheader505(f,"M505TIM1",8,3,3ULL*2*N505);
        for(int rep=-1;rep<3;rep++)for(int slot=0;slot<2;slot++){
            int arm=rep<0?slot:((rep%2)+slot)%2;
            for(int i=0;i<N505;i++){
                double before=cost_clock();if(arm)candidate505(&b,t+i,w);else baseline505(model,t+i,w);
                double elapsed=cost_clock()-before;require(isfinite(elapsed)&&elapsed>0,"505_positive_time");
                require(!memcmp(w->wire.down,target+(size_t)i*D505,D505*4),"505_every_timed_output_BYTE");checksum505+=(double)w->wire.down[i%D505];
                if(rep>=0)times[((size_t)rep*2+arm)*N505+i]=elapsed;
            }
        }
        require(fwrite(times,8,3ULL*2*N505,f)==3ULL*2*N505,"505_timing_payload");free(times);
    }
    require(fclose(f)==0,"505_output_close");PROCESS_MEMORY_COUNTERS mem={0};mem.cb=sizeof(mem);
    require(GetProcessMemoryInfo(GetCurrentProcess(),&mem,sizeof(mem))&&mem.PeakWorkingSetSize<=((SIZE_T)2<<30),"505_native2GiB");
    printf("{\"mode\":\"%s\",\"native_seconds\":%.9f,\"native_peak_bytes\":%llu,\"bank_bytes\":%llu,\"workspace_bytes\":%llu,\"primal_controls\":%s,\"timed_output_comparisons\":%d,\"checksum\":%.17g",argv[1],cost_clock()-start,(unsigned long long)mem.PeakWorkingSetSize,(unsigned long long)b.mapped.bytes,(unsigned long long)sizeof(Work505),primal?"true":"false",primal?0:8*N505,(double)checksum505);
    worker_json();printf("}\n");if(model)unload(model);close505(b.mapped);close505(input);close505(gold);free(w);return 0;
}
