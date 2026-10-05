/* ONE new full-domain cached-router oracle. No model/capture/timing. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#define D 768
#define N 128
#define JOBS 1536
#define QUERIES 387036
static void need(int v,const char *why){if(!v){fprintf(stderr,"source_router_error:%s\n",why);fflush(stderr);exit(2);}}
static void readn(FILE *f,void *p,size_t n){need(fread(p,1,n,f)==n,"short_read");}
static uint32_t u32(FILE *f){uint32_t v;readn(f,&v,4);return v;}
static uint64_t u64(FILE *f){uint64_t v;readn(f,&v,8);return v;}
static FILE *openr(const char *p){FILE *f=fopen(p,"rb");need(f!=NULL,"input_open");return f;}
static char *string(FILE *f){uint32_t n=u32(f);need(n>0&&n<4096,"path_length");char *p=calloc(n+1,1);need(p!=NULL,"allocation");readn(f,p,n);return p;}
static void end(FILE *f){need(fgetc(f)==EOF&&fclose(f)==0,"bounded_EOF");}
static int normalzero(float v){uint32_t b;memcpy(&b,&v,4);return isfinite(v)&&((b&0x7f800000u)!=0||(b&0x7fffffffu)==0);}
static double dot(const float *a,const float *b){
    __m256d l=_mm256_setzero_pd(),r=_mm256_setzero_pd();
    for(int i=0;i<D;i+=8){l=_mm256_add_pd(l,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),_mm256_cvtps_pd(_mm_loadu_ps(b+i))));r=_mm256_add_pd(r,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),_mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));}
    double a4[4];_mm256_storeu_pd(a4,_mm256_add_pd(l,r));double total=0.;for(int j=0;j<4;j++)total+=a4[j];return total;
}
static int winner(const float *s,int n){int k=0;for(int j=1;j<n;j++)if(s[j]>s[k])k=j;return k;}
static double mass(const float *s,int n,int chosen){double z=0.;for(int j=0;j<n;j++){float d=s[j]-s[chosen];float v=(float)exp((double)d);z+=(double)v;}need(z>0.&&isfinite(z),"positive_mass");return z;}
static float weights[12][N][D];
static void load(const char *path){
    FILE *f=openr(path);char magic[8];readn(f,magic,8);need(!memcmp(magic,"M479WGT1",8),"weight_ledger_magic");need(u32(f)==12&&u32(f)==D,"weight_dimensions");
    uint32_t seen=0;
    for(int b=0;b<12;b++){uint32_t bank=u32(f);uint64_t off=u64(f);char *p=string(f);need(bank<12&&!(seen&(1u<<bank)),"weight_bank");seen|=1u<<bank;FILE *w=openr(p);need(_fseeki64(w,off,SEEK_SET)==0,"weight_seek");readn(w,weights[bank],N*D*4);need(fclose(w)==0,"weight_close");free(p);for(int e=0;e<N;e++)for(int d=0;d<D;d++)need(normalzero(weights[bank][e][d]),"weight_normal_or_zero");}
    end(f);need(seen==4095,"ALL12banks");
}
typedef struct {int32_t expert,accepted;float probability;} Route;
typedef struct {uint32_t meta[12];double value[3];} Record;
_Static_assert(sizeof(Record)==72,"wire72");
static void source(const char *jobs_path,const char *out_path){
    FILE *jobs=openr(jobs_path);char magic[8];readn(jobs,magic,8);need(!memcmp(magic,"M479JOB1",8),"jobs_magic");need(u32(jobs)==JOBS&&u32(jobs)==QUERIES,"jobs_dimensions");
    FILE *out=fopen(out_path,"wb");need(out!=NULL,"output_open");uint32_t width=72,reserved=0;uint64_t total=QUERIES;need(fwrite("M479SRC1",1,8,out)==8,"output_header");fwrite(&width,4,1,out);fwrite(&reserved,4,1,out);fwrite(&total,8,1,out);
    uint32_t serial=0;
    for(uint32_t ordinal=0;ordinal<JOBS;ordinal++){
        uint32_t book=u32(jobs),case_id=u32(jobs),mode=u32(jobs),s=u32(jobs),t=u32(jobs),role=u32(jobs),count=u32(jobs);char *trace=string(jobs),*whole=string(jobs);
        need(book<192&&case_id<4&&mode<2&&s==29&&t>0&&t<=64&&count==6*(s+t),"job_fields");need(role==(book<64?0:book<128?1:2),"explicit_book_role");need(mode||t==14,"teacher_length");
        need(book==ordinal/8&&case_id==(ordinal/2)%4&&mode==ordinal%2,"job_full_order");
        FILE *wf=openr(whole);char wm[8];readn(wf,wm,8);need(!memcmp(wm,"SWR32O01",8),"whole_magic");uint32_t wh[7];readn(wf,wh,28);need(wh[0]==s&&wh[1]==t&&wh[2]==D&&wh[3]==12&&wh[4]==12&&wh[5]==32128&&wh[6]==count,"whole_dimensions");
        uint64_t offset=36+4*((uint64_t)14*s*D+(uint64_t)t*14*D+(uint64_t)t*32128);need(_fseeki64(wf,offset,SEEK_SET)==0,"whole_seek");Route *routes=calloc(count,sizeof(Route));need(routes!=NULL,"route_allocation");readn(wf,routes,count*sizeof(Route));end(wf);
        FILE *tf=openr(trace);char tm[8];readn(tf,tm,8);need(!memcmp(tm,"SWRTA001",8)&&u32(tf)==N&&u32(tf)==D,"trace_header");float x[D],score[N],actual[N];
        for(uint32_t i=0;i<count;i++){
            need(u32(tf)==i&&u32(tf)==(uint32_t)(i>=6*s),"trace_index_phase");readn(tf,x,D*4);readn(tf,score,N*4);for(int d=0;d<D;d++)need(normalzero(x[d]),"input_normal_or_zero");for(int e=0;e<N;e++)need(isfinite(score[e]),"score_finite");
            uint32_t bank=i<6*s?i/s:6+(i-6*s)%6,pos=i<6*s?i%s:(i-6*s)/6;
            for(int e=0;e<N;e++)actual[e]=(float)dot(weights[bank][e],x);
            need(!memcmp(actual,score,N*4),"ALL_original_score_BYTE");int chosen=winner(actual,N);double z=mass(actual,N,chosen);float p=(float)(1./z);
            need(routes[i].expert==chosen&&!memcmp(&p,&routes[i].probability,4)&&(routes[i].accepted==0||routes[i].accepted==1),"ALL_original_ID_probability_BYTE");
            Record record={{serial++,N,book,case_id,mode,role,bank,pos,i,(uint32_t)chosen,(uint32_t)routes[i].accepted,ordinal},{(double)actual[chosen],(double)p,z}};
            need(fwrite(&record,1,sizeof(record),out)==sizeof(record),"record_write");
        }
        end(tf);free(routes);free(trace);free(whole);need(fflush(out)==0,"output_flush");
        if((ordinal+1)%128==0){printf("{\"jobs_completed\":%u,\"queries_completed\":%u}\n",ordinal+1,serial);fflush(stdout);}
    }
    end(jobs);need(serial==QUERIES&&fclose(out)==0,"complete_query_EOF");
    printf("{\"terminal\":true,\"source_queries\":%u,\"source_score_values\":%u,\"scores_BYTE_exact\":true,\"ID_probability_BYTE_exact\":true,\"CPU_affinity_mask\":1,\"MXCSR\":%u}\n",serial,serial*N,_mm_getcsr());
}
static void controls(void){
    const float toys[6][3]={{0,1,-1},{10000,9999,-10000},{0,0,0},{1000,999,-1000},{0,-1,-2},{0,0,-1}};
    for(int k=0;k<6;k++){int e=winner(toys[k],3);double z=mass(toys[k],3,e);printf("{\"control\":%d,\"chosen\":%d,\"denominator\":%.17g,\"probability\":%.17g}\n",k,e,z,(double)(float)(1./z));}
    float a[D],x[D];for(int i=0;i<D;i++){a[i]=(i%7-3)*0x1p-10f;x[i]=(i%5-2)*0x1p-9f;}double value=dot(a,x);printf("{\"dot_control\":0,\"double_result\":%.17g,\"f32_result\":%.17g}\n",value,(double)(float)value);
    for(int i=0;i<D;i++){a[i]=i%2?0x1p-100f:0x1p80f;x[i]=i%2?0x1p-20f:(i%4?-0x1p-80f:0x1p-80f);}value=dot(a,x);printf("{\"dot_control\":1,\"double_result\":%.17g,\"f32_result\":%.17g}\n",value,(double)(float)value);
    printf("{\"CPU_affinity_mask\":1,\"rounding\":%d,\"MXCSR\":%u}\n",fegetround(),_mm_getcsr());
}
int main(int argc,char **argv){
    need(SetProcessAffinityMask(GetCurrentProcess(),1)!=0,"CPU_affinity_set");DWORD_PTR p=0,s=0;need(GetProcessAffinityMask(GetCurrentProcess(),&p,&s)&&p==1,"CPU_affinity_readback");need(fegetround()==FE_TONEAREST&&(_mm_getcsr()&0x8040u)==0,"RNE_noFTZ_noDAZ");
    need(argc>=2,"arguments");if(!strcmp(argv[1],"controls")){need(argc==2,"controls_arguments");controls();return 0;}
    need(argc==5&&!strcmp(argv[1],"source"),"source_arguments");load(argv[2]);source(argv[3],argv[4]);return 0;
}
