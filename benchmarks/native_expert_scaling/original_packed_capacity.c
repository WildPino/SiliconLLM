/* Original engine computational bodies, runtime-n byte-pair banks and full u32 vocabulary.
   The generated header is extracted and hashed from the frozen phase60 source.
   No expert float or unpacked reference allocation exists in this deployment path. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#include <windows.h>
#include <malloc.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define D 256
#define N 96
#define H 8
#define HD (D/H)
#define L 6
#define DN 512
#define DTR 16
#define CONV 4
#define WIN 128
#define SWA_LAYER 5
#define AQ 63
#define HID_E 128
#define KTOP 8
#define TUP (D/2)
#define TDE (HID_E/2)
#define GH (E*HID_E)
#define MPAD_GU ((GH+31)&~31)
#define MPAD_D D
#define NPLANES(T) (T)
#define MV_ROWS(c,l,y,r0,M,Mp,T) matvec_lut_rows(c,l,y,r0,M,Mp,T)
#define OMP_PFOR
static int E, V, g_moe=1, g_router_shortlist=0;
static void fail(const char*s){fprintf(stderr,"%s\n",s);exit(2);}
static double now_s(void){LARGE_INTEGER a,b;QueryPerformanceCounter(&a);QueryPerformanceFrequency(&b);return (double)a.QuadPart/b.QuadPart;}
typedef struct {float *in_proj,*conv_w,*conv_b,*x_proj,*dt_proj,*dt_b,*A,*Dskip,*out_proj,*norm;} SSML;
typedef struct {float *qkv,*o,*norm;} SWAL;
static float *emb,*head,*normf; static SSML ssm[L]; static SWAL swa; static int is_swa[L];
static float *mlp_n2[L],*router_w[L],*router_b[L],*egate_f[L],*eup_f[L],*eWd_f[L];
static float *router_prob;static int *router_used;
static int8_t *egate_cd[L],*eup_cd[L],*eWd_cd[L];static float *egate_sc[L],*eup_sc[L],*eWd_sc[L];
static float (*hstate)[DN][N],(*convbuf)[DN][CONV],*kring,*vring;static int kvpos,kvcnt;
typedef struct {float xraw[L][DN];float dt[L][DN];float Bm[L][N];float kk[D];float vv[D];} ActPos;
typedef struct {double scan,scan_other,swa,mlp,head,norm,router;} Tacc;
static ActPos *g_cap;static int *g_esel_cap;
static int actual_ids[L][KTOP];static float actual_mass[L][KTOP];
static void router_int8_topk(int l,const float*x,int*i,float*w){(void)l;(void)x;(void)i;(void)w;fail("shortlist unsupported in this contract");}
static void mlp_dense(int l,const float*x,float*y,int a,int b){(void)l;(void)x;(void)y;(void)a;(void)b;fail("dense unsupported in this contract");}
#include "original_packed_bodies.h"

typedef struct {char name[64];uint32_t dtype,rank,shape[4];uint64_t offset,bytes;} Field;
_Static_assert(sizeof(Field)==104,"field ABI");
static uint8_t *blob;static uint64_t blob_bytes,code_bytes,f32_bytes,state_bytes,heap_bytes;
static Field *fields;static uint32_t nf;static uint8_t used[128];
static uint64_t mul(uint64_t a,uint64_t b){if(b&&a>UINT64_MAX/b)fail("size multiplication overflow");return a*b;}
static void *zeros(uint64_t n){uint64_t bytes=mul(n,4);if(bytes>SIZE_MAX)fail("state size");void*p=calloc((size_t)n,4);if(!p)fail("state allocation");state_bytes+=bytes;heap_bytes+=_msize(p);return p;}
static void *field(const char*name,uint32_t type,uint32_t rank,uint32_t a,uint32_t b,uint32_t c){
    uint32_t dims[4]={a,b,c,0};
    for(uint32_t i=0;i<nf;i++)if(!strcmp(fields[i].name,name)){
        Field*f=fields+i;if(used[i]++)fail("duplicate field use");
        if(f->dtype!=type||f->rank!=rank)fail("field dtype/rank");uint64_t n=1;
        for(int j=0;j<4;j++){if(f->shape[j]!=dims[j])fail("field shape");if((uint32_t)j<rank)n=mul(n,dims[j]);}
        if(f->bytes!=mul(n,type==1?4:1))fail("field byte count");
        if(type==1){float*p=(float*)(blob+f->offset);for(uint64_t j=0;j<n;j++)if(!isfinite(p[j]))fail("nonfinite organ");f32_bytes+=f->bytes;}
        else{uint8_t*p=blob+f->offset;for(uint64_t j=0;j<n;j++)if(p[j]>8)fail("invalid pair code");code_bytes+=f->bytes;}
        return blob+f->offset;
    }fail("missing field");return NULL;
}
static void *lf(int l,const char*s,uint32_t t,uint32_t r,uint32_t a,uint32_t b,uint32_t c){char name[64];snprintf(name,sizeof(name),"layers.%d.%s",l,s);return field(name,t,r,a,b,c);}
static void load_model(const char*path){
    FILE*f=fopen(path,"rb");if(!f)fail("model open");uint8_t magic[8];uint32_t h[16];uint64_t size;
    if(fread(magic,1,8,f)!=8||fread(h,4,16,f)!=16||fread(&size,8,1,f)!=1)fail("short header");
    if(memcmp(magic,"E4BPv001",8)||h[0]!=1||h[2]!=D||h[3]!=N||h[4]!=H||h[5]!=L||h[6]!=DN||h[7]!=DTR||h[8]!=CONV||h[9]!=WIN||h[10]!=SWA_LAYER||h[12]!=HID_E||h[13]!=KTOP||h[14]!=1)fail("geometry contract");
    if(h[1]<1||h[1]>INT_MAX||h[11]<KTOP||h[11]>(uint32_t)((INT_MAX-31)/HID_E)||h[15]<1||h[15]>128)fail("dimension bounds");
    V=(int)h[1];E=(int)h[11];nf=h[15];uint64_t table=80+mul(nf,104);
    if(size<table||size>SIZE_MAX||_fseeki64(f,0,SEEK_END)||_ftelli64(f)<0||(uint64_t)_ftelli64(f)!=size)fail("file extent");
    if(_fseeki64(f,0,SEEK_SET))fail("seek");blob_bytes=size;blob=malloc((size_t)size);if(!blob)fail("model allocation");heap_bytes+=_msize(blob);
    if(fread(blob,1,(size_t)size,f)!=(size_t)size||fgetc(f)!=EOF)fail("short payload");fclose(f);fields=(Field*)(blob+80);
    uint64_t cursor=table;
    for(uint32_t i=0;i<nf;i++){Field*z=fields+i;
        if(!memchr(z->name,0,64)||z->offset!=cursor||z->offset%4||z->bytes>size-cursor)fail("field offsets");
        for(uint32_t j=0;j<i;j++)if(!strcmp(z->name,fields[j].name))fail("duplicate field name");cursor+=z->bytes;
    }if(cursor!=size)fail("payload extent");
    emb=field("embed",1,2,V,D,0);head=field("head",1,2,V,D,0);normf=field("final_norm",1,1,D,0,0);
    for(int l=0;l<L;l++){
        is_swa[l]=(l==SWA_LAYER);
        if(is_swa[l]){swa.norm=lf(l,"norm",1,1,D,0,0);swa.qkv=lf(l,"qkv",1,2,3*D,D,0);swa.o=lf(l,"o",1,2,D,D,0);}
        else{SSML*s=ssm+l;s->norm=lf(l,"norm",1,1,D,0,0);s->in_proj=lf(l,"in_proj",1,2,2*DN,D,0);s->conv_w=lf(l,"conv_w",1,2,DN,CONV,0);s->conv_b=lf(l,"conv_b",1,1,DN,0,0);
            s->x_proj=lf(l,"x_proj",1,2,DTR+2*N,DN,0);s->dt_proj=lf(l,"dt_proj",1,2,DN,DTR,0);s->dt_b=lf(l,"dt_b",1,1,DN,0,0);s->A=lf(l,"A_log",1,2,DN,N,0);
            for(int i=0;i<DN*N;i++){s->A[i]=-expf(s->A[i]);if(!isfinite(s->A[i]))fail("A overflow");}
            s->Dskip=lf(l,"Dskip",1,1,DN,0,0);s->out_proj=lf(l,"out_proj",1,2,D,DN,0);}
        mlp_n2[l]=lf(l,"ff_norm",1,1,D,0,0);router_w[l]=lf(l,"router",1,2,E,D,0);router_b[l]=lf(l,"router_bias",1,1,E,0,0);
        egate_cd[l]=lf(l,"gate_code",2,2,TUP,GH,0);eup_cd[l]=lf(l,"up_code",2,2,TUP,GH,0);eWd_cd[l]=lf(l,"down_code",2,3,E,TDE,D);
        egate_sc[l]=lf(l,"gate_scale",1,2,E,HID_E,0);eup_sc[l]=lf(l,"up_scale",1,2,E,HID_E,0);eWd_sc[l]=lf(l,"down_scale",1,2,E,D,0);
        for(int i=0;i<GH;i++)if(egate_sc[l][i]<=0||eup_sc[l][i]<=0)fail("GU scale");for(uint64_t i=0;i<(uint64_t)E*D;i++)if(eWd_sc[l][i]<=0)fail("down scale");
        if(egate_f[l]||eup_f[l]||eWd_f[l])fail("expert reference allocation");
    }
    for(uint32_t i=0;i<nf;i++)if(!used[i])fail("unused field");
    if(code_bytes!=mul(mul(mul(mul(3,L),E),D),HID_E)/2)fail("code accounting");
    router_prob=zeros(E);router_used=zeros(E);hstate=zeros((uint64_t)L*DN*N);convbuf=zeros((uint64_t)L*DN*CONV);kring=zeros((uint64_t)WIN*D);vring=zeros((uint64_t)WIN*D);state_reset();
}
static FILE *output(const char*p){int fd=_open(p,_O_BINARY|_O_WRONLY|_O_CREAT|_O_EXCL,_S_IREAD|_S_IWRITE);if(fd<0)fail("output exists");FILE*f=_fdopen(fd,"wb");if(!f)fail("output stream");return f;}
static void put(FILE*f,const void*p,size_t n){if(fwrite(p,1,n,f)!=n)fail("output write");}
static uint64_t witnesses(const char*path){
    FILE*f=output(path);int candidates[6]={0,31,32,1023,1024,E-1},done[6],nd=0;int8_t q[D],qh[HID_E],lut[TUP*16],luth[TDE*16];int32_t y[D];
    for(int i=0;i<D;i++)q[i]=(int8_t)(i%127-63);for(int i=0;i<HID_E;i++)qh[i]=(int8_t)((i*7)%127-63);
    build_lut_t3(q,TUP,lut);build_lut_t3(qh,TDE,luth);
    for(int j=0;j<6;j++){int e=candidates[j],dup=0;if(e<0||e>=E)continue;for(int k=0;k<nd;k++)dup|=(done[k]==e);if(dup)continue;done[nd++]=e;
        for(int l=0;l<L;l++){matvec_lut_rows(egate_cd[l],lut,y,e*HID_E,HID_E,MPAD_GU,TUP);put(f,y,HID_E*4);matvec_lut_rows(eup_cd[l],lut,y,e*HID_E,HID_E,MPAD_GU,TUP);put(f,y,HID_E*4);
            matvec_lut_rows(eWd_cd[l]+(size_t)e*TDE*D,luth,y,0,D,D,TDE);put(f,y,D*4);}}
    fclose(f);return (uint64_t)nd*L*(2*HID_E+D);
}
int main(int argc,char**argv){
    if(argc!=7)fail("usage: model queries logits routes witnesses report");if(fesetround(FE_TONEAREST))fail("round mode");
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    double start=now_s();load_model(argv[1]);double loaded=now_s()-start;uint64_t checks=witnesses(argv[5]);
    FILE*q=fopen(argv[2],"rb");uint32_t n;if(!q||fread(&n,4,1,q)!=1||n<1||n>16384)fail("query header");
    FILE*lg=output(argv[3]),*rt=output(argv[4]);float*logits=malloc((size_t)V*4);if(!logits)fail("logits allocation");heap_bytes+=_msize(logits);uint32_t max_id=0;
    for(uint32_t j=0;j<n;j++){uint32_t id;if(fread(&id,4,1,q)!=1||id>=(uint32_t)V)fail("query ID");if(id>max_id)max_id=id;
        forward_token(id,logits,1,0,1,NULL);for(int i=0;i<V;i++)if(!isfinite(logits[i]))fail("nonfinite logits");put(lg,logits,(size_t)V*4);
        for(int l=0;l<L;l++){float sum=0;for(int k=0;k<KTOP;k++){if(actual_ids[l][k]<0||actual_ids[l][k]>=E||!isfinite(actual_mass[l][k])||actual_mass[l][k]<0)fail("route support");sum+=actual_mass[l][k];}if(fabsf(sum-1)>1e-6f)fail("route mass");put(rt,actual_ids[l],KTOP*4);put(rt,actual_mass[l],KTOP*4);}
    }if(fgetc(q)!=EOF)fail("query extent");fclose(q);fclose(lg);fclose(rt);
    FILE*r=output(argv[6]);fprintf(r,"{\"n\":%d,\"V\":%d,\"inputs\":%u,\"max_input_id\":%u,\"blob_bytes\":%llu,\"code_bytes\":%llu,\"f32_bytes\":%llu,\"allocated_state_and_router_bytes\":%llu,\"heap_allocated_bytes\":%llu,\"integer_coordinates\":%llu,\"expert_reference_bytes\":0,\"load_seconds\":%.9g,\"elapsed_seconds\":%.9g}\n",E,V,n,max_id,(unsigned long long)blob_bytes,(unsigned long long)code_bytes,(unsigned long long)f32_bytes,(unsigned long long)state_bytes,(unsigned long long)heap_bytes,(unsigned long long)checks,loaded,now_s()-start);fclose(r);return 0;
}
