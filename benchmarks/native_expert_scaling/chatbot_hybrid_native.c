/* Compact learner deployment using extracted ORIGINAL phase60 matrix kernels.
   Byte-pair experts ONLY; source-compatible per-head scan and bounded SWA.
   First fixed-prefix numerical qualification, not chatbot quality admission. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#include <windows.h>
#include <malloc.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define D 512
#define L 12
#define V 65537
#define E 72
#define H 128
#define DN 768
#define N 256
#define NH 48
#define HD 16
#define CONV 4
#define CD (DN+2*N)
#define PROJ (2*DN+2*N+NH)
#define WIN 128
#define AQ 63
#define OMP_PFOR
#include "chatbot_hybrid_original_kernels.h"
#define K 8

static void fail(const char *s){fprintf(stderr,"%s\n",s);exit(2);}
static void read_exact(FILE*f,void*p,size_t n){if(fread(p,1,n,f)!=n)fail("short input");}
static void write_exact(FILE*f,const void*p,size_t n){if(fwrite(p,1,n,f)!=n)fail("short output");}
static FILE* exclusive(const char*p){int fd=_open(p,_O_BINARY|_O_WRONLY|_O_CREAT|_O_EXCL,_S_IREAD|_S_IWRITE);
    if(fd<0)fail("output exists or unavailable");FILE*f=_fdopen(fd,"wb");if(!f)fail("fdopen");return f;}
static double now_s(void){LARGE_INTEGER t,f;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart/f.QuadPart;}

typedef struct {char name[64];uint32_t dtype,rank,shape[4];uint64_t offset,bytes;} Field;
typedef struct {float *n1,*n2,*rw,*rb,*gs,*us,*ds,*wi,*wo,*cw,*cb,*al,*dt,*skip,*gn,*q,*k,*v,*o;
                const int8_t *gc,*uc,*dc;float *state,*conv,*keys,*vals;float a[NH];} Layer;
static Layer layers[L];
static uint8_t *blob;
static uint64_t blob_bytes,code_bytes,payload_bytes,state_bytes,heap_state_bytes;
static Field *fields;static int nfields,used[256];
static float *emb,*head,*final_norm,*freq;

static void* field(const char*name,uint32_t type,int rank,uint32_t a,uint32_t b,uint32_t c){
    uint32_t dims[4]={a,b,c,0};
    for(int i=0;i<nfields;i++)if(!strcmp(fields[i].name,name)){
        Field*f=fields+i;if(used[i]++)fail("duplicate field use");
        if(f->dtype!=type||f->rank!=(uint32_t)rank)fail("field dtype/rank");
        uint64_t elements=1;for(int j=0;j<4;j++){if(f->shape[j]!=dims[j])fail("field shape");if(j<rank)elements*=dims[j];}
        if(f->bytes!=elements*(type==1?4:1))fail("field byte count");
        if(type==1){float*p=(float*)(blob+f->offset);for(uint64_t j=0;j<elements;j++)if(!isfinite(p[j]))fail("nonfinite coefficient");}
        else {uint8_t*p=blob+f->offset;for(uint64_t j=0;j<elements;j++)if(p[j]>8)fail("invalid pair code");code_bytes+=f->bytes;}
        return blob+f->offset;
    } fail(name);return NULL;
}
static void* lf(int l,const char*s,uint32_t type,int rank,uint32_t a,uint32_t b,uint32_t c){
    char name[64];snprintf(name,sizeof(name),"layers.%d.%s",l,s);return field(name,type,rank,a,b,c);
}
static float* zeros(size_t n){float*p=calloc(n,4);if(!p)fail("state OOM");state_bytes+=n*4;heap_state_bytes+=_msize(p);return p;}
static void load_model(const char*path){
    FILE*f=fopen(path,"rb");if(!f)fail("model open");char magic[8];uint32_t h[16];
    read_exact(f,magic,8);read_exact(f,h,64);read_exact(f,&blob_bytes,8);
    uint32_t want[14]={1,D,L,V,E,K,H,DN,N,NH,HD,CONV,WIN,(1u<<5)|(1u<<11)};
    if(memcmp(magic,"SLH1PK01",8)||memcmp(h,want,56)||h[15]!=0x01020304)fail("model contract");
    nfields=(int)h[14];if(nfields!=212||sizeof(Field)!=104||blob_bytes>512ULL*1024*1024)fail("model table");
    blob=malloc((size_t)blob_bytes);if(!blob)fail("model OOM");memcpy(blob,magic,8);memcpy(blob+8,h,64);memcpy(blob+72,&blob_bytes,8);
    read_exact(f,blob+80,(size_t)blob_bytes-80);if(fgetc(f)!=EOF)fail("model tail");fclose(f);
    fields=(Field*)(blob+80);uint64_t cursor=80+(uint64_t)nfields*104;
    for(int i=0;i<nfields;i++){Field*z=fields+i;if(!memchr(z->name,0,64)||z->offset!=cursor||z->offset+z->bytes>blob_bytes||z->offset%4)fail("model offsets");cursor+=z->bytes;}
    if(cursor!=blob_bytes)fail("model extent");payload_bytes=blob_bytes-80-(uint64_t)nfields*104;
    emb=field("embed.weight",1,2,V,D,0);head=field("head.weight",1,2,V,D,0);
    final_norm=field("final_norm.weight",1,1,D,0,0);freq=field("rope_inv_freq",1,1,64,0,0);
    for(int l=0;l<L;l++){
        Layer*s=layers+l;
        s->n1=lf(l,"input_norm.weight",1,1,D,0,0);s->n2=lf(l,"ff_norm.weight",1,1,D,0,0);
        s->rw=lf(l,"banks.router.weight",1,2,E,D,0);s->rb=lf(l,"banks.router.bias",1,1,E,0,0);
        s->gs=lf(l,"banks.gate_scale",1,2,E,H,0);s->us=lf(l,"banks.up_scale",1,2,E,H,0);s->ds=lf(l,"banks.down_scale",1,2,E,D,0);
        for(int i=0;i<E*H;i++)if(s->gs[i]<1e-8f||s->us[i]<1e-8f)fail("nonpositive GU scale");
        for(int i=0;i<E*D;i++)if(s->ds[i]<1e-8f)fail("nonpositive down scale");
        s->gc=lf(l,"banks.gate",2,3,E,D/2,H);s->uc=lf(l,"banks.up",2,3,E,D/2,H);s->dc=lf(l,"banks.down",2,3,E,H/2,D);
        if(l==5||l==11){
            s->q=lf(l,"core.q.weight",1,2,D,D,0);s->k=lf(l,"core.k.weight",1,2,D,D,0);
            s->v=lf(l,"core.v.weight",1,2,D,D,0);s->o=lf(l,"core.o.weight",1,2,D,D,0);
            s->keys=zeros(WIN*D);s->vals=zeros(WIN*D);
        }else{
            s->wi=lf(l,"core.in_proj.weight",1,2,PROJ,D,0);s->wo=lf(l,"core.out_proj.weight",1,2,D,DN,0);
            s->cw=lf(l,"core.conv1d.weight",1,3,CD,1,CONV);s->cb=lf(l,"core.conv1d.bias",1,1,CD,0,0);
            s->al=lf(l,"core.A_log",1,1,NH,0,0);s->dt=lf(l,"core.dt_bias",1,1,NH,0,0);s->skip=lf(l,"core.D",1,1,NH,0,0);
            s->gn=lf(l,"core.norm.weight",1,1,DN,0,0);s->state=zeros(DN*N);s->conv=zeros(CD*CONV);
            for(int i=0;i<NH;i++)s->a[i]=-expf(s->al[i]);
        }
    } for(int i=0;i<nfields;i++)if(used[i]!=1)fail("unused field");
    if(code_bytes!=84934656ULL||payload_bytes!=425188608ULL||state_bytes!=9117696ULL)fail("physical accounting");
}
static void reset(void){for(int l=0;l<L;l++){Layer*s=layers+l;if(s->state){memset(s->state,0,DN*N*4);memset(s->conv,0,CD*CONV*4);}
    else{memset(s->keys,0,WIN*D*4);memset(s->vals,0,WIN*D*4);}}}
static void norm(const float*x,const float*w,float*y,int n){float sum=0;for(int i=0;i<n;i++)sum+=x[i]*x[i];
    float inv=1.0f/sqrtf(sum/(float)n+1e-5f);for(int i=0;i<n;i++)y[i]=(x[i]*inv)*w[i];}
static float quant_target(const float*x,int n,int8_t*q){
    float a=0;for(int i=0;i<n;i++)if(fabsf(x[i])>a)a=fabsf(x[i]);
    if(a>=1e-12f)return quant_i8(x,n,q); // ORIGINAL AQ63 reciprocal/rounding kernel.
    float s=1e-12f/63.0f,iv=1.0f/s;for(int i=0;i<n;i++){int v=(int)lrintf(x[i]*iv);q[i]=(int8_t)(v>63?63:v<-63?-63:v);}return s;
}
static void emit_trace(FILE*f,const void*p,size_t n){if(f)write_exact(f,p,n);}

typedef struct {double core,mlp,head,total;} Timings;
static void forward(uint32_t tok,int pos,float*logits,int readout,FILE*trace,Timings*tm){
    if(tok>=V)fail("token out of range");double total0=now_s();
    float x[D],xn[D],tmp[D],proj[PROJ],convolved[CD],y[DN];
    memcpy(x,emb+(size_t)tok*D,D*4);
    for(int l=0;l<L;l++){
        Layer*s=layers+l;double t0=now_s();
        emit_trace(trace,x,D*4);norm(x,s->n1,xn,D);emit_trace(trace,xn,D*4);
        if(s->state){
            matvec(s->wi,xn,proj,PROJ,D);
            for(int c=0;c<CD;c++){
                float*b=s->conv+(size_t)c*CONV;for(int j=0;j<CONV-1;j++)b[j]=b[j+1];b[CONV-1]=proj[DN+c];
                float z=0;for(int j=0;j<CONV;j++)z+=b[j]*s->cw[(size_t)c*CONV+j];convolved[c]=silu(z+s->cb[c]);
            }
            for(int h=0;h<NH;h++){
                float dt=softplus(proj[2*DN+2*N+h]+s->dt[h]),decay=expf(dt*s->a[h]);
                for(int z=0;z<HD;z++){
                    int c=h*HD+z;float *state=s->state+(size_t)c*N,dx=dt*convolved[c],acc=0;
                    for(int j=0;j<N;j++){state[j]=decay*state[j]+convolved[DN+j]*dx;acc+=state[j]*convolved[DN+N+j];}
                    y[c]=(acc+s->skip[h]*convolved[c])*silu(proj[c]);
                }
            }
            norm(y,s->gn,y,DN);matvec(s->wo,y,tmp,D,DN);
        }else{
            float q[D],key[D],value[D],att[D]={0},scores[WIN];
            matvec(s->q,xn,q,D,D);matvec(s->k,xn,key,D,D);matvec(s->v,xn,value,D,D);
            for(int h=0;h<4;h++)for(int j=0;j<64;j++){
                float a=(float)pos*freq[j],co=cosf(a),si=sinf(a);int i=h*128+j;
                float q0=q[i],q1=q[i+64],k0=key[i],k1=key[i+64];
                q[i]=q0*co+(-q1)*si;q[i+64]=q1*co+q0*si;key[i]=k0*co+(-k1)*si;key[i+64]=k1*co+k0*si;
            }
            memcpy(s->keys+(size_t)(pos%WIN)*D,key,D*4);memcpy(s->vals+(size_t)(pos%WIN)*D,value,D*4);
            int first=pos>=WIN?pos-WIN+1:0,count=pos-first+1;
            for(int h=0;h<4;h++){
                float mx=-INFINITY;for(int t=0;t<count;t++){int slot=(first+t)%WIN;
                    float z=dotf(q+h*128,s->keys+(size_t)slot*D+h*128,128)/sqrtf(128.0f);scores[t]=z;if(z>mx)mx=z;}
                float sum=0;for(int t=0;t<count;t++){scores[t]=expf(scores[t]-mx);sum+=scores[t];}
                for(int t=0;t<count;t++){float mass=scores[t]/sum;const float*v=s->vals+(size_t)((first+t)%WIN)*D+h*128;
                    for(int j=0;j<128;j++)att[h*128+j]+=mass*v[j];}
            }matvec(s->o,att,tmp,D,D);
        }
        emit_trace(trace,tmp,D*4);for(int i=0;i<D;i++)x[i]+=tmp[i];tm->core+=now_s()-t0;
        t0=now_s();norm(x,s->n2,xn,D);emit_trace(trace,xn,D*4);
        float scores[E],mass[K],out[D]={0};int ids[K];matvec(s->rw,xn,scores,E,D);
        for(int e=0;e<E;e++)scores[e]+=s->rb[e];
        for(int k=0;k<K;k++){int id=-1;float best=-INFINITY;for(int e=0;e<E;e++){
                int used=0;for(int j=0;j<k;j++)if(ids[j]==e)used=1;
                if(!used&&(id<0||scores[e]>best)){best=scores[e];id=e;}}
            ids[k]=id;mass[k]=best;}
        float sum=0,mx=mass[0];for(int k=0;k<K;k++){mass[k]=expf(mass[k]-mx);sum+=mass[k];}for(int k=0;k<K;k++)mass[k]/=sum;
        emit_trace(trace,scores,E*4);emit_trace(trace,ids,K*4);emit_trace(trace,mass,K*4);
        int8_t qx[D],lut[D/2*16],qz[H],lutd[H/2*16];int32_t g[H],u[H],d[D];float z[H];
        float sx=quant_target(xn,D,qx);build_lut_t3(qx,D/2,lut);
        // Torch index_add is issued in ascending expert ID, not top-score order.
        for(int e=0;e<E;e++){
            int slot=-1;for(int k=0;k<K;k++)if(ids[k]==e)slot=k;if(slot<0)continue;
            matvec_lut_full(s->gc+(size_t)e*D/2*H,lut,g,H,H,D/2);
            matvec_lut_full(s->uc+(size_t)e*D/2*H,lut,u,H,H,D/2);
            for(int j=0;j<H;j++){float gv=((float)g[j]*s->gs[e*H+j])*sx,uv=((float)u[j]*s->us[e*H+j])*sx;z[j]=silu(gv)*uv;}
            float sz=quant_target(z,H,qz);build_lut_t3(qz,H/2,lutd);matvec_lut_full(s->dc+(size_t)e*H/2*D,lutd,d,D,D,H/2);
            for(int j=0;j<D;j++)out[j]+=(((float)d[j]*s->ds[e*D+j])*sz)*mass[slot];
        }
        emit_trace(trace,out,D*4);for(int i=0;i<D;i++){x[i]+=out[i];if(!isfinite(x[i]))fail("nonfinite state");}
        emit_trace(trace,x,D*4);tm->mlp+=now_s()-t0;
    }
    if(readout){double t0=now_s();norm(x,final_norm,xn,D);matvec(head,xn,logits,V,D);tm->head+=now_s()-t0;
        for(int i=0;i<V;i++)if(!isfinite(logits[i]))fail("nonfinite logit");}
    tm->total+=now_s()-total0;
}

static int fixtures(void){
    int8_t w[32*18],codes[9*32],x[18],lut[9*16];int32_t a[32],b[32];
    for(int r=0;r<32;r++)for(int t=0;t<9;t++){int code=(r+t)%9;w[r*18+2*t]=(int8_t)(code/3-1);w[r*18+2*t+1]=(int8_t)(code%3-1);}
    for(int k=0;k<18;k++)x[k]=(int8_t)((k%3-1)*63);
    bc_tm(w,32,18,32,codes);build_lut_t3(x,9,lut);ref_t3(w,x,a,32,18);matvec_lut_full(codes,lut,b,32,32,9);
    if(memcmp(a,b,sizeof(a)))fail("original LUT fixture");
    for(int t=0;t<9;t++)for(int c=0;c<9;c++)if(lut[t*16+c]!=(c/3-1)*x[2*t]+(c%3-1)*x[2*t+1])fail("pair fixture");
    float z[8]={-63,-2.5f,-1.5f,-.5f,.5f,1.5f,2.5f,63};int8_t q[8],want[8]={-63,-2,-2,0,0,2,2,63};
    if(quant_target(z,8,q)!=1||memcmp(q,want,8))fail("ties-to-even fixture");
    memset(z,0,sizeof(z));quant_target(z,8,q);for(int i=0;i<8;i++)if(q[i])fail("zero fixture");
    if(!(silu(-1)<0))fail("signed gate fixture");return 1;
}
int main(int argc,char**argv){
    if(argc!=7){fputs("model queries logits trace final_states report\n",stderr);return 1;}
    if(fesetround(FE_TONEAREST))fail("round mode");_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    double start=now_s();fixtures();load_model(argv[1]);
    FILE*in=fopen(argv[2],"rb");if(!in)fail("queries open");uint32_t h[4];read_exact(in,h,16);
    if(h[0]!=0x31514853||h[1]!=6||h[2]!=32||h[3]!=V)fail("query header");
    FILE*out=exclusive(argv[3]),*trace=exclusive(argv[4]),*states=exclusive(argv[5]);
    uint32_t oh[4]={0x314c4853,6,32,V};write_exact(out,oh,16);
    float*logits=malloc(V*4);if(!logits)fail("logits OOM");Timings tm={0};int total_ids=0,rows=0;
    for(int c=0;c<6;c++){
        uint32_t sizes[2],ids[128],positions[8];read_exact(in,sizes,8);
        if(sizes[0]>128||sizes[0]<1||sizes[1]>8||sizes[1]<1)fail("query dimensions");
        read_exact(in,ids,sizes[0]*4);read_exact(in,positions,sizes[1]*4);
        for(uint32_t j=0;j<sizes[1];j++)if(positions[j]>=sizes[0]||(j&&positions[j]<=positions[j-1]))fail("query position");
        reset();uint32_t next=0;double t0=now_s();
        for(uint32_t t=0;t<sizes[0];t++){
            int take=next<sizes[1]&&positions[next]==t;forward(ids[t],(int)t,logits,take,trace,&tm);
            if(take){write_exact(out,logits,V*4);rows++;next++;}total_ids++;
        }
        for(int l=0;l<L;l++){Layer*s=layers+l;if(s->state){write_exact(states,s->state,DN*N*4);write_exact(states,s->conv,CD*CONV*4);}
            else{write_exact(states,s->keys,WIN*D*4);write_exact(states,s->vals,WIN*D*4);}}
        printf("case=%d ids=%u rows=%u seconds=%.6f\n",c,sizes[0],sizes[1],now_s()-t0);fflush(stdout);
    }
    if(fgetc(in)!=EOF||rows!=32||total_ids!=261)fail("query extent");fclose(in);
    if(fclose(out)||fclose(trace)||fclose(states))fail("close outputs");
    FILE*r=exclusive(argv[6]);fprintf(r,"{\"schema\":\"HYBRID_NATIVE_C_V1\",\"case_count\":6,\"rows\":%d,\"input_ids\":%d,"
        "\"model_file_bytes\":%llu,\"model_heap_bytes\":%llu,\"coefficient_payload_bytes\":%llu,\"expert_code_bytes\":%llu,"
        "\"expert_master_or_unpacked_bytes\":0,\"state_bytes\":%llu,\"state_heap_bytes\":%llu,\"derived_A_bytes\":%u,"
        "\"original_kernel_fixtures\":true,\"threads\":1,\"core_seconds\":%.9f,\"mlp_seconds\":%.9f,\"head_seconds\":%.9f,"
        "\"forward_seconds\":%.9f,\"process_seconds\":%.9f}\n",rows,total_ids,
        (unsigned long long)blob_bytes,(unsigned long long)_msize(blob),(unsigned long long)payload_bytes,(unsigned long long)code_bytes,
        (unsigned long long)state_bytes,(unsigned long long)heap_state_bytes,10*NH*4,tm.core,tm.mlp,tm.head,tm.total,now_s()-start);
    if(fclose(r))fail("report close");return 0;
}
