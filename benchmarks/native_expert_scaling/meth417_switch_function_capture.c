#define _WIN32_WINNT 0x0601
/* METH-356: ALL I8 matrices use A16 inputs; SAME serialized338 weights. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <fenv.h>
#include <limits.h>
#define CTX 256
#define MAX_L 24
#define MAX_FILES 8

static void require(int ok,const char *why){if(!ok){fprintf(stderr,"switch_reference_error:%s\n",why);exit(2);}}
static void *alloc(size_t count,size_t width){void *p=calloc(count,width);require(p!=NULL,"allocation");return p;}
static uint32_t u32(FILE *f){uint32_t v;require(fread(&v,4,1,f)==1,"u32");return v;}
static uint64_t u64(FILE *f){uint64_t v;require(fread(&v,8,1,f)==1,"u64");return v;}
static void string(FILE *f,char *s,size_t bound){uint32_t n=u32(f);require(n>0&&n<bound,"string_length");require(fread(s,1,n,f)==n,"string");s[n]=0;}
typedef struct {HANDLE file,map;unsigned char *data;uint64_t bytes;} Mapping;
typedef struct {const float *f,*scales;const int8_t *q;uint32_t rows,cols;} Matrix;
typedef struct {char name[192];uint32_t dims,shape[2],encoding;const float *value,*scales;const int8_t *codes;} Tensor;
typedef struct {int d,ff,heads,dk,enc,dec,n,capacity,vocab,buckets,distance,encstep,decstep;float epsilon;} Config;
typedef struct {Matrix q,k,v,o;} Attention;
typedef struct {const float *selfnorm,*crossnorm,*ffnorm;Matrix wi,wo,router,*expertwi,*expertwo;Attention self,cross;int sparse;} Block;
typedef struct {Config c;Mapping files[MAX_FILES];int nfiles,ntensors;Tensor *tensors;Block enc[MAX_L],dec[MAX_L];const float *embed,*encnorm,*decnorm,*encbias,*decbias;Matrix head;} Model;
typedef struct {int32_t expert,accepted;float probability;} Route;
typedef struct {float *selfk,*selfv,*crossk,*crossv;} Cache;
static Route *routes;static int route_count,route_bound,fault;
#include "meth388_switch_thread_binding.h"

static float dot(const float *a,const float *b,int n){
    __m256 x=_mm256_setzero_ps(),y=_mm256_setzero_ps();int i=0;
    for(;i+16<=n;i+=16){x=_mm256_add_ps(x,_mm256_mul_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));y=_mm256_add_ps(y,_mm256_mul_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8)));}
    float v[8];_mm256_storeu_ps(v,_mm256_add_ps(x,y));float sum=0.f;
    for(int j=0;j<8;j++)sum+=v[j];for(;i<n;i++)sum+=a[i]*b[i];return sum;
}
static double dot64(const float *a,const float *b,int n){worker_bind();__m256d x=_mm256_setzero_pd(),y=_mm256_setzero_pd();int i=0;for(;i+8<=n;i+=8){x=_mm256_add_pd(x,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),_mm256_cvtps_pd(_mm_loadu_ps(b+i))));y=_mm256_add_pd(y,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),_mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));}double values[4];_mm256_storeu_pd(values,_mm256_add_pd(x,y));double sum=0.;for(int j=0;j<4;j++)sum+=values[j];for(;i<n;i++)sum+=(double)a[i]*(double)b[i];return sum;}
static int32_t integer_dot(const int8_t *a,const int8_t *b,int n){
    __m256i acc=_mm256_setzero_si256();int i=0;
    for(;i+16<=n;i+=16){__m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a+i)));__m256i y=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b+i)));acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x,y));}
    int32_t values[8];_mm256_storeu_si256((__m256i *)values,acc);int32_t sum=0;for(int j=0;j<8;j++)sum+=values[j];for(;i<n;i++)sum+=(int32_t)a[i]*(int32_t)b[i];return sum;
}
static float activation_codes(const float *x,int8_t *codes,int cols){
    require(cols>0&&cols<=4096&&fegetround()==FE_TONEAREST,"integer_activation_contract");float maximum=0.f;
    for(int i=0;i<cols;i++){require(isfinite(x[i]),"activation_finite");float v=fabsf(x[i]);if(v>maximum)maximum=v;}
    float scale=maximum==0.f?1.f:maximum/127.f;require(isfinite(scale)&&scale>0.f,"activation_scale");
    for(int i=0;i<cols;i++){float divided=x[i]/scale;long code=lrintf(divided);if(code>127)code=127;if(code< -127)code= -127;codes[i]=(int8_t)code;}return scale;
}
typedef struct {uint64_t calls,codes,f32,scales;double matrix_seconds;} CostCounter;
static CostCounter cost_counts[2][4];
static int cost_enabled,cost_profile,cost_phase,cost_kind;
static double cost_frequency;
static double cost_clock(void){LARGE_INTEGER value;require(QueryPerformanceCounter(&value),"performance_counter");return (double)value.QuadPart/cost_frequency;}
static int64_t head_integer_dot(const int8_t *a,const int16_t *b,int n);
static float head_activation_codes(const float *x,int16_t *codes,int cols);
static void mv(Matrix w,const float *x,float *y,int rows,int cols){
    double cost_start=cost_enabled&&cost_profile?cost_clock():0.;
    if(cost_enabled){CostCounter *counter=&cost_counts[cost_phase][cost_kind];counter->calls++;
        if(w.q){counter->codes+=(uint64_t)rows*cols;counter->scales+=(uint64_t)rows*4;}
        else counter->f32+=(uint64_t)rows*cols*4;}

    require(w.rows==(uint32_t)rows&&w.cols==(uint32_t)cols,"matrix_dimensions");
    if(w.q){int16_t codes[4096];float scale=head_activation_codes(x,codes,cols);
        #pragma omp parallel for schedule(static) if(rows>64)
        for(int row=0;row<rows;row++){int64_t value=head_integer_dot(w.q+(size_t)row*cols,codes,cols);y[row]=(float)(((double)value*(double)w.scales[row])*(double)scale);}
    }else{
        require(w.f!=NULL,"f32_matrix");
        #pragma omp parallel for schedule(static) if(rows>64)
        for(int row=0;row<rows;row++)y[row]=(float)dot64(w.f+(size_t)row*cols,x,cols);
    }
    if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;
}
/* For cols<=4096 each I32 lane has <=512 products: 512*127*32767
   =2130641408<INT32_MAX. Global sum can reach17045131264: use I64. */
static int64_t head_integer_dot(const int8_t *a,const int16_t *b,int n){
    worker_bind();
    require(n>0&&n<=4096,"head_integer_bound");__m256i acc=_mm256_setzero_si256();int i=0;
    for(;i+16<=n;i+=16){__m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a+i)));__m256i y=_mm256_loadu_si256((const __m256i *)(b+i));acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x,y));}
    int32_t values[8];_mm256_storeu_si256((__m256i *)values,acc);int64_t sum=0;for(int j=0;j<8;j++)sum+=(int64_t)values[j];for(;i<n;i++)sum+=(int64_t)a[i]*(int64_t)b[i];return sum;
}
static float head_activation_codes(const float *x,int16_t *codes,int cols){
    require(cols>0&&cols<=4096&&fegetround()==FE_TONEAREST,"head_activation_contract");float maximum=0.f;
    for(int i=0;i<cols;i++){require(isfinite(x[i]),"head_activation_finite");float v=fabsf(x[i]);if(v>maximum)maximum=v;}
    float scale=maximum==0.f?1.f:maximum/32767.f;require(isfinite(scale)&&scale>0.f,"head_activation_scale");
    for(int i=0;i<cols;i++){float divided=x[i]/scale;long code=lrintf(divided);if(code>32767)code=32767;if(code< -32767)code= -32767;codes[i]=(int16_t)code;}return scale;
}
static void head_mv(Matrix w,const float *x,float *y,int rows,int cols){
    require(w.q&&w.rows==(uint32_t)rows&&w.cols==(uint32_t)cols,"head_matrix_dimensions");
    double cost_start=cost_enabled&&cost_profile?cost_clock():0.;
    if(cost_enabled){CostCounter *counter=&cost_counts[cost_phase][cost_kind];counter->calls++;counter->codes+=(uint64_t)rows*cols;counter->scales+=(uint64_t)rows*4;}
    int16_t codes[4096];float scale=head_activation_codes(x,codes,cols);
    #pragma omp parallel for schedule(static) if(rows>64)
    for(int row=0;row<rows;row++){int64_t value=head_integer_dot(w.q+(size_t)row*cols,codes,cols);y[row]=(float)(((double)value*(double)w.scales[row])*(double)scale);}
    if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;
}
static void mv_batch(Matrix w,const float *x,float *y,int rows,int cols,int tokens){
    require(w.q&&w.rows==(uint32_t)rows&&w.cols==(uint32_t)cols&&tokens>0&&tokens<=CTX,"batch_matrix_contract");
    double cost_start=cost_enabled&&cost_profile?cost_clock():0.;
    if(cost_enabled){CostCounter *counter=&cost_counts[cost_phase][cost_kind];counter->calls+=tokens;counter->codes+=(uint64_t)rows*cols*tokens;counter->scales+=(uint64_t)rows*4*tokens;}
    int16_t *codes=alloc((size_t)tokens*cols,2);float scales[CTX];for(int token=0;token<tokens;token++)scales[token]=head_activation_codes(x+(size_t)token*cols,codes+(size_t)token*cols,cols);
    #pragma omp parallel for schedule(static) if(rows>64)
    for(int row=0;row<rows;row++)for(int token=0;token<tokens;token++){
        int64_t value=head_integer_dot(w.q+(size_t)row*cols,codes+(size_t)token*cols,cols);
        y[(size_t)token*rows+row]=(float)(((double)value*(double)w.scales[row])*(double)scales[token]);
    }
    free(codes);if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;
}
static void norm(const float *x,const float *w,float *y,int d,float epsilon){double sum=0.;for(int j=0;j<d;j++){float square=x[j]*x[j];sum+=(double)square;}float mean=(float)(sum/(double)d);float scale=1.f/sqrtf(mean+epsilon);for(int j=0;j<d;j++)y[j]=x[j]*scale*w[j];}
static int bucket(int relative,int bidi,int count,int distance){
    int base=0;if(bidi){count/=2;if(relative>0)base=count;relative=abs(relative);}else relative=relative<0?-relative:0;
    int exact=count/2;if(relative<exact)return base+relative;
    int large=exact+(int)(logf((float)relative/(float)exact)/logf((float)distance/(float)exact)*(float)(count-exact));
    if(large>=count)large=count-1;return base+large;
}
static Tensor *find_tensor(Model *m,const char *name){
    int low=0,high=m->ntensors;while(low<high){int middle=low+(high-low)/2;int order=strcmp(m->tensors[middle].name,name);if(order<0)low=middle+1;else high=middle;}
    if(low<m->ntensors&&!strcmp(m->tensors[low].name,name))return m->tensors+low;fprintf(stderr,"missing_tensor:%s\n",name);exit(2);
}
static const float *weight(Model *m,const char *name,int rows,int cols,int dimensions){Tensor *t=find_tensor(m,name);require(t->dims==(uint32_t)dimensions&&t->shape[0]==(uint32_t)rows&&t->shape[1]==(uint32_t)cols&&t->encoding==0,"f32_organ_shape_encoding");return t->value;}
static Matrix matrix(Model *m,const char *name,int rows,int cols){Tensor *t=find_tensor(m,name);require(t->dims==2&&t->shape[0]==(uint32_t)rows&&t->shape[1]==(uint32_t)cols,"matrix_shape");Matrix value={t->value,t->scales,t->codes,(uint32_t)rows,(uint32_t)cols};return value;}
static Matrix matrix_organ(Model *m,const char *prefix,const char *suffix,int rows,int cols){char name[192];require(snprintf(name,sizeof(name),"%s.%s",prefix,suffix)<(int)sizeof(name),"matrix_name");return matrix(m,name,rows,cols);}
static const float *organ(Model *m,const char *prefix,const char *suffix,int rows,int cols,int dimensions){char name[192];require(snprintf(name,sizeof(name),"%s.%s",prefix,suffix)<(int)sizeof(name),"tensor_name");return weight(m,name,rows,cols,dimensions);}
static Attention attention_weights(Model *m,const char *prefix){int d=m->c.d;Attention a={matrix_organ(m,prefix,"q.weight",d,d),matrix_organ(m,prefix,"k.weight",d,d),matrix_organ(m,prefix,"v.weight",d,d),matrix_organ(m,prefix,"o.weight",d,d)};return a;}
static Block block_weights(Model *m,int decoder,int index){
    Config c=m->c;Block b={0};char prefix[192],part[192];const char *stack=decoder?"decoder":"encoder";
    snprintf(prefix,sizeof(prefix),"%s.block.%d.layer.0",stack,index);b.selfnorm=organ(m,prefix,"layer_norm.weight",c.d,1,1);
    snprintf(part,sizeof(part),"%s.SelfAttention",prefix);b.self=attention_weights(m,part);
    if(decoder){snprintf(prefix,sizeof(prefix),"decoder.block.%d.layer.1",index);b.crossnorm=organ(m,prefix,"layer_norm.weight",c.d,1,1);snprintf(part,sizeof(part),"%s.EncDecAttention",prefix);b.cross=attention_weights(m,part);}
    snprintf(prefix,sizeof(prefix),"%s.block.%d.layer.%d",stack,index,decoder?2:1);b.ffnorm=organ(m,prefix,"layer_norm.weight",c.d,1,1);
    int step=decoder?c.decstep:c.encstep;b.sparse=step>0&&(index%step==1||step==1);
    snprintf(part,sizeof(part),"%s.mlp",prefix);
    if(b.sparse){b.expertwi=alloc(c.n,sizeof(Matrix));b.expertwo=alloc(c.n,sizeof(Matrix));b.router=matrix_organ(m,part,"router.classifier.weight",c.n,c.d);for(int e=0;e<c.n;e++){char suffix[80];snprintf(suffix,sizeof(suffix),"experts.expert_%d.wi.weight",e);b.expertwi[e]=matrix_organ(m,part,suffix,c.ff,c.d);snprintf(suffix,sizeof(suffix),"experts.expert_%d.wo.weight",e);b.expertwo[e]=matrix_organ(m,part,suffix,c.d,c.ff);}}
    else{b.wi=matrix_organ(m,part,"wi.weight",c.ff,c.d);b.wo=matrix_organ(m,part,"wo.weight",c.d,c.ff);}return b;
}
static Model *load(const char *spec){
    Model *m=alloc(1,sizeof(Model));FILE *f=fopen(spec,"rb");require(f!=NULL,"manifest_open");char magic[8];require(fread(magic,1,8,f)==8&&!memcmp(magic,"SWI8A001",8),"manifest_magic");
    Config *c=&m->c;c->d=u32(f);c->ff=u32(f);c->heads=u32(f);c->dk=u32(f);c->enc=u32(f);c->dec=u32(f);c->n=u32(f);c->capacity=u32(f);c->vocab=u32(f);c->buckets=u32(f);c->distance=u32(f);c->encstep=u32(f);c->decstep=u32(f);require(fread(&c->epsilon,4,1,f)==1,"epsilon");
    require(c->d>0&&c->d<=1024&&c->ff>0&&c->ff<=4096&&c->heads>0&&c->heads*c->dk==c->d&&c->enc>0&&c->enc<=MAX_L&&c->dec>0&&c->dec<=MAX_L&&c->n>0&&(uint64_t)c->n*c->d<((uint64_t)1<<31)&&c->capacity>0&&c->vocab>0&&c->vocab<=65536&&c->buckets==32&&c->distance==128&&c->epsilon>0.f&&c->epsilon<1.f,"configuration");
    m->nfiles=u32(f);m->ntensors=u32(f);require(m->nfiles>0&&m->nfiles<=MAX_FILES&&m->ntensors>0&&m->ntensors<=INT_MAX,"manifest_counts");
    MEMORYSTATUSEX memory={0};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory),"physical_memory");require((uint64_t)m->ntensors*sizeof(Tensor)<memory.ullTotalPhys,"tensor_metadata_memory");uint64_t mappedbytes=0;
    for(int i=0;i<m->nfiles;i++){char path[1024];string(f,path,sizeof(path));Mapping *p=m->files+i;p->file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);require(p->file!=INVALID_HANDLE_VALUE,"source_file_open");LARGE_INTEGER size;require(GetFileSizeEx(p->file,&size)&&size.QuadPart>0,"source_size");p->bytes=(uint64_t)size.QuadPart;require(p->bytes<=memory.ullTotalPhys-mappedbytes,"complete_target_exceeds_physical_RAM");mappedbytes+=p->bytes;p->map=CreateFileMappingA(p->file,NULL,PAGE_READONLY,0,0,NULL);require(p->map!=NULL,"source_mapping");p->data=MapViewOfFile(p->map,FILE_MAP_READ,0,0,0);require(p->data!=NULL,"source_view");}
    m->tensors=alloc(m->ntensors,sizeof(Tensor));
    for(int i=0;i<m->ntensors;i++){Tensor *t=m->tensors+i;string(f,t->name,sizeof(t->name));uint32_t file=u32(f);t->dims=u32(f);t->shape[0]=u32(f);t->shape[1]=u32(f);t->encoding=u32(f);uint64_t offset=u64(f),scaleoffset=u64(f),elements=u64(f);require(file<(uint32_t)m->nfiles&&(t->dims==1||t->dims==2)&&elements==(uint64_t)t->shape[0]*t->shape[1]&&elements>0&&elements<((uint64_t)1<<31)&&offset%64==0&&t->encoding<=1,"tensor_record");uint64_t bytes=elements*(t->encoding?1:4);require(offset<=m->files[file].bytes&&bytes<=m->files[file].bytes-offset,"target_tensor_bounds");
        if(t->encoding){require(t->dims==2&&t->shape[1]<=4096&&scaleoffset%64==0&&scaleoffset<=m->files[file].bytes&&(uint64_t)t->shape[0]*4<=m->files[file].bytes-scaleoffset,"target_scale_bounds");t->codes=(const int8_t *)(m->files[file].data+offset);t->scales=(const float *)(m->files[file].data+scaleoffset);}else{require(scaleoffset==0,"f32_no_scale");t->value=(const float *)(m->files[file].data+offset);}}
    for(int i=1;i<m->ntensors;i++)require(strcmp(m->tensors[i-1].name,m->tensors[i].name)<0,"sorted_unique_tensor_names");
    require(fgetc(f)==EOF,"manifest_trailing_bytes");fclose(f);
    m->embed=weight(m,"shared.weight",c->vocab,c->d,2);m->head=matrix(m,"lm_head.weight",c->vocab,c->d);m->encnorm=weight(m,"encoder.final_layer_norm.weight",c->d,1,1);m->decnorm=weight(m,"decoder.final_layer_norm.weight",c->d,1,1);
    m->encbias=weight(m,"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",c->buckets,c->heads,2);m->decbias=weight(m,"decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",c->buckets,c->heads,2);
    for(int l=0;l<c->enc;l++)m->enc[l]=block_weights(m,0,l);for(int l=0;l<c->dec;l++)m->dec[l]=block_weights(m,1,l);return m;
}
static void unload(Model *m){for(int i=0;i<m->c.enc;i++){free(m->enc[i].expertwi);free(m->enc[i].expertwo);}for(int i=0;i<m->c.dec;i++){free(m->dec[i].expertwi);free(m->dec[i].expertwo);}for(int i=0;i<m->nfiles;i++){UnmapViewOfFile(m->files[i].data);CloseHandle(m->files[i].map);CloseHandle(m->files[i].file);}free(m->tensors);free(m);}
static void attend(Config c,const float *q,const float *k,const float *v,int length,int position,int bidi,int cross,const float *bias,float *output){
    float scores[CTX],probabilities[CTX];for(int head=0;head<c.heads;head++){int start=head*c.dk;float maximum=-INFINITY;
        for(int t=0;t<length;t++){float score=(float)dot64(q+start,k+(size_t)t*c.d+start,c.dk);if(!cross&&fault!=1)score+=bias[(size_t)bucket(t-position,bidi,c.buckets,c.distance)*c.heads+head];scores[t]=score;if(score>maximum)maximum=score;}
        double sum=0.;for(int t=0;t<length;t++){scores[t]=(float)exp((double)(scores[t]-maximum));sum+=(double)scores[t];}require(sum>0.&&isfinite(sum),"attention_softmax");
        for(int t=0;t<length;t++)probabilities[t]=(float)((double)scores[t]/sum);
        for(int dim=0;dim<c.dk;dim++){double value=0.;for(int t=0;t<length;t++)value+=(double)probabilities[t]*(double)v[(size_t)t*c.d+start+dim];output[start+dim]=(float)value;}
    }
}
#include "meth393_switch_router_trace.h"
/* M417 observer begin */
#include "meth417_switch_function_capture.h"
/* M417 observer end */
static void feed(Config c,Block *b,float *hidden,int tokens){
    float *x=alloc(c.d,sizeof(float)),*up=alloc(c.ff,sizeof(float)),*down=alloc(c.d,sizeof(float));int *counts=b->sparse?alloc(c.n,sizeof(int)):NULL;
    for(int token=0;token<tokens;token++){float *h=hidden+(size_t)token*c.d;norm(h,b->ffnorm,x,c.d,c.epsilon);/* M417 observer begin */function_capture_input(c,x);/* M417 observer end */Matrix wi=b->wi,wo=b->wo;float probability=1.f;int accepted=1;
        if(b->sparse){float *logits=alloc(c.n,sizeof(float));cost_kind=2;mv(b->router,x,logits,c.n,c.d);router_audit(c,x,logits);int chosen=0;for(int e=1;e<c.n;e++)if(logits[e]>logits[chosen])chosen=e;double sum=0.;for(int e=0;e<c.n;e++)sum+=(double)(float)exp((double)(logits[e]-logits[chosen]));probability=(float)(1./sum);counts[chosen]++;accepted=counts[chosen]<=c.capacity||fault==4;require(route_count<route_bound,"route_buffer");routes[route_count++]=(Route){chosen,accepted,probability};wi=b->expertwi[chosen];wo=b->expertwo[chosen];if(fault==3)probability=1.f;free(logits);}
        if(accepted){cost_kind=b->sparse?1:0;mv(wi,x,up,c.ff,c.d);/* M417 observer begin */function_capture_up(c,up);/* M417 observer end */for(int j=0;j<c.ff;j++)if(up[j]<0.f)up[j]=0.f;/* M417 observer begin */function_capture_relu(c,up);/* M417 observer end */mv(wo,up,down,c.d,c.ff);/* M417 observer begin */function_capture_down(c,down);/* M417 observer end */for(int j=0;j<c.d;j++)h[j]+=probability*down[j];}/* M417 observer begin */function_capture_post(c,h);/* M417 observer end */
    }free(x);free(up);free(down);free(counts);cost_kind=0;
}
static float *encode(Model *m,const int *ids,int tokens,float *snap){
    Config c=m->c;size_t width=(size_t)tokens*c.d;float *h=alloc(width,sizeof(float)),*x=alloc(width,sizeof(float)),*q=alloc(width,sizeof(float)),*k=alloc(width,sizeof(float)),*v=alloc(width,sizeof(float)),*att=alloc(c.d,sizeof(float)),*out=alloc(c.d,sizeof(float));
    for(int t=0;t<tokens;t++)memcpy(h+(size_t)t*c.d,m->embed+(size_t)ids[t]*c.d,c.d*4);memcpy(snap,h,width*4);
    for(int l=0;l<c.enc;l++){Block *b=m->enc+l;for(int t=0;t<tokens;t++){size_t off=(size_t)t*c.d;norm(h+off,b->selfnorm,x+off,c.d,c.epsilon);}mv_batch(b->self.q,x,q,c.d,c.d,tokens);mv_batch(b->self.k,x,k,c.d,c.d,tokens);mv_batch(b->self.v,x,v,c.d,c.d,tokens);
        for(int t=0;t<tokens;t++){size_t off=(size_t)t*c.d;attend(c,q+off,k,v,tokens,t,1,0,m->encbias,att);mv(b->self.o,att,out,c.d,c.d);for(int j=0;j<c.d;j++)h[off+j]+=out[j];}
        feed(c,b,h,tokens);memcpy(snap+(size_t)(l+1)*width,h,width*4);
    }for(int t=0;t<tokens;t++)norm(h+(size_t)t*c.d,m->encnorm,x+(size_t)t*c.d,c.d,c.epsilon);memcpy(snap+(size_t)(c.enc+1)*width,x,width*4);memcpy(h,x,width*4);
    free(x);free(q);free(k);free(v);free(att);free(out);return h;
}
static Cache *cache_init(Model *m,const float *encoder,int tokens){Config c=m->c;Cache *cache=alloc(c.dec,sizeof(Cache));for(int l=0;l<c.dec;l++){Cache *a=cache+l;a->selfk=alloc((size_t)CTX*c.d,4);a->selfv=alloc((size_t)CTX*c.d,4);a->crossk=alloc((size_t)tokens*c.d,4);a->crossv=alloc((size_t)tokens*c.d,4);mv_batch(m->dec[l].cross.k,encoder,a->crossk,c.d,c.d,tokens);mv_batch(m->dec[l].cross.v,encoder,a->crossv,c.d,c.d,tokens);if(fault==2)memset(a->crossv,0,(size_t)tokens*c.d*4);}return cache;}
static void decode(Model *m,Cache *cache,int id,int position,int source_tokens,float *snap,float *logits){
    Config c=m->c;float *h=alloc(c.d,4),*x=alloc(c.d,4),*q=alloc(c.d,4),*att=alloc(c.d,4),*out=alloc(c.d,4);memcpy(h,m->embed+(size_t)id*c.d,c.d*4);memcpy(snap,h,c.d*4);
    for(int l=0;l<c.dec;l++){Block *b=m->dec+l;Cache *a=cache+l;norm(h,b->selfnorm,x,c.d,c.epsilon);mv(b->self.q,x,q,c.d,c.d);mv(b->self.k,x,a->selfk+(size_t)position*c.d,c.d,c.d);mv(b->self.v,x,a->selfv+(size_t)position*c.d,c.d,c.d);attend(c,q,a->selfk,a->selfv,position+1,position,0,0,m->decbias,att);mv(b->self.o,att,out,c.d,c.d);for(int j=0;j<c.d;j++)h[j]+=out[j];
        norm(h,b->crossnorm,x,c.d,c.epsilon);mv(b->cross.q,x,q,c.d,c.d);attend(c,q,a->crossk,a->crossv,source_tokens,position,0,1,NULL,att);mv(b->cross.o,att,out,c.d,c.d);for(int j=0;j<c.d;j++)h[j]+=out[j];/* M417 observer begin */if(l==c.dec-1){require(b->sparse,"function_capture_final_sparse");function_capture_begin(c,h,position,id,source_tokens);}/* M417 observer end */feed(c,b,h,1);memcpy(snap+(size_t)(l+1)*c.d,h,c.d*4);
    }norm(h,m->decnorm,x,c.d,c.epsilon);/* M417 observer begin */function_capture_final(c,x);/* M417 observer end */memcpy(snap+(size_t)(c.dec+1)*c.d,x,c.d*4);float scale=(float)(1./sqrt((double)c.d));for(int j=0;j<c.d;j++)x[j]*=scale;cost_kind=3;head_mv(m->head,x,logits,c.vocab,c.d);/* M417 observer begin */function_capture_finish(c,x);/* M417 observer end */cost_kind=0;if(fault==5)memset(logits,0,c.vocab*4);free(h);free(x);free(q);free(att);free(out);
}
static int ids(const char *text,int *result,int vocab){char copy[4096];require(strlen(text)<sizeof(copy),"ids_string");strcpy(copy,text);int n=0;char *token=strtok(copy,",");while(token){require(n<CTX,"context_bound");char *end;long value=strtol(token,&end,10);require(*end==0&&value>=0&&value<vocab,"token_id");result[n++]=(int)value;token=strtok(NULL,",");}require(n>0,"empty_ids");return n;}
static int integer_controls(const char *input,const char *output){
    FILE *f=fopen(input,"rb"),*out=fopen(output,"wb");require(f&&out,"head_control_files");char magic[8];require(fread(magic,1,8,f)==8&&!memcmp(magic,"SWI8D001",8),"head_control_magic");uint32_t cases=u32(f);require(cases>0&&cases<=32,"head_control_count");fwrite("SW16R001",1,8,out);fwrite(&cases,4,1,out);
    for(uint32_t i=0;i<cases;i++){uint32_t rows=u32(f),cols=u32(f);require(rows>0&&rows<=65536&&cols>0&&cols<=4096,"head_control_dims");int8_t *weights=alloc((size_t)rows*cols,1);int16_t *codes=alloc(cols,2);float *scales=alloc(rows,4),*x=alloc(cols,4),*y=alloc(rows,4);require(fread(weights,1,(size_t)rows*cols,f)==(size_t)rows*cols&&fread(scales,4,rows,f)==rows&&fread(x,4,cols,f)==cols,"head_control_payload");for(size_t j=0;j<(size_t)rows*cols;j++)require(weights[j]!=-128,"head_control_weight_codes");for(uint32_t j=0;j<rows;j++)require(isfinite(scales[j])&&scales[j]>0.f,"head_control_scales");Matrix w={NULL,scales,weights,rows,cols};mv(w,x,y,rows,cols);float scale=head_activation_codes(x,codes,cols);fwrite(&rows,4,1,out);fwrite(&cols,4,1,out);fwrite(y,4,rows,out);fwrite(codes,2,cols,out);fwrite(&scale,4,1,out);for(uint32_t row=0;row<rows;row++){int64_t value=head_integer_dot(weights+(size_t)row*cols,codes,cols);fwrite(&value,8,1,out);}free(weights);free(codes);free(scales);free(x);free(y);}
    require(fgetc(f)==EOF&&fclose(f)==0&&fclose(out)==0,"head_control_close");return 0;
}
static int head_integer_controls(const char *input,const char *output){
    FILE *f=fopen(input,"rb"),*out=fopen(output,"wb");require(f&&out,"head_control_files");char magic[8];require(fread(magic,1,8,f)==8&&!memcmp(magic,"SWI8D001",8),"head_control_magic");uint32_t cases=u32(f);require(cases>0&&cases<=32,"head_control_count");fwrite("SW16R001",1,8,out);fwrite(&cases,4,1,out);
    for(uint32_t i=0;i<cases;i++){uint32_t rows=u32(f),cols=u32(f);require(rows>0&&rows<=65536&&cols>0&&cols<=4096,"head_control_dims");int8_t *weights=alloc((size_t)rows*cols,1);int16_t *codes=alloc(cols,2);float *scales=alloc(rows,4),*x=alloc(cols,4),*y=alloc(rows,4);require(fread(weights,1,(size_t)rows*cols,f)==(size_t)rows*cols&&fread(scales,4,rows,f)==rows&&fread(x,4,cols,f)==cols,"head_control_payload");for(size_t j=0;j<(size_t)rows*cols;j++)require(weights[j]!=-128,"head_control_weight_codes");for(uint32_t j=0;j<rows;j++)require(isfinite(scales[j])&&scales[j]>0.f,"head_control_scales");Matrix w={NULL,scales,weights,rows,cols};head_mv(w,x,y,rows,cols);float scale=head_activation_codes(x,codes,cols);fwrite(&rows,4,1,out);fwrite(&cols,4,1,out);fwrite(y,4,rows,out);fwrite(codes,2,cols,out);fwrite(&scale,4,1,out);for(uint32_t row=0;row<rows;row++){int64_t value=head_integer_dot(weights+(size_t)row*cols,codes,cols);fwrite(&value,8,1,out);}free(weights);free(codes);free(scales);free(x);free(y);}
    require(fgetc(f)==EOF&&fclose(f)==0&&fclose(out)==0,"head_control_close");return 0;
}
int meth417_contract_entry(int argc,char **argv){
    if(argc==4&&!strcmp(argv[1],"--head-int-dot"))return head_integer_controls(argv[2],argv[3]);
    if(argc==4&&!strcmp(argv[1],"--int-dot"))return integer_controls(argv[2],argv[3]);
    if(argc==2&&!strcmp(argv[1],"--buckets")){int positions[]={-4096,-2048,-128,-64,-16,-8,-1,0,1,8,16,64,128,2048,4096};printf("[");for(int bidi=0;bidi<=1;bidi++)for(int i=0;i<15;i++)printf("%s%d",bidi||i?",":"",bucket(positions[i],bidi,32,128));puts("]");return 0;}
    require(argc==6,"arguments:manifest source_csv decoder_csv output fault");fault=atoi(argv[5]);require(fault>=0&&fault<=5,"fault");omp_set_dynamic(0);omp_set_num_threads(1);Model *m=load(argv[1]);Config c=m->c;int encids[CTX],decids[CTX];int s=ids(argv[2],encids,c.vocab),t=ids(argv[3],decids,c.vocab);
    route_bound=(c.enc*s+c.dec*t);routes=alloc(route_bound,sizeof(Route));size_t enc_count=(size_t)(c.enc+2)*s*c.d,dec_count=(size_t)t*(c.dec+2)*c.d,logit_count=(size_t)t*c.vocab;float *encsnap=alloc(enc_count,4),*decsnap=alloc(dec_count,4),*logits=alloc(logit_count,4);float *encoder=encode(m,encids,s,encsnap);Cache *cache=cache_init(m,encoder,s);for(int step=0;step<t;step++)decode(m,cache,decids[step],step,s,decsnap+(size_t)step*(c.dec+2)*c.d,logits+(size_t)step*c.vocab);
    for(size_t i=0;i<enc_count;i++)require(isfinite(encsnap[i]),"encoder_finite");for(size_t i=0;i<dec_count;i++)require(isfinite(decsnap[i]),"decoder_finite");for(size_t i=0;i<logit_count;i++)require(isfinite(logits[i]),"logits_finite");FILE *f=fopen(argv[4],"wb");require(f!=NULL,"output_open");fwrite("SWR32O01",1,8,f);uint32_t header[]={s,t,c.d,c.enc,c.dec,c.vocab,route_count};fwrite(header,4,7,f);fwrite(encsnap,4,enc_count,f);fwrite(decsnap,4,dec_count,f);fwrite(logits,4,logit_count,f);require(sizeof(Route)==12,"route_layout");fwrite(routes,sizeof(Route),route_count,f);require(fclose(f)==0,"output_close");printf("{\"encoder_tokens\":%d,\"decoder_tokens\":%d,\"routes\":%d,\"source_experts\":%d}\n",s,t,route_count,c.n);
    for(int l=0;l<c.dec;l++){free(cache[l].selfk);free(cache[l].selfv);free(cache[l].crossk);free(cache[l].crossv);}free(cache);free(encoder);free(encsnap);free(decsnap);free(logits);free(routes);unload(m);return 0;
}
