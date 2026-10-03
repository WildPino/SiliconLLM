/* Complete Switch/T5 FP32 arithmetic bridge; no accepted-rate claim. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#define CTX 256
#define MAX_L 24
#define MAX_E 256
#define MAX_FILES 8

static void require(int ok,const char *why){if(!ok){fprintf(stderr,"switch_reference_error:%s\n",why);exit(2);}}
static void *alloc(size_t count,size_t width){void *p=calloc(count,width);require(p!=NULL,"allocation");return p;}
static uint32_t u32(FILE *f){uint32_t v;require(fread(&v,4,1,f)==1,"u32");return v;}
static uint64_t u64(FILE *f){uint64_t v;require(fread(&v,8,1,f)==1,"u64");return v;}
static void string(FILE *f,char *s,size_t bound){uint32_t n=u32(f);require(n>0&&n<bound,"string_length");require(fread(s,1,n,f)==n,"string");s[n]=0;}
typedef struct {HANDLE file,map;unsigned char *data;uint64_t bytes;} Mapping;
typedef struct {char name[192];uint32_t dims,shape[2];const float *value;} Tensor;
typedef struct {int d,ff,heads,dk,enc,dec,n,capacity,vocab,buckets,distance,encstep,decstep;float epsilon;} Config;
typedef struct {const float *q,*k,*v,*o;} Attention;
typedef struct {const float *selfnorm,*crossnorm,*ffnorm,*wi,*wo,*router,*expertwi[MAX_E],*expertwo[MAX_E];Attention self,cross;int sparse;} Block;
typedef struct {Config c;Mapping files[MAX_FILES];int nfiles,ntensors;Tensor *tensors;Block enc[MAX_L],dec[MAX_L];const float *embed,*encnorm,*decnorm,*encbias,*decbias;} Model;
typedef struct {int32_t expert,accepted;float probability;} Route;
typedef struct {float *selfk,*selfv,*crossk,*crossv;} Cache;
static Route *routes;static int route_count,route_bound,fault;

static float dot(const float *a,const float *b,int n){
    __m256 x=_mm256_setzero_ps(),y=_mm256_setzero_ps();int i=0;
    for(;i+16<=n;i+=16){x=_mm256_add_ps(x,_mm256_mul_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));y=_mm256_add_ps(y,_mm256_mul_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8)));}
    float v[8];_mm256_storeu_ps(v,_mm256_add_ps(x,y));float sum=0.f;
    for(int j=0;j<8;j++)sum+=v[j];for(;i<n;i++)sum+=a[i]*b[i];return sum;
}
static double dot64(const float *a,const float *b,int n){__m256d x=_mm256_setzero_pd(),y=_mm256_setzero_pd();int i=0;for(;i+8<=n;i+=8){x=_mm256_add_pd(x,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),_mm256_cvtps_pd(_mm_loadu_ps(b+i))));y=_mm256_add_pd(y,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),_mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));}double values[4];_mm256_storeu_pd(values,_mm256_add_pd(x,y));double sum=0.;for(int j=0;j<4;j++)sum+=values[j];for(;i<n;i++)sum+=(double)a[i]*(double)b[i];return sum;}
static void mv(const float *w,const float *x,float *y,int rows,int cols){
    #pragma omp parallel for schedule(static) if(rows>64)
    for(int row=0;row<rows;row++)y[row]=(float)dot64(w+(size_t)row*cols,x,cols);
}
static void norm(const float *x,const float *w,float *y,int d,float epsilon){double sum=0.;for(int j=0;j<d;j++){float square=x[j]*x[j];sum+=(double)square;}float mean=(float)(sum/(double)d);float scale=1.f/sqrtf(mean+epsilon);for(int j=0;j<d;j++)y[j]=x[j]*scale*w[j];}
static int bucket(int relative,int bidi,int count,int distance){
    int base=0;if(bidi){count/=2;if(relative>0)base=count;relative=abs(relative);}else relative=relative<0?-relative:0;
    int exact=count/2;if(relative<exact)return base+relative;
    int large=exact+(int)(logf((float)relative/(float)exact)/logf((float)distance/(float)exact)*(float)(count-exact));
    if(large>=count)large=count-1;return base+large;
}
static const float *weight(Model *m,const char *name,int rows,int cols,int dimensions){
    for(int i=0;i<m->ntensors;i++)if(!strcmp(m->tensors[i].name,name)){Tensor *t=m->tensors+i;require(t->dims==(uint32_t)dimensions&&t->shape[0]==(uint32_t)rows&&t->shape[1]==(uint32_t)cols,"tensor_shape");return t->value;}
    fprintf(stderr,"missing_tensor:%s\n",name);exit(2);
}
static const float *organ(Model *m,const char *prefix,const char *suffix,int rows,int cols,int dimensions){char name[192];require(snprintf(name,sizeof(name),"%s.%s",prefix,suffix)<(int)sizeof(name),"tensor_name");return weight(m,name,rows,cols,dimensions);}
static Attention attention_weights(Model *m,const char *prefix){int d=m->c.d;Attention a={organ(m,prefix,"q.weight",d,d,2),organ(m,prefix,"k.weight",d,d,2),organ(m,prefix,"v.weight",d,d,2),organ(m,prefix,"o.weight",d,d,2)};return a;}
static Block block_weights(Model *m,int decoder,int index){
    Config c=m->c;Block b={0};char prefix[192],part[192];const char *stack=decoder?"decoder":"encoder";
    snprintf(prefix,sizeof(prefix),"%s.block.%d.layer.0",stack,index);b.selfnorm=organ(m,prefix,"layer_norm.weight",c.d,1,1);
    snprintf(part,sizeof(part),"%s.SelfAttention",prefix);b.self=attention_weights(m,part);
    if(decoder){snprintf(prefix,sizeof(prefix),"decoder.block.%d.layer.1",index);b.crossnorm=organ(m,prefix,"layer_norm.weight",c.d,1,1);snprintf(part,sizeof(part),"%s.EncDecAttention",prefix);b.cross=attention_weights(m,part);}
    snprintf(prefix,sizeof(prefix),"%s.block.%d.layer.%d",stack,index,decoder?2:1);b.ffnorm=organ(m,prefix,"layer_norm.weight",c.d,1,1);
    int step=decoder?c.decstep:c.encstep;b.sparse=step>0&&(index%step==1||step==1);
    snprintf(part,sizeof(part),"%s.mlp",prefix);
    if(b.sparse){b.router=organ(m,part,"router.classifier.weight",c.n,c.d,2);for(int e=0;e<c.n;e++){char suffix[80];snprintf(suffix,sizeof(suffix),"experts.expert_%d.wi.weight",e);b.expertwi[e]=organ(m,part,suffix,c.ff,c.d,2);snprintf(suffix,sizeof(suffix),"experts.expert_%d.wo.weight",e);b.expertwo[e]=organ(m,part,suffix,c.d,c.ff,2);}}
    else{b.wi=organ(m,part,"wi.weight",c.ff,c.d,2);b.wo=organ(m,part,"wo.weight",c.d,c.ff,2);}return b;
}
static Model *load(const char *spec){
    Model *m=alloc(1,sizeof(Model));FILE *f=fopen(spec,"rb");require(f!=NULL,"manifest_open");char magic[8];require(fread(magic,1,8,f)==8&&!memcmp(magic,"SWF32A01",8),"manifest_magic");
    Config *c=&m->c;c->d=u32(f);c->ff=u32(f);c->heads=u32(f);c->dk=u32(f);c->enc=u32(f);c->dec=u32(f);c->n=u32(f);c->capacity=u32(f);c->vocab=u32(f);c->buckets=u32(f);c->distance=u32(f);c->encstep=u32(f);c->decstep=u32(f);require(fread(&c->epsilon,4,1,f)==1,"epsilon");
    require(c->d>0&&c->d<=1024&&c->ff>0&&c->ff<=4096&&c->heads>0&&c->heads*c->dk==c->d&&c->enc>0&&c->enc<=MAX_L&&c->dec>0&&c->dec<=MAX_L&&c->n>0&&c->n<=MAX_E&&c->capacity>0&&c->vocab>0&&c->vocab<=65536&&c->buckets==32&&c->distance==128&&c->epsilon>0.f&&c->epsilon<1.f,"configuration");
    m->nfiles=u32(f);m->ntensors=u32(f);require(m->nfiles>0&&m->nfiles<=MAX_FILES&&m->ntensors>0&&m->ntensors<=15000,"manifest_counts");
    for(int i=0;i<m->nfiles;i++){char path[1024];string(f,path,sizeof(path));Mapping *p=m->files+i;p->file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);require(p->file!=INVALID_HANDLE_VALUE,"source_file_open");LARGE_INTEGER size;require(GetFileSizeEx(p->file,&size)&&size.QuadPart>0,"source_size");p->bytes=(uint64_t)size.QuadPart;p->map=CreateFileMappingA(p->file,NULL,PAGE_READONLY,0,0,NULL);require(p->map!=NULL,"source_mapping");p->data=MapViewOfFile(p->map,FILE_MAP_READ,0,0,0);require(p->data!=NULL,"source_view");}
    m->tensors=alloc(m->ntensors,sizeof(Tensor));
    for(int i=0;i<m->ntensors;i++){Tensor *t=m->tensors+i;string(f,t->name,sizeof(t->name));uint32_t file=u32(f);t->dims=u32(f);t->shape[0]=u32(f);t->shape[1]=u32(f);uint64_t offset=u64(f),elements=u64(f);require(file<(uint32_t)m->nfiles&&(t->dims==1||t->dims==2)&&elements==(uint64_t)t->shape[0]*t->shape[1]&&elements>0&&elements<((uint64_t)1<<31)&&offset%4==0,"tensor_record");require(offset<=m->files[file].bytes&&elements*4<=m->files[file].bytes-offset,"source_tensor_bounds");t->value=(const float *)(m->files[file].data+offset);}
    require(fgetc(f)==EOF,"manifest_trailing_bytes");fclose(f);
    m->embed=weight(m,"shared.weight",c->vocab,c->d,2);m->encnorm=weight(m,"encoder.final_layer_norm.weight",c->d,1,1);m->decnorm=weight(m,"decoder.final_layer_norm.weight",c->d,1,1);
    m->encbias=weight(m,"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",c->buckets,c->heads,2);m->decbias=weight(m,"decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",c->buckets,c->heads,2);
    for(int l=0;l<c->enc;l++)m->enc[l]=block_weights(m,0,l);for(int l=0;l<c->dec;l++)m->dec[l]=block_weights(m,1,l);return m;
}
static void unload(Model *m){for(int i=0;i<m->nfiles;i++){UnmapViewOfFile(m->files[i].data);CloseHandle(m->files[i].map);CloseHandle(m->files[i].file);}free(m->tensors);free(m);}
static void attend(Config c,const float *q,const float *k,const float *v,int length,int position,int bidi,int cross,const float *bias,float *output){
    float scores[CTX];for(int head=0;head<c.heads;head++){int start=head*c.dk;float maximum=-INFINITY;
        for(int t=0;t<length;t++){float score=dot(q+start,k+(size_t)t*c.d+start,c.dk);if(!cross&&fault!=1)score+=bias[(size_t)bucket(t-position,bidi,c.buckets,c.distance)*c.heads+head];scores[t]=score;if(score>maximum)maximum=score;}
        float sum=0.f;for(int t=0;t<length;t++){scores[t]=expf(scores[t]-maximum);sum+=scores[t];}require(sum>0.f&&isfinite(sum),"attention_softmax");
        for(int dim=0;dim<c.dk;dim++){float value=0.f;for(int t=0;t<length;t++)value+=(scores[t]/sum)*v[(size_t)t*c.d+start+dim];output[start+dim]=value;}
    }
}
static void feed(Config c,Block *b,float *hidden,int tokens){
    float *x=alloc(c.d,sizeof(float)),*up=alloc(c.ff,sizeof(float)),*down=alloc(c.d,sizeof(float));int counts[MAX_E]={0};
    for(int token=0;token<tokens;token++){float *h=hidden+(size_t)token*c.d;norm(h,b->ffnorm,x,c.d,c.epsilon);const float *wi=b->wi,*wo=b->wo;float probability=1.f;int accepted=1;
        if(b->sparse){float logits[MAX_E];mv(b->router,x,logits,c.n,c.d);int chosen=0;for(int e=1;e<c.n;e++)if(logits[e]>logits[chosen])chosen=e;float sum=0.f;for(int e=0;e<c.n;e++)sum+=expf(logits[e]-logits[chosen]);probability=1.f/sum;counts[chosen]++;accepted=counts[chosen]<=c.capacity||fault==4;require(route_count<route_bound,"route_buffer");routes[route_count++]=(Route){chosen,accepted,probability};wi=b->expertwi[chosen];wo=b->expertwo[chosen];if(fault==3)probability=1.f;}
        if(accepted){mv(wi,x,up,c.ff,c.d);for(int j=0;j<c.ff;j++)if(up[j]<0.f)up[j]=0.f;mv(wo,up,down,c.d,c.ff);for(int j=0;j<c.d;j++)h[j]+=probability*down[j];}
    }free(x);free(up);free(down);
}
static float *encode(Model *m,const int *ids,int tokens,float *snap){
    Config c=m->c;size_t width=(size_t)tokens*c.d;float *h=alloc(width,sizeof(float)),*x=alloc(width,sizeof(float)),*q=alloc(width,sizeof(float)),*k=alloc(width,sizeof(float)),*v=alloc(width,sizeof(float)),*att=alloc(c.d,sizeof(float)),*out=alloc(c.d,sizeof(float));
    for(int t=0;t<tokens;t++)memcpy(h+(size_t)t*c.d,m->embed+(size_t)ids[t]*c.d,c.d*4);memcpy(snap,h,width*4);
    for(int l=0;l<c.enc;l++){Block *b=m->enc+l;for(int t=0;t<tokens;t++){size_t off=(size_t)t*c.d;norm(h+off,b->selfnorm,x+off,c.d,c.epsilon);mv(b->self.q,x+off,q+off,c.d,c.d);mv(b->self.k,x+off,k+off,c.d,c.d);mv(b->self.v,x+off,v+off,c.d,c.d);}
        for(int t=0;t<tokens;t++){size_t off=(size_t)t*c.d;attend(c,q+off,k,v,tokens,t,1,0,m->encbias,att);mv(b->self.o,att,out,c.d,c.d);for(int j=0;j<c.d;j++)h[off+j]+=out[j];}
        feed(c,b,h,tokens);memcpy(snap+(size_t)(l+1)*width,h,width*4);
    }for(int t=0;t<tokens;t++)norm(h+(size_t)t*c.d,m->encnorm,x+(size_t)t*c.d,c.d,c.epsilon);memcpy(snap+(size_t)(c.enc+1)*width,x,width*4);memcpy(h,x,width*4);
    free(x);free(q);free(k);free(v);free(att);free(out);return h;
}
static Cache *cache_init(Model *m,const float *encoder,int tokens){Config c=m->c;Cache *cache=alloc(c.dec,sizeof(Cache));for(int l=0;l<c.dec;l++){Cache *a=cache+l;a->selfk=alloc((size_t)CTX*c.d,4);a->selfv=alloc((size_t)CTX*c.d,4);a->crossk=alloc((size_t)tokens*c.d,4);a->crossv=alloc((size_t)tokens*c.d,4);for(int t=0;t<tokens;t++){mv(m->dec[l].cross.k,encoder+(size_t)t*c.d,a->crossk+(size_t)t*c.d,c.d,c.d);mv(m->dec[l].cross.v,encoder+(size_t)t*c.d,a->crossv+(size_t)t*c.d,c.d,c.d);}if(fault==2)memset(a->crossv,0,(size_t)tokens*c.d*4);}return cache;}
static void decode(Model *m,Cache *cache,int id,int position,int source_tokens,float *snap,float *logits){
    Config c=m->c;float *h=alloc(c.d,4),*x=alloc(c.d,4),*q=alloc(c.d,4),*att=alloc(c.d,4),*out=alloc(c.d,4);memcpy(h,m->embed+(size_t)id*c.d,c.d*4);memcpy(snap,h,c.d*4);
    for(int l=0;l<c.dec;l++){Block *b=m->dec+l;Cache *a=cache+l;norm(h,b->selfnorm,x,c.d,c.epsilon);mv(b->self.q,x,q,c.d,c.d);mv(b->self.k,x,a->selfk+(size_t)position*c.d,c.d,c.d);mv(b->self.v,x,a->selfv+(size_t)position*c.d,c.d,c.d);attend(c,q,a->selfk,a->selfv,position+1,position,0,0,m->decbias,att);mv(b->self.o,att,out,c.d,c.d);for(int j=0;j<c.d;j++)h[j]+=out[j];
        norm(h,b->crossnorm,x,c.d,c.epsilon);mv(b->cross.q,x,q,c.d,c.d);attend(c,q,a->crossk,a->crossv,source_tokens,position,0,1,NULL,att);mv(b->cross.o,att,out,c.d,c.d);for(int j=0;j<c.d;j++)h[j]+=out[j];feed(c,b,h,1);memcpy(snap+(size_t)(l+1)*c.d,h,c.d*4);
    }norm(h,m->decnorm,x,c.d,c.epsilon);memcpy(snap+(size_t)(c.dec+1)*c.d,x,c.d*4);float scale=1.f/sqrtf((float)c.d);for(int j=0;j<c.d;j++)x[j]*=scale;mv(m->embed,x,logits,c.vocab,c.d);if(fault==5)memset(logits,0,c.vocab*4);free(h);free(x);free(q);free(att);free(out);
}
static int ids(const char *text,int *result,int vocab){char copy[4096];require(strlen(text)<sizeof(copy),"ids_string");strcpy(copy,text);int n=0;char *token=strtok(copy,",");while(token){require(n<CTX,"context_bound");char *end;long value=strtol(token,&end,10);require(*end==0&&value>=0&&value<vocab,"token_id");result[n++]=(int)value;token=strtok(NULL,",");}require(n>0,"empty_ids");return n;}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--buckets")){int positions[]={-4096,-2048,-128,-64,-16,-8,-1,0,1,8,16,64,128,2048,4096};printf("[");for(int bidi=0;bidi<=1;bidi++)for(int i=0;i<15;i++)printf("%s%d",bidi||i?",":"",bucket(positions[i],bidi,32,128));puts("]");return 0;}
    require(argc==6,"arguments:manifest source_csv decoder_csv output fault");fault=atoi(argv[5]);require(fault>=0&&fault<=5,"fault");omp_set_dynamic(0);omp_set_num_threads(1);Model *m=load(argv[1]);Config c=m->c;int encids[CTX],decids[CTX];int s=ids(argv[2],encids,c.vocab),t=ids(argv[3],decids,c.vocab);
    route_bound=(c.enc*s+c.dec*t);routes=alloc(route_bound,sizeof(Route));size_t enc_count=(size_t)(c.enc+2)*s*c.d,dec_count=(size_t)t*(c.dec+2)*c.d,logit_count=(size_t)t*c.vocab;float *encsnap=alloc(enc_count,4),*decsnap=alloc(dec_count,4),*logits=alloc(logit_count,4);float *encoder=encode(m,encids,s,encsnap);Cache *cache=cache_init(m,encoder,s);for(int step=0;step<t;step++)decode(m,cache,decids[step],step,s,decsnap+(size_t)step*(c.dec+2)*c.d,logits+(size_t)step*c.vocab);
    for(size_t i=0;i<enc_count;i++)require(isfinite(encsnap[i]),"encoder_finite");for(size_t i=0;i<dec_count;i++)require(isfinite(decsnap[i]),"decoder_finite");for(size_t i=0;i<logit_count;i++)require(isfinite(logits[i]),"logits_finite");FILE *f=fopen(argv[4],"wb");require(f!=NULL,"output_open");fwrite("SWR32O01",1,8,f);uint32_t header[]={s,t,c.d,c.enc,c.dec,c.vocab,route_count};fwrite(header,4,7,f);fwrite(encsnap,4,enc_count,f);fwrite(decsnap,4,dec_count,f);fwrite(logits,4,logit_count,f);require(sizeof(Route)==12,"route_layout");fwrite(routes,sizeof(Route),route_count,f);require(fclose(f)==0,"output_close");printf("{\"encoder_tokens\":%d,\"decoder_tokens\":%d,\"routes\":%d,\"source_experts\":%d}\n",s,t,route_count,c.n);
    for(int l=0;l<c.dec;l++){free(cache[l].selfk);free(cache[l].selfv);free(cache[l].crossk);free(cache[l].crossv);}free(cache);free(encoder);free(encsnap);free(decsnap);free(logits);free(routes);unload(m);return 0;
}
