/* Frozen support2/root16 physical-vector inquiry; no model replay or timing claim. */
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
#define U 238872
#define VECTORS 268
static void need(int v,const char *why){if(!v){fprintf(stderr,"learned_router_error:%s\n",why);fflush(stderr);exit(2);}}
static void readn(FILE *f,void *p,size_t n){need(fread(p,1,n,f)==n,"short_read");}
static void writen(FILE *f,const void *p,size_t n){need(fwrite(p,1,n,f)==n,"short_write");}
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
static double source_mass(const float *s,int n,int chosen){double z=0.;for(int j=0;j<n;j++){float d=s[j]-s[chosen];float v=(float)exp((double)d);z+=(double)v;}need(z>0.&&isfinite(z),"positive_mass");return z;}
typedef struct {uint32_t left,right,parent,count,first,k,offset,reserved;uint32_t ids[N];} Node;
typedef struct {float vectors[VECTORS][D];double log_b[16];Node nodes[255];} Bank;
static Bank banks[12];
static void load(const char *path){
    FILE *ledger=openr(path);char magic[8];readn(ledger,magic,8);need(!memcmp(magic,"M480LED1",8),"model_ledger_magic");need(u32(ledger)==12&&u32(ledger)==0,"ledger_dimensions");
    uint32_t seen=0;
    for(uint32_t ordinal=0;ordinal<12;ordinal++){
        uint32_t bank=u32(ledger);need(bank==ordinal&&!(seen&(1u<<bank)),"bank_order");seen|=1u<<bank;char *p=string(ledger);FILE *f=openr(p);free(p);
        readn(f,magic,8);need(!memcmp(magic,"M480MOD1",8),"model_magic");uint32_t header[6];readn(f,header,24);need(header[0]==N&&header[1]==bank&&header[2]==D&&header[3]==255&&header[4]==VECTORS&&header[5]==16,"model_dimensions");
        Bank *b=&banks[bank];readn(f,b->vectors,sizeof(b->vectors));readn(f,b->log_b,sizeof(b->log_b));for(int v=0;v<VECTORS;v++)for(int d=0;d<D;d++)need(isfinite(b->vectors[v][d]),"vector_finite");
        double total_b=0.;for(int j=0;j<16;j++){need(isfinite(b->log_b[j]),"log_b_finite");total_b+=exp(b->log_b[j]);}need(fabs(total_b-128.)<=1e-10,"mass_sum128");
        uint64_t proto_seen=0;
        for(uint32_t j=0;j<255;j++){
            Node *g=&b->nodes[j];readn(f,g,32);need(g->reserved==0&&g->count>0&&g->count<=N&&!(g->count&(g->count-1)),"node_fields");readn(f,g->ids,g->count*4);
            need(g->first==g->ids[0]&&g->ids[g->count-1]<N,"member_range");for(uint32_t k=1;k<g->count;k++)need(g->ids[k-1]<g->ids[k],"member_order");
            if(!j)need(g->count==N&&g->parent==UINT32_MAX&&g->k==0&&g->offset==UINT32_MAX,"root");
            else if(g->count<=2)need(g->k==g->count&&g->offset==UINT32_MAX,"exact_group");
            else{need(g->k==2&&g->offset>=128&&g->offset<252&&!(g->offset&1),"prototype_group");uint32_t k=(g->offset-128)/2;need(!(proto_seen&(1ull<<k)),"unique_prototype_group");proto_seen|=1ull<<k;}
        }
        need(proto_seen==((1ull<<62)-1),"ALL62prototype_groups");
        for(uint32_t j=0;j<255;j++){
            Node *g=&b->nodes[j];if(j){need(g->parent<j,"parent_order");Node *pa=&b->nodes[g->parent];need(pa->left==j||pa->right==j,"parent_child");}
            if(g->count==1){need(g->left==UINT32_MAX&&g->right==UINT32_MAX,"leaf");continue;}
            need(g->left==j+1&&g->right==j+g->count&&g->right<255,"preorder_children");Node *l=&b->nodes[g->left],*r=&b->nodes[g->right];
            need(l->parent==j&&r->parent==j&&l->count==g->count/2&&r->count==g->count/2,"balanced_children");
            uint32_t li=0,ri=0;for(uint32_t k=0;k<g->count;k++){uint32_t value;if(li<l->count&&(ri==r->count||l->ids[li]<r->ids[ri]))value=l->ids[li++];else value=r->ids[ri++];need(value==g->ids[k],"child_partition");}
        }
        for(uint32_t j=0;j<N;j++)need(b->nodes[0].ids[j]==j,"root_all_IDs");end(f);
    }
    end(ledger);need(seen==4095,"ALL12banks");
}
typedef struct {uint32_t meta[12];float value[3];double root[4];uint16_t vector_ids[42];float score[42];uint32_t direction[7];} Prediction;
static float evaluate(Bank *b,uint32_t vector,const float *x,Prediction *out){
    need(vector<VECTORS&&out->meta[4]<42,"vector_slot");uint32_t at=out->meta[4]++;float value=(float)dot(b->vectors[vector],x);need(isfinite(value),"dot_finite");out->vector_ids[at]=(uint16_t)vector;out->score[at]=value;out->meta[7]+=D;out->meta[8]+=D*4;return value;
}
static float support(Bank *b,uint32_t node,const float *x,Prediction *out,uint32_t *key){
    Node *g=&b->nodes[node];out->meta[9]+=32;float best=-INFINITY;uint32_t chosen=g->first;
    if(g->count<=2){out->meta[10]+=g->count*4;for(uint32_t j=0;j<g->count;j++){uint32_t e=g->ids[j];float v=evaluate(b,e,x,out);if(j==0||v>best||(v==best&&e<chosen)){best=v;chosen=e;}}}
    else{for(uint32_t j=0;j<2;j++){float v=evaluate(b,g->offset+j,x,out);if(j==0||v>best)best=v;}}
    *key=chosen;return best;
}
static double partition_value(const float *scores,const double *log_b,int count,double *top,double *z){
    *top=-INFINITY;for(int j=0;j<count;j++){double v=(double)scores[j]+log_b[j];if(v>*top)*top=v;}
    *z=0.;for(int j=0;j<count;j++)*z+=exp((double)scores[j]+log_b[j]-*top);
    need(*z>=1.&&*z<=count&&isfinite(*z)&&isfinite(*top),"stable_partition");return *top+log(*z);
}
static void predict(const char *unique_path,const char *out_path){
    FILE *in=openr(unique_path);char magic[8];readn(in,magic,8);need(!memcmp(magic,"M479UNI1",8)&&u32(in)==3720&&u32(in)==D&&u64(in)==U,"unique_header");
    FILE *out=fopen(out_path,"wb");need(out!=NULL,"output_open");writen(out,"M480PRD1",8);uint32_t width=372,reserved=0;uint64_t count=U;writen(out,&width,4);writen(out,&reserved,4);writen(out,&count,8);
    for(uint32_t uid=0;uid<U;uid++){
        uint32_t meta[12];unsigned char sha[64];double original_root[3];float x[D],original_scores[N];readn(in,meta,48);readn(in,sha,64);readn(in,original_root,24);readn(in,x,D*4);readn(in,original_scores,N*4);
        need(meta[0]==uid&&meta[1]==N&&meta[2]<12&&meta[7]<N,"unique_meta");for(int d=0;d<D;d++)need(normalzero(x[d]),"source_input_normal_or_zero");Bank *b=&banks[meta[2]];
        Prediction record;memset(&record,0,sizeof(record));record.meta[0]=uid;record.meta[1]=meta[2];record.meta[3]=meta[7];record.value[2]=(float)original_root[1];
        uint32_t node=0;
        for(uint32_t depth=0;depth<7;depth++){
            Node *g=&b->nodes[node];need(g->count>1,"path_internal");record.meta[9]+=32;uint32_t lk,rk;float l=support(b,g->left,x,&record,&lk),r=support(b,g->right,x,&record,&rk);uint32_t right=r>l||(r==l&&rk<lk);
            record.meta[11]+=(uint32_t)(r==l);record.direction[depth]=right;node=right?g->right:g->left;record.meta[5]++;
        }
        need(b->nodes[node].count==1&&record.meta[4]==26,"selected_leaf");uint32_t chosen=b->nodes[node].first;record.meta[2]=chosen;float chosen_score=0;int found=0;
        for(uint32_t j=0;j<26;j++)if(record.vector_ids[j]==chosen){chosen_score=record.score[j];found=1;}need(found,"chosen_original_score_available");record.value[0]=chosen_score;
        float mass_scores[16];for(uint32_t j=0;j<16;j++)mass_scores[j]=evaluate(b,252+j,x,&record);
        record.meta[6]=16;record.meta[8]+=16*8;record.root[2]=partition_value(mass_scores,b->log_b,16,&record.root[0],&record.root[1]);record.root[3]=exp((double)chosen_score-record.root[2]);record.value[1]=(float)record.root[3];
        need(isfinite(record.value[1])&&record.value[1]>=0&&isfinite(record.root[3]),"candidate_probability_finite");
        for(uint32_t j=0;j<42;j++)if(record.vector_ids[j]<128)need(!memcmp(&record.score[j],&original_scores[record.vector_ids[j]],4),"visited_original_score_BYTE");
        need(record.meta[4]==42&&record.meta[5]==7&&record.meta[7]==32256&&record.meta[8]==129152&&record.meta[9]==672&&record.meta[10]==24,"paid_work_counts");
        writen(out,record.meta,48);writen(out,record.value,12);writen(out,record.root,32);writen(out,record.vector_ids,84);writen(out,record.score,168);writen(out,record.direction,28);
        if((uid+1)%32768==0){need(fflush(out)==0,"output_flush");printf("{\"unique_completed\":%u}\n",uid+1);fflush(stdout);}
    }
    end(in);need(fclose(out)==0,"output_close");printf("{\"terminal\":true,\"unique_queries\":%u,\"visited_dot_fields\":%llu,\"coefficient_values\":%llu,\"source_visited_scores_BYTE\":true,\"CPU_affinity_mask\":1,\"MXCSR\":%u}\n",U,(unsigned long long)U*42,(unsigned long long)U*32256,_mm_getcsr());fflush(stdout);
}
static void controls(void){
    const float toys[6][3]={{0,1,-1},{10000,9999,-10000},{0,0,0},{1000,999,-1000},{0,-1,-2},{0,0,-1}};
    for(int k=0;k<6;k++){int e=winner(toys[k],3);double z=source_mass(toys[k],3,e);printf("{\"control\":%d,\"chosen\":%d,\"denominator\":%.17g,\"probability\":%.17g}\n",k,e,z,(double)(float)(1./z));}
    float a[D],x[D];for(int i=0;i<D;i++){a[i]=(i%7-3)*0x1p-10f;x[i]=(i%5-2)*0x1p-9f;}double value=dot(a,x);printf("{\"dot_control\":0,\"double_result\":%.17g,\"f32_result\":%.17g}\n",value,(double)(float)value);
    for(int i=0;i<D;i++){a[i]=i%2?0x1p-100f:0x1p80f;x[i]=i%2?0x1p-20f:(i%4?-0x1p-80f:0x1p-80f);}value=dot(a,x);printf("{\"dot_control\":1,\"double_result\":%.17g,\"f32_result\":%.17g}\n",value,(double)(float)value);
    printf("{\"CPU_affinity_mask\":1,\"rounding\":%d,\"MXCSR\":%u}\n",fegetround(),_mm_getcsr());
    const double b[3]={log(2.),log(3.),log(5.)};for(int k=0;k<2;k++){double top,z,A=partition_value(toys[k],b,3,&top,&z);float p=(float)exp((double)toys[k][winner(toys[k],3)]-A);printf("{\"weighted_partition_control\":%d,\"A\":%.17g,\"probability\":%.17g}\n",k,A,(double)p);}
}
int main(int argc,char **argv){
    need(SetProcessAffinityMask(GetCurrentProcess(),1)!=0,"CPU_affinity_set");DWORD_PTR p=0,s=0;need(GetProcessAffinityMask(GetCurrentProcess(),&p,&s)&&p==1,"CPU_affinity_readback");need(fegetround()==FE_TONEAREST&&(_mm_getcsr()&0x8040u)==0,"RNE_noFTZ_noDAZ");
    need(argc>=2,"arguments");if(!strcmp(argv[1],"controls")){need(argc==2,"controls_arguments");controls();return 0;}
    need(argc==5&&!strcmp(argv[1],"predict"),"predict_arguments");load(argv[2]);predict(argv[3],argv[4]);return 0;
}
