/* Saved-query geometric routing feasibility; never a model/rate benchmark. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <immintrin.h>
#include <stddef.h>

#define D 768
#define MAXN 256
#define MAXNODE 511
#define TOTALQ 96186
static void require(int yes,const char *why){if(!yes){fprintf(stderr,"grouped_router_error:%s\n",why);fflush(stderr);exit(2);}}
static void *alloc(size_t n,size_t size){void *p=calloc(n,size);require(p!=NULL,"allocation");return p;}
static void readn(FILE *f,void *p,size_t n){require(fread(p,1,n,f)==n,"short_read");}
static uint32_t u32(FILE *f){uint32_t v;readn(f,&v,4);return v;}
static uint64_t u64(FILE *f){uint64_t v;readn(f,&v,8);return v;}
static FILE *openr(const char *p){FILE *f=fopen(p,"rb");require(f!=NULL,"open_input");return f;}
static char *string(FILE *f){uint32_t n=u32(f);require(n>0&&n<4096,"path_length");char *p=alloc(n+1,1);readn(f,p,n);return p;}
static void eofclose(FILE *f){require(fgetc(f)==EOF,"bounded_EOF");require(fclose(f)==0,"input_close");}
static double gamma_q(int q){double v=q*0x1p-53;return v/(1.-v);}
static int normalzero(float v){uint32_t b;memcpy(&b,&v,4);return isfinite(v)&&((b&0x7f800000u)!=0||(b&0x7fffffffu)==0);}

/* Original393 reduction and final cast, not an arbitrary BLAS contraction. */
static double dot_source(const float *a,const float *b,int n){
    __m256d x=_mm256_setzero_pd(),y=_mm256_setzero_pd();int i=0;
    for(;i+8<=n;i+=8){x=_mm256_add_pd(x,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),_mm256_cvtps_pd(_mm_loadu_ps(b+i))));y=_mm256_add_pd(y,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),_mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));}
    double values[4];_mm256_storeu_pd(values,_mm256_add_pd(x,y));double sum=0.;for(int j=0;j<4;j++)sum+=values[j];for(;i<n;i++)sum+=(double)a[i]*(double)b[i];return sum;
}
static double dot_center(const double *a,const float *b){
    __m256d x=_mm256_setzero_pd(),y=_mm256_setzero_pd();
    for(int i=0;i<D;i+=8){x=_mm256_add_pd(x,_mm256_mul_pd(_mm256_loadu_pd(a+i),_mm256_cvtps_pd(_mm_loadu_ps(b+i))));y=_mm256_add_pd(y,_mm256_mul_pd(_mm256_loadu_pd(a+i+4),_mm256_cvtps_pd(_mm_loadu_ps(b+i+4))));}
    double values[4];_mm256_storeu_pd(values,_mm256_add_pd(x,y));double sum=0.;for(int j=0;j<4;j++)sum+=values[j];return sum;
}
static int chosen(const float *s,int n){int k=0;for(int j=1;j<n;j++)if(s[j]>s[k])k=j;return k;}
static float term(float score,float shift){float d=score-shift;return (float)exp((double)d);}
static float probability(const float *s,int n,int k){double total=0.;for(int j=0;j<n;j++)total+=(double)term(s[j],s[k]);require(total>0.&&isfinite(total),"native_probability_sum");return (float)(1./total);}

typedef struct {uint32_t left,right,parent,m,first,reserved0,dirmode,reserved1;double radius,maxnorm,centernorm,meanerror,s0,s1;double *center,*direction;uint32_t *ids;} Node;
typedef struct {int n,bank,nodes;float *w;Node node[MAXNODE];} Bank;
static Bank banks[24];
static Bank *bankof(int n,int bank){require((n==128||n==256)&&bank>=0&&bank<12,"bank_identity");return &banks[(n==256?0:12)+bank];}
static void load_weights(const char *path){
    FILE *f=openr(path);char magic[8];readn(f,magic,8);require(!memcmp(magic,"M478WGT1",8),"weight_ledger_magic");require(u32(f)==24&&u32(f)==D,"weight_ledger_dimensions");
    for(int k=0;k<24;k++){uint32_t n=u32(f),bank=u32(f);uint64_t offset=u64(f);char *payload=string(f);Bank *b=bankof(n,bank);require(b->w==NULL,"weight_duplicate_bank");b->n=n;b->bank=bank;b->w=alloc((size_t)n*D,4);FILE *p=openr(payload);require(_fseeki64(p,offset,SEEK_SET)==0,"weight_seek");readn(p,b->w,(size_t)n*D*4);require(fclose(p)==0,"weight_close");free(payload);for(int j=0;j<(int)n*D;j++)require(normalzero(b->w[j]),"weights_normal_or_zero");}
    eofclose(f);
}
static void load_trees(const char *directory){
    for(int k=0;k<24;k++){
        Bank *b=&banks[k];char path[4096];snprintf(path,sizeof(path),"%s/n%d.bank%d.tree.bin",directory,b->n,b->bank);FILE *f=openr(path);char magic[8];readn(f,magic,8);require(!memcmp(magic,"M478TRE1",8),"tree_magic");require(u32(f)==(uint32_t)b->n&&u32(f)==(uint32_t)b->bank&&u32(f)==D,"tree_identity");b->nodes=u32(f);require(b->nodes==2*b->n-1&&u32(f)==80&&u32(f)==8,"tree_dimensions");
        float *w=alloc((size_t)b->n*D,4);readn(f,w,(size_t)b->n*D*4);require(!memcmp(w,b->w,(size_t)b->n*D*4),"tree_weights_BYTE");free(w);
        for(int j=0;j<b->nodes;j++){
            Node *v=&b->node[j];readn(f,v,80);require(v->m>0&&v->m<=(uint32_t)b->n&&v->first<(uint32_t)b->n&&!v->reserved0&&!v->reserved1,"node_metadata");
            require(isfinite(v->radius)&&v->radius>=0.&&isfinite(v->maxnorm)&&v->maxnorm>=0.&&isfinite(v->centernorm)&&v->centernorm>=0.&&isfinite(v->meanerror)&&v->meanerror>=0.,"node_bounds");
            if(v->m>1){require(v->left>j&&v->left<(uint32_t)b->nodes&&v->right>j&&v->right<(uint32_t)b->nodes,"node_children");v->center=alloc(D,8);v->direction=alloc(D,8);readn(f,v->center,D*8);readn(f,v->direction,D*8);}
            else require(v->left==UINT32_MAX&&v->right==UINT32_MAX,"leaf_children");
            v->ids=alloc(v->m,4);readn(f,v->ids,v->m*4);require(v->ids[0]==v->first,"node_first_ID");for(uint32_t q=0;q<v->m;q++)require(v->ids[q]<(uint32_t)b->n&&(!q||v->ids[q]>v->ids[q-1]),"node_sorted_IDs");
        }
        eofclose(f);require(b->node[0].m==(uint32_t)b->n&&b->node[0].parent==UINT32_MAX,"root");
        for(int j=1;j<b->nodes;j++){uint32_t p=b->node[j].parent;require(p<(uint32_t)j&&(b->node[p].left==(uint32_t)j||b->node[p].right==(uint32_t)j),"parent_link");}
    }
}
typedef struct {uint32_t n,book,case_id,mode,s,t;char *trace,*whole;} Job;
static uint32_t jobs_left;
static FILE *jobs_open(const char *path){FILE *f=openr(path);char m[8];readn(f,m,8);require(!memcmp(m,"M478JOB1",8),"jobs_magic");jobs_left=u32(f);require(jobs_left==384&&u32(f)==TOTALQ,"jobs_counts");return f;}
static Job nextjob(FILE *f){Job j;require(jobs_left>0,"job_remaining");jobs_left--;readn(f,&j,24);j.trace=string(f);j.whole=string(f);require((j.n==128||j.n==256)&&j.book<24&&j.case_id<4&&j.mode<2&&j.s==29&&j.t>0&&j.t<=64,"job_fields");if(!j.mode)require(j.t==14,"teacher_length");return j;}
static void query_metadata(const Job *j,int index,int *bank,int *position){if(index<6*(int)j->s){*bank=index/j->s;*position=index%j->s;}else{int r=index-6*j->s;*bank=6+r%6;*position=r/6;}}
typedef struct {int32_t expert,accepted;float p;} Route;
static Route *routes_open(const Job *j){FILE *f=openr(j->whole);char m[8];readn(f,m,8);require(!memcmp(m,"SWR32O01",8),"whole_magic");uint32_t h[7];readn(f,h,28);uint32_t count=6*(j->s+j->t);require(h[0]==j->s&&h[1]==j->t&&h[2]==D&&h[3]==12&&h[4]==12&&h[5]==32128&&h[6]==count,"whole_dimensions");uint64_t off=36+4*((uint64_t)14*j->s*D+(uint64_t)j->t*14*D+(uint64_t)j->t*32128);require(_fseeki64(f,off,SEEK_SET)==0,"whole_route_seek");Route *r=alloc(count,sizeof(Route));readn(f,r,count*sizeof(Route));eofclose(f);return r;}
static FILE *trace_open(const Job *j){FILE *f=openr(j->trace);char m[8];readn(f,m,8);require(!memcmp(m,"SWRTA001",8),"trace_magic");require(u32(f)==j->n&&u32(f)==D,"trace_dimensions");return f;}
static void trace_query(FILE *f,const Job *j,int index,float *x,float *s){require(u32(f)==(uint32_t)index&&u32(f)==(uint32_t)(index>=6*(int)j->s),"trace_index_phase");readn(f,x,D*4);readn(f,s,j->n*4);for(int i=0;i<D;i++)require(normalzero(x[i]),"input_normal_or_zero");for(uint32_t i=0;i<j->n;i++)require(isfinite(s[i]),"score_finite");}
static void source_oracle(const char *path){
    FILE *jobs=jobs_open(path);uint64_t query_count=0,score_count=0;
    while(jobs_left){Job j=nextjob(jobs);Route *r=routes_open(&j);FILE *f=trace_open(&j);int count=6*(j.s+j.t);float x[D],s[MAXN],actual[MAXN];
        for(int i=0;i<count;i++){trace_query(f,&j,i,x,s);int bank,pos;query_metadata(&j,i,&bank,&pos);Bank *b=bankof(j.n,bank);for(int e=0;e<b->n;e++)actual[e]=(float)dot_source(b->w+(size_t)e*D,x,D);require(!memcmp(actual,s,j.n*4),"ALL_source_scores_BYTE");int k=chosen(actual,b->n);float p=probability(actual,b->n,k);require(k==r[i].expert&&!memcmp(&p,&r[i].p,4)&&(r[i].accepted==0||r[i].accepted==1),"ALL_source_probability_ID_BYTE");query_count++;score_count+=j.n;}
        eofclose(f);free(r);free(j.trace);free(j.whole);
    }
    eofclose(jobs);require(query_count==TOTALQ,"source_query_total");printf("{\"source_queries\":%llu,\"source_score_values\":%llu,\"scores_BYTE_exact\":true,\"probability_ID_BYTE_exact\":true}\n",(unsigned long long)query_count,(unsigned long long)score_count);
}
static void controls(void){
    const float scores[6][3]={{0,1,-1},{10000,9999,-10000},{0,0,0},{1000,999,-1000},{0,-1,-2},{0,0,-1}};
    for(int i=0;i<6;i++){int k=chosen(scores[i],3);double z=0.;for(int q=0;q<3;q++)z+=(double)term(scores[i][q],scores[i][k]);printf("{\"control\":%d,\"chosen\":%d,\"denominator\":%.17g,\"probability\":%.17g}\n",i,k,z,(double)probability(scores[i],3,k));}
    float a[D],b[D];for(int i=0;i<D;i++){a[i]=(i%7-3)*0x1p-10f;b[i]=(i%5-2)*0x1p-9f;}
    printf("{\"dot_control\":0,\"double_result\":%.17g,\"f32_result\":%.17g}\n",dot_source(a,b,D),(double)(float)dot_source(a,b,D));
    for(int i=0;i<D;i++){a[i]=i%2?0x1p-100f:0x1p80f;b[i]=i%2?0x1p-20f:(i%4?-0x1p-80f:0x1p-80f);}
    printf("{\"dot_control\":1,\"double_result\":%.17g,\"f32_result\":%.17g}\n",dot_source(a,b,D),(double)(float)dot_source(a,b,D));
    printf("{\"CPU_affinity_mask\":1,\"rounding\":%d,\"MXCSR\":%u}\n",fegetround(),_mm_getcsr());
}

_Static_assert(offsetof(Node,center)==80,"node_disk_prefix");
typedef struct {double center,upper,rho,eta,lower_mass,upper_mass,agg_lower,agg_upper;float score;int expanded;} State;
typedef struct {uint32_t meta[20];double value[8];unsigned char frontier[64];} Observation;
_Static_assert(sizeof(Observation)==208,"observation_wire");
static State state[MAXNODE];
static int heap[MAXNODE],heap_size,heap_mode,visited[MAXNODE],visit_count;
static uint32_t rows_source,rows_center,splits,heap_pushes,heap_comparisons,aggregate_updates,native_exp_calls,bound_exp_calls;
static Bank *current;
static const float *query,*oracle_scores;
static double xnorm;
static float winner_score;
static double safe_exp(double x){bound_exp_calls++;if(x>709.)return INFINITY;if(x< -745.)return 0.;return exp(x);}
static void enclosure(double lo,double hi,double value,const char *why){double tolerance=1e-11+1e-10*fabs(value);require(!isnan(lo)&&!isnan(hi)&&lo<=hi&&lo<=value+tolerance&&hi+tolerance>=value,why);}
static int better(int a,int b){heap_comparisons++;double ka=heap_mode?state[a].upper_mass-state[a].lower_mass:state[a].upper,kb=heap_mode?state[b].upper_mass-state[b].lower_mass:state[b].upper;if(ka!=kb)return ka>kb;return current->node[a].first<current->node[b].first;}
static void push(int id){require(heap_size<MAXNODE,"heap_capacity");heap_pushes++;int at=heap_size++;while(at){int parent=(at-1)/2;if(!better(id,heap[parent]))break;heap[at]=heap[parent];at=parent;}heap[at]=id;}
static int pop(void){require(heap_size>0,"heap_empty");int answer=heap[0],last=heap[--heap_size],at=0;while(2*at+1<heap_size){int child=2*at+1;if(child+1<heap_size&&better(heap[child+1],heap[child]))child++;if(!better(heap[child],last))break;heap[at]=heap[child];at=child;}if(heap_size)heap[at]=last;return answer;}
static void evaluate(int id){
    Node *v=&current->node[id];State *p=&state[id];memset(p,0,sizeof(*p));require(visit_count<MAXNODE,"visited_capacity");visited[visit_count++]=id;
    if(v->m==1){p->score=(float)dot_source(current->w+(size_t)v->first*D,query,D);require(!memcmp(&p->score,&oracle_scores[v->first],4),"candidate_leaf_score_BYTE");p->center=p->score;p->upper=p->score;rows_source++;}
    else{p->center=dot_center(v->center,query);p->rho=v->radius*xnorm;double ec=gamma_q(2*D+8)*v->centernorm*xnorm;double en=(gamma_q(D+8)+0x1p-24*(1.+gamma_q(D+8)))*v->maxnorm*xnorm+0x1p-150;
        p->eta=ec+en;p->upper=p->center+p->rho+p->eta;rows_center++;double mn=INFINITY,mx=-INFINITY;for(uint32_t k=0;k<v->m;k++){double a=oracle_scores[v->ids[k]];if(a<mn)mn=a;if(a>mx)mx=a;}enclosure(p->center-p->rho-p->eta,p->upper,mn,"group_score_min_enclosure");enclosure(p->center-p->rho-p->eta,p->upper,mx,"group_score_max_enclosure");}
}
static void mass(int id){
    Node *v=&current->node[id];State *p=&state[id];double lo,hi;
    if(v->m==1){lo=hi=(double)term(p->score,winner_score);native_exp_calls++;}
    else{double mean=v->meanerror*xnorm;double esub=0x1p-24*(v->maxnorm*xnorm+p->eta+fabs((double)winner_score))+0x1p-149;double eta=p->eta+esub;double a=p->rho>0.?fmin(mean/p->rho,1.):0.;
        double lcw=p->rho+log(.5*(1.+a)+.5*(1.-a)*safe_exp(-2.*p->rho));double center=p->center-(double)winner_score;
        lo=fmax(0.,safe_exp(log((double)v->m)+center-mean-eta+log1p(-0x1p-21))-v->m*0x1p-149);
        hi=safe_exp(log((double)v->m)+center+eta+lcw+log1p(0x1p-21))+v->m*0x1p-149;
    }
    double actual=0.;for(uint32_t k=0;k<v->m;k++)actual+=(double)term(oracle_scores[v->ids[k]],winner_score);
    enclosure(lo,hi,actual,"group_native_mass_enclosure");p->lower_mass=p->agg_lower=lo;p->upper_mass=p->agg_upper=hi;
}
static void initial_aggregate(int id){Node *v=&current->node[id];State *p=&state[id];if(p->expanded){initial_aggregate(v->left);initial_aggregate(v->right);p->agg_lower=state[v->left].agg_lower+state[v->right].agg_lower;p->agg_upper=state[v->left].agg_upper+state[v->right].agg_upper;aggregate_updates++;}}
static void update_ancestors(int id){while(id>=0){Node *v=&current->node[id];State *p=&state[id];require(p->expanded,"aggregate_expanded");p->agg_lower=state[v->left].agg_lower+state[v->right].agg_lower;p->agg_upper=state[v->left].agg_upper+state[v->right].agg_upper;aggregate_updates++;id=v->parent==UINT32_MAX?-1:(int)v->parent;}}
static Observation traverse(const Job *job,int bank,int position,const float *x,const float *scores,const Route *reference,uint32_t serial){
    current=bankof(job->n,bank);query=x;oracle_scores=scores;visit_count=heap_size=heap_mode=0;rows_source=rows_center=splits=heap_pushes=heap_comparisons=aggregate_updates=native_exp_calls=bound_exp_calls=0;
    xnorm=nextafter(sqrt(dot_source(x,x,D))*(1.+gamma_q(8*D+16)),INFINITY);evaluate(0);push(0);
    while(current->node[heap[0]].m>1){int id=pop();Node *v=&current->node[id];state[id].expanded=1;splits++;evaluate(v->left);evaluate(v->right);push(v->left);push(v->right);}
    int winning_node=heap[0],winning=current->node[winning_node].first;winner_score=state[winning_node].score;require(winning==reference->expert,"ALL_candidate_winner_exact");
    heap_size=0;heap_mode=1;for(int k=0;k<visit_count;k++){int id=visited[k];if(!state[id].expanded){mass(id);if(current->node[id].m>1)push(id);}}initial_aggregate(0);
    double lo,hi,ratio;int full=0;
    for(;;){double g=gamma_q(current->n+8);lo=fmax(0.,state[0].agg_lower*(1.-g));hi=state[0].agg_upper*(1.+g);require(lo>0.&&!isnan(hi),"positive_aggregate");ratio=hi/lo;
        if(!heap_size){full=1;break;}if(ratio<=1.01)break;
        int id=pop();Node *v=&current->node[id];state[id].expanded=1;splits++;evaluate(v->left);evaluate(v->right);mass(v->left);mass(v->right);if(current->node[v->left].m>1)push(v->left);if(current->node[v->right].m>1)push(v->right);update_ancestors(id);
    }
    float predicted,lp,up;
    if(full){float actual[MAXN];for(int k=0;k<visit_count;k++){int id=visited[k];if(current->node[id].m==1)actual[current->node[id].first]=state[id].score;}predicted=probability(actual,current->n,winning);native_exp_calls+=current->n;lp=up=predicted;ratio=1.;require(rows_source==(uint32_t)current->n,"full_leaf_coverage");}
    else{predicted=(float)(1./hi);lp=nextafterf(predicted,-INFINITY);up=nextafterf((float)(1./lo),INFINITY);}
    enclosure(lp,up,reference->p,"ALL_probability_interval");require(fabs((double)predicted/(double)reference->p-1.)<=.01+1e-10,"ALL_probability_1percent");
    Observation result;memset(&result,0,sizeof(result));uint32_t m[20]={serial,job->n,job->book,job->case_id,job->mode,(uint32_t)bank,(uint32_t)position,(uint32_t)winning,(uint32_t)reference->expert,rows_source,rows_center,(uint32_t)visit_count,splits,(uint32_t)full,heap_pushes,heap_comparisons,aggregate_updates,native_exp_calls,bound_exp_calls,(uint32_t)visit_count};memcpy(result.meta,m,sizeof(m));
    double val[8]={(double)reference->p,(double)predicted,(double)lp,(double)up,((double)rows_source+rows_center+1)/current->n,xnorm,current->node[0].radius*xnorm,ratio};memcpy(result.value,val,sizeof(val));for(int k=0;k<visit_count;k++){int id=visited[k];if(!state[id].expanded)result.frontier[id/8]|=1u<<(id%8);}
    require(visit_count==1+2*(int)splits&&rows_source+rows_center==(uint32_t)visit_count,"work_cardinality");return result;
}
static void candidate(const char *jobs_path,const char *destination){
    FILE *jobs=jobs_open(jobs_path),*out=fopen(destination,"wb");require(out!=NULL,"observation_open");require(fwrite("M478OBS1",1,8,out)==8,"observation_magic");uint32_t size=208,reserved=0;uint64_t total=TOTALQ;fwrite(&size,4,1,out);fwrite(&reserved,4,1,out);fwrite(&total,8,1,out);uint32_t serial=0;
    while(jobs_left){Job j=nextjob(jobs);Route *r=routes_open(&j);FILE *f=trace_open(&j);int count=6*(j.s+j.t);float x[D],s[MAXN];for(int i=0;i<count;i++){trace_query(f,&j,i,x,s);int bank,pos;query_metadata(&j,i,&bank,&pos);Observation row=traverse(&j,bank,pos,x,s,&r[i],serial++);require(fwrite(&row,1,sizeof(row),out)==sizeof(row),"observation_write");}require(fflush(out)==0,"observation_flush");eofclose(f);free(r);free(j.trace);free(j.whole);if((384-jobs_left)%32==0){printf("{\"jobs_completed\":%u,\"queries_completed\":%u}\n",384-jobs_left,serial);fflush(stdout);}}
    eofclose(jobs);require(serial==TOTALQ&&fclose(out)==0,"candidate_complete_EOF");printf("{\"terminal\":true,\"queries\":%u,\"winner_exact\":true,\"ALL_group_enclosures\":true,\"ALL_probability_intervals\":true,\"ALL_probability_relative_error_le1percent\":true,\"CPU_affinity_mask\":1,\"MXCSR\":%u}\n",serial,_mm_getcsr());
}
int main(int argc,char **argv){
    require(SetProcessAffinityMask(GetCurrentProcess(),1)!=0,"CPU_affinity_set");DWORD_PTR a=0,s=0;require(GetProcessAffinityMask(GetCurrentProcess(),&a,&s)&&a==1,"CPU_affinity_readback");require(fegetround()==FE_TONEAREST&&(_mm_getcsr()&0x8040u)==0,"RNE_no_FTZ_DAZ");
    require(argc>=2,"arguments");if(!strcmp(argv[1],"controls")){require(argc==2,"controls_args");controls();return 0;}
    require(argc==4||argc==6,"probe_args");load_weights(argv[2]);if(!strcmp(argv[1],"source")){require(argc==4,"source_args");source_oracle(argv[3]);return 0;}
    require(!strcmp(argv[1],"candidate")&&argc==6,"candidate_args");load_trees(argv[4]);candidate(argv[3],argv[5]);return 0;
}
