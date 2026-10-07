/* Factorized candidate: unweighted hybrid functions and explicit variable mass. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#define D 768
#define R 512
#define N 17540
#define E 128
#define Q 1281
static void need(int v,const char*s){if(!v){fprintf(stderr,"factor521_error:%s\n",s);fflush(stderr);exit(2);}}
static void rd(FILE*f,void*p,size_t n){need(fread(p,1,n,f)==n,"short_read");}
static void wr(FILE*f,const void*p,size_t n){need(fwrite(p,1,n,f)==n,"short_write");}
static FILE*in(const char*p){FILE*f=fopen(p,"rb");need(f!=NULL,"input_open");return f;}
static FILE*out(const char*p){need(GetFileAttributesA(p)==INVALID_FILE_ATTRIBUTES,"exclusive_output");FILE*f=fopen(p,"wb");need(f!=NULL,"output_open");return f;}
static void end(FILE*f){need(fgetc(f)==EOF&&fclose(f)==0,"EOF");}
static void header(FILE*f,const char*m,uint32_t w,uint32_t r,uint64_t n,int write){
    if(write){wr(f,m,8);wr(f,&w,4);wr(f,&r,4);wr(f,&n,8);}else{
        char a[8];uint32_t b,c;uint64_t d;rd(f,a,8);rd(f,&b,4);rd(f,&c,4);rd(f,&d,8);
        need(!memcmp(a,m,8),"wire_magic");need(b==w&&c==r&&d==n,"wire_shape");}
}
static float quant(const float*x,int16_t*q,int n){float mx=0;for(int i=0;i<n;i++){need(isfinite(x[i]),"finite_quant_input");if(fabsf(x[i])>mx)mx=fabsf(x[i]);}
    float a=mx==0?1.f:mx/32767.f;need(isfinite(a)&&a>0,"quant_scale");
    for(int i=0;i<n;i++){float z=x[i]/a;long t=lrintf(z);if(t>32767)t=32767;if(t< -32767)t= -32767;q[i]=(int16_t)t;}return a;}
static int64_t idot(const int8_t*w,const int16_t*x,int n){
    __m256i s=_mm256_setzero_si256();for(int i=0;i<n;i+=16){
        __m256i a=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w+i)));
        s=_mm256_add_epi32(s,_mm256_madd_epi16(a,_mm256_loadu_si256((const __m256i*)(x+i))));}
    int32_t v[8];_mm256_storeu_si256((__m256i*)v,s);int64_t answer=0;for(int j=0;j<8;j++)answer+=v[j];return answer;}
static uint8_t packed[R][D/4];static float sigma[R],keys[24][Q];static unsigned char*private;
#define EB 992256
static void bank(const char*p,const char*m){FILE*f=in(p);char magic[8];uint32_t shape[8];rd(f,magic,8);rd(f,shape,32);
    uint32_t wanted[8]={D,3072,R,E,11,4,8,16};need(!memcmp(magic,m,8),"wire_magic");need(!memcmp(shape,wanted,32),"bank_shape");
    rd(f,packed,sizeof(packed));rd(f,sigma,sizeof(sigma));rd(f,keys,sizeof(keys));private=malloc((size_t)EB*E);need(private!=NULL,"bank_memory");rd(f,private,(size_t)EB*E);end(f);
    for(int j=0;j<R;j++){need(isfinite(sigma[j])&&sigma[j]>0,"dictionary_scale");for(int g=0;g<D/4;g++)need(packed[j][g]<=80,"packed_range");}}
static void source(FILE*f,float*x,int16_t*q,float*a,uint32_t*e,uint32_t*accepted){
    rd(f,e,4);rd(f,accepted,4);rd(f,a,4);rd(f,x,D*4);rd(f,q,D*2);need(*e<E&&*accepted<=1,"source_meta");
    int16_t recomputed[D];float qa=quant(x,recomputed,D);need(!memcmp(&qa,a,4)&&!memcmp(q,recomputed,D*2),"source_A16_BYTE");}
static void phi(const int16_t*x,float alpha,float*f,int16_t*q,float*a){int32_t table[D/4][81];
    for(int g=0;g<D/4;g++)for(int p=0;p<81;p++){int code=p,sum=0;for(int j=0;j<4;j++){sum+=(code%3-1)*x[4*g+j];code/=3;}table[g][p]=sum;}
    for(int j=0;j<R;j++){int32_t sum=0;for(int g=0;g<D/4;g++)sum+=table[g][packed[j][g]];
        f[j]=fabsf((float)(((double)sum*(double)sigma[j])*(double)alpha));need(isfinite(f[j]),"finite_phi");}*a=quant(f,q,R);}
static void linear(uint32_t e,const int16_t*x,float a,float*y){const unsigned char*p=private+(size_t)e*EB;
    const int8_t*w=(const int8_t*)p;const float*s=(const float*)(p+D*D);
    for(int j=0;j<D;j++){need(isfinite(s[j])&&s[j]>0,"Lscale");y[j]=(float)(((double)idot(w+j*D,x,D)*(double)s[j])*(double)a);need(isfinite(y[j]),"finite_L");}}
static void output(uint32_t e,const int16_t*x,float ax,const int16_t*q,float aq,float*y){const unsigned char*p=private+(size_t)e*EB;
    const int8_t*w=(const int8_t*)(p+D*D+D*4);const float*s=(const float*)(p+D*D+D*4+D*R);const float*b=s+D;
    linear(e,x,ax,y);for(int j=0;j<D;j++){int64_t dot=idot(w+j*R,q,R);need(dot>=INT32_MIN&&dot<=INT32_MAX,"B_I32_bound");
        need(isfinite(s[j])&&s[j]>0&&isfinite(b[j]),"Bscale_bias");float z=(float)(((double)dot*(double)s[j])*(double)aq);
        float sum=y[j]+z;y[j]=sum+b[j];need(isfinite(y[j]),"finite_output");}}
static void choice(const float*x,const float*p,float*logits,uint32_t*id,float*mass){
    for(int j=0;j<24;j++){double lanes[8]={0};for(int i=0;i<D+R;i++){double value=i<D?x[i]:p[i-D];lanes[i%8]+=(double)keys[j][i]*value;}
        double v=0;for(int i=0;i<4;i++)v+=lanes[i]+lanes[i+4];v+=(double)keys[j][D+R];logits[j]=(float)v;need(isfinite(logits[j]),"finite_key");}
    int picked[2];float prob[2];for(int h=0;h<2;h++){int start=h?8:0,n=h?16:8,winner=0;
        for(int j=1;j<n;j++)if(logits[start+j]>logits[start+winner])winner=j;
        float ex[16];double total=0;for(int j=0;j<n;j++){ex[j]=(float)exp((double)logits[start+j]-(double)logits[start+winner]);total+=(double)ex[j];}
        picked[h]=winner;prob[h]=(float)((double)ex[winner]/total);}
    *id=16*picked[0]+picked[1];*mass=prob[0]*prob[1];need(isfinite(*mass)&&*mass>0&&*mass<=1,"candidate_mass");}
static float product(float p,float f,uint32_t accepted){return accepted?p*f:0.f;}

typedef struct {uint32_t e,accepted;float alpha,x[D];int16_t q[D];float p;} Input521;
typedef struct {uint16_t e,owner;uint32_t uid;} Pair521;
typedef struct {uint64_t wi,ws,wo,os;} Extent521;
typedef struct {uint32_t uid,e,owner,nonzero;float alpha;int16_t codes[3072];float down[D];} Response521;
_Static_assert(sizeof(Input521)==4624,"source_input_wire");
_Static_assert(sizeof(Pair521)==8,"source_pair_wire");
_Static_assert(sizeof(Extent521)==32,"source_extent_wire");
_Static_assert(sizeof(Response521)==9236,"source_response_wire");
_Static_assert(192LL*2*128*32767<INT32_MAX,"source_WO_lane_bound");

static void capture521(const char *payload_path,const char *extent_path,const char *input_path,const char *pair_path,const char *output_path){
    FILE *ext=in(extent_path);header(ext,"M521EX01",32,D,E,0);Extent521 offsets[E];rd(ext,offsets,sizeof(offsets));end(ext);
    FILE *fi=in(input_path);header(fi,"M499INP1",4624,D,N,0);Input521 *inputs=malloc((size_t)N*sizeof(Input521));need(inputs!=NULL,"input_memory");rd(fi,inputs,(size_t)N*sizeof(Input521));end(fi);
    FILE *pf=in(pair_path);header(pf,"M521PA01",8,D,53943,0);Pair521 *pairs=malloc(53943ULL*sizeof(Pair521));need(pairs!=NULL,"pair_memory");rd(pf,pairs,53943ULL*sizeof(Pair521));end(pf);
    FILE *payload=in(payload_path);need(_fseeki64(payload,0,SEEK_END)==0,"payload_seek_end");int64_t payload_bytes=_ftelli64(payload);need(payload_bytes==7541946880LL,"original_payload_size");
    int8_t *wi=malloc(3072ULL*D),*wo=malloc(3072ULL*D);float *ws=malloc(3072ULL*4),*os=malloc(D*4);need(wi&&wo&&ws&&os,"source_memory");
    FILE *g=out(output_path);header(g,"M521SR01",sizeof(Response521),D,53943,1);Response521 response;float hidden[3072];uint32_t cursor=0;
    for(uint32_t e=0;e<E;e++){
        uint64_t starts[4]={offsets[e].wi,offsets[e].ws,offsets[e].wo,offsets[e].os};size_t lengths[4]={3072ULL*D,3072ULL*4,3072ULL*D,D*4};void *buffers[4]={wi,ws,wo,os};
        for(int k=0;k<4;k++){need(starts[k]<=7541946880ULL&&lengths[k]<=7541946880ULL-starts[k],"source_extent_range");need(_fseeki64(payload,(int64_t)starts[k],SEEK_SET)==0,"source_seek");rd(payload,buffers[k],lengths[k]);}
        for(int j=0;j<3072;j++)need(isfinite(ws[j])&&ws[j]>0,"source_wi_scale");for(int j=0;j<D;j++)need(isfinite(os[j])&&os[j]>0,"source_wo_scale");
        need(cursor<53943&&pairs[cursor].e==e,"source_parent_complete_order");
        while(cursor<53943&&pairs[cursor].e==e){
            Pair521 p=pairs[cursor];need(p.uid<N&&p.owner<E&&p.owner!=e,"source_pair_identity");Input521 *input=inputs+p.uid;
            need(input->e==p.owner&&input->accepted==1&&isfinite(input->alpha)&&input->alpha>0,"source_input_owner_accept");
            int16_t checked[D];float alpha=quant(input->x,checked,D);need(!memcmp(&alpha,&input->alpha,4)&&!memcmp(checked,input->q,sizeof(checked)),"source_original_A16_BYTE");
            for(int j=0;j<3072;j++){hidden[j]=(float)(((double)idot(wi+(size_t)j*D,input->q,D)*(double)ws[j])*(double)alpha);need(isfinite(hidden[j]),"source_finite_WI");if(hidden[j]<0.f)hidden[j]=0.f;}
            response.uid=p.uid;response.e=e;response.owner=p.owner;response.alpha=quant(hidden,response.codes,3072);response.nonzero=0;
            for(int j=0;j<3072;j++){need(response.codes[j]>=0,"relu_nonnegative_codes");response.nonzero+=(response.codes[j]!=0);}
            for(int r=0;r<D;r++){response.down[r]=(float)(((double)idot(wo+(size_t)r*3072,response.codes,3072)*(double)os[r])*(double)response.alpha);need(isfinite(response.down[r]),"source_finite_WO");}
            wr(g,&response,sizeof(response));cursor++;
        }
        if(e%8==7){printf("{\"source_parent_terminal\":%u,\"source_functions\":%u}\n",e,cursor);fflush(stdout);}
    }
    need(cursor==53943,"all_new_source_pairs");need(fclose(g)==0&&fclose(payload)==0,"source_output_close");free(inputs);free(pairs);free(wi);free(wo);free(ws);free(os);
    printf("{\"new_source_function_calls\":53943,\"source_response_width\":9236}\n");
}

int main(int argc,char **argv){
    need(fesetround(FE_TONEAREST)==0&&fegetround()==FE_TONEAREST,"RNE");need(!(_mm_getcsr()&0x8040u),"FTZ_DAZ_off");
    DWORD_PTR mask,system;need(GetProcessAffinityMask(GetCurrentProcess(),&mask,&system)&&mask==1024,"CPU10");
    if(argc==2&&!strcmp(argv[1],"controls")){
        int8_t w[3072];int16_t q[3072];int64_t expected=0;
        for(int i=0;i<3072;i++){w[i]=i%3==0?-128:i%3==1?127:-17;q[i]=i%2?32767:-32767;expected+=(int64_t)w[i]*q[i];}
        need(idot(w,q,3072)==expected,"source_full_width_scalar_I64");
        for(int i=0;i<3072;i++){w[i]=-128;q[i]=32767;}expected=-128LL*32767*3072;
        need(idot(w,q,3072)==expected&&expected<INT32_MIN,"source_full_width_non_I32");
        float zero[3072]={0};float za=quant(zero,q,3072);need(za==1.f,"source_zero_alpha");for(int i=0;i<3072;i++)need(q[i]==0,"source_zero_codes");
        float tie[8]={32767.f,-32767.f,2.5f,3.5f,-2.5f,-3.5f,0.f,1.f};float a=quant(tie,q,8);uint32_t bits;memcpy(&bits,&a,4);
        printf("{\"source_full_width_I64\":%lld,\"zero_alpha\":%.1f,\"alpha_bits\":%u,\"codes\":[",(long long)expected,za,bits);
        for(int i=0;i<8;i++)printf("%s%d",i?",":"",q[i]);printf("],\"CPU10\":%llu,\"RNE\":%d,\"MXCSR\":%u}\n",(unsigned long long)mask,fegetround(),_mm_getcsr());
        int16_t x[4]={32765,-19875,301,-101};printf("{\"new_table81\":[");for(int p=0;p<81;p++){int code=p,s=0;for(int j=0;j<4;j++){s+=(code%3-1)*x[j];code/=3;}printf("%s%d",p?",":"",s);}printf("]}\n");
        uint32_t bits_in[6]={7,9,10,14,0x8000000a,0x3f000003};printf("{\"new_quarter_product_bits\":[");
        for(int j=0;j<6;j++){float f;uint32_t b;memcpy(&f,bits_in+j,4);float y=product(.25f,f,1);memcpy(&b,&y,4);printf("%s%u",j?",":"",b);}printf("]}\n");return 0;
    }
    if(argc==7&&!strcmp(argv[1],"capture")){capture521(argv[2],argv[3],argv[4],argv[5],argv[6]);return 0;}
    need(argc==5&&!strcmp(argv[1],"predict"),"arguments");bank(argv[2],"M521BNK1");FILE *f=in(argv[3]);header(f,"M499INP1",4624,D,N,0);
    FILE *g=out(argv[4]);header(g,"M521PRE1",15464,D,N,1);
    for(int k=0;k<N;k++){float x[D],ax,p[R],ap,psrc;int16_t q[D],qp[R];uint32_t e,accepted;
        source(f,x,q,&ax,&e,&accepted);rd(f,&psrc,4);need(isfinite(psrc)&&psrc>0&&psrc<=1,"source_mass");phi(q,ax,p,qp,&ap);
        float logits[24],mass,fo[D],fc[D],go[D],same[D],choicep[D],coupled[D];uint32_t id;choice(x,p,logits,&id,&mass);
        if(accepted){output(e,q,ax,qp,ap,fo);if(id==e)memcpy(fc,fo,sizeof(fo));else output(id,q,ax,qp,ap,fc);}else{memset(fo,0,sizeof(fo));memset(fc,0,sizeof(fc));}
        for(int j=0;j<D;j++){go[j]=product(psrc,fo[j],accepted);same[j]=product(mass,fo[j],accepted);choicep[j]=product(psrc,fc[j],accepted);coupled[j]=product(mass,fc[j],accepted);need(isfinite(go[j])&&isfinite(same[j])&&isfinite(choicep[j])&&isfinite(coupled[j]),"finite_weighted_output");}
        wr(g,&id,4);wr(g,&mass,4);wr(g,logits,96);wr(g,fo,D*4);wr(g,go,D*4);wr(g,same,D*4);wr(g,choicep,D*4);wr(g,coupled,D*4);
    }
    end(f);need(fclose(g)==0,"output_close");free(private);return 0;
}
