/* Candidate hybrid only: integer LUT features, private affine/even output. */
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
static void need(int v,const char*s){if(!v){fprintf(stderr,"hybrid495_error:%s\n",s);fflush(stderr);exit(2);}}
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
int main(int argc,char**argv){
    need(fesetround(FE_TONEAREST)==0&&fegetround()==FE_TONEAREST,"RNE");need(!(_mm_getcsr()&0x8040u),"FTZ_DAZ_off");
    DWORD_PTR mask,system;need(GetProcessAffinityMask(GetCurrentProcess(),&mask,&system)&&mask==1,"CPU0");
    if(argc==2&&!strcmp(argv[1],"controls")){
        const int16_t x[4]={32767,-17213,43,-8191};printf("{\"table81\":[");for(int p=0;p<81;p++){int code=p,s=0;for(int j=0;j<4;j++){s+=(code%3-1)*x[j];code/=3;}printf("%s%d",p?",":"",s);}printf("]}\n");
        float z[8]={65534,-65534,1,3,5,-1,-3,-5};int16_t q[8];float a=quant(z,q,8);uint32_t bits;memcpy(&bits,&a,4);
        printf("{\"alpha_bits\":%u,\"codes\":[",bits);for(int j=0;j<8;j++)printf("%s%d",j?",":"",q[j]);printf("]}\n");
        int8_t w[D];int16_t v[D];for(int i=0;i<D;i++){w[i]=126;v[i]=32766;}
        printf("{\"L_I64\":%lld,\"B_I32\":%lld,\"RNE\":%d,\"MXCSR\":%u,\"CPU0\":%llu}\n",(long long)idot(w,v,D),(long long)idot(w,v,R),fegetround(),_mm_getcsr(),(unsigned long long)mask);return 0;}
    need(argc==5,"arguments");int features=!strcmp(argv[1],"features");need(features||!strcmp(argv[1],"predict"),"mode");
    bank(argv[2],features?"M494BNK1":"M495BNK1");FILE*f=in(argv[3]);header(f,"M495INP1",4620,D,N,0);
    FILE*g=out(argv[4]);header(g,features?"M495FEA1":"M495PRE1",features?6148:6248,features?R:D,N,1);
    for(int k=0;k<N;k++){float x[D],ax,p[R],ap,l[D];int16_t q[D],qp[R];uint32_t e,accepted;source(f,x,q,&ax,&e,&accepted);phi(q,ax,p,qp,&ap);
        if(features){if(accepted)linear(e,q,ax,l);else memset(l,0,sizeof(l));wr(g,p,R*4);wr(g,qp,R*2);wr(g,&ap,4);wr(g,l,D*4);}
        else{float logits[24],mass,oracle[D],coupled[D];uint32_t id;choice(x,p,logits,&id,&mass);
            if(accepted){output(e,q,ax,qp,ap,oracle);if(id==e)memcpy(coupled,oracle,sizeof(oracle));else output(id,q,ax,qp,ap,coupled);}
            else{memset(oracle,0,sizeof(oracle));memset(coupled,0,sizeof(coupled));}
            wr(g,&id,4);wr(g,&mass,4);wr(g,logits,96);wr(g,oracle,D*4);wr(g,coupled,D*4);}}
    end(f);need(fclose(g)==0,"output_close");free(private);return 0;
}
