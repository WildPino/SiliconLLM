/* One affine sigmoid root head; original AVX F64 dot/F32 input and coefficients. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#define D 768
#define U 238872
static void need(int v,const char *s){if(!v){fprintf(stderr,"root_mass_error:%s\n",s);fflush(stderr);exit(2);}}
static void readn(FILE *f,void *p,size_t n){need(fread(p,1,n,f)==n,"short_read");}
static void writen(FILE *f,const void *p,size_t n){need(fwrite(p,1,n,f)==n,"short_write");}
static FILE *openr(const char *s){FILE *f=fopen(s,"rb");need(f!=NULL,"input_open");return f;}
static void eof(FILE *f){need(fgetc(f)==EOF&&fclose(f)==0,"bounded_EOF");}
static double dot(const float *a,const float *x){
    __m256d l=_mm256_setzero_pd(),r=_mm256_setzero_pd();
    for(int i=0;i<D;i+=8){
        l=_mm256_add_pd(l,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),_mm256_cvtps_pd(_mm_loadu_ps(x+i))));
        r=_mm256_add_pd(r,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),_mm256_cvtps_pd(_mm_loadu_ps(x+i+4))));
    }
    double v[4];_mm256_storeu_pd(v,_mm256_add_pd(l,r));double s=0.;for(int j=0;j<4;j++)s+=v[j];return s;
}
static void pair(float z,float *l,float *r){
    double t=exp(-fabs((double)z)),small=t/(1.+t),large=1./(1.+t);
    *l=(float)(z>=0?small:large);*r=(float)(z>=0?large:small);
    need(isfinite(*l)&&isfinite(*r)&&*l>=0&&*l<=1&&*r>=0&&*r<=1,"pair_range");
}
static void header(FILE *f,const char *magic,uint32_t width,uint32_t reserved,uint64_t count){
    char m[8];uint32_t w,r;uint64_t c;readn(f,m,8);readn(f,&w,4);readn(f,&r,4);readn(f,&c,8);
    need(!memcmp(m,magic,8),"wire_magic");need(w==width&&r==reserved&&c==count,"wire_shape");
}
int main(int argc,char **argv){
    need(fesetround(FE_TONEAREST)==0&&fegetround()==FE_TONEAREST,"RNE");
    need(!(_mm_getcsr()&0x8040u),"FTZ_DAZ_off");DWORD_PTR mask,system;
    need(GetProcessAffinityMask(GetCurrentProcess(),&mask,&system)&&mask==1,"CPU0_affinity");
    if(argc==2&&!strcmp(argv[1],"controls")){
        float zs[]={-1000,-80,-20,-1,0,1,20,80,1000};
        for(int j=0;j<9;j++){float l,r;pair(zs[j],&l,&r);printf("{\"control\":%d,\"z\":%.17g,\"left\":%.17g,\"right\":%.17g}\n",j,(double)zs[j],(double)l,(double)r);}
        for(int k=0;k<2;k++){
            float a[D],x[D];for(int i=0;i<D;i++){
                a[i]=k?(i%2?0x1p-100f:0x1p80f):(i%7-3)*0x1p-10f;
                x[i]=k?(i%2?0x1p-20f:(i%4?-0x1p-80f:0x1p-80f)):(i%5-2)*0x1p-9f;
            }
            double v=dot(a,x);printf("{\"dot_control\":%d,\"double_result\":%.17g,\"f32_result\":%.17g}\n",k,v,(double)(float)v);
        }
        printf("{\"rounding\":%d,\"MXCSR\":%u,\"CPU_affinity_mask\":%llu}\n",fegetround(),_mm_getcsr(),(unsigned long long)mask);return 0;
    }
    need(argc==5&&!strcmp(argv[1],"predict"),"arguments");
    FILE *hf=openr(argv[2]);header(hf,"M490HED1",4*(D+1),D,12);
    float heads[12][D+1];readn(hf,heads,sizeof(heads));eof(hf);
    for(int b=0;b<12;b++)for(int d=0;d<=D;d++)need(isfinite(heads[b][d]),"finite_head");
    FILE *f=openr(argv[3]);header(f,"M479UNI1",3720,D,U);
    need(GetFileAttributesA(argv[4])==INVALID_FILE_ATTRIBUTES,"exclusive_output");
    FILE *out=fopen(argv[4],"wb");need(out!=NULL,"output_open");
    uint32_t width=16,reserved=0;uint64_t count=U;
    writen(out,"M490PRD1",8);writen(out,&width,4);writen(out,&reserved,4);writen(out,&count,8);
    unsigned char row[3720];float x[D];uint32_t meta[12];
    for(uint32_t i=0;i<U;i++){
        readn(f,row,sizeof(row));memcpy(meta,row,sizeof(meta));need(meta[0]==i&&meta[2]<12,"uid_bank");
        memcpy(x,row+136,sizeof(x));for(int d=0;d<D;d++)need(isfinite(x[d]),"finite_input");
        uint32_t b=meta[2];float z=(float)(dot(heads[b],x)+(double)heads[b][D]);need(isfinite(z),"finite_logit");
        float l,r;pair(z,&l,&r);writen(out,&i,4);writen(out,&z,4);writen(out,&l,4);writen(out,&r,4);
    }
    eof(f);need(fclose(out)==0,"output_close");return 0;
}
