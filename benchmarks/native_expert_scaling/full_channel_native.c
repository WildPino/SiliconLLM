/* Fixed source-state96 with every trained channel;engine dot primitives unchanged. */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#define API __declspec(dllexport)
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static double now(void){LARGE_INTEGER q,f;QueryPerformanceCounter(&q);QueryPerformanceFrequency(&f);return (double)q.QuadPart/(double)f.QuadPart;}
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
static float unbf(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
API int full_abi(int i){int v[]={3072,96,48,2048,6,4};return i>=0&&i<6?v[i]:-1;}
/* weights Ws,Vs,Alog,D,gamma,O,in_proj;inputs x,B,C,delta,gate.
 * float outputs B',C',latent,finalstate,source/own scores;bits source/own norm,source/own output. */
API int full_history(int n,int ownonly,const float*const*w,const float*const*inp,const double*sourceinv,float*const*out,uint16_t*const*bits,double*times,int timed){
    if(n<=0||n>1507)return 1;float*state=calloc(3072*96,sizeof(float));if(!state)return 2;
    memset(times,0,4*sizeof(double));float decay[48],norm[3072];double u[3072];
    for(int t=0;t<n;t++){
        const float*x=inp[0]+(size_t)t*3072,*bb=inp[1]+(size_t)t*256,*cc=inp[2]+(size_t)t*256,*dt=inp[3]+(size_t)t*48,*gate=inp[4]+(size_t)t*3072;
        float*bp=out[0]+(size_t)t*96,*cp=out[1]+(size_t)t*96,*lat=out[2]+(size_t)t*3072;double start=timed?now():0;
        for(int j=0;j<96;j++){bp[j]=dotf(w[0]+j*256,bb,256);cp[j]=dotf(w[1]+j*256,cc,256);}
        if(timed){times[0]+=now()-start;start=now();}
        for(int h=0;h<48;h++)decay[h]=expf(dt[h]*(-expf(w[2][h])));
        for(int c=0;c<3072;c++){
            int h=c/64;float tx=dt[h]*x[c],*s=state+c*96;__m256 a=_mm256_set1_ps(decay[h]),b=_mm256_set1_ps(tx),acc=_mm256_setzero_ps();
            for(int j=0;j<96;j+=8){__m256 z=_mm256_fmadd_ps(a,_mm256_loadu_ps(s+j),_mm256_mul_ps(b,_mm256_loadu_ps(bp+j)));_mm256_storeu_ps(s+j,z);acc=_mm256_fmadd_ps(z,_mm256_loadu_ps(cp+j),acc);}lat[c]=hsum256(acc)+w[3][h]*x[c];
        }
        if(timed){times[1]+=now()-start;start=now();}
        double ms=0;for(int c=0;c<3072;c++){u[c]=(double)lat[c]*((double)gate[c]/(1.0+exp(-(double)gate[c])));ms+=u[c]*u[c];}double inv=1.0/sqrt(ms/3072.0+1e-5);
        if(timed){times[2]+=now()-start;start=now();}
        for(int mode=ownonly?1:0;mode<2;mode++){
            double r=mode?inv:sourceinv[t];for(int c=0;c<3072;c++){uint16_t b=bf((float)((u[c]*r)*(double)w[4][c]));bits[mode][(size_t)t*3072+c]=b;norm[c]=unbf(b);}
            for(int o=0;o<2048;o++){float f=dotf(w[5]+(size_t)o*3072,norm,3072);out[4+mode][(size_t)t*2048+o]=f;bits[2+mode][(size_t)t*2048+o]=bf(f);}
        }
        if(timed)times[3]+=now()-start;
    }
    memcpy(out[3],state,3072*96*sizeof(float));free(state);return 0;
}
API int full_generator(int n,const float*w,const float*x,float*y){if(n<=0||n>256)return 1;for(int t=0;t<n;t++)for(int o=0;o<6704;o++)y[(size_t)t*6704+o]=dotf(w+(size_t)o*2048,x+(size_t)t*2048,2048);return 0;}
