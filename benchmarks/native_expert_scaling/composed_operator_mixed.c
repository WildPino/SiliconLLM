/* Mixed precision correction: F64 gate/RMS before BF16,original F32 dotf after. */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define API __declspec(dllexport)
static inline float hsum256(__m256 v){ float o[8]; _mm256_storeu_ps(o,v); return o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; }
static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float r=hsum256(s); for(;i<n;i++) r+=a[i]*b[i]; return r; }
static double now(void){LARGE_INTEGER c,f;QueryPerformanceCounter(&c);QueryPerformanceFrequency(&f);return (double)c.QuadPart/(double)f.QuadPart;}
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
static float unbf(uint16_t v){uint32_t u=(uint32_t)v<<16;float f;memcpy(&f,&u,4);return f;}
static double dot64(const float*w,const float*x){__m256d a=_mm256_setzero_pd();for(int j=0;j<3072;j+=4)a=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(w+j)),_mm256_cvtps_pd(_mm_loadu_ps(x+j)),a);double s[4];_mm256_storeu_pd(s,a);return s[0]+s[1]+s[2]+s[3];}
API int composed_abi(int i){const int v[]={3072,384,96,48,2048,5,9,8};return i>=0&&i<8?v[i]:-1;}
/* weights:Wc rows,Vc rows,Ws rows,Vs rows,Alog,D,gamma,O.
 * inputs:x,B,C,delta,gate; float outputs:px,B',C',latent,finalstate,4 scores;
 * u16 outputs:4 normalized inputs,4 outputs. flags1=F64,2=F32,4=ownnorm only. */
API int composed_history(int n,int flags,const int*ranks,const int*offs,const float*const*w,const float*const*inp,const double*sourceinv,float*const*out,uint16_t*const*bits,double*times,int timed){
    if(n<=0||n>1507||offs[0]!=0||offs[48]!=384||!(flags&3))return 1;
    int ids[384];for(int h=0;h<48;h++){if(ranks[h]<=0||ranks[h]>64||offs[h+1]-offs[h]!=ranks[h])return 2;for(int k=offs[h];k<offs[h+1];k++)ids[k]=h;}
    float*state=calloc(384*96,sizeof(float));if(!state)return 3;memset(times,0,8*sizeof(double));
    float qr[384],y32[3072],u32[3072],norm[3072],decay[48];double qd[384],y64[3072],u64[3072];
    for(int t=0;t<n;t++){
        double start=timed?now():0;const float*x=inp[0]+(size_t)t*3072,*B=inp[1]+(size_t)t*256,*C=inp[2]+(size_t)t*256,*delta=inp[3]+(size_t)t*48,*gate=inp[4]+(size_t)t*3072;float*px=out[0]+(size_t)t*384,*bp=out[1]+(size_t)t*96,*cp=out[2]+(size_t)t*96,*lat=out[3]+(size_t)t*384;
        for(int h=0;h<48;h++)for(int k=offs[h];k<offs[h+1];k++)px[k]=dotf(w[0]+(size_t)k*64,x+h*64,64);
        for(int j=0;j<96;j++){bp[j]=dotf(w[2]+j*256,B,256);cp[j]=dotf(w[3]+j*256,C,256);}
        if(timed){times[0]+=now()-start;start=now();}
        for(int h=0;h<48;h++)decay[h]=expf(delta[h]*(-expf(w[4][h])));
        for(int k=0;k<384;k++){int h=ids[k];float tx=delta[h]*px[k];__m256 d=_mm256_set1_ps(decay[h]),vtx=_mm256_set1_ps(tx),acc=_mm256_setzero_ps();float*s=state+k*96;
            for(int j=0;j<96;j+=8){__m256 z=_mm256_fmadd_ps(d,_mm256_loadu_ps(s+j),_mm256_mul_ps(vtx,_mm256_loadu_ps(bp+j)));_mm256_storeu_ps(s+j,z);acc=_mm256_fmadd_ps(z,_mm256_loadu_ps(cp+j),acc);}lat[k]=hsum256(acc)+w[5][h]*px[k];}
        if(timed){times[1]+=now()-start;start=now();}
        if(flags&1){
            for(int k=0;k<384;k++)qd[k]=(double)lat[k]-(double)w[5][ids[k]]*(double)px[k];
            for(int h=0;h<48;h++){int r=ranks[h];for(int c=0;c<64;c++){double z=0;const float*v=w[1]+(size_t)offs[h]*64+c*r;for(int j=0;j<r;j++)z+=(double)v[j]*qd[offs[h]+j];y64[h*64+c]=z+(double)w[5][h]*(double)x[h*64+c];}}
            if(timed){times[2]+=now()-start;start=now();}
            double ms=0;for(int c=0;c<3072;c++){u64[c]=y64[c]*((double)gate[c]/(1.0+exp(-(double)gate[c])));ms+=u64[c]*u64[c];}double inv=1.0/sqrt(ms/3072.0+1e-5);
            if(timed){times[3]+=now()-start;start=now();}
            for(int mode=(flags&4)?1:0;mode<2;mode++){
                double rr=mode?inv:sourceinv[t];for(int c=0;c<3072;c++){uint16_t q=bf((float)((u64[c]*rr)*(double)w[6][c]));bits[mode][(size_t)t*3072+c]=q;norm[c]=unbf(q);}
                for(int o=0;o<2048;o++){float s=dotf(w[7]+(size_t)o*3072,norm,3072);out[5+mode][(size_t)t*2048+o]=s;bits[4+mode][(size_t)t*2048+o]=bf(s);}
            }
            if(timed){times[4]+=now()-start;start=now();}
        }
        if(flags&2){
            for(int k=0;k<384;k++)qr[k]=lat[k]-w[5][ids[k]]*px[k];
            for(int h=0;h<48;h++){int r=ranks[h];for(int c=0;c<64;c++)y32[h*64+c]=dotf(w[1]+(size_t)offs[h]*64+c*r,qr+offs[h],r)+w[5][h]*x[h*64+c];}
            if(timed){times[5]+=now()-start;start=now();}
            float ms=0;for(int c=0;c<3072;c++){u32[c]=y32[c]*(gate[c]/(1.0f+expf(-gate[c])));ms=fmaf(u32[c],u32[c],ms);}float inv=1.0f/sqrtf(ms/3072.0f+1e-5f);
            if(timed){times[6]+=now()-start;start=now();}
            for(int mode=(flags&4)?1:0;mode<2;mode++){
                float rr=mode?inv:(float)sourceinv[t];for(int c=0;c<3072;c++){uint16_t q=bf((u32[c]*rr)*w[6][c]);bits[2+mode][(size_t)t*3072+c]=q;norm[c]=unbf(q);}
                for(int o=0;o<2048;o++){float s=dotf(w[7]+(size_t)o*3072,norm,3072);out[7+mode][(size_t)t*2048+o]=s;bits[6+mode][(size_t)t*2048+o]=bf(s);}
            }
            if(timed)times[7]+=now()-start;
        }
    }
    memcpy(out[4],state,384*96*sizeof(float));free(state);return 0;
}
/* Kernel-only retained generator cost;arbitrary supplied vectors,not source quality. */
API int retained_generator(int n,const float*w,const float*x,float*y){if(n<=0||n>256)return 1;for(int t=0;t<n;t++)for(int o=0;o<6704;o++)y[(size_t)t*6704+o]=dotf(w+(size_t)o*2048,x+(size_t)t*2048,2048);return 0;}
