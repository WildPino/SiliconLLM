/* Same qualified p6 source artifact;new five-trit F64 blocked LUT backend. */
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
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
API int tr_abi(int i){int v[]={3072,2048,615,6,8,243};return i>=0&&i<6?v[i]:-1;}
API void tr_table(const float*x,double*table){
    for(int code=0;code<243;code++){int rem=code;double z=0;
        for(int k=0;k<5;k++){int t=rem%3-1;rem/=3;z=z+(double)t*(double)x[k];}table[code]=z;}
}
/* Timing added only after full correctness audit;all outputs archived here. */
API int tr_history(int n,const uint8_t*codes,const double*scale,const float*O,const float*X,double*Y,float*dense,uint16_t*bits){
    if(n<1||n>1507)return 1;
    double*acc=calloc(6*2048,sizeof(double));if(!acc)return 2;double table[8*243];
    const double power[6]={1,3,9,27,81,243};
    for(int t=0;t<n;t++){
        const float*x=X+(size_t)t*3072;memset(acc,0,6*2048*sizeof(double));
        for(int base=0;base<615;base+=8){int count=615-base;if(count>8)count=8;
            for(int g=0;g<count;g++){float q[5];for(int k=0;k<5;k++){int c=(base+g)*5+k;q[k]=c<3072?x[c]:0;}tr_table(q,table+g*243);}
            for(int p=0;p<6;p++)for(int o=0;o<2048;o+=4){
                __m256d a=_mm256_loadu_pd(acc+p*2048+o);
                for(int g=0;g<count;g++){const uint8_t*cr=codes+((size_t)p*615+base+g)*2048+o;
                    uint32_t word;memcpy(&word,cr,4);__m128i ix=_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)word));
                    a=_mm256_add_pd(a,_mm256_i32gather_pd(table+g*243,ix,8));}
                _mm256_storeu_pd(acc+p*2048+o,a);
            }
        }
        for(int o=0;o<2048;o++){double z=0;for(int p=0;p<6;p++)z=z+acc[p*2048+o]*power[p];
            z=z*scale[o];Y[(size_t)t*2048+o]=z;bits[(size_t)t*2048+o]=bf((float)z);
            dense[(size_t)t*2048+o]=dotf(O+(size_t)o*3072,x,3072);}
    }
    free(acc);return 0;
}
