/* Qualified p6 integer weights,new explicit I16 input approximation. */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#define API __declspec(dllexport)
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
API int ir_abi(int i){int v[]={3072,2048,128,364,32767};return i>=0&&i<5?v[i]:-1;}
static int64_t dot16(const int16_t*w,const int16_t*x){int64_t total=0;
    for(int base=0;base<3072;base+=128){__m256i sum=_mm256_setzero_si256();
        for(int j=0;j<128;j+=16){__m256i p=_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(w+base+j)),_mm256_loadu_si256((const __m256i*)(x+base+j)));sum=_mm256_add_epi32(sum,p);}
        __m128i s=_mm_add_epi32(_mm256_castsi256_si128(sum),_mm256_extracti128_si256(sum,1));s=_mm_hadd_epi32(s,s);s=_mm_hadd_epi32(s,s);total+=(int64_t)_mm_cvtsi128_si32(s);
    }return total;
}
API int ir_history(int n,const int16_t*w,const double*scale,const int16_t*x,const double*a,int64_t*dots,double*y,uint16_t*bits){
    if(n<1||n>1507)return 1;
    for(int t=0;t<n;t++)for(int o=0;o<2048;o++){int64_t z=dot16(w+(size_t)o*3072,x+(size_t)t*3072);dots[(size_t)t*2048+o]=z;double f=(double)z*a[t];f=f*scale[o];y[(size_t)t*2048+o]=f;bits[(size_t)t*2048+o]=bf((float)f);}return 0;
}
