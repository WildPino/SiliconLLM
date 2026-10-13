/* Exact p6 / four input digits, original two-trit nibble/pshufb kernel. */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define API __declspec(dllexport)
#define OMP_PFOR
/* This function is copied literally from phase60/engine.c. */
static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
#include "../donor_adaptation/nibble_pack.h"
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
static int digit(uint16_t u,int s){return s==3?(int)(u>>12)-((u&0x8000u)?16:0):(int)((u>>(4*s))&15u);}
API int el_abi(int i){int v[]={3072,2048,6,4,1536,768,98304,32768,8192};return i>=0&&i<9?v[i]:-1;}
API int el_digits(int8_t*out){for(int s=0;s<4;s++)for(int u=0;u<65536;u++)out[(size_t)s*65536+u]=(int8_t)digit((uint16_t)u,s);return 0;}
API int el_tables(int pairs,const int16_t*x,int8_t*out){
    if(pairs<1||pairs>32768)return 1;
    for(int s=0;s<4;s++)for(int t=0;t<pairs;t++){
        int a=digit((uint16_t)x[2*t],s),b=digit((uint16_t)x[2*t+1],s);
        for(int c=0;c<16;c++)out[((size_t)s*pairs+t)*16+c]=(int8_t)(c<9?(c/3-1)*a+(c%3-1)*b:0);
    }return 0;
}
API int el_history(int n,const uint8_t*codes,const double*scale,const int16_t*x,const double*a,int64_t*dots,double*y,uint16_t*bits){
    if(n<1||n>1507)return 1;
    int8_t*lut=(int8_t*)malloc(98304);int32_t*tmp=(int32_t*)malloc(8192);int32_t*tot=(int32_t*)malloc(32768);
    if(!lut||!tmp||!tot){free(lut);free(tmp);free(tot);return 2;}
    for(int t=0;t<n;t++){
        el_tables(1536,x+(size_t)t*3072,lut);memset(tot,0,32768);
        int power=1;
        for(int p=0;p<6;p++,power*=3)for(int s=0;s<4;s++){
            matvec_lut_full_n((const int8_t*)(codes+(size_t)p*768*2048),lut+(size_t)s*1536*16,tmp,2048,2048,1536);
            for(int o=0;o<2048;o++)tot[s*2048+o]+=power*tmp[o];
        }
        for(int o=0;o<2048;o++){
            int64_t z=(int64_t)tot[o]+16*(int64_t)tot[2048+o]+256*(int64_t)tot[4096+o]+4096*(int64_t)tot[6144+o];
            dots[(size_t)t*2048+o]=z;double f=(double)z*a[t];f=f*scale[o];y[(size_t)t*2048+o]=f;bits[(size_t)t*2048+o]=bf((float)f);
        }
    }free(lut);free(tmp);free(tot);return 0;
}
/* Counts architectural code-vector loads only, not cache/DRAM events. */
API int64_t el_counted_once(const uint8_t*codes,const double*scale,const int16_t*x,const double*a,int64_t*dots,double*y,uint16_t*bits){
    g_np_code_bytes=0;g_np_count=1;int rc=el_history(1,codes,scale,x,a,dots,y,bits);g_np_count=0;return rc?-rc:g_np_code_bytes;
}
