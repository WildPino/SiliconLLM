/* Exact p6 dot products: widen SIMD lanes after 1024 columns.
 * Caller domain: abs(weight)<=364, abs(input)<=32767, 3072 columns.
 * Each int32 lane sums <=128 scalar products, bound 1526680064.
 * Full dot bound 36640321536; ordered F64 scales/BF conversion unchanged.
 */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#define API __declspec(dllexport)
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)((u+0x7fffu+((u>>16)&1u))>>16);}
API int ir_abi(int i){int v[]={3072,2048,1024,364,32767};return i>=0&&i<5?v[i]:-1;}
static int64_t dot16(const int16_t*w,const int16_t*x){
    __m256i wide=_mm256_setzero_si256();
    for(int base=0;base<3072;base+=1024){
        __m256i left=_mm256_setzero_si256(),right=_mm256_setzero_si256();
        for(int j=0;j<1024;j+=32){
            left=_mm256_add_epi32(left,_mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i*)(w+base+j)),
                _mm256_loadu_si256((const __m256i*)(x+base+j))));
            right=_mm256_add_epi32(right,_mm256_madd_epi16(
                _mm256_loadu_si256((const __m256i*)(w+base+j+16)),
                _mm256_loadu_si256((const __m256i*)(x+base+j+16))));
        }
        __m256i sum=_mm256_add_epi32(left,right);
        wide=_mm256_add_epi64(wide,_mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum)));
        wide=_mm256_add_epi64(wide,_mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum,1)));
    }
    __m128i pair=_mm_add_epi64(_mm256_castsi256_si128(wide),_mm256_extracti128_si256(wide,1));
    pair=_mm_add_epi64(pair,_mm_unpackhi_epi64(pair,pair));
    return _mm_cvtsi128_si64(pair);
}
API int ir_history(int n,const int16_t*w,const double*scale,const int16_t*x,const double*a,int64_t*dots,double*y,uint16_t*bits){
    if(n<1||n>1507)return 1;
    for(int t=0;t<n;t++)for(int o=0;o<2048;o++){int64_t z=dot16(w+(size_t)o*3072,x+(size_t)t*3072);dots[(size_t)t*2048+o]=z;double f=(double)z*a[t];f=f*scale[o];y[(size_t)t*2048+o]=f;bits[(size_t)t*2048+o]=bf((float)f);}return 0;
}
/* Small control vectors bypass only outer output-row iteration, not arithmetic. */
API int ir_dot_controls(int n,const int16_t*w,const int16_t*x,int64_t*out){
    if(n<1||n>256)return 1;
    for(int i=0;i<n;i++)out[i]=dot16(w+(size_t)i*3072,x+(size_t)i*3072);
    return 0;
}
