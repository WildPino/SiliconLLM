/* Lossless BF weight storage, decode in AVX registers; supplied true BF h. */
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#define API __declspec(dllexport)
static float decode(uint16_t b){uint32_t v=(uint32_t)b<<16;float f;memcpy(&f,&v,4);return f;}
static uint16_t round_bf(float f){uint32_t v;memcpy(&v,&f,4);return (uint16_t)((v+0x7fff+((v>>16)&1))>>16);}
API int cg_abi(int i){int v[]={2048,6704,8,23};return i>=0&&i<4?v[i]:-1;}
API unsigned cg_control(void){return _mm_getcsr()&0xffc0u;}
API int cg_step(const uint16_t*w,const uint16_t*h,const uint16_t*mup,float*raw,uint16_t*gg,uint16_t*fields){
    if(!w||!h||!mup||!raw||!gg||!fields)return 1;
    if(fegetround()!=FE_TONEAREST||cg_control()!=0x1f80u)return 3;
    float input[2048];for(int i=0;i<2048;i++){if((h[i]&0x7f80u)==0x7f80u)return 4;input[i]=decode(h[i]);}
    for(int o=0;o<6704;o++){
        const uint16_t*row=w+(size_t)o*2048;__m256 sum=_mm256_setzero_ps();
        for(int j=0;j<2048;j+=8){
            __m256i bits=_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(row+j))),16);
            sum=_mm256_fmadd_ps(_mm256_castsi256_ps(bits),_mm256_loadu_ps(input+j),sum);
        }
        float lane[8];_mm256_storeu_ps(lane,sum);
        float value=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
        raw[o]=value;gg[o]=round_bf(value);fields[o]=round_bf(decode(gg[o])*decode(mup[o]));
    }return 0;
}
API int cg_noop(const uint16_t*w,const uint16_t*h,const uint16_t*mup,float*raw,uint16_t*gg,uint16_t*fields){
    return w&&h&&mup&&raw&&gg&&fields?0:1;
}
