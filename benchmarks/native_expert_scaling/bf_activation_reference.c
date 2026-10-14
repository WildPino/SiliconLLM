/* Separate scalar implementation and manual RNE BF packing; libm shared. */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <immintrin.h>
#define API __declspec(dllexport)
static float decode_word(uint16_t word){uint32_t bits=((uint32_t)word)*65536u;float result;memcpy(&result,&bits,4);return result;}
static uint16_t round_word(double value){
    float f=(float)value;uint32_t bits;memcpy(&bits,&f,4);
    uint32_t hi=bits/65536u,lo=bits%65536u;
    if(lo>32768u||(lo==32768u&&(hi%2u)))hi++;
    return (uint16_t)hi;
}
API unsigned ref_control(void){return _mm_getcsr()&0xffc0u;}
API int act_reference(int n,const uint16_t*input,uint16_t*conv,double*gate,uint16_t*delta){
    if(n<1||n>65536||fegetround()!=FE_TONEAREST||(_mm_getcsr()&0x1f80u)!=0x1f80u)return 6;
    for(int i=0;i<n;i++){
        double v=(double)decode_word(input[i]);
        double softnegative=fmax(0.0,-v)+log1p(exp(-fabs(v)));
        conv[i]=round_word(v*exp(-softnegative));
        double denominator=1.0+exp(-v);gate[i]=v/denominator;
        double softpositive=v>20.0?v:fmax(v,0.0)+log1p(exp(-fabs(v)));
        delta[i]=round_word(softpositive);
    }return 0;
}
