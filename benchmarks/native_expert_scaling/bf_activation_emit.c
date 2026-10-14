/* Emit the literal qualified row-state expressions on the finite BF domain. */
#include "full_channel_native.c"
#include <fenv.h>
API unsigned act_control(void){return _mm_getcsr()&0xffc0u;}
API int act_emit(uint16_t*conv,double*gate,uint16_t*delta){
    if(fegetround()!=FE_TONEAREST||(_mm_getcsr()&0x1f80u)!=0x1f80u)return 6;
    for(unsigned i=0;i<65536;i++){
        if((i&0x7f80u)==0x7f80u){conv[i]=0;gate[i]=0;delta[i]=0;continue;}
        double v=unbf((uint16_t)i);
        double z=fmax(0.,-v)+log1p(exp(-fabs(v)));
        conv[i]=bf((float)(v*exp(-z)));
        gate[i]=v/(1.0+exp(-v));
        double q=v>20?v:fmax(v,0.)+log1p(exp(-fabs(v)));
        delta[i]=bf((float)q);
    }return 0;
}
