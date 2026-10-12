#include <math.h>
#include <stddef.h>
#define D 256
#define V 65537

/* Independent scalar reference: explicit float FMA, eight logical lanes.
   No original engine bodies, SIMD intrinsics, or calls to producer DLL. */
__declspec(dllexport) int reference_readout(const float* head,const float* bias,
    const float* states,int count,float epsilon,float* normalized,
    float* inverse,float* scores){
    if(count<1||count>128||epsilon<=0.0f||!isfinite(epsilon))return 1;
    for(int sample=0;sample<count;sample++){
        const float* t=states+(size_t)sample*D;
        float square=0.0f;
        for(int col=0;col<D;col++)square=fmaf(t[col],t[col],square);
        float denominator=square*0.00390625f;
        denominator=denominator+epsilon;
        float inv=1.0f/sqrtf(denominator);
        inverse[sample]=inv;
        float* u=normalized+(size_t)sample*D;
        for(int col=0;col<D;col++)u[col]=t[col]*inv;
        for(int row=0;row<V;row++){
            const float* w=head+(size_t)row*D;
            float lane[8]={0,0,0,0,0,0,0,0};
            for(int col=0;col<D;col++){
                int index=col%8;
                lane[index]=fmaf(w[col],u[col],lane[index]);
            }
            float sum=lane[0];
            for(int index=1;index<8;index++)sum=sum+lane[index];
            scores[(size_t)sample*V+row]=fmaf(bias[row],inv,sum);
        }
    }
    return 0;
}
