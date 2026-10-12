/* Independent scalar eight-FMA-lane output contraction;no native source include. */
#include <stdint.h>
#include <math.h>
#include <string.h>
#define API __declspec(dllexport)
API int scalar_output(int n,const float*w,const uint16_t*x,float*y){
    if(n<=0||n>1507)return 1;
    for(int t=0;t<n;t++)for(int o=0;o<2048;o++){
        float a[8]={0};const float*q=w+(size_t)o*3072;
        for(int j=0;j<3072;j+=8)for(int k=0;k<8;k++){uint32_t u=(uint32_t)x[(size_t)t*3072+j+k]<<16;float v;memcpy(&v,&u,4);a[k]=fmaf(q[j+k],v,a[k]);}
        float z=a[0];for(int k=1;k<8;k++)z+=a[k];y[(size_t)t*2048+o]=z;
    }
    return 0;
}
