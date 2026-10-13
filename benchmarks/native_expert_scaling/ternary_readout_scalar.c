/* Separate scalar reference of the same integer-q response;no LUT/gather. */
#include <stdint.h>
#include <string.h>
#include <windows.h>
#define API __declspec(dllexport)
API int tr_scalar(int n,const int16_t*q,const double*s,const float*x,double*y){
    if(n<1||n>16)return 1;
    for(int t=0;t<n;t++)for(int o=0;o<2048;o++){double z=0;
        for(int i=0;i<3072;i++)z=z+(double)q[(size_t)o*3072+i]*(double)x[(size_t)t*3072+i];
        y[(size_t)t*2048+o]=z*s[o];}return 0;
}
