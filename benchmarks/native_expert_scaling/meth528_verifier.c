/* Independent scalar verifier: original cached signed dots, NEW selected WO only. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>
#include <float.h>
#define D 768
#define H 3072
#define API __declspec(dllexport)

API int m528_quant(int width,const float *hidden,int16_t *code,float *alpha) {
    if(width<1 || width>H || fesetround(FE_TONEAREST)!=0 || sizeof(float)!=4 || sizeof(double)!=8 || FLT_RADIX!=2) return -1;
    float mx=0.f;
    for(int j=0;j<width;j++){ if(!isfinite(hidden[j]))return -2; if(fabsf(hidden[j])>mx)mx=fabsf(hidden[j]); }
    volatile float scale=mx/32767.f; if(mx==0.f)scale=1.f;
    if(!(scale>0.f) || !isfinite(scale))return -3; *alpha=scale;
    for(int j=0;j<width;j++){
        volatile float divided=hidden[j]/scale; double floorv=floor((double)divided); double rem=(double)divided-floorv;
        double rounded=floorv+((rem>.5 || (rem==.5 && fmod(floorv,2.)!=0.))?1.:0.);
        if(rounded>32767.)rounded=32767.; if(rounded< -32767.)rounded=-32767.; code[j]=(int16_t)rounded;
    }
    return 0;
}

API int m528_eval(int rows,int width,const int64_t *signed_dots,const float *input_alpha,
    const float *si,const int8_t *wo,const float *so,int16_t *hidden_codes,float *hidden_alpha,
    int64_t *integer_out,float *physical_out,double *continuous_out,double *absolute_out) {
    if(rows<1 || width<0 || width>H || fesetround(FE_TONEAREST)!=0)return -10;
    double hidden[H]; float h32[H]; int16_t codes[H];
    for(int i=0;i<rows;i++){
        for(int j=0;j<width;j++){
            if(!(si[j]>0.f) || !(input_alpha[i]>0.f))return -11;
            volatile double first=(double)signed_dots[(size_t)i*width+j]*(double)si[j];
            volatile double value=first*(double)input_alpha[i]; hidden[j]=value>0.?value:0.; h32[j]=(float)hidden[j];
        }
        float ah=1.f;
        if(physical_out){
            if(width==0)return -12; int rc=m528_quant(width,h32,codes,&ah); if(rc)return rc;
            if(hidden_alpha)hidden_alpha[i]=ah;
            if(hidden_codes)for(int j=0;j<width;j++)hidden_codes[(size_t)i*width+j]=codes[j];
        }
        for(int d=0;d<D;d++){
            int64_t acc=0; double sum=0.,absolute=0.;
            for(int j=0;j<width;j++){
                int8_t w=wo[(size_t)d*width+j];
                if(physical_out)acc+=(int64_t)codes[j]*(int64_t)w;
                if(continuous_out){
                    volatile double weight=(double)w*(double)so[d]; volatile double term=hidden[j]*weight;
                    sum+=term;absolute+=fabs(term);
                }
            }
            if(physical_out){ volatile double first=(double)acc*(double)so[d]; volatile double second=first*(double)ah;
                physical_out[(size_t)i*D+d]=(float)second;integer_out[(size_t)i*D+d]=acc; }
            if(continuous_out){continuous_out[(size_t)i*D+d]=sum;absolute_out[(size_t)i*D+d]=absolute;}
        }
    }
    return 0;
}
