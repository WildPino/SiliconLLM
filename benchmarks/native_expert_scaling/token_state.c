/* Token-only F0 -> persistent conv -> unchanged state96 core -> integer readout. */
#include "full_channel_native.c"
#define bf integer_bf
#include "integer_readout.c"
#undef bf
#include <fenv.h>
typedef struct { uint64_t seen;uint32_t last_id;uint16_t ring[3*3584];float state[3072*96]; } TokenState;
API int ts_abi(int i){int v[]={65537,6704,3072,96,3584,3,2048};return i>=0&&i<7?v[i]:-1;}
API void* ts_create(void){return calloc(1,sizeof(TokenState));}
API void ts_destroy(void*p){free(p);}
API uint32_t ts_last(const void*p){const TokenState*s=p;return s&&s->seen?s->last_id:UINT32_MAX;}
API int ts_snapshot(const void*p,float*state,uint16_t*ring,uint64_t*seen){if(!p)return 1;const TokenState*s=p;memcpy(state,s->state,sizeof(s->state));memcpy(ring,s->ring,sizeof(s->ring));*seen=s->seen;return 0;}
/* w Ws,Vs,Alog,D,gamma;conv cw/ cb/ db DOUBLE. f B',C',latent;
 * bits fields6704,norm3072,output2048;int16 input;int64 dots;F64 a/raw. */
API int ts_step(void*p,uint32_t id,const uint16_t*table,const float*const*w,const double*const*conv,const int16_t*q,const double*scale,float*const*f,uint16_t*const*bits,int16_t*r,int64_t*dots,double*a,double*raw){
    if(!p||id>=65537)return 1;if(fegetround()!=FE_TONEAREST)return 3;TokenState*s=p;
    const uint16_t*row=table+(size_t)id*6704;float x[3072],bb[256],cc[256],dt[48],gate[3072],decay[48];double u[3072];
    memcpy(bits[0],row,3072*sizeof(uint16_t));for(int c=0;c<3072;c++)gate[c]=unbf(row[c]);
    for(int c=0;c<3584;c++){
        double value=conv[1][c];
        for(int tap=0;tap<4;tap++){
            int lag=3-tap;double v=0;if(lag==0)v=unbf(row[3072+c]);else if(s->seen>=(uint64_t)lag)v=unbf(s->ring[((s->seen-lag)%3)*3584+c]);
            value+=v*conv[0][c*4+tap];
        }
        double v=unbf(bf((float)value));double z=fmax(0.,-v)+log1p(exp(-fabs(v)));uint16_t b=bf((float)(v*exp(-z)));bits[0][3072+c]=b;
        if(c<3072)x[c]=unbf(b);else if(c<3328)bb[c-3072]=unbf(b);else cc[c-3328]=unbf(b);
    }
    for(int h=0;h<48;h++){
        double d=unbf(bf((float)((double)unbf(row[6656+h])+conv[2][h])));double z=d>20?d:fmax(d,0.)+log1p(exp(-fabs(d)));uint16_t b=bf((float)z);bits[0][6656+h]=b;dt[h]=unbf(b);
    }
    memcpy(s->ring+(s->seen%3)*3584,row+3072,3584*sizeof(uint16_t));
    for(int j=0;j<96;j++){f[0][j]=dotf(w[0]+j*256,bb,256);f[1][j]=dotf(w[1]+j*256,cc,256);}
    for(int h=0;h<48;h++)decay[h]=expf(dt[h]*(-expf(w[2][h])));
    for(int c=0;c<3072;c++){
        int h=c/64;float tx=dt[h]*x[c],*ss=s->state+c*96;__m256 aa=_mm256_set1_ps(decay[h]),b=_mm256_set1_ps(tx),acc=_mm256_setzero_ps();
        for(int j=0;j<96;j+=8){__m256 z=_mm256_fmadd_ps(aa,_mm256_loadu_ps(ss+j),_mm256_mul_ps(b,_mm256_loadu_ps(f[0]+j)));_mm256_storeu_ps(ss+j,z);acc=_mm256_fmadd_ps(z,_mm256_loadu_ps(f[1]+j),acc);}f[2][c]=hsum256(acc)+w[3][h]*x[c];
    }
    double ms=0;for(int c=0;c<3072;c++){u[c]=(double)f[2][c]*((double)gate[c]/(1.0+exp(-(double)gate[c])));ms+=u[c]*u[c];}double inv=1.0/sqrt(ms/3072.0+1e-5),peak=0;
    for(int c=0;c<3072;c++){uint16_t b=bf((float)((u[c]*inv)*(double)w[4][c]));bits[1][c]=b;double v=fabs((double)unbf(b));if(v>peak)peak=v;}
    *a=peak/32767.;for(int c=0;c<3072;c++){double z=*a?nearbyint((double)unbf(bits[1][c])/(*a)):0;if(z>32767)z=32767;if(z< -32767)z= -32767;r[c]=(int16_t)z;}
    int rc=ir_history(1,q,scale,r,a,dots,raw,bits[2]);s->seen++;s->last_id=id;return rc;
}
API int ts_block(void*p,int n,const uint32_t*ids,const uint16_t*table,const float*const*w,const double*const*conv,const int16_t*q,const double*scale,float*const*f,uint16_t*const*bits,int16_t*r,int64_t*dots,double*a,double*raw){
    if(n<1||n>1507)return 1;
    for(int t=0;t<n;t++){
        float*ff[]={f[0]+(size_t)t*96,f[1]+(size_t)t*96,f[2]+(size_t)t*3072};uint16_t*bb[]={bits[0]+(size_t)t*6704,bits[1]+(size_t)t*3072,bits[2]+(size_t)t*2048};
        int rc=ts_step(p,ids[t],table,w,conv,q,scale,ff,bb,r+(size_t)t*3072,dots+(size_t)t*2048,a+t,raw+(size_t)t*2048);if(rc)return rc;
    }return 0;
}
