/* Exact reduction used by the pinned generic ggml_vec_dot_f16 build. */
#ifndef STRAT01_F16_VECTOR_DOT_H
#define STRAT01_F16_VECTOR_DOT_H

static uint64_t strat01_f16vec_qk_invocations=0;
static uint64_t strat01_f16vec_value_invocations=0;

static float strat01_f16vec_half_to_float(uint16_t h) {
    uint32_t sign=((uint32_t)h&0x8000U)<<16,exp=((uint32_t)h>>10)&31U,mant=(uint32_t)h&1023U,u;float x;
    if(exp==0){if(!mant)u=sign;else{int e=-14;while(!(mant&0x400U)){mant<<=1;--e;}mant&=0x3ffU;u=sign|((uint32_t)(e+127)<<23)|(mant<<13);}}
    else if(exp==31)u=sign|0x7f800000U|(mant<<13);
    else u=sign|((exp+112U)<<23)|(mant<<13);
    memcpy(&x,&u,4);return x;
}

static float strat01_f16vec_dot(unsigned n,const uint16_t *x,const uint16_t *y) {
    double out=0.0;for(unsigned i=0;i<n;++i)out+=(double)(strat01_f16vec_half_to_float(x[i])*strat01_f16vec_half_to_float(y[i]));return (float)out;
}

static float strat01_f16vec_scalar_dot(unsigned n,const uint16_t *x,const uint16_t *y) {
    float out=0.0f;for(unsigned i=0;i<n;++i)out+=strat01_f16vec_half_to_float(x[i])*strat01_f16vec_half_to_float(y[i]);return out;
}

static void strat01_f16vec_reset_counts(void){strat01_f16vec_qk_invocations=0;strat01_f16vec_value_invocations=0;}

#endif
