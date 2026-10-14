/* Included after qualified bf/unbf/readout definitions; owners stay in host. */
static const uint16_t *act_conv_table=NULL,*act_delta_table=NULL;
static const double *act_gate_table=NULL;
static unsigned act_bound_control=0;
API unsigned act_control(void){return _mm_getcsr()&0xffc0u;}
API void act_unbind_for_test(void){act_conv_table=NULL;act_gate_table=NULL;act_delta_table=NULL;act_bound_control=0;}
API void act_set_control_for_test(unsigned value){_mm_setcsr((_mm_getcsr()&0x3fu)|(value&0xffc0u));}
API int act_bind(const uint16_t*c,const double*g,const uint16_t*d,unsigned control){
    if(!c||!g||!d||fegetround()!=FE_TONEAREST||control!=act_control()||(control&0x1f80u)!=0x1f80u)return 6;
    act_conv_table=c;act_gate_table=g;act_delta_table=d;act_bound_control=control;return 0;
}
static int act_valid(void){return act_conv_table&&act_gate_table&&act_delta_table&&
    act_bound_control==act_control()&&fegetround()==FE_TONEAREST;}
static uint16_t act_conv(uint16_t code){
    if((code&0x7f80u)!=0x7f80u)return act_conv_table[code];
    double v=unbf(code);double z=fmax(0.,-v)+log1p(exp(-fabs(v)));return bf((float)(v*exp(-z)));
}
static double act_gate(uint16_t code){
    if((code&0x7f80u)!=0x7f80u)return act_gate_table[code];
    double v=unbf(code);return v/(1.0+exp(-v));
}
static uint16_t act_delta(uint16_t code){
    if((code&0x7f80u)!=0x7f80u)return act_delta_table[code];
    double d=unbf(code);double z=d>20?d:fmax(d,0.)+log1p(exp(-fabs(d)));return bf((float)z);
}
API int act_probe(int n,const uint16_t*input,uint16_t*c,double*g,uint16_t*d){
    if(!act_valid())return 5;if(n<1||n>65536)return 1;
    for(int i=0;i<n;i++){c[i]=act_conv(input[i]);g[i]=act_gate(input[i]);d[i]=act_delta(input[i]);}return 0;
}
/* Adversarial controls reuse the UNCHANGED accepted128-column readout. */
API int ir_dot_controls(int n,const int16_t*w,const int16_t*x,int64_t*out){
    if(n<1||n>256)return 1;for(int i=0;i<n;i++)out[i]=dot16(w+(size_t)i*3072,x+(size_t)i*3072);return 0;
}
