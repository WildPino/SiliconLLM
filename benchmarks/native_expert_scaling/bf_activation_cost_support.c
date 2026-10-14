/* Evict shared activation payload once after the last parameter range. */
#define cost_evict cost_parameter_evict
#include "native_24_cost_support.c"
#undef cost_evict
static const void *final_parameter=NULL,*table_conv=NULL,*table_gate=NULL,*table_delta=NULL;
static uint64_t table_flush_groups=0;
API void cost_activation_bind(const void*p,const void*c,const void*g,const void*d){
    final_parameter=p;table_conv=c;table_gate=g;table_delta=d;table_flush_groups=0;
}
API uint64_t cost_activation_groups(void){return table_flush_groups;}
API void cost_evict(const void*p,size_t n){
    cost_parameter_evict(p,n);
    if(p==final_parameter&&table_conv&&table_gate&&table_delta){
        cost_parameter_evict(table_conv,65536*2);cost_parameter_evict(table_gate,65536*8);
        cost_parameter_evict(table_delta,65536*2);table_flush_groups++;
    }
}
