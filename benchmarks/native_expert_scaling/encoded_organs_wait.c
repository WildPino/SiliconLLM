/* One runtime-policy intervention; every numerical primitive remains included. */
#include "contextual_generator_codec.c"
API int ew_setup(void){worker_setup(3);return 0;}
API int ew_blocktime(void){return kmp_get_blocktime();}
