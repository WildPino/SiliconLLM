/* Capture unchanged330 router arithmetic; original output must remain byte exact. */
#define main meth330_entry
#include "meth331_switch_router_trace.c"
#undef main
int main(int argc,char **argv){require(argc==7,"arguments:manifest source decoder output fault trace");router_trace=fopen(argv[6],"wb");require(router_trace!=NULL,"trace_open");int result=meth330_entry(6,argv);require(fclose(router_trace)==0,"trace_close");return result;}
