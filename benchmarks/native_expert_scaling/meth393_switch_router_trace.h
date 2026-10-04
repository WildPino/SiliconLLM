/* Diagnostic capture only: scores/input are copied after the unchanged mv. */
static FILE *router_audit_file;
static void router_audit_close(void){
    if(router_audit_file){FILE *file=router_audit_file;router_audit_file=NULL;require(fclose(file)==0,"router_audit_close");}
}
static void router_audit(Config c,const float *input,const float *scores){
    if(!router_audit_file){
        const char *path=getenv("SILICON_ROUTER_AUDIT_PATH");
        require(path&&path[0],"router_audit_path");router_audit_file=fopen(path,"wb");
        require(router_audit_file!=NULL,"router_audit_open");
        require(fwrite("SWRTA001",1,8,router_audit_file)==8,"router_audit_magic");
        uint32_t dimensions[]={(uint32_t)c.n,(uint32_t)c.d};
        require(fwrite(dimensions,4,2,router_audit_file)==2,"router_audit_dimensions");
        require(atexit(router_audit_close)==0,"router_audit_atexit");
    }
    uint32_t record[]={(uint32_t)route_count,(uint32_t)cost_phase};
    require(fwrite(record,4,2,router_audit_file)==2,"router_audit_index");
    require(fwrite(input,4,c.d,router_audit_file)==(size_t)c.d,"router_audit_input");
    require(fwrite(scores,4,c.n,router_audit_file)==(size_t)c.n,"router_audit_scores");
}
