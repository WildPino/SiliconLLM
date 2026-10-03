// New direct Windows affinity wrapper; immutable306 computation retained.
#define main meth306_immutable_main
#include "meth306_packed_i4_cpu.c"
#undef main
static unsigned physical_cpu[6];
static void parse_cpus(const char *s){
    for(unsigned i=0;i<6;i++){
        char *end;unsigned long cpu=strtoul(s,&end,10);if(end==s||cpu>=64)die("CPU list range");
        physical_cpu[i]=(unsigned)cpu;for(unsigned j=0;j<i;j++)if(physical_cpu[j]==cpu)die("duplicate CPU");
        if(i<5){if(*end!=',')die("CPU list separator");s=end+1;}else if(*end)die("CPU list trailing data");
    }
}
static void native_affinity(unsigned phase,int set){
    unsigned current[6]={0},group[6]={0};uint64_t masks[6]={0};int success[6]={0};
    #pragma omp parallel num_threads(6)
    {
        unsigned t=(unsigned)omp_get_thread_num();GROUP_AFFINITY a;memset(&a,0,sizeof(a));
        DWORD_PTR wanted=(DWORD_PTR)UINT64_C(1)<<physical_cpu[t];
        int ok=!set||SetThreadAffinityMask(GetCurrentThread(),wanted)!=0;
        if(ok)ok=GetThreadGroupAffinity(GetCurrentThread(),&a)!=0;
        if(ok){group[t]=a.Group;masks[t]=(uint64_t)a.Mask;current[t]=GetCurrentProcessorNumber();}success[t]=ok;
    }
    for(unsigned t=0;t<6;t++)if(!success[t]||group[t]!=0||masks[t]!=(UINT64_C(1)<<physical_cpu[t])||(phase&&current[t]!=physical_cpu[t]))die("actual Windows affinity mismatch");
    printf("{\"event\":\"native_affinity\",\"phase\":%u,\"threads\":[",phase);
    for(unsigned t=0;t<6;t++)printf("%s{\"thread\":%u,\"group\":%u,\"mask\":%llu,\"current_cpu\":%u}",t?",":"",t,group[t],(unsigned long long)masks[t],current[t]);
    printf("]}\n");fflush(stdout);
}
int main(int argc,char **argv){
    if(argc!=4)die("usage: SPEC MODEL CPU_LIST");parse_cpus(argv[3]);fesetround(FE_TONEAREST);omp_set_dynamic(0);omp_set_num_threads(6);started=clock_s();native_affinity(0,1);initialize(argv[1],argv[2]);
    printf("{\"event\":\"ready\",\"parent_count\":64,\"children_per_parent\":10,\"functions_per_layer\":640,\"threads\":6,\"code_bits\":4,\"row_tile\":4,\"serial_if_coefficients_less_than\":32768,\"allocated_bytes\":%llu,\"active_coefficients\":%llu,\"active_row_scale_bytes\":%llu,\"stored_coefficients\":%llu,\"initialization_seconds\":%.9f}\n",(unsigned long long)allocated,(unsigned long long)active_i8,(unsigned long long)active_scales,(unsigned long long)code_capacity,clock_s()-started);fflush(stdout);
    selftest();unsigned char selected[26][BANKS];memset(selected,0,sizeof(selected));
    for(unsigned rep=0;rep<3;rep++){
        native_affinity(rep*2+1,0);
        for(unsigned token=0;token<10;token++){
            prepare(token);double t=clock_s();execute();double seconds=clock_s()-t;uint64_t hash=output_hash();
            for(unsigned l=1;l<LAYERS;l++)for(unsigned k=0;k<4;k++)selected[l][layers[l].flat[k]]=1;
            printf("{\"event\":\"operator\",\"rep\":%u,\"input\":%u,\"warmup\":%s,\"seconds\":%.12f,\"output_route_hash\":\"%016llx\"}\n",rep,token,token<2?"true":"false",seconds,(unsigned long long)hash);fflush(stdout);
        }
        native_affinity(rep*2+2,0);guards();
    }
    unsigned lo=BANKS,hi=0;for(unsigned l=1;l<LAYERS;l++){unsigned n=0;for(unsigned j=0;j<BANKS;j++)n+=selected[l][j];if(n<lo)lo=n;if(n>hi)hi=n;}
    printf("{\"event\":\"finished\",\"minimum_selected_child_union\":%u,\"maximum_selected_child_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f}\n",lo,hi,(unsigned long long)peak_rss(),clock_s()-started);return 0;
}
