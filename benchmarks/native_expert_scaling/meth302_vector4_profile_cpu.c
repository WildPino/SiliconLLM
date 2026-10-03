// Instrument immutable301 operators; no arithmetic/layout/fixture changes.
#define main m301_original_main
#include "meth301_vector4_lut_cpu.c"
#undef main

static unsigned organ(const Proj *p){
    if(p->kind==3)return 5; // router
    if(p->kind==1 || p->kind==2)return 1; // routed
    if(p->layer==26 && p->d==1536 && p->o==128256)return 4; // head
    if(p->layer==0 && (p->d==8960 || p->o==8960))return 3; // dense
    if((p->d==1536 && p->o==1280) || (p->d==1280 && p->o==1536))return 2; // shared
    if(p->b==32 || (p->d==1536 && (p->o==6144 || p->o==576)) || (p->d==6144 && p->o==1536))return 0; // MLA
    fail("profile organ mapping");return 0;
}
static void profile_execute(double build[6],double matvec[6]){
    memset(build,0,6*sizeof(double));memset(matvec,0,6*sizeof(double));
    for(unsigned i=0;i<NT;i++)if(proj[i].kind==3){double t=now();router(&proj[i]);matvec[5]+=now()-t;}
    for(unsigned i=0;i<NT;i++)if(proj[i].kind!=3){
        unsigned k=organ(&proj[i]);double t=now();tables(&proj[i]);build[k]+=now()-t;
        t=now();project(&proj[i]);matvec[k]+=now()-t;
    }
}
static void print_array(const double *v){fputc('[',stdout);for(unsigned i=0;i<6;i++)printf("%s%.12f",i?",":"",v[i]);fputc(']',stdout);}
int main(int argc,char **argv){
    if(argc!=3 || strcmp(argv[2],"64"))fail("profile SPEC n64 only");
    started=now();omp_set_dynamic(0);omp_set_num_threads(6);load(argv[1],64);
    unsigned counts[6]={0};for(unsigned i=0;i<NT;i++)counts[organ(&proj[i])]++;
    const unsigned expected[6]={130,75,75,3,1,25};if(memcmp(counts,expected,sizeof(counts)))fail("profile inventory");
    printf("{\"event\":\"ready\",\"experts\":64,\"threads\":6,\"allocated_fixture_bytes\":%llu,\"weight_bytes_per_token\":%llu,\"lookups_per_token\":%llu,\"table_entries_per_token\":%llu,\"router_coefficients_per_token\":%llu,\"initialization_seconds\":%.9f}\n",(unsigned long long)fixture_bytes,(unsigned long long)weight_bytes,(unsigned long long)lookup_count,(unsigned long long)table_entries,(unsigned long long)router_coeffs,now()-started);fflush(stdout);
    selftest();unsigned char touched[26][640];memset(touched,0,sizeof(touched));
    for(unsigned rep=0;rep<3;rep++){
        for(unsigned tok=0;tok<10;tok++){
            double build[6],matvec[6];inputs(tok);double t=now();profile_execute(build,matvec);double seconds=now()-t;
            uint64_t h=output_hash();
            for(unsigned l=1;l<26;l++)for(unsigned k=0;k<4;k++)touched[l][routes[l][k]]=1;
            printf("{\"event\":\"token\",\"rep\":%u,\"token\":%u,\"warmup\":%s,\"seconds\":%.12f,\"output_hash\":\"%016llx\",\"table_seconds_by_organ\":",rep,tok,tok<2?"true":"false",seconds,(unsigned long long)h);
            print_array(build);fputs(",\"matvec_seconds_by_organ\":",stdout);print_array(matvec);fputs("}\n",stdout);fflush(stdout);
        }
        guards();
    }
    unsigned min_union=64,max_union=0;
    for(unsigned l=1;l<26;l++){unsigned sum=0;for(unsigned j=0;j<64;j++)sum+=touched[l][j];if(sum<min_union)min_union=sum;if(sum>max_union)max_union=sum;}
    printf("{\"event\":\"finished\",\"min_selected_expert_union\":%u,\"max_selected_expert_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f}\n",min_union,max_union,(unsigned long long)rss(),now()-started);return 0;
}
