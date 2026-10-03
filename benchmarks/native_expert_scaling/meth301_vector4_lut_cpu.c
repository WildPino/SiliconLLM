// Source-shaped four-coefficient U8 LUT cost preflight; not model inference.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>
#include <psapi.h>

#define NT 309
#define NC 256
#define TILE 32
typedef struct {
    uint32_t layer,kind,d,o,b,a,q;
    uint8_t *codes;
    float *scales,*palette,*x,*table,*y,*router;
    uint64_t bytes;
} Proj;
static Proj proj[NT];
static unsigned routes[26][4];
static uint64_t weight_bytes,lookup_count,table_entries,router_coeffs,fixture_bytes;
static double started;

static void fail(const char *s){fprintf(stderr,"M301 apparatus failure: %s\n",s);exit(2);}
static double now(void){LARGE_INTEGER a,b;QueryPerformanceFrequency(&a);QueryPerformanceCounter(&b);return (double)b.QuadPart/(double)a.QuadPart;}
static uint64_t mix(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static float val(uint64_t key){return ((int)(mix(key)&511)-256)*(1.0f/256.0f);}
static void *alloc(uint64_t bytes){
    if(!bytes || bytes>UINT64_C(40)*1024*1024*1024)fail("allocation bound");
    void *p=_aligned_malloc((size_t)bytes,64);if(!p)fail("allocation failed");
    fixture_bytes+=bytes;return p;
}
static uint64_t rss(void){PROCESS_MEMORY_COUNTERS p;memset(&p,0,sizeof(p));p.cb=sizeof(p);if(!GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p)))fail("RSS query");return p.PeakWorkingSetSize;}
static void guards(void){if(now()-started>600.0 || rss()>UINT64_C(40)*1024*1024*1024)fail("resource stop");}
static size_t code_offset(const Proj *p,unsigned bank,unsigned row,unsigned group){
    return (size_t)bank*p->o*(p->d/4)+(size_t)(row/TILE)*TILE*(p->d/4)+(size_t)group*TILE+row%TILE;
}
static void load(const char *path,unsigned experts){
    FILE *f=fopen(path,"rb");if(!f)fail("spec open");
    char magic[8];uint32_t count,source_n;
    if(fread(magic,1,8,f)!=8 || memcmp(magic,"M301SPC1",8) || fread(&count,4,1,f)!=1 || fread(&source_n,4,1,f)!=1 || count!=NT || source_n!=64)fail("spec header");
    unsigned routers=0,routed=0;
    for(unsigned i=0;i<NT;i++){
        Proj *p=&proj[i];uint32_t v[7];if(fread(v,4,7,f)!=7)fail("short spec");
        p->layer=v[0];p->kind=v[1];p->d=v[2];p->o=v[3];p->b=v[4];p->a=v[5];p->q=v[6];
        if(p->layer>26 || p->kind>3 || p->d<4 || p->d>8960 || p->d%4 || !p->o || p->o%32 || p->o>128256 || !p->b || !p->a || !p->q)fail("spec dimensions");
        if(p->kind==3){
            if(p->d!=1536 || p->o!=64 || p->b!=1 || p->a!=1 || p->q!=1 || p->layer<1 || p->layer>25)fail("router spec");
            routers++;p->o=experts;p->router=alloc((uint64_t)p->d*p->o*4);
            #pragma omp parallel for schedule(static)
            for(int64_t j=0;j<(int64_t)p->d*p->o;j++)p->router[j]=val((uint64_t)j+UINT64_C(50000000000)+i*UINT64_C(100000000));
            p->x=alloc((uint64_t)p->d*4);p->y=alloc((uint64_t)p->o*4);
            router_coeffs+=(uint64_t)p->d*p->o;weight_bytes+=(uint64_t)p->d*p->o*4;
            continue;
        }
        if(p->kind==1 || p->kind==2){
            if(p->b!=64 || p->a!=4 || p->q!=(p->kind==1?1U:4U) || p->layer<1 || p->layer>25)fail("routed spec");
            routed++;p->b=experts;
        }else if(p->b!=p->a || p->q!=p->a)fail("projection bank spec");
        p->bytes=(uint64_t)p->d*p->o*p->b/4;
        p->codes=alloc(p->bytes);p->scales=alloc((uint64_t)p->o*p->b*4);
        p->palette=alloc(4096);p->x=alloc((uint64_t)p->d*p->q*4);
        p->table=alloc((uint64_t)p->d/4*p->q*NC*4);p->y=alloc((uint64_t)p->o*p->a*4);
        #pragma omp parallel for schedule(static)
        for(int64_t j=0;j<(int64_t)p->bytes/8;j++)((uint64_t*)p->codes)[j]=mix((uint64_t)j+i*UINT64_C(10000000000));
        #pragma omp parallel for schedule(static)
        for(int64_t j=0;j<(int64_t)p->o*p->b;j++)p->scales[j]=1.0f+(float)(mix((uint64_t)j+i*UINT64_C(10000000000))&127)*(1.0f/128.0f);
        for(unsigned j=0;j<1024;j++)p->palette[j]=val(j+i*UINT64_C(10000000));
        lookup_count+=(uint64_t)p->d*p->o*p->a/4;
        table_entries+=(uint64_t)p->d/4*p->q*NC;
        weight_bytes+=(uint64_t)p->d*p->o*p->a/4+(uint64_t)p->o*p->a*4+4096;
    }
    if(fgetc(f)!=EOF || routers!=25 || routed!=75)fail("spec inventory");fclose(f);
    if(lookup_count!=406405120 || table_entries!=70352896)fail("source geometry counts");
    guards();
}
static void inputs(unsigned token){
    // Fixtures are input-ready operator probes, not a composed causal model.
    for(unsigned i=0;i<NT;i++){
        Proj *p=&proj[i];
        for(unsigned bank=0;bank<p->q;bank++)for(unsigned j=0;j<p->d;j++){
            uint64_t seed=(uint64_t)token*UINT64_C(1000000000)+(uint64_t)p->layer*UINT64_C(10000000)+j;
            // Routed gate/up and router probes at a layer see the same input.
            if(p->kind==2)seed+=UINT64_C(500000)+bank*UINT64_C(100000);
            else if(p->kind==0)seed+=i*UINT64_C(100000)+bank*UINT64_C(10000);
            p->x[(size_t)bank*p->d+j]=val(seed);
        }
    }
}
static void tables(Proj *p){
    unsigned groups=p->d/4;
    #pragma omp parallel for schedule(static)
    for(int g=0;g<(int)(groups*p->q);g++){
        const float *x=p->x+(size_t)g*4;
        float *t=p->table+(size_t)g*NC;
        __m256 x0=_mm256_set1_ps(x[0]),x1=_mm256_set1_ps(x[1]),x2=_mm256_set1_ps(x[2]),x3=_mm256_set1_ps(x[3]);
        for(unsigned c=0;c<NC;c+=8){
            __m256 v=_mm256_mul_ps(x0,_mm256_load_ps(p->palette+c));
            v=_mm256_add_ps(v,_mm256_mul_ps(x1,_mm256_load_ps(p->palette+NC+c)));
            v=_mm256_add_ps(v,_mm256_mul_ps(x2,_mm256_load_ps(p->palette+2*NC+c)));
            v=_mm256_add_ps(v,_mm256_mul_ps(x3,_mm256_load_ps(p->palette+3*NC+c)));
            _mm256_store_ps(t+c,v);
        }
    }
}
static unsigned stored_bank(const Proj *p,unsigned active){return (p->kind==1 || p->kind==2)?routes[p->layer][active]:active;}
static void project(Proj *p){
    unsigned groups=p->d/4,tiles=p->o/TILE;
    #pragma omp parallel for schedule(static)
    for(int tile=0;tile<(int)(tiles*p->a);tile++){
        unsigned active=(unsigned)tile/tiles,row=((unsigned)tile%tiles)*TILE;
        unsigned bank=stored_bank(p,active),query=p->q==1?0:active;
        const uint8_t *code=p->codes+code_offset(p,bank,row,0);
        const float *tab=p->table+(size_t)query*groups*NC;
        __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;
        for(unsigned g=0;g<groups;g++,code+=TILE,tab+=NC){
            __m256i q0=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(code)));
            __m256i q1=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(code+8)));
            __m256i q2=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(code+16)));
            __m256i q3=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(code+24)));
            a0=_mm256_add_ps(a0,_mm256_i32gather_ps(tab,q0,4));
            a1=_mm256_add_ps(a1,_mm256_i32gather_ps(tab,q1,4));
            a2=_mm256_add_ps(a2,_mm256_i32gather_ps(tab,q2,4));
            a3=_mm256_add_ps(a3,_mm256_i32gather_ps(tab,q3,4));
        }
        const float *s=p->scales+(size_t)bank*p->o+row;
        float *y=p->y+(size_t)active*p->o+row;
        _mm256_store_ps(y,_mm256_mul_ps(a0,_mm256_load_ps(s)));
        _mm256_store_ps(y+8,_mm256_mul_ps(a1,_mm256_load_ps(s+8)));
        _mm256_store_ps(y+16,_mm256_mul_ps(a2,_mm256_load_ps(s+16)));
        _mm256_store_ps(y+24,_mm256_mul_ps(a3,_mm256_load_ps(s+24)));
    }
}
static void router(Proj *p){
    #pragma omp parallel for schedule(static)
    for(int r=0;r<(int)p->o;r++){
        __m256 sum=_mm256_setzero_ps();const float *w=p->router+(size_t)r*p->d;
        for(unsigned j=0;j<p->d;j+=8)sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_load_ps(w+j),_mm256_load_ps(p->x+j)));
        float lanes[8];_mm256_storeu_ps(lanes,sum);float v=0;for(unsigned j=0;j<8;j++)v+=lanes[j];p->y[r]=v;
    }
    for(unsigned k=0;k<4;k++){
        unsigned best=UINT32_MAX;
        for(unsigned r=0;r<p->o;r++){
            int used=0;for(unsigned j=0;j<k;j++)if(routes[p->layer][j]==r)used=1;
            if(!used && (best==UINT32_MAX || p->y[r]>p->y[best]))best=r;
        }
        if(best==UINT32_MAX)fail("top4 selection");routes[p->layer][k]=best;
    }
}
static void execute(void){
    for(unsigned i=0;i<NT;i++)if(proj[i].kind==3)router(&proj[i]);
    for(unsigned i=0;i<NT;i++)if(proj[i].kind!=3){tables(&proj[i]);project(&proj[i]);}
}
static uint64_t output_hash(void){
    uint64_t h=UINT64_C(0x123456789abcdef0);
    for(unsigned i=0;i<NT;i++){
        Proj *p=&proj[i];unsigned n=p->kind==3?p->o:p->o*p->a;
        for(unsigned j=0;j<n;j++){uint32_t u;memcpy(&u,p->y+j,4);if(!isfinite(p->y[j]))fail("nonfinite output");h=mix(h^u);}
    }
    for(unsigned l=1;l<26;l++)for(unsigned k=0;k<4;k++)h=mix(h^routes[l][k]);
    return h;
}
static void selftest(void){
    inputs(13);execute();double err2=0,norm2=0;uint64_t exact=0,decoded=0;int detected=0;
    for(unsigned i=0;i<NT;i++){
        Proj *p=&proj[i];
        if(p->kind==3){
            for(unsigned r=0;r<p->o;r++){
                double v=0;for(unsigned j=0;j<p->d;j++)v+=(double)p->router[(size_t)r*p->d+j]*p->x[j];
                double e=p->y[r]-v;err2+=e*e;norm2+=v*v;decoded++;
            }
            continue;
        }
        for(unsigned active=0;active<p->a;active++)for(unsigned j=0;j<8;j++){
            unsigned row=j*(p->o-1)/7,bank=stored_bank(p,active),query=p->q==1?0:active;
            float sum=0;double ref=0;unsigned groups=p->d/4;
            for(unsigned g=0;g<groups;g++){
                unsigned c=p->codes[code_offset(p,bank,row,g)];
                sum+=p->table[((size_t)query*groups+g)*NC+c];
                for(unsigned z=0;z<4;z++)ref+=(double)p->palette[z*NC+c]*p->x[(size_t)query*p->d+g*4+z];
            }
            float s=p->scales[(size_t)bank*p->o+row];sum*=s;ref*=s;
            if(memcmp(&sum,p->y+(size_t)active*p->o+row,4))fail("scalar/AVX LUT mismatch");exact++;
            double e=p->y[(size_t)active*p->o+row]-ref;err2+=e*e;norm2+=ref*ref;decoded++;
        }
    }
    // A wrong query-table value must be detected against the unchanged palette.
    Proj *p=&proj[0];if(p->kind==3)fail("first descriptor contract");
    unsigned c=p->codes[0];float before=p->y[0],saved=p->table[c];
    p->table[c]+=128.0f;project(p);detected=p->y[0]!=before;p->table[c]=saved;project(p);
    if(!detected || norm2<=0 || sqrt(err2/norm2)>1e-5)fail("decoded FP64/negative control");
    printf("{\"event\":\"selftest\",\"exact_scalar_rows\":%llu,\"decoded_reference_rows\":%llu,\"pooled_relative_l2\":%.17g,\"bad_table_detected\":true}\n",(unsigned long long)exact,(unsigned long long)decoded,sqrt(err2/norm2));fflush(stdout);
}
int main(int argc,char **argv){
    if(argc!=3)fail("usage: SPEC experts64or640");
    unsigned n=(unsigned)strtoul(argv[2],NULL,10);if(n!=64 && n!=640)fail("unsupported expert count");
    started=now();omp_set_dynamic(0);omp_set_num_threads(6);load(argv[1],n);
    printf("{\"event\":\"ready\",\"experts\":%u,\"threads\":6,\"allocated_fixture_bytes\":%llu,\"weight_bytes_per_token\":%llu,\"lookups_per_token\":%llu,\"table_entries_per_token\":%llu,\"router_coefficients_per_token\":%llu,\"initialization_seconds\":%.9f}\n",n,(unsigned long long)fixture_bytes,(unsigned long long)weight_bytes,(unsigned long long)lookup_count,(unsigned long long)table_entries,(unsigned long long)router_coeffs,now()-started);fflush(stdout);
    selftest();unsigned char touched[26][640];memset(touched,0,sizeof(touched));
    for(unsigned rep=0;rep<3;rep++){
        for(unsigned tok=0;tok<10;tok++){
            inputs(tok);double t=now();execute();double seconds=now()-t;
            uint64_t h=output_hash();
            for(unsigned l=1;l<26;l++)for(unsigned k=0;k<4;k++)touched[l][routes[l][k]]=1;
            printf("{\"event\":\"token\",\"rep\":%u,\"token\":%u,\"warmup\":%s,\"seconds\":%.12f,\"output_hash\":\"%016llx\"}\n",rep,tok,tok<2?"true":"false",seconds,(unsigned long long)h);fflush(stdout);
        }
        guards();
    }
    unsigned min_union=n,max_union=0;
    for(unsigned l=1;l<26;l++){unsigned sum=0;for(unsigned j=0;j<n;j++)sum+=touched[l][j];if(sum<min_union)min_union=sum;if(sum>max_union)max_union=sum;}
    printf("{\"event\":\"finished\",\"min_selected_expert_union\":%u,\"max_selected_expert_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f}\n",min_union,max_union,(unsigned long long)rss(),now()-started);return 0;
}
