// METH-226: one full-input BF16 affine donor tangent per fixture layer.
#define main meth182_unused_main
#include "meth182_group64_ffn_cpu.c"
#undef main
typedef struct { const uint16_t *w; const float *bias; } Tangent;
static Tangent tangents[L];
// Same BF16 decode and four-accumulator reduction as the frozen METH-224.
static __m256 tangent_bf8(const uint16_t *p) {
    __m128i raw=_mm_loadu_si128((const __m128i*)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw),16));
}
static void tangent_forward(int li,const uint16_t *bits,float *out) {
    float x[D]; for(int j=0;j<D;j++) x[j]=bf16_to_float(bits[j]);
    #pragma omp parallel for schedule(static)
    for(int row=0;row<D;row++) {
        const uint16_t *w=tangents[li].w+(size_t)row*D;
        __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
        for(int j=0;j<D;j+=32) {
            a=_mm256_fmadd_ps(tangent_bf8(w+j),_mm256_loadu_ps(x+j),a);
            b=_mm256_fmadd_ps(tangent_bf8(w+j+8),_mm256_loadu_ps(x+j+8),b);
            c=_mm256_fmadd_ps(tangent_bf8(w+j+16),_mm256_loadu_ps(x+j+16),c);
            d=_mm256_fmadd_ps(tangent_bf8(w+j+24),_mm256_loadu_ps(x+j+24),d);
        }
        __m256 total=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
        float lanes[8]; _mm256_storeu_ps(lanes,total);
        float sum=0; for(int j=0;j<8;j++) sum+=lanes[j];
        out[row]=sum+tangents[li].bias[row];
    }
    for(int j=0;j<D;j++) if(!isfinite(out[j])) die("nonfinite tangent");
}
int main(int argc,char **argv) {
    if(argc!=4) die("usage: meth226 weights vectors check");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("ISA unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6);
    size_t wn,vn; uint8_t *weights=read_file(argv[1],&wn),*vectors=read_file(argv[2],&vn);
    if(wn<24 || memcmp(weights,"M226BF01",8) || u32le(weights+8)!=L ||
       u32le(weights+12)!=D || u32le(weights+16)!=D || u32le(weights+20)!=32) die("weight binding");
    size_t pos=24;
    for(int li=0;li<L;li++) {
        size_t wb=(size_t)D*D*2;
        if(pos>wn || wb>wn-pos) die("short weight");
        tangents[li].w=(const uint16_t*)(weights+pos); pos+=wb;
        if(pos>wn || D*4>wn-pos) die("short bias");
        tangents[li].bias=(const float*)(weights+pos); pos+=D*4;
    }
    if(pos!=wn || vn!=24+(size_t)TOKENS*L*D*2 || memcmp(vectors,"M125HX01",8) ||
       u32le(vectors+8)!=L || u32le(vectors+12)!=TOKENS || u32le(vectors+16)!=D ||
       u32le(vectors+20)!=1280) die("vector binding");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *out=fopen(argv[3],"wb"); if(!out) die("check open");
    if(fwrite("M226OUT1",1,8,out)!=8) die("header write");
    uint32_t dims[3]={CHECK_TOKENS,L,D};
    if(fwrite(dims,1,sizeof(dims),out)!=sizeof(dims)) die("dims write");
    float y[D]; double checksum=0;
    for(int t=0;t<CHECK_TOKENS;t++) for(int li=0;li<L;li++) {
        tangent_forward(li,states+((size_t)t*L+li)*D,y);
        if(fwrite(y,1,sizeof(y),out)!=sizeof(y)) die("check write");
    }
    if(fclose(out)) die("check close");
    double elapsed[3];
    for(int rep=0;rep<3;rep++) {
        double start=seconds();
        for(int t=0;t<TOKENS;t++) for(int li=0;li<L;li++) {
            tangent_forward(li,states+((size_t)t*L+li)*D,y); checksum+=y[(t+li)%D];
        }
        elapsed[rep]=seconds()-start;
    }
    PROCESS_MEMORY_COUNTERS memory;
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))) die("RSS read");
    printf("{\"threads\":6,\"tokens\":256,\"layers\":24,\"input_width\":896,\"weight_bytes\":%llu,\"pass_ms_per_token\":[%.6f,%.6f,%.6f],\"checksum\":%.9f,\"peak_working_set_bytes\":%llu}\n",
        (unsigned long long)wn,elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,checksum,
        (unsigned long long)memory.PeakWorkingSetSize);
    free(weights); free(vectors); return 0;
}
