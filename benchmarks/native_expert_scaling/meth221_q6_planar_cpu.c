// METH-221: lossless planar group64 Q6, float input/hidden, fixed AVX2 kernel.
#define main meth182_unused_main
#include "meth182_group64_ffn_cpu.c"
#undef main

typedef struct { const uint8_t *q; const uint16_t *scale; int rows,cols; } Q6Matrix;
typedef struct { Q6Matrix gate,up,down; } Q6Layer;
static Q6Layer q6model[L];

static __m256i decode32(const uint8_t *group, int half) {
    __m128i low=_mm_loadu_si128((const __m128i*)(group+half*16));
    __m128i mask=_mm_set1_epi8(15);
    __m128i even=_mm_and_si128(low,mask);
    __m128i odd=_mm_and_si128(_mm_srli_epi16(low,4),mask);
    __m256i lo=_mm256_set_m128i(_mm_unpackhi_epi8(even,odd),_mm_unpacklo_epi8(even,odd));
    __m128i high=_mm_loadl_epi64((const __m128i*)(group+32+half*8));
    __m256i repeated=_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(high),
        _mm256_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,
                        4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7));
    __m256i m0=_mm256_set1_epi32(0x00000003);
    __m256i m1=_mm256_set1_epi32(0x00000300);
    __m256i m2=_mm256_set1_epi32(0x00030000);
    __m256i m3=_mm256_set1_epi32(0x03000000);
    __m256i hi=_mm256_or_si256(_mm256_and_si256(repeated,m0),
        _mm256_and_si256(_mm256_srli_epi16(repeated,2),m1));
    hi=_mm256_or_si256(hi,_mm256_and_si256(_mm256_srli_epi16(repeated,4),m2));
    hi=_mm256_or_si256(hi,_mm256_and_si256(_mm256_srli_epi16(repeated,6),m3));
    __m256i unsigned_codes=_mm256_or_si256(lo,_mm256_slli_epi16(hi,4));
    return _mm256_sub_epi8(unsigned_codes,_mm256_set1_epi8(32));
}
static void codec_check(void) {
    for(int phase=0;phase<64;phase++) {
        uint8_t u[64],packed[48]={0}; int8_t decoded[64];
        for(int j=0;j<64;j++) {
            u[j]=(uint8_t)((phase+j*13)&63);
            packed[j/2]|=(u[j]&15) << ((j&1)*4);
            packed[32+j/4]|=(u[j]>>4) << ((j&3)*2);
        }
        _mm256_storeu_si256((__m256i*)decoded,decode32(packed,0));
        _mm256_storeu_si256((__m256i*)(decoded+32),decode32(packed,1));
        for(int j=0;j<64;j++) if(decoded[j]!=(int)u[j]-32) die("Q6 SIMD codec check");
    }
}
static void bind_q6(Q6Matrix *m, const uint8_t *file, size_t length,
                    size_t *position, int rows, int cols) {
    size_t qb=(size_t)rows*cols/4*3,sb=(size_t)rows*(cols/G)*2;
    if(*position>length || qb>length-*position) die("short Q6 codes");
    m->q=file+*position; *position+=qb;
    if(sb>length-*position) die("short Q6 scales");
    m->scale=(const uint16_t*)(file+*position); *position+=sb;
    m->rows=rows; m->cols=cols;
}
static void q6_matvec(const Q6Matrix *m, const float *input, float *output) {
    int groups=m->cols/G;
    #pragma omp parallel for schedule(static)
    for(int row=0;row<m->rows;row++) {
        const uint8_t *codes=m->q+(size_t)row*groups*48;
        const uint16_t *scales=m->scale+(size_t)row*groups;
        float total=0;
        for(int g=0;g<groups;g++) {
            __m256 acc=_mm256_setzero_ps();
            for(int half=0;half<2;half++) {
                __m256i decoded=decode32(codes+(size_t)g*48,half);
                __m128i bytes0=_mm256_castsi256_si128(decoded);
                __m128i bytes1=_mm256_extracti128_si256(decoded,1);
                __m256 f0=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes0));
                __m256 f1=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes0,8)));
                __m256 f2=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes1));
                __m256 f3=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bytes1,8)));
                const float *x=input+g*G+half*32;
                acc=_mm256_fmadd_ps(f0,_mm256_loadu_ps(x),acc);
                acc=_mm256_fmadd_ps(f1,_mm256_loadu_ps(x+8),acc);
                acc=_mm256_fmadd_ps(f2,_mm256_loadu_ps(x+16),acc);
                acc=_mm256_fmadd_ps(f3,_mm256_loadu_ps(x+24),acc);
            }
            float lanes[8]; _mm256_storeu_ps(lanes,acc);
            float sum=0; for(int j=0;j<8;j++) sum+=lanes[j];
            total+=sum*f16_to_float(scales[g]);
        }
        output[row]=total;
    }
}
static void q6_forward(int li, const uint16_t *bits, float *out) {
    float input[D],gate[H],up[H],hidden[H];
    for(int j=0;j<D;j++) input[j]=bf16_to_float(bits[j]);
    q6_matvec(&q6model[li].gate,input,gate);
    q6_matvec(&q6model[li].up,input,up);
    for(int j=0;j<H;j++) hidden[j]=(gate[j]/(1.0f+expf(-gate[j])))*up[j];
    q6_matvec(&q6model[li].down,hidden,out);
    for(int j=0;j<D;j++) if(!isfinite(out[j])) die("nonfinite Q6 output");
}
int main(int argc, char **argv) {
    if(argc!=4) die("usage: meth221 weights vectors check");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("ISA unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6); codec_check();
    size_t weight_size,vector_size;
    uint8_t *weights=read_file(argv[1],&weight_size),*vectors=read_file(argv[2],&vector_size);
    if(weight_size<24 || memcmp(weights,"M221Q6P1",8) || u32le(weights+8)!=L ||
       u32le(weights+12)!=D || u32le(weights+16)!=H || u32le(weights+20)!=G) die("weight binding");
    size_t position=24;
    for(int li=0;li<L;li++) {
        bind_q6(&q6model[li].gate,weights,weight_size,&position,H,D);
        bind_q6(&q6model[li].up,weights,weight_size,&position,H,D);
        bind_q6(&q6model[li].down,weights,weight_size,&position,D,H);
    }
    if(position!=weight_size || vector_size!=24+(size_t)TOKENS*L*D*2 ||
       memcmp(vectors,"M125HX01",8) || u32le(vectors+8)!=L || u32le(vectors+12)!=TOKENS ||
       u32le(vectors+16)!=D || u32le(vectors+20)!=1280) die("vector binding");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *out=fopen(argv[3],"wb"); if(!out) die("output open");
    if(fwrite("M221OUT1",1,8,out)!=8) die("header write");
    uint32_t dims[3]={CHECK_TOKENS,L,D};
    if(fwrite(dims,1,sizeof(dims),out)!=sizeof(dims)) die("dims write");
    float output[D]; double checksum=0;
    for(int t=0;t<CHECK_TOKENS;t++) for(int li=0;li<L;li++) {
        q6_forward(li,states+((size_t)t*L+li)*D,output);
        if(fwrite(output,1,sizeof(output),out)!=sizeof(output)) die("output write");
    }
    if(fclose(out)) die("output close");
    double elapsed[3];
    for(int rep=0;rep<3;rep++) {
        double start=seconds();
        for(int t=0;t<TOKENS;t++) for(int li=0;li<L;li++) {
            q6_forward(li,states+((size_t)t*L+li)*D,output); checksum+=output[(t+li)%D];
        }
        elapsed[rep]=seconds()-start;
    }
    PROCESS_MEMORY_COUNTERS memory;
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))) die("RSS read");
    printf("{\"threads\":6,\"tokens\":256,\"layers\":24,\"codec_patterns\":64,\"weight_bytes\":%llu,\"pass_ms_per_token\":[%.6f,%.6f,%.6f],\"checksum\":%.9f,\"peak_working_set_bytes\":%llu}\n",
        (unsigned long long)weight_size,elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,checksum,
        (unsigned long long)memory.PeakWorkingSetSize);
    free(weights); free(vectors); return 0;
}
