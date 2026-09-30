// METH-182: actual stored group-64 R8 Qwen FFN on varied BF16 states.
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <psapi.h>

#define L 24
#define D 896
#define H 4864
#define G 64
#define TOKENS 256
#define CHECK_TOKENS 16

typedef struct { const int8_t *q; const uint16_t *scale; int rows, cols; } Matrix;
typedef struct { Matrix gate, up, down; } Layer;
static Layer model[L];

static void die(const char *message) { fprintf(stderr, "%s\n", message); exit(2); }
static uint32_t u32le(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint8_t *read_file(const char *path, size_t *length) {
    FILE *f=fopen(path,"rb"); if(!f) die("open failed");
    if(fseek(f,0,SEEK_END)) die("seek failed");
    long n=ftell(f); if(n<=0 || fseek(f,0,SEEK_SET)) die("file size failed");
    uint8_t *p=(uint8_t*)malloc((size_t)n); if(!p) die("allocation failed");
    if(fread(p,1,(size_t)n,f)!=(size_t)n) die("short file read");
    if(fclose(f)) die("close failed"); *length=(size_t)n; return p;
}
static void bind_matrix(Matrix *m, const uint8_t *file, size_t length,
                        size_t *position, int rows, int cols) {
    size_t qb=(size_t)rows*cols, sb=(size_t)rows*(cols/G)*2;
    if(*position > length || qb > length-*position) die("short q matrix");
    m->q=(const int8_t*)(file+*position); *position+=qb;
    if(sb > length-*position) die("short scale matrix");
    m->scale=(const uint16_t*)(file+*position); *position+=sb;
    m->rows=rows; m->cols=cols;
}
static float bf16_to_float(uint16_t word) {
    uint32_t bits=(uint32_t)word<<16; float value; memcpy(&value,&bits,4); return value;
}
static float f16_to_float(uint16_t bits) {
    uint32_t sign=((uint32_t)bits&0x8000u)<<16;
    int exp=(bits>>10)&31;
    uint32_t mantissa=bits&1023u;
    uint32_t out;
    if(exp==0) {
        if(mantissa==0) out=sign;
        else {
            exp=-14;
            while((mantissa&1024u)==0) { mantissa<<=1; exp--; }
            mantissa&=1023u;
            out=sign | (uint32_t)(exp+127)<<23 | mantissa<<13;
        }
    } else if(exp==31) out=sign | 0x7f800000u | mantissa<<13;
    else out=sign | (uint32_t)(exp-15+127)<<23 | mantissa<<13;
    float value; memcpy(&value,&out,4); return value;
}
static int32_t dot64(const int8_t *weights, const int8_t *input) {
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i sum=_mm256_setzero_si256();
    for(int block=0;block<2;block++) {
        __m256i x=_mm256_loadu_si256((const __m256i*)(input+block*32));
        __m256i w=_mm256_loadu_si256((const __m256i*)(weights+block*32));
        __m256i positive_x=_mm256_abs_epi8(x);
        __m256i signed_w=_mm256_sign_epi8(w,x);
        __m256i pairs=_mm256_maddubs_epi16(positive_x,signed_w);
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(pairs,ones));
    }
    int32_t lanes[8]; _mm256_storeu_si256((__m256i*)lanes,sum);
    int32_t total=0; for(int i=0;i<8;i++) total+=lanes[i]; return total;
}
static void quantize_groups(const float *input, int n, int8_t *codes, float *scale) {
    for(int g=0;g<n/G;g++) {
        float maximum=0;
        for(int j=0;j<G;j++) {
            float v=input[g*G+j];
            if(!isfinite(v)) die("nonfinite input");
            float a=fabsf(v); if(a>maximum) maximum=a;
        }
        float s=maximum>0 ? maximum/127.0f : 1.0f;
        scale[g]=s;
        for(int j=0;j<G;j++) {
            long q=lrintf(input[g*G+j]/s);
            if(q>127) q=127; if(q< -127) q=-127;
            codes[g*G+j]=(int8_t)q;
        }
    }
}
static void matvec(const Matrix *m, const int8_t *input,
                   const float *input_scale, float *output) {
    int groups=m->cols/G;
    #pragma omp parallel for schedule(static)
    for(int row=0;row<m->rows;row++) {
        const int8_t *w=m->q+(size_t)row*m->cols;
        const uint16_t *s=m->scale+(size_t)row*groups;
        float sum=0;
        for(int g=0;g<groups;g++) {
            float ws=f16_to_float(s[g]);
            sum += (float)dot64(w+(size_t)g*G,input+(size_t)g*G)*ws*input_scale[g];
        }
        output[row]=sum;
    }
}
static void forward(int layer, const uint16_t *bf16_input, float *output) {
    float input[D], hidden[H], gate[H], up[H];
    int8_t qinput[D], qhidden[H];
    float si[D/G], sh[H/G];
    for(int i=0;i<D;i++) input[i]=bf16_to_float(bf16_input[i]);
    quantize_groups(input,D,qinput,si);
    matvec(&model[layer].gate,qinput,si,gate);
    matvec(&model[layer].up,qinput,si,up);
    for(int i=0;i<H;i++) hidden[i]=(gate[i]/(1.0f+expf(-gate[i])))*up[i];
    quantize_groups(hidden,H,qhidden,sh);
    matvec(&model[layer].down,qhidden,sh,output);
    for(int i=0;i<D;i++) if(!isfinite(output[i])) die("nonfinite output");
}
static double seconds(void) {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency); QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart/(double)frequency.QuadPart;
}
int main(int argc, char **argv) {
    if(argc!=4) die("usage: meth182_group64_ffn_cpu weights.bin vectors.bin check.bin");
    if(!__builtin_cpu_supports("avx2")) die("AVX2 unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6);
    size_t weight_size, vector_size;
    uint8_t *weights=read_file(argv[1],&weight_size);
    uint8_t *vectors=read_file(argv[2],&vector_size);
    if(weight_size<24 || memcmp(weights,"M182FFN1",8) ||
       u32le(weights+8)!=L || u32le(weights+12)!=D ||
       u32le(weights+16)!=H || u32le(weights+20)!=G) die("weight header mismatch");
    size_t position=24;
    for(int l=0;l<L;l++) {
        bind_matrix(&model[l].gate,weights,weight_size,&position,H,D);
        bind_matrix(&model[l].up,weights,weight_size,&position,H,D);
        bind_matrix(&model[l].down,weights,weight_size,&position,D,H);
    }
    if(position!=weight_size) die("weight trailer mismatch");
    if(vector_size != 24+(size_t)TOKENS*L*D*2 ||
       memcmp(vectors,"M125HX01",8) || u32le(vectors+8)!=L ||
       u32le(vectors+12)!=TOKENS || u32le(vectors+16)!=D ||
       u32le(vectors+20)!=1280) die("vector binding mismatch");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *check=fopen(argv[3],"wb"); if(!check) die("check output open failed");
    const uint8_t header[20]={'M','1','8','2','O','U','T','1',16,0,0,0,24,0,0,0,128,3,0,0};
    if(fwrite(header,1,20,check)!=20) die("check header write failed");
    float output[D]; double checksum=0;
    for(int t=0;t<CHECK_TOKENS;t++) for(int l=0;l<L;l++) {
        forward(l,states+((size_t)t*L+l)*D,output);
        if(fwrite(output,sizeof(float),D,check)!=D) die("check output write failed");
        for(int i=0;i<D;i++) checksum+=output[i]*(1.0+(i%13)*0.01);
    }
    if(fclose(check)) die("check output close failed");
    double elapsed[3];
    for(int repetition=0;repetition<3;repetition++) {
        double start=seconds();
        for(int t=0;t<TOKENS;t++) for(int l=0;l<L;l++) {
            forward(l,states+((size_t)t*L+l)*D,output);
            checksum+=output[(t+l)%D];
        }
        elapsed[repetition]=seconds()-start;
    }
    PROCESS_MEMORY_COUNTERS info;
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&info,sizeof(info))) die("RSS query failed");
    printf("{\"threads\":6,\"tokens\":%d,\"layers\":%d,\"pass_seconds\":[%.9f,%.9f,%.9f],\"pass_ms_per_token\":[%.6f,%.6f,%.6f],\"peak_working_set_bytes\":%llu,\"checksum\":%.9f}\n",
           TOKENS,L,elapsed[0],elapsed[1],elapsed[2],
           elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,
           (unsigned long long)info.PeakWorkingSetSize,checksum);
    free(weights);free(vectors);return 0;
}
