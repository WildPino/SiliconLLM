// METH-220: unchanged METH-182 tape and float-input grouped-Q8 alternative.
#define main meth182_unused_main
#include "meth182_group64_ffn_cpu.c"
#undef main

static void write_bytes(FILE *out, const void *p, size_t n) {
    if(fwrite(p,1,n,out)!=n) die("short intermediate write");
}
static void float_matvec(const Matrix *m, const float *input, float *output) {
    int groups=m->cols/G;
    #pragma omp parallel for schedule(static)
    for(int row=0;row<m->rows;row++) {
        const int8_t *w=m->q+(size_t)row*m->cols;
        const uint16_t *scales=m->scale+(size_t)row*groups;
        float total=0;
        for(int g=0;g<groups;g++) {
            __m256 sums=_mm256_setzero_ps();
            for(int j=0;j<G;j+=8) {
                __m128i raw=_mm_loadl_epi64((const __m128i*)(w+g*G+j));
                __m256 weights=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw));
                sums=_mm256_fmadd_ps(weights,_mm256_loadu_ps(input+g*G+j),sums);
            }
            float lanes[8]; _mm256_storeu_ps(lanes,sums);
            float group=0; for(int j=0;j<8;j++) group+=lanes[j];
            total+=group*f16_to_float(scales[g]);
        }
        output[row]=total;
    }
}
static void float_forward(int li, const float *x, float *out) {
    float gate[H],up[H],hidden[H];
    float_matvec(&model[li].gate,x,gate);
    float_matvec(&model[li].up,x,up);
    for(int j=0;j<H;j++) hidden[j]=(gate[j]/(1.0f+expf(-gate[j])))*up[j];
    float_matvec(&model[li].down,hidden,out);
}
int main(int argc, char **argv) {
    if(argc!=4) die("usage: meth220 weights vectors tape");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("ISA unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6);
    size_t weight_size,vector_size;
    uint8_t *weights=read_file(argv[1],&weight_size);
    uint8_t *vectors=read_file(argv[2],&vector_size);
    if(weight_size<24 || memcmp(weights,"M182FFN1",8) ||
       u32le(weights+8)!=L || u32le(weights+12)!=D ||
       u32le(weights+16)!=H || u32le(weights+20)!=G) die("weight binding");
    size_t position=24;
    for(int li=0;li<L;li++) {
        bind_matrix(&model[li].gate,weights,weight_size,&position,H,D);
        bind_matrix(&model[li].up,weights,weight_size,&position,H,D);
        bind_matrix(&model[li].down,weights,weight_size,&position,D,H);
    }
    if(position!=weight_size || vector_size!=24+(size_t)TOKENS*L*D*2 ||
       memcmp(vectors,"M125HX01",8) || u32le(vectors+8)!=L ||
       u32le(vectors+12)!=TOKENS || u32le(vectors+16)!=D ||
       u32le(vectors+20)!=1280) die("vector binding");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *tape=fopen(argv[3],"wb"); if(!tape) die("tape open");
    write_bytes(tape,"M220INT1",8);
    uint32_t dims[5]={CHECK_TOKENS,L,D,H,G}; write_bytes(tape,dims,sizeof(dims));
    float x[D],gate[H],up[H],hidden[H],si[D/G],sh[H/G],out[D],float_out[D];
    int8_t qx[D],qh[H];
    for(int t=0;t<CHECK_TOKENS;t++) for(int li=0;li<L;li++) {
        for(int d=0;d<D;d++) x[d]=bf16_to_float(states[((size_t)t*L+li)*D+d]);
        quantize_groups(x,D,qx,si);
        matvec(&model[li].gate,qx,si,gate); matvec(&model[li].up,qx,si,up);
        for(int j=0;j<H;j++) hidden[j]=(gate[j]/(1.0f+expf(-gate[j])))*up[j];
        quantize_groups(hidden,H,qh,sh);
        matvec(&model[li].down,qh,sh,out);
        float_forward(li,x,float_out);
        write_bytes(tape,qx,sizeof(qx)); write_bytes(tape,si,sizeof(si));
        write_bytes(tape,gate,sizeof(gate)); write_bytes(tape,up,sizeof(up));
        write_bytes(tape,hidden,sizeof(hidden)); write_bytes(tape,qh,sizeof(qh));
        write_bytes(tape,sh,sizeof(sh)); write_bytes(tape,out,sizeof(out));
        write_bytes(tape,float_out,sizeof(float_out));
    }
    if(fclose(tape)) die("tape close");
    double elapsed[3],checksum=0;
    for(int rep=0;rep<3;rep++) {
        double start=seconds();
        for(int t=0;t<TOKENS;t++) for(int li=0;li<L;li++) {
            for(int d=0;d<D;d++) x[d]=bf16_to_float(states[((size_t)t*L+li)*D+d]);
            float_forward(li,x,out); checksum+=out[(t+li)%D];
        }
        elapsed[rep]=seconds()-start;
    }
    printf("{\"threads\":6,\"tokens\":256,\"layers\":24,\"float_input_pass_ms_per_token\":[%.6f,%.6f,%.6f],\"checksum\":%.9f}\n",
        elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,checksum);
    free(weights); free(vectors); return 0;
}
