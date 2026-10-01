// METH-229: full affine plus H2048 original-source nonlinear branch.
#define main meth182_unused_main
#include "meth182_group64_ffn_cpu.c"
#undef main
#define NH 2048
typedef struct { const uint16_t *w; int rows,cols; } BFMatrix;
typedef struct { BFMatrix affine,gate,up,down; const float *bias; } CurvatureLayer;
static CurvatureLayer curvature[L];
static void bind_bf(BFMatrix *m,const uint8_t *file,size_t length,size_t *pos,int rows,int cols) {
    size_t bytes=(size_t)rows*cols*2;
    if(*pos>length || bytes>length-*pos) die("short BF16 matrix");
    m->w=(const uint16_t*)(file+*pos); *pos+=bytes; m->rows=rows; m->cols=cols;
}
static __m256 bf8(const uint16_t *p) {
    __m128i raw=_mm_loadu_si128((const __m128i*)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw),16));
}
static void bf_matvec(const BFMatrix *m,const float *x,float *out) {
    #pragma omp parallel for schedule(static)
    for(int row=0;row<m->rows;row++) {
        const uint16_t *w=m->w+(size_t)row*m->cols;
        __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
        for(int j=0;j<m->cols;j+=32) {
            a=_mm256_fmadd_ps(bf8(w+j),_mm256_loadu_ps(x+j),a);
            b=_mm256_fmadd_ps(bf8(w+j+8),_mm256_loadu_ps(x+j+8),b);
            c=_mm256_fmadd_ps(bf8(w+j+16),_mm256_loadu_ps(x+j+16),c);
            d=_mm256_fmadd_ps(bf8(w+j+24),_mm256_loadu_ps(x+j+24),d);
        }
        __m256 total=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
        float lanes[8]; _mm256_storeu_ps(lanes,total);
        float sum=0; for(int j=0;j<8;j++) sum+=lanes[j]; out[row]=sum;
    }
}
static void curvature_forward(int li,const uint16_t *bits,float *out) {
    float x[D],gate[NH],up[NH],hidden[NH],linear[D],nonlinear[D];
    for(int j=0;j<D;j++) x[j]=bf16_to_float(bits[j]);
    bf_matvec(&curvature[li].affine,x,linear);
    bf_matvec(&curvature[li].gate,x,gate); bf_matvec(&curvature[li].up,x,up);
    for(int j=0;j<NH;j++) hidden[j]=(gate[j]/(1.0f+expf(-gate[j])))*up[j];
    bf_matvec(&curvature[li].down,hidden,nonlinear);
    for(int j=0;j<D;j++) {
        out[j]=(linear[j]+nonlinear[j])+curvature[li].bias[j];
        if(!isfinite(out[j])) die("nonfinite combined output");
    }
}
int main(int argc,char **argv) {
    if(argc!=4) die("usage: meth229 weights vectors check");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("ISA unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6);
    size_t wn,vn; uint8_t *weights=read_file(argv[1],&wn),*vectors=read_file(argv[2],&vn);
    if(wn<24 || memcmp(weights,"M229BF01",8) || u32le(weights+8)!=L ||
       u32le(weights+12)!=D || u32le(weights+16)!=NH || u32le(weights+20)!=32) die("weight binding");
    size_t pos=24;
    for(int li=0;li<L;li++) {
        bind_bf(&curvature[li].affine,weights,wn,&pos,D,D);
        bind_bf(&curvature[li].gate,weights,wn,&pos,NH,D);
        bind_bf(&curvature[li].up,weights,wn,&pos,NH,D);
        bind_bf(&curvature[li].down,weights,wn,&pos,D,NH);
        if(pos>wn || D*4>wn-pos) die("short bias");
        curvature[li].bias=(const float*)(weights+pos); pos+=D*4;
    }
    if(pos!=wn || vn!=24+(size_t)TOKENS*L*D*2 || memcmp(vectors,"M125HX01",8) ||
       u32le(vectors+8)!=L || u32le(vectors+12)!=TOKENS || u32le(vectors+16)!=D ||
       u32le(vectors+20)!=1280) die("vector binding");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *out=fopen(argv[3],"wb"); if(!out) die("check open");
    if(fwrite("M229OUT1",1,8,out)!=8) die("header write");
    uint32_t dims[3]={CHECK_TOKENS,L,D};
    if(fwrite(dims,1,sizeof(dims),out)!=sizeof(dims)) die("dims write");
    float y[D]; double checksum=0;
    for(int t=0;t<CHECK_TOKENS;t++) for(int li=0;li<L;li++) {
        curvature_forward(li,states+((size_t)t*L+li)*D,y);
        if(fwrite(y,1,sizeof(y),out)!=sizeof(y)) die("check write");
    }
    if(fclose(out)) die("check close");
    double elapsed[3];
    for(int rep=0;rep<3;rep++) {
        double start=seconds();
        for(int t=0;t<TOKENS;t++) for(int li=0;li<L;li++) {
            curvature_forward(li,states+((size_t)t*L+li)*D,y); checksum+=y[(t+li)%D];
        }
        elapsed[rep]=seconds()-start;
    }
    PROCESS_MEMORY_COUNTERS memory;
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))) die("RSS read");
    printf("{\"threads\":6,\"tokens\":256,\"layers\":24,\"hidden\":2048,\"weight_bytes\":%llu,\"pass_ms_per_token\":[%.6f,%.6f,%.6f],\"checksum\":%.9f,\"peak_working_set_bytes\":%llu}\n",
        (unsigned long long)wn,elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,checksum,
        (unsigned long long)memory.PeakWorkingSetSize);
    free(weights); free(vectors); return 0;
}
