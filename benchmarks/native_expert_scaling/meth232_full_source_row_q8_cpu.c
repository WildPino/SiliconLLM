// METH-232: all original source features, row-scaled int8/FP32 arithmetic.
#define main meth182_unused_main
#include "meth182_group64_ffn_cpu.c"
#undef main
typedef struct { const int8_t *q; const float *scale; int rows,cols; } RowMatrix;
typedef struct { RowMatrix gate,up,down; const float *bias; } RowLayer;
static RowLayer rowsource[L];
static void bind_row(RowMatrix *m,const uint8_t *file,size_t length,size_t *pos,int rows,int cols) {
    size_t qb=(size_t)rows*cols,sb=(size_t)rows*4;
    if(*pos>length || qb>length-*pos) die("short row codes");
    m->q=(const int8_t*)(file+*pos); *pos+=qb;
    if(sb>length-*pos) die("short row scales");
    m->scale=(const float*)(file+*pos); *pos+=sb; m->rows=rows; m->cols=cols;
    for(size_t i=0;i<qb;i++) if(m->q[i]==-128) die("invalid row code");
    for(int i=0;i<rows;i++) if(!isfinite(m->scale[i]) || m->scale[i]<=0) die("invalid row scale");
}
static __m256 q8float(const int8_t *p) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)p)));
}
static void row_matvec(const RowMatrix *m,const float *x,float *out) {
    #pragma omp parallel for schedule(static)
    for(int row=0;row<m->rows;row++) {
        const int8_t *q=m->q+(size_t)row*m->cols;
        __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
        for(int j=0;j<m->cols;j+=32) {
            a=_mm256_fmadd_ps(q8float(q+j),_mm256_loadu_ps(x+j),a);
            b=_mm256_fmadd_ps(q8float(q+j+8),_mm256_loadu_ps(x+j+8),b);
            c=_mm256_fmadd_ps(q8float(q+j+16),_mm256_loadu_ps(x+j+16),c);
            d=_mm256_fmadd_ps(q8float(q+j+24),_mm256_loadu_ps(x+j+24),d);
        }
        __m256 total=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
        float lanes[8]; _mm256_storeu_ps(lanes,total);
        float sum=0; for(int j=0;j<8;j++) sum+=lanes[j];
        out[row]=sum*m->scale[row];
    }
}
static void row_forward(int li,const uint16_t *bits,float *out) {
    float x[D],gate[H],up[H],hidden[H];
    for(int j=0;j<D;j++) x[j]=bf16_to_float(bits[j]);
    row_matvec(&rowsource[li].gate,x,gate); row_matvec(&rowsource[li].up,x,up);
    for(int j=0;j<H;j++) hidden[j]=(gate[j]/(1.0f+expf(-gate[j])))*up[j];
    row_matvec(&rowsource[li].down,hidden,out);
    for(int j=0;j<D;j++) {
        out[j]+=rowsource[li].bias[j];
        if(!isfinite(out[j])) die("nonfinite row output");
    }
}
int main(int argc,char **argv) {
    if(argc!=4) die("usage: meth232 weights vectors check");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("ISA unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6);
    size_t wn,vn; uint8_t *weights=read_file(argv[1],&wn),*vectors=read_file(argv[2],&vn);
    if(wn<24 || memcmp(weights,"M232RQ01",8) || u32le(weights+8)!=L ||
       u32le(weights+12)!=D || u32le(weights+16)!=H || u32le(weights+20)!=32) die("weight binding");
    size_t pos=24;
    for(int li=0;li<L;li++) {
        bind_row(&rowsource[li].gate,weights,wn,&pos,H,D);
        bind_row(&rowsource[li].up,weights,wn,&pos,H,D);
        bind_row(&rowsource[li].down,weights,wn,&pos,D,H);
        if(pos>wn || D*4>wn-pos) die("short bias");
        rowsource[li].bias=(const float*)(weights+pos); pos+=D*4;
    }
    if(pos!=wn || vn!=24+(size_t)TOKENS*L*D*2 || memcmp(vectors,"M125HX01",8) ||
       u32le(vectors+8)!=L || u32le(vectors+12)!=TOKENS || u32le(vectors+16)!=D ||
       u32le(vectors+20)!=1280) die("vector binding");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *out=fopen(argv[3],"wb"); if(!out) die("check open");
    if(fwrite("M232OUT1",1,8,out)!=8) die("header write");
    uint32_t dims[3]={CHECK_TOKENS,L,D};
    if(fwrite(dims,1,sizeof(dims),out)!=sizeof(dims)) die("dims write");
    float y[D]; double checksum=0;
    for(int t=0;t<CHECK_TOKENS;t++) for(int li=0;li<L;li++) {
        row_forward(li,states+((size_t)t*L+li)*D,y);
        if(fwrite(y,1,sizeof(y),out)!=sizeof(y)) die("check write");
    }
    if(fclose(out)) die("check close");
    double elapsed[3];
    for(int rep=0;rep<3;rep++) {
        double start=seconds();
        for(int t=0;t<TOKENS;t++) for(int li=0;li<L;li++) {
            row_forward(li,states+((size_t)t*L+li)*D,y); checksum+=y[(t+li)%D];
        }
        elapsed[rep]=seconds()-start;
    }
    PROCESS_MEMORY_COUNTERS memory;
    if(!GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory))) die("RSS read");
    printf("{\"threads\":6,\"tokens\":256,\"layers\":24,\"hidden\":4864,\"weight_bytes\":%llu,\"pass_ms_per_token\":[%.6f,%.6f,%.6f],\"checksum\":%.9f,\"peak_working_set_bytes\":%llu}\n",
        (unsigned long long)wn,elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,checksum,
        (unsigned long long)memory.PeakWorkingSetSize);
    free(weights); free(vectors); return 0;
}
