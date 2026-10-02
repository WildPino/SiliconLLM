// METH-272: same private128 function, fused shared gate/up input dots.
#define RESIDUAL_RANK 32
#define PRIVATE_UNITS 128
#define main meth182_unused_main
#include "meth182_group64_ffn_cpu.c"
#undef main
typedef struct { const int8_t *q; const float *scale; int rows,cols; } RowMatrix;
typedef struct { RowMatrix gate,up,down; const uint16_t *ids,*escape; const float *bias,*residual_bias; const uint16_t *right,*left,*private_ids,*private_gate,*private_up; } RowLayer;
static RowLayer rowsource[L];
static const float *silu_table;
static float silu_lookup(float gate) {
    if(gate<=-16.0f) return 0.0f;
    if(gate>=16.0f) return gate;
    float position=(gate+16.0f)*16.0f;
    if(position>=512.0f) return gate;
    int index=(int)position;
    float fraction=position-(float)index;
    return silu_table[index]+fraction*(silu_table[index+1]-silu_table[index]);
}
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
static __m256 escape8(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)),16));
}
static __m256 gather8(const float *x,const uint16_t *p) {
    __m256i index=_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p));
    return _mm256_i32gather_ps(x,index,4);
}
static float bf16_dot(const uint16_t *weight,const float *x,int cols) {
    __m256 a=_mm256_setzero_ps(),b=a,c=a,d=a;
    for(int j=0;j<cols;j+=32) {
        a=_mm256_fmadd_ps(escape8(weight+j),_mm256_loadu_ps(x+j),a);
        b=_mm256_fmadd_ps(escape8(weight+j+8),_mm256_loadu_ps(x+j+8),b);
        c=_mm256_fmadd_ps(escape8(weight+j+16),_mm256_loadu_ps(x+j+16),c);
        d=_mm256_fmadd_ps(escape8(weight+j+24),_mm256_loadu_ps(x+j+24),d);
    }
    __m256 total=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
    float lanes[8]; _mm256_storeu_ps(lanes,total);
    float sum=0; for(int j=0;j<8;j++) sum+=lanes[j];
    return sum;
}
static float row_readout(const RowMatrix *m,const float *x,int row,const uint16_t *ids,const uint16_t *escape) {
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
        float base=sum*m->scale[row];
        if(ids) {
            const uint16_t *index=ids+(size_t)row*32,*weight=escape+(size_t)row*32;
            __m256 a=_mm256_mul_ps(escape8(weight),gather8(x,index));
            __m256 b=_mm256_mul_ps(escape8(weight+8),gather8(x,index+8));
            __m256 c=_mm256_mul_ps(escape8(weight+16),gather8(x,index+16));
            __m256 d=_mm256_mul_ps(escape8(weight+24),gather8(x,index+24));
            __m256 total=_mm256_add_ps(_mm256_add_ps(a,b),_mm256_add_ps(c,d));
            _mm256_storeu_ps(lanes,total);
            float extra=0; for(int j=0;j<8;j++) extra+=lanes[j];
            return base+extra;
        } else return base;
}
static void row_input_pair(const RowMatrix *gate,const RowMatrix *up,const float *x,
                           int row,float *gate_value,float *up_value) {
    const int8_t *g=gate->q+(size_t)row*gate->cols;
    const int8_t *u=up->q+(size_t)row*up->cols;
    __m256 ga=_mm256_setzero_ps(),gb=ga,gc=ga,gd=ga;
    __m256 ua=ga,ub=ga,uc=ga,ud=ga;
    for(int j=0;j<gate->cols;j+=32) {
        __m256 x0=_mm256_loadu_ps(x+j),x1=_mm256_loadu_ps(x+j+8);
        __m256 x2=_mm256_loadu_ps(x+j+16),x3=_mm256_loadu_ps(x+j+24);
        ga=_mm256_fmadd_ps(q8float(g+j),x0,ga);
        ua=_mm256_fmadd_ps(q8float(u+j),x0,ua);
        gb=_mm256_fmadd_ps(q8float(g+j+8),x1,gb);
        ub=_mm256_fmadd_ps(q8float(u+j+8),x1,ub);
        gc=_mm256_fmadd_ps(q8float(g+j+16),x2,gc);
        uc=_mm256_fmadd_ps(q8float(u+j+16),x2,uc);
        gd=_mm256_fmadd_ps(q8float(g+j+24),x3,gd);
        ud=_mm256_fmadd_ps(q8float(u+j+24),x3,ud);
    }
    __m256 gt=_mm256_add_ps(_mm256_add_ps(ga,gb),_mm256_add_ps(gc,gd));
    __m256 ut=_mm256_add_ps(_mm256_add_ps(ua,ub),_mm256_add_ps(uc,ud));
    float glanes[8],ulanes[8];_mm256_storeu_ps(glanes,gt);_mm256_storeu_ps(ulanes,ut);
    float gsum=0,usum=0;
    for(int j=0;j<8;j++) {gsum+=glanes[j];usum+=ulanes[j];}
    *gate_value=gsum*gate->scale[row];*up_value=usum*up->scale[row];
}
static void row_forward(int li,const uint16_t *bits,float *out) {
    float x[D],hidden[H],latent[RESIDUAL_RANK];
    const RowLayer *layer=&rowsource[li];
    for(int j=0;j<D;j++) x[j]=bf16_to_float(bits[j]);
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int row=0;row<H;row++) {
            float gate,up;
            row_input_pair(&layer->gate,&layer->up,x,row,&gate,&up);
            hidden[row]=silu_lookup(gate)*up;
        }
        #pragma omp for schedule(static)
        for(int slot=0;slot<PRIVATE_UNITS;slot++) {
            int row=layer->private_ids[slot];
            float gate=bf16_dot(layer->private_gate+(size_t)slot*D,x,D);
            float up=bf16_dot(layer->private_up+(size_t)slot*D,x,D);
            hidden[row]=silu_lookup(gate)*up;
        }
        #pragma omp for schedule(static) nowait
        for(int row=0;row<D;row++) out[row]=row_readout(&layer->down,hidden,row,layer->ids,layer->escape);
        #pragma omp for schedule(static)
        for(int row=0;row<RESIDUAL_RANK;row++) latent[row]=bf16_dot(layer->right+(size_t)row*H,hidden,H);
        #pragma omp for schedule(static)
        for(int row=0;row<D;row++) {
            out[row]+=bf16_dot(layer->left+(size_t)row*RESIDUAL_RANK,latent,RESIDUAL_RANK);
            out[row]+=layer->bias[row];
            out[row]+=layer->residual_bias[row];
            if(!isfinite(out[row])) die("nonfinite row output");
        }
    }
}
int main(int argc,char **argv) {
    if(argc!=4) die("usage: meth272 weights vectors check");
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) die("ISA unsupported");
    omp_set_dynamic(0); omp_set_num_threads(6);
    size_t wn,vn; uint8_t *weights=read_file(argv[1],&wn),*vectors=read_file(argv[2],&vn);
    if(wn<32 || memcmp(weights,"M252PF01",8) || u32le(weights+8)!=L ||
       u32le(weights+12)!=D || u32le(weights+16)!=H || u32le(weights+20)!=32 ||
       u32le(weights+24)!=RESIDUAL_RANK || u32le(weights+28)!=PRIVATE_UNITS) die("weight binding");
    size_t pos=32;
    if(513*4>wn-pos) die("short SiLU table");
    silu_table=(const float*)(weights+pos); pos+=513*4;
    for(int i=0;i<513;i++) if(!isfinite(silu_table[i])) die("nonfinite table");
    for(int li=0;li<L;li++) {
        bind_row(&rowsource[li].gate,weights,wn,&pos,H,D);
        bind_row(&rowsource[li].up,weights,wn,&pos,H,D);
        bind_row(&rowsource[li].down,weights,wn,&pos,D,H);
        if(pos>wn || (size_t)D*32*4>wn-pos) die("short escape arrays");
        rowsource[li].ids=(const uint16_t*)(weights+pos); pos+=(size_t)D*32*2;
        rowsource[li].escape=(const uint16_t*)(weights+pos); pos+=(size_t)D*32*2;
        for(int row=0;row<D;row++) for(int j=0;j<32;j++) {
            uint16_t id=rowsource[li].ids[row*32+j];
            if(id>=H || rowsource[li].down.q[(size_t)row*H+id]!=0) die("escape ID/code binding");
            if(!isfinite(bf16_to_float(rowsource[li].escape[row*32+j]))) die("nonfinite escape");
            for(int k=0;k<j;k++) if(rowsource[li].ids[row*32+k]==id) die("duplicate escape ID");
        }
        if(pos>wn || D*4>wn-pos) die("short bias");
        rowsource[li].bias=(const float*)(weights+pos); pos+=D*4;
        size_t rb=(size_t)RESIDUAL_RANK*H*2,lb=(size_t)D*RESIDUAL_RANK*2;
        if(pos>wn || rb+lb+D*4>wn-pos) die("short residual");
        rowsource[li].right=(const uint16_t*)(weights+pos); pos+=rb;
        rowsource[li].left=(const uint16_t*)(weights+pos); pos+=lb;
        rowsource[li].residual_bias=(const float*)(weights+pos); pos+=D*4;
        for(size_t i=0;i<(size_t)RESIDUAL_RANK*H;i++) if(!isfinite(bf16_to_float(rowsource[li].right[i]))) die("nonfinite right");
        for(size_t i=0;i<(size_t)D*RESIDUAL_RANK;i++) if(!isfinite(bf16_to_float(rowsource[li].left[i]))) die("nonfinite left");
        for(int i=0;i<D;i++) if(!isfinite(rowsource[li].bias[i]) || !isfinite(rowsource[li].residual_bias[i])) die("nonfinite bias");
        size_t pb=(size_t)PRIVATE_UNITS*D*2;
        if(pos>wn || PRIVATE_UNITS*2+2*pb>wn-pos) die("short private feature rows");
        rowsource[li].private_ids=(const uint16_t*)(weights+pos); pos+=PRIVATE_UNITS*2;
        rowsource[li].private_gate=(const uint16_t*)(weights+pos); pos+=pb;
        rowsource[li].private_up=(const uint16_t*)(weights+pos); pos+=pb;
        for(int slot=0;slot<PRIVATE_UNITS;slot++) {
            int row=rowsource[li].private_ids[slot];
            if(row>=H) die("private unit ID binding");
            if(slot>0 && row<=rowsource[li].private_ids[slot-1]) die("private unit ID order");
            for(int j=0;j<D;j++) {
                if(!isfinite(bf16_to_float(rowsource[li].private_gate[(size_t)slot*D+j])) ||
                   !isfinite(bf16_to_float(rowsource[li].private_up[(size_t)slot*D+j]))) die("nonfinite private feature");
            }
        }
    }
    if(pos!=wn || vn!=24+(size_t)TOKENS*L*D*2 || memcmp(vectors,"M125HX01",8) ||
       u32le(vectors+8)!=L || u32le(vectors+12)!=TOKENS || u32le(vectors+16)!=D ||
       u32le(vectors+20)!=1280) die("vector binding");
    const uint16_t *states=(const uint16_t*)(vectors+24);
    FILE *out=fopen(argv[3],"wb"); if(!out) die("check open");
    if(fwrite("M272OUT1",1,8,out)!=8) die("header write");
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
    printf("{\"threads\":6,\"tokens\":256,\"layers\":24,\"hidden\":4864,\"residual_rank\":32,\"private_units\":128,\"weight_bytes\":%llu,\"pass_ms_per_token\":[%.6f,%.6f,%.6f],\"checksum\":%.9f,\"peak_working_set_bytes\":%llu}\n",
        (unsigned long long)wn,elapsed[0]*1000/TOKENS,elapsed[1]*1000/TOKENS,elapsed[2]*1000/TOKENS,checksum,
        (unsigned long long)memory.PeakWorkingSetSize);
    free(weights); free(vectors); return 0;
}
