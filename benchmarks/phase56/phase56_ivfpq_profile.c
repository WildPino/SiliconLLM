// Phase 56 STAGE 1 (re-spec v2) - faithful CPU cost of the IVFPQ recall probe, NOW with PQ4 fast-scan.
//   v1 (8-bit, scalar ADC) = 38.98us@128K = near-miss FAIL (const 18.5 + ADC-scan 18.75 both scalar-inflated).
//   v2 fixes BOTH dominant terms with standard FAISS SIMD (apples-to-apples 128 bit/vec budget):
//     * 4-bit PQ fast-scan (M=32 sub-quantizers x 4 bit = 128 bit = 16 B/vec, SAME size as 8-bit M=16):
//       ADC LUT per query = M x 16 uint8 (256 B, L1). Codes in FAISS blocked layout (32 vec/block, 2 sub-q
//       packed/byte). Inner loop = _mm256_shuffle_epi8 does 32 table lookups/op -> ~1-2 ops/candidate.
//     * Vectorized + 16x-smaller ADC-table build: 4-bit table = M x 16 sub-dists (vs 8-bit M x 256), uint8.
//   Only top-k=16 winners gather RAW V (16 random DRAM reads, constant). Same store cliff measured @128K.
//   Both paths measured in ONE binary (A/B). MODEL 128K with explicit const + n_cand*ADC + k*DRAM (NOT slope).
//   GATE 1 unchanged: IVFPQ(fast)@128K <= ~30us AND large margin vs naive. Pass -> graduate to 3-arm quality.
//
// Build: gcc -O3 -march=native -mavx2 -mfma benchmarks/phase56/phase56_ivfpq_profile.c -o bin/phase56_ivfpq.exe -lm
// Run:   bin/phase56_ivfpq.exe [--dim 64] [--V 256] [--nprobe 4] [--k 16] [--tokens 4000]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static inline float dotf(const float*a,const float*b,int n){ __m256 s=_mm256_setzero_ps(); int i=0;
    for(;i<=n-8;i+=8) s=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),s);
    float o[8]; _mm256_storeu_ps(o,s); float r=o[0]+o[1]+o[2]+o[3]+o[4]+o[5]+o[6]+o[7]; for(;i<n;i++) r+=a[i]*b[i]; return r; }
static inline float l2f(const float*a,const float*b,int n){ float r=0; for(int i=0;i<n;i++){ float d=a[i]-b[i]; r+=d*d; } return r; }
static uint64_t rs=0x9e3779b9ULL; static inline uint32_t xr(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (uint32_t)(rs>>32); }
static inline float frand(){ return ((float)(xr()&0xFFFFF)/(float)0x100000-0.5f); }
static double now(){ return (double)clock()/CLOCKS_PER_SEC; }
static inline uint16_t hmin_epu16(__m256i v){ uint16_t t[16]; _mm256_storeu_si256((__m256i*)t,v); uint16_t m=t[0];
    for(int i=1;i<16;i++) if(t[i]<m)m=t[i]; return m; }

int main(int argc,char**argv){
    int D=64,V=256,NPROBE=4,K=16; long ntok=4000;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"--dim")&&i+1<argc)D=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--V")&&i+1<argc)V=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--nprobe")&&i+1<argc)NPROBE=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--k")&&i+1<argc)K=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--tokens")&&i+1<argc)ntok=atol(argv[++i]); }
    int M8=D/4, DS8=4;                                    // 8-bit path: DS8=4 dims/sub-q -> M8=D/4
    int M4=D/2, DS4=2;                                    // 4-bit path: DS4=2 dims/sub-q -> M4=D/2 (same 128-bit budget as 8b)
    long maxctx=131072; size_t vecB=(size_t)D*4;
    printf("==== Phase 56 STAGE 1 v2 | IVFPQ + PQ4 fast-scan | D=%d V=%d nprobe=%d k=%d | 8b:M=%d 4b:M=%d (both 128bit=16B/vec) ====\n",
           D,V,NPROBE,K,M8,M4);
    printf("  code store@128K = 2.1 MB (vs raw keys %.1f MB); nprobe/V=%.4f -> n_cand(ctx)=ctx*%.4f\n",
           (double)maxctx*vecB/1e6,(double)NPROBE/V,(double)NPROBE/V);
    if(DS8*M8!=D||DS4*M4!=D){ printf("  ERROR: D not divisible\n"); return 1; }

    float* C=malloc((size_t)V*D*4);
    float* sub8=malloc((size_t)M8*256*DS8*4);  float* tbl8=malloc((size_t)M8*256*4);
    float* sub4=malloc((size_t)M4*16*DS4*4);   uint8_t* lut4=malloc((size_t)M4*16);
    float* Vst=malloc((size_t)maxctx*D*4);
    uint8_t* codes8=malloc((size_t)maxctx*M8);
    long nblk=(maxctx+31)/32; uint8_t* codes4=malloc((size_t)nblk*(M4/2)*32);   // FAISS blocked: 32 vec/block, 2 sub-q/byte
    float* q=malloc(vecB); float* nk=malloc(vecB); float* acc=malloc(vecB); float* cs=malloc((size_t)V*4);
    uint8_t* nc8=malloc(M8); uint8_t* nc4=malloc(M4);
    for(long i=0;i<(long)V*D;i++) C[i]=frand();
    for(long i=0;i<(long)M8*256*DS8;i++) sub8[i]=frand();
    for(long i=0;i<(long)M4*16*DS4;i++) sub4[i]=frand();
    for(long i=0;i<maxctx*D;i++) Vst[i]=frand();
    for(long i=0;i<maxctx*M8;i++) codes8[i]=(uint8_t)(xr()&0xFF);
    for(long i=0;i<(long)nblk*(M4/2)*32;i++) codes4[i]=(uint8_t)(xr()&0xFF);

    // ---------- (A) raw-V random-read latency vs store size (only k=16/token) ----------
    printf("\n  -- (A) raw-V random-read latency vs store size (k=%d gather/token) --\n",K);
    long sizes[]={1024,16384,131072,0}; double vread_ns_128k=0;
    for(int si=0;sizes[si];si++){ long S=sizes[si]; double t0=now(); volatile float sink=0;
        for(long it=0;it<ntok;it++){ float a=0; for(long j=0;j<4096;j++){ long idx=xr()%S; const float* v=Vst+(size_t)idx*D; for(int d=0;d<D;d++) a+=v[d]; } sink+=a; }
        double ns=(now()-t0)/((double)ntok*4096)*1e9; printf("  %8ld vec %8.1f MB | %7.2f ns/read\n",S,(double)S*vecB/1e6,ns);
        if(S==maxctx) vread_ns_128k=ns; }
    double vgather_us=K*vread_ns_128k/1e3;

    volatile float sink=0; double t0;
    // ---------- (B8) v1 const: 8-bit scalar (insert argmax + PQ-encode + probe + scalar ADC table) ----------
    t0=now();
    for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++){ q[d]=frand(); nk[d]=frand(); }
        float best=-1e30f; for(int j=0;j<V;j++){ float s=dotf(nk,C+(size_t)j*D,D); if(s>best)best=s; }
        for(int m=0;m<M8;m++){ const float* sv=nk+m*DS8; float bb=1e30f; int bi=0; for(int c=0;c<256;c++){ float dd=l2f(sv,sub8+((size_t)m*256+c)*DS8,DS8); if(dd<bb){bb=dd;bi=c;} } nc8[m]=(uint8_t)bi; }
        for(int j=0;j<V;j++) cs[j]=dotf(q,C+(size_t)j*D,D);
        for(int p=0;p<NPROBE;p++){ float bb=-1e30f; int bi=0; for(int j=0;j<V;j++) if(cs[j]>bb){bb=cs[j];bi=j;} cs[bi]=-1e30f; }
        for(int m=0;m<M8;m++){ const float* sv=q+m*DS8; for(int c=0;c<256;c++) tbl8[m*256+c]=l2f(sv,sub8+((size_t)m*256+c)*DS8,DS8); }
        sink+=best+nc8[0]+tbl8[0]; }
    double const8_us=(now()-t0)/ntok*1e6;
    // (C8) v1 scalar ADC per candidate
    t0=now();
    for(long it=0;it<ntok;it++){ float a=0; for(long j=0;j<2048;j++){ const uint8_t* cc=codes8+(size_t)j*M8; float dd=0; for(int m=0;m<M8;m++) dd+=tbl8[m*256+cc[m]]; a+=dd; } sink+=a; }
    double adc8_us=(now()-t0)/((double)ntok*2048)*1e6;

    // ---------- (B4) v2 const: 4-bit, vectorized + 16x-smaller table build ----------
    t0=now();
    for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++){ q[d]=frand(); nk[d]=frand(); }
        float best=-1e30f; for(int j=0;j<V;j++){ float s=dotf(nk,C+(size_t)j*D,D); if(s>best)best=s; }          // insert argmax (AVX2 dotf)
        for(int m=0;m<M4;m++){ const float* sv=nk+m*DS4; float bb=1e30f; int bi=0;                              // PQ-encode: M4 x 16 sub-dists (2-dim)
            for(int c=0;c<16;c++){ float dd=l2f(sv,sub4+((size_t)m*16+c)*DS4,DS4); if(dd<bb){bb=dd;bi=c;} } nc4[m]=(uint8_t)bi; }
        for(int j=0;j<V;j++) cs[j]=dotf(q,C+(size_t)j*D,D);                                                      // probe centroid (AVX2)
        for(int p=0;p<NPROBE;p++){ float bb=-1e30f; int bi=0; for(int j=0;j<V;j++) if(cs[j]>bb){bb=cs[j];bi=j;} cs[bi]=-1e30f; }
        for(int m=0;m<M4;m++){ const float* sv=q+m*DS4; float mn=1e30f,mx=-1e30f; float tmp[16];               // build LUT (M4 x 16), quantize to uint8
            for(int c=0;c<16;c++){ float dd=l2f(sv,sub4+((size_t)m*16+c)*DS4,DS4); tmp[c]=dd; if(dd<mn)mn=dd; if(dd>mx)mx=dd; }
            float sc=(mx>mn)?255.f/(mx-mn):0.f; for(int c=0;c<16;c++) lut4[m*16+c]=(uint8_t)((tmp[c]-mn)*sc); }
        sink+=best+nc4[0]+lut4[0]; }
    double const4_us=(now()-t0)/ntok*1e6;

    // ---------- (C4) v2 PQ4 fast-scan: _mm256_shuffle_epi8, 32 lookups/op, blocked codes ----------
    const __m256i mask0f=_mm256_set1_epi8(0x0F); const __m256i zero=_mm256_setzero_si256();
    __m256i lr[128];                                      // broadcast 16-byte LUT of each sub-q (M4 up to D/2) into both 128-bit lanes
    t0=now();
    long scan_blocks=2048/32;                             // scan 2048 candidates = 64 blocks of 32
    for(long it=0;it<ntok;it++){
        for(int m=0;m<M4;m++) lr[m]= _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut4+m*16)));
        __m256i gmin=_mm256_set1_epi16((short)0xFFFF);
        for(long b=0;b<scan_blocks;b++){ const uint8_t* bc=codes4+(size_t)b*(M4/2)*32;
            __m256i a0=zero,a1=zero;
            for(int p=0;p<M4/2;p++){ __m256i c=_mm256_loadu_si256((const __m256i*)(bc+(size_t)p*32));
                __m256i lo=_mm256_and_si256(c,mask0f); __m256i hi=_mm256_and_si256(_mm256_srli_epi16(c,4),mask0f);
                __m256i dlo=_mm256_shuffle_epi8(lr[2*p],lo); __m256i dhi=_mm256_shuffle_epi8(lr[2*p+1],hi);
                __m256i ds=_mm256_adds_epu8(dlo,dhi);
                a0=_mm256_add_epi16(a0,_mm256_unpacklo_epi8(ds,zero)); a1=_mm256_add_epi16(a1,_mm256_unpackhi_epi8(ds,zero)); }
            gmin=_mm256_min_epu16(gmin,_mm256_min_epu16(a0,a1)); }
        sink+=hmin_epu16(gmin); }
    double adc4_us=(now()-t0)/((double)ntok*2048)*1e6;

    // ---------- (E) MODEL 128K: v1(8b scalar) vs v2(4b fast-scan) ----------
    t0=now(); for(long it=0;it<ntok;it++){ for(int d=0;d<D;d++) q[d]=frand(); float a=0; for(long s=0;s<4096;s++) a+=dotf(q,Vst+(size_t)s*D,D); sink+=a; }
    double perdot_us=(now()-t0)/((double)ntok*4096)*1e6;
    long nc128=(long)NPROBE*131072/V;
    double ivf8=const8_us+(double)nc128*adc8_us+vgather_us;
    double ivf4=const4_us+(double)nc128*adc4_us+vgather_us;
    double naive128=(double)131072*perdot_us;
    printf("\n  -- per-token cost terms (us) --\n");
    printf("     %-22s  v1(8b scalar)   v2(4b fast-scan)\n","");
    printf("     %-22s  %10.3f   %10.3f\n","const (ins+enc+probe+LUT)",const8_us,const4_us);
    printf("     %-22s  %10.5f   %10.5f   (%.2f vs %.2f ns/cand)\n","ADC per candidate",adc8_us,adc4_us,adc8_us*1e3,adc4_us*1e3);
    printf("     %-22s  %10.3f   %10.3f   (k=%d raw-V DRAM)\n","V-gather (constant)",vgather_us,vgather_us,K);
    printf("\n  -- (E) EXPLICIT MODEL @128K (const + n_cand*ADC + k*DRAM ; NOT slope) | n_cand=%ld naive=%.1fus --\n",nc128,naive128);
    printf("     v1 8-bit scalar : %8.2f us/tok/layer  (%.1fx vs naive)\n",ivf8,naive128/ivf8);
    printf("     v2 4-bit fast   : %8.2f us/tok/layer  (%.1fx vs naive)\n",ivf4,naive128/ivf4);
    printf("        v2 breakdown : const %.2f (%.0f%%) + ADC-scan %.2f (%.0f%%, %ld cand x %.2f ns) + V-gather %.2f (%.0f%%)\n",
           const4_us,100*const4_us/ivf4, (double)nc128*adc4_us,100*(double)nc128*adc4_us/ivf4,nc128,adc4_us*1e3, vgather_us,100*vgather_us/ivf4);

    printf("\n  ==== GATE 1 (pre-registered): IVFPQ(fast)@128K <= ~30 us AND >> margin vs naive ====\n");
    int pass=(ivf4<=30.0);
    printf("     v2 4-bit fast-scan @128K = %.2f us ; naive = %.2f us ; speedup = %.1fx ; v1->v2 = %.2fx faster\n",
           ivf4,naive128,naive128/ivf4,ivf8/ivf4);
    printf("     VERDICT: %s (threshold 30 us)\n", pass?"PASS -> graduate to STAGE 2 quality run (3-arm InfoNCE/kmeans/random)":"FAIL -> IVFPQ dies here, clean");
    printf("  (sink=%.1f)\nSTOP. no commit.\n",(double)sink);
    free(C);free(sub8);free(tbl8);free(sub4);free(lut4);free(Vst);free(codes8);free(codes4);free(q);free(nk);free(acc);free(cs);free(nc8);free(nc4);
    return 0;
}
