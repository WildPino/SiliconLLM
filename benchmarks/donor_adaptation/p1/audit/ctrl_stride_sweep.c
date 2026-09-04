/* CONTROLLER P1 AUDIT probe. Byte arm only. Discriminates two mechanisms for the +64 stride recovery:
     (H1, the report's)  L1d set conflict: recovery iff (stride/64) mod 64 == 0 -> independent of plane count
     (H2, the auditor's) loss of the 64B half-line reuse between base and base+32, when the reuse working set
                         W = planes*64 exceeds the conflict-restricted L2/L3 capacity -> depends on planes
   Arms: A stride=Mpad ; B stride=Mpad+64 ; C stride=Mpad, base pointer +64 (alignment/alloc control). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>
static double now_s(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart/(double)f.QuadPart;}
static inline void acc_add_i8x32(__m256i* acc,__m256i p){
    __m128i lo=_mm256_castsi256_si128(p),hi=_mm256_extracti128_si256(p,1);
    acc[0]=_mm256_add_epi32(acc[0],_mm256_cvtepi8_epi32(lo)); acc[1]=_mm256_add_epi32(acc[1],_mm256_cvtepi8_epi32(_mm_srli_si128(lo,8)));
    acc[2]=_mm256_add_epi32(acc[2],_mm256_cvtepi8_epi32(hi)); acc[3]=_mm256_add_epi32(acc[3],_mm256_cvtepi8_epi32(_mm_srli_si128(hi,8)));
}
static void matvec_lut_full(const int8_t* codes,const int8_t* lut,int32_t* y,int M,int Mpad,int T){
    for(int base=0;base<M;base+=32){
        __m256i acc[4]={_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
        for(int t=0;t<T;t++){ __m256i tbl=_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut+(size_t)t*16)));
            __m256i idx=_mm256_loadu_si256((const __m256i*)(codes+(size_t)t*Mpad+base)); acc_add_i8x32(acc,_mm256_shuffle_epi8(tbl,idx)); }
        int32_t tmp[32]; _mm256_storeu_si256((__m256i*)(tmp+0),acc[0]); _mm256_storeu_si256((__m256i*)(tmp+8),acc[1]);
        _mm256_storeu_si256((__m256i*)(tmp+16),acc[2]); _mm256_storeu_si256((__m256i*)(tmp+24),acc[3]);
        for(int r=0;r<32&&base+r<M;r++) y[base+r]=tmp[r]; }
}
static uint64_t rs=0x9E3779B97F4A7C15ULL;
static inline uint32_t r32(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (uint32_t)(rs>>32);}
#define M 8192
static int32_t y[M];
volatile int64_t g_sink=0;
static void cell(const char* arm,int planes,int stride,int ptr_off,double target_gb,int inv){
    size_t block=(size_t)planes*stride + 128;
    int8_t* lut=malloc((size_t)planes*16); for(size_t i=0;i<(size_t)planes*16;i++) lut[i]=(int8_t)((int)(r32()%127)-63);
    int R=(int)((512ull*1048576 + block-1)/block); if(R<2)R=2;
    char* pool=malloc((size_t)R*block); if(!pool){printf("OOM\n");exit(1);}
    for(size_t i=0;i<(size_t)R*block;i++) pool[i]=(int8_t)(r32()%9u);
    long long moved=(long long)planes*(M/32)*32;
    long reps=(long)(target_gb*1e9/(double)moved); if(reps<10)reps=10; if(reps>20000)reps=20000;
    for(int w=0;w<3;w++){ matvec_lut_full((const int8_t*)(pool+ptr_off),lut,y,M,stride,planes); g_sink+=y[0]+y[M-1]; }
    double* dt=malloc(sizeof(double)*reps);
    for(long i=0;i<reps;i++){ const int8_t* c=(const int8_t*)(pool+(size_t)(i%R)*block+ptr_off);
        double t0=now_s(); matvec_lut_full(c,lut,y,M,stride,planes); dt[i]=now_s()-t0;
        int64_t sk=0; for(int q=0;q<M;q+=512) sk+=y[q]; g_sink+=sk; }
    double s=0; for(long i=0;i<reps;i++) s+=dt[i]; double mean=s/reps;
    double v=0; for(long i=0;i<reps;i++){double d=dt[i]-mean; v+=d*d;} double sd=sqrt(v/(reps>1?reps-1:1));
    printf("CTRL,%d,%s,%d,%d,%d,%zu,%ld,%d,%.4f,%.2f,%.2f,%.3f\n",inv,arm,planes,stride,ptr_off,
           (size_t)planes*64,reps,R,mean*1e6,sd/mean*100.0,1.0/mean,(double)moved/mean/1e9);
    fflush(stdout); free(pool); free(dt); free(lut);
}
int main(int argc,char**argv){
    int inv=argc>1?atoi(argv[1]):1; rs=0x1000+inv;
    printf("CTRLHDR,inv,arm,planes,stride,ptr_off,W_bytes,reps,replicas,mean_us,cv_pct,matvecs_per_s,charged_GBs\n");
    int PAD[]={0,64,128,256,512,1024,2048,4096}; char nm[64];
    for(int i=0;i<8;i++){ snprintf(nm,sizeof nm,"S_pad%d",PAD[i]); cell(nm,4096,M+PAD[i],0,8.0,inv); }
    return 0;
}
