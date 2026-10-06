/* Source I8/A16 column representation. Caller supplies original quantizer codes.
   Preconditions: D=768,H=3072,abs(code)<=32767,finite positive scales/alpha.
   Zero terms deleted;512*128*32767=2147418112<INT32_MAX; totals I64.
   Algebra/order/block/vector width unchanged from qualified METH505. */
#ifndef SILICON_EXACT_I8_COLUMNS_H
#define SILICON_EXACT_I8_COLUMNS_H
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <immintrin.h>
#define EXACT_COLUMN_D 768
#define EXACT_COLUMN_H 3072
#define EXACT_COLUMN_CHUNK 512
_Static_assert(EXACT_COLUMN_CHUNK*128*32767==2147418112,"exact_column_partial_bound");
typedef struct {
    int16_t codes[EXACT_COLUMN_H];uint16_t indices[EXACT_COLUMN_H];
    int32_t partial[EXACT_COLUMN_D];int64_t total[EXACT_COLUMN_D];
} ExactColumnWork;
static uint32_t exact_i8_columns(const int8_t *columns,const float *scales,
                                float alpha,float *output,ExactColumnWork *w){
    uint32_t nonzero=0;for(int j=0;j<EXACT_COLUMN_H;j++)if(w->codes[j]!=0)w->indices[nonzero++]=(uint16_t)j;
    memset(w->total,0,sizeof(w->total));
    for(uint32_t start=0;start<nonzero;start+=EXACT_COLUMN_CHUNK){
        uint32_t end=start+EXACT_COLUMN_CHUNK;if(end>nonzero)end=nonzero;memset(w->partial,0,sizeof(w->partial));
        for(uint32_t p=start;p<end;p++){
            uint32_t j=w->indices[p];const int8_t *q=columns+(size_t)j*EXACT_COLUMN_D;
            __m256i c=_mm256_set1_epi32((int32_t)w->codes[j]);
            for(int r=0;r<EXACT_COLUMN_D;r+=8){
                __m256i v=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q+r)));
                __m256i partial=_mm256_loadu_si256((const __m256i *)(w->partial+r));
                _mm256_storeu_si256((__m256i *)(w->partial+r),_mm256_add_epi32(partial,_mm256_mullo_epi32(v,c)));
            }
        }
        for(int r=0;r<EXACT_COLUMN_D;r+=4){
            __m256i part=_mm256_cvtepi32_epi64(_mm_loadu_si128((const __m128i *)(w->partial+r)));
            __m256i total=_mm256_loadu_si256((const __m256i *)(w->total+r));
            _mm256_storeu_si256((__m256i *)(w->total+r),_mm256_add_epi64(total,part));
        }
    }
    for(int r=0;r<EXACT_COLUMN_D;r++)output[r]=(float)(((double)w->total[r]*(double)scales[r])*(double)alpha);
    return nonzero;
}
#endif
