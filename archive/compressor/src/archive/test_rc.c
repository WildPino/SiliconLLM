#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "range_coder.h"

int main() {
    FILE* fout = fopen("rc_test.bin", "wb");
    RangeEncoder re;
    rc_encoder_init(&re, fout);
    
    // Encode some frequencies
    // CDF_SCALE = 16384
    rc_encode(&re, 0, 100, 16384);
    rc_encode(&re, 100, 200, 16384);
    rc_encode(&re, 300, 16084, 16384);
    rc_encode(&re, 100, 200, 16384);
    
    rc_encoder_flush(&re);
    fclose(fout);
    
    FILE* fin = fopen("rc_test.bin", "rb");
    RangeDecoder rd;
    rc_decoder_init(&rd, fin);
    
    uint64_t f;
    
    f = rc_get_freq(&rd, 16384);
    printf("1: f = %llu\n", (unsigned long long)f);
    assert(f >= 0 && f < 100);
    rc_decode(&rd, 0, 100, 16384);
    
    f = rc_get_freq(&rd, 16384);
    printf("2: f = %llu\n", (unsigned long long)f);
    assert(f >= 100 && f < 300);
    rc_decode(&rd, 100, 200, 16384);
    
    f = rc_get_freq(&rd, 16384);
    printf("3: f = %llu\n", (unsigned long long)f);
    assert(f >= 300 && f < 16384);
    rc_decode(&rd, 300, 16084, 16384);
    
    f = rc_get_freq(&rd, 16384);
    printf("4: f = %llu\n", (unsigned long long)f);
    assert(f >= 100 && f < 300);
    rc_decode(&rd, 100, 200, 16384);
    
    fclose(fin);
    printf("Range Coder Test Passed!\n");
    return 0;
}
