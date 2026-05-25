#include "range_coder.h"

void rc_encoder_init(RangeEncoder* rc, FILE* file) {
    rc->low = 0;
    rc->range = 0xFFFFFFFFFFFFFFFFull;
    rc->cache = 0;
    rc->cache_size = 1;
    rc->file = file;
}

void rc_encode(RangeEncoder* rc, uint64_t cum_freq, uint64_t freq, uint64_t total_freq) {
    rc->range /= total_freq;
    rc->low += (__uint128_t)cum_freq * rc->range;
    rc->range *= freq;
    
    while (rc->range < 0x0100000000000000ull) {
        if ((uint64_t)(rc->low >> 64) != 0 || (uint64_t)rc->low < 0xFF00000000000000ull) {
            uint8_t temp = rc->cache;
            do {
                fputc(temp + (uint8_t)(rc->low >> 64), rc->file);
                temp = 0xFF;
            } while (--rc->cache_size != 0);
            rc->cache = (uint8_t)((uint64_t)rc->low >> 56);
        }
        rc->cache_size++;
        rc->low = (uint64_t)(rc->low << 8);
        rc->range <<= 8;
    }
}

void rc_encoder_flush(RangeEncoder* rc) {
    for (int i = 0; i < 9; i++) {
        if ((uint64_t)(rc->low >> 64) != 0 || (uint64_t)rc->low < 0xFF00000000000000ull) {
            uint8_t temp = rc->cache;
            do {
                fputc(temp + (uint8_t)(rc->low >> 64), rc->file);
                temp = 0xFF;
            } while (--rc->cache_size != 0);
            rc->cache = (uint8_t)((uint64_t)rc->low >> 56);
        }
        rc->cache_size++;
        rc->low = (uint64_t)(rc->low << 8);
    }
}

void rc_decoder_init(RangeDecoder* rc, FILE* file) {
    rc->file = file;
    rc->code = 0;
    rc->range = 0xFFFFFFFFFFFFFFFFull;
    int b = fgetc(rc->file); // Read dummy byte
    (void)b; // Ignore
    for (int i = 0; i < 8; i++) {
        b = fgetc(rc->file);
        if (b == EOF) b = 0;
        rc->code = (rc->code << 8) | b;
    }
}

uint64_t rc_get_freq(RangeDecoder* rc, uint64_t total_freq) {
    uint64_t range_per_freq = rc->range / total_freq;
    return rc->code / range_per_freq;
}

void rc_decode(RangeDecoder* rc, uint64_t cum_freq, uint64_t freq, uint64_t total_freq) {
    uint64_t range_per_freq = rc->range / total_freq;
    rc->code -= cum_freq * range_per_freq;
    rc->range = freq * range_per_freq;
    
    while (rc->range < 0x0100000000000000ull) {
        rc->range <<= 8;
        int b = fgetc(rc->file);
        if (b == EOF) b = 0;
        rc->code = (rc->code << 8) | b;
    }
}
