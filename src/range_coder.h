#pragma once
#include <stdint.h>
#include <stdio.h>

typedef struct {
    __uint128_t low;
    uint64_t range;
    uint64_t cache_size;
    uint8_t cache;
    FILE* file;
} RangeEncoder;

typedef struct {
    uint64_t range;
    uint64_t code;
    FILE* file;
} RangeDecoder;

void rc_encoder_init(RangeEncoder* rc, FILE* file);
void rc_encode(RangeEncoder* rc, uint64_t cum_freq, uint64_t freq, uint64_t total_freq);
void rc_encoder_flush(RangeEncoder* rc);

void rc_decoder_init(RangeDecoder* rc, FILE* file);
uint64_t rc_get_freq(RangeDecoder* rc, uint64_t total_freq);
void rc_decode(RangeDecoder* rc, uint64_t cum_freq, uint64_t freq, uint64_t total_freq);
