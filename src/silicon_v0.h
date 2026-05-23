#ifndef SILICON_V0_H
#define SILICON_V0_H

#include <stdint.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Silicon Sequence Compressor V0
// Architecture: G128_T16_C16
// Dynamics: Persistent Wave (sat_add / avg)
// Readout: Lane-aware pooled 32D
// ----------------------------------------------------------------------------

typedef struct {
    // Spatial state of the wave (128 AVX2 blocks = 4096 bytes)
    __m256i state[128];
    
    // Physical Codebook (Random Binary Single-Block)
    __m256i codebook[256];
    
    // M4 History Buffer (circular)
    uint8_t m4_buf[256];
    int m4_head;
} SiliconV0;

/**
 * Inizializza l'engine.
 * Genera il Codebook (Random Binary Single-Block) usando il seed fornito,
 * e azzera lo stato spaziale e il buffer storico.
 */
void silicon_v0_init(SiliconV0* e, int codebook_seed);

/**
 * Resetta esclusivamente lo stato differenziale della Wave.
 * Utile per ablazioni o per testare il motore senza integrazione storica,
 * o all'inizio di un nuovo documento.
 */
void silicon_v0_reset(SiliconV0* e);

/**
 * Esegue un tick dell'engine per il byte in ingresso.
 * 1. Accoda il byte in M4.
 * 2. Reinietta la finestra T3 spazialmente shiftata.
 * 3. Applica il damping e i 4 step di diffusione d'onda.
 */
void silicon_v0_tick(SiliconV0* e, uint8_t input_byte);

/**
 * Estrae le feature Lane-Aware Pooled 32D dalla griglia spaziale.
 * Esegue un sum-pooling lane-wise sui 16 canali spaziali.
 * L'output e' un vettore di 32 double pronto per regressione/layer lineare.
 */
void silicon_v0_extract_32d(const SiliconV0* e, double* out_32d);

#ifdef __cplusplus
}
#endif

#endif // SILICON_V0_H
