#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Shared hardware parameters
#define GRID_SIZE 256
#define BRIDGE_SPACING 64
#define NUM_ZONES (GRID_SIZE / BRIDGE_SPACING)
#define NUM_RULES 4
#define HISTORY_SIZE 1024

// Shared evaluation metrics
#define TOTAL_TICKS 100000

typedef struct {
    uint8_t input_sym;
    uint8_t target_sym;
} TaskData;

static inline TaskData generate_task(int task_type, int t, uint8_t* hist) {
    TaskData d;
    d.input_sym = rand() % 2;
    d.target_sym = 0;
    
    if (task_type == 0) { // XOR-2
        d.target_sym = hist[0] ^ hist[1];
    } else if (task_type == 1) { // Period-7
        d.target_sym = (t % 7 == 0) ? 1 : 0;
        d.input_sym = 0; 
    } else if (task_type == 2) { // Echo-5
        d.target_sym = hist[4];
    }
    
    // Shift history
    memmove(&hist[1], &hist[0], 15);
    hist[0] = d.input_sym;
    return d;
}

static inline void print_accuracy(int t, int correct, int window) {
    double acc = (double)correct / window;
    printf("  Step %5d | Accuracy: %5.1f%%\n", t, acc * 100.0);
}

#endif
