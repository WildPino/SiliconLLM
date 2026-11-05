/* Stage 2 test 5, amendment 2: two engine costs at Ling-flash's shapes, with c50_engine.c's own code.  No model is read.
 *
 * The engine is included whole, built with C50_FLASH_SHAPES (D 4096, 32 query heads on 4 KV heads, 32 layers), so these are its functions and its
 * thread pool, not copies.
 *   --cell kv --kv f16|q8_0   each layer runs the attention job on a filled cache at position --ctx - 1: the KV read
 *   --cell fixed              every step of forward() that reads no weights: norms, RoPE, the router's selection, activation prepares, SwiGLU, sums,
 *                             with an empty job where forward() runs a weight job and where it runs attention.  k = 4, as the desk.
 * Steps: --steps after 8 warm-up; prints the median.  CONFIG states the shapes.
 *
 * Build: clang -O2 -mavx2 -mfma -mf16c -DC50_FLASH_SHAPES -I../chatbot50 -o bin/s2k_engine.exe s2k_engine.c -lpsapi
 */
#define main c50_engine_main
#include "c50_engine.c"
#undef main

static state_t S;
static act_t Eact[NU + 1];

static void noise(float *p, size_t n) {
    uint64_t s = 20261105u;
    size_t i;
    for (i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        p[i] = (float)((double)(s >> 11) / 9007199254740992.0) - 0.5f;
    }
}

static void step_kv(unsigned ctx) {
    unsigned l;
    for (l = 0; l < NL; ++l) {
        S.att.kc = S.kc ? S.kc + (size_t)l * MAXCTX * NKV * HD : NULL;
        S.att.vc = S.vc ? S.vc + (size_t)l * MAXCTX * NKV * HD : NULL;
        S.att.kq = S.kq ? S.kq + (size_t)l * MAXCTX * NKV * KVROW : NULL;
        S.att.vq = S.vq ? S.vq + (size_t)l * MAXCTX * NKV * KVROW : NULL;
        S.att.q = S.qkv; S.att.npos = ctx; S.att.out = S.ao;
        run(att_worker, &S.att);
        att_combine(&S.att);
    }
}

static float norm_w[D], head_w[HD], bias[NE0];

static void step_fixed(void) {
    unsigned l, h, i, e, sel[NU];
    float w[NU];
    swiglu_job sj;
    float *x = S.x;
    rope_table(2047, S.rope_base);
    for (l = 0; l < NL; ++l) {
        rmsnorm_w(S.h, x, norm_w, D, S.eps);
        act_set(&S.xa, S.h, D);
        act_prepare(&S.xa, GT_Q4_K, D);
        run(empty_job, NULL);                                /* qkv */
        for (h = 0; h < NH + NKV; ++h) {
            float *v = S.qkv + h * HD;
            rmsnorm_w(v, v, head_w, HD, S.eps);
            rope(v);
        }
        run(empty_job, NULL);                                /* attention */
        act_set(&S.xa, S.ao, D);
        act_prepare(&S.xa, GT_Q4_K, D);
        run(empty_job, NULL);                                /* output projection */
        for (i = 0; i < D; ++i) x[i] += S.tmp[i];
        rmsnorm_w(S.h, x, norm_w, D, S.eps);
        act_set(&S.xa, S.h, D);
        act_prepare(&S.xa, GT_Q4_K, D);
        run(empty_job, NULL);                                /* router */
        route(S.logits_r, bias, sel, w, S.wscale, NE0);
        run(empty_job, NULL);                                /* gate and up */
        sj.gu = S.gu; sj.ea = Eact; sj.ff = FFE; sj.n = g_ku + 1;
        run(swiglu_worker, &sj);
        for (e = 0; e <= g_ku; ++e) act_prepare(&Eact[e], GT_Q4_K, FFE);
        run(empty_job, NULL);                                /* down */
        for (i = 0; i < D; ++i) {
            float s = 0.0f;
            for (e = 0; e < g_ku; ++e) s += w[e] * S.y[(size_t)e * D + i];
            x[i] = 0.999f * x[i] + 1e-3f * (s + S.y[(size_t)g_ku * D + i]);
        }
    }
    rmsnorm_w(S.h, x, norm_w, D, S.eps);
    act_set(&S.xa, S.h, D);
    act_prepare(&S.xa, GT_Q4_K, D);
}

/* --cell corr: the fourth code's -0.5 * s * sum(x) per row, on one thread, as the upper bound of a term the engine would fold into its jobs: per layer, the sum of
 * the input and of k = 4 expert inputs, then gate and up (2 x 1024 rows) and down (4096 rows) of each expert; every layer holds its own scales and outputs. */
#define CROWS ((2u * FFE + D) * 4u)
static float *c_s, *c_y;

static void step_corr(void) {
    unsigned l, e, r, i;
    for (l = 0; l < NL; ++l) {
        float *s = c_s + (size_t)l * CROWS, *y = c_y + (size_t)l * CROWS;
        float sx = 0.0f;
        for (i = 0; i < D; ++i) sx += S.x[i];
        for (e = 0; e < 4; ++e) {
            float se = 0.0f;
            for (i = 0; i < FFE; ++i) se += S.gu[(size_t)e * FFE + i];
            for (r = 0; r < 2 * FFE; ++r) y[r] -= 0.5f * s[r] * sx;
            s += 2 * FFE; y += 2 * FFE;
            for (r = 0; r < D; ++r) y[r] -= 0.5f * s[r] * se;
            s += D; y += D;
        }
    }
}

int main(int argc, char **argv) {
    const char *cell = "kv", *kv = "f16";
    unsigned ctx = 2048, steps = 64, i, e;
    int a;
    double *t, sum = 0;
    for (a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--cell")) cell = argv[++a];
        else if (!strcmp(argv[a], "--kv")) kv = argv[++a];
        else if (!strcmp(argv[a], "--ctx")) ctx = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--steps")) steps = (unsigned)atoi(argv[++a]);
        else { fprintf(stderr, "unknown argument %s\n", argv[a]); return 2; }
    }
    start_pool();
    S.eps = 1e-6f; S.rope_base = 600000.0f; S.wscale = 2.5f;
    noise(S.x, D); noise(S.qkv, QKV_ROWS); noise(S.logits_r, NE0); noise(S.gu, sizeof S.gu / sizeof S.gu[0]); noise(S.y, (NU + 1) * D); noise(S.tmp, D); noise(S.ao, D);
    for (i = 0; i < D; ++i) norm_w[i] = 1.0f;
    for (i = 0; i < HD; ++i) head_w[i] = 1.0f;
    g_ku = 4;
    for (e = 0; e <= NU; ++e) act_prepare(&Eact[e], GT_Q4_K, FFE), act_prepare(&S.ea[e], GT_Q4_K, FFE);
    if (!strcmp(cell, "corr")) {
        c_s = malloc(sizeof(float) * (size_t)NL * CROWS);
        c_y = malloc(sizeof(float) * (size_t)NL * CROWS);
        noise(c_s, (size_t)NL * CROWS);
        noise(c_y, (size_t)NL * CROWS);
    }
    if (!strcmp(cell, "kv")) {
        size_t nb =!strcmp(kv, "q8_0") ? (size_t)NL * MAXCTX * NKV * KVROW : (size_t)NL * MAXCTX * NKV * HD * 2, b;
        uint8_t *k = VirtualAlloc(NULL, nb, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE), *v = VirtualAlloc(NULL, nb, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!strcmp(kv, "q8_0")) {
            g_kvq8 = 1; S.kq = k; S.vq = v;
            for (b = 0; b + KVB <= nb; b += KVB) { uint16_t hs = _cvtss_sh(0.01f, 0); memcpy(k + b, &hs, 2); memcpy(v + b, &hs, 2); memset(k + b + 2, (int)(b & 63), 32); memset(v + b + 2, (int)(b & 31), 32); }
        } else {
            uint16_t *kk = (uint16_t *)k, *vv = (uint16_t *)v;
            S.kc = kk; S.vc = vv;
            for (b = 0; b < nb / 2; ++b) { kk[b] = _cvtss_sh(0.01f * (float)(b & 63), 0); vv[b] = _cvtss_sh(0.01f * (float)(b & 31), 0); }
        }
    }
    printf("CONFIG {\"cell\": \"%s\", \"kv\": \"%s\", \"ctx\": %u, \"steps\": %u, \"d\": %u, \"nh\": %u, \"nkv\": %u, \"hd\": %u, \"nl\": %u, \"ffe\": %u, \"k\": %u, \"threads\": %d}\n",
           cell, kv, ctx, steps, D, NH, NKV, HD, NL, FFE, g_ku, g_nt);
    fflush(stdout);
    t = malloc(sizeof(double) * steps);
    for (i = 0; i < steps + 8; ++i) {
        double t0 = now_s();
        if (!strcmp(cell, "kv")) step_kv(ctx);
        else if (!strcmp(cell, "corr")) step_corr();
        else step_fixed();
        if (i >= 8) t[i - 8] = now_s() - t0;
    }
    for (i = 0; i < steps; ++i) sum += t[i];
    qsort(t, steps, sizeof(double), cmpd);
    {
        double bytes = !strcmp(cell, "kv") ? 2.0 * NL * ctx * NKV * (!strcmp(kv, "q8_0") ? (double)KVROW : HD * 2.0) : 0.0;
        printf("RESULT {\"cell\": \"%s\", \"kv\": \"%s\", \"ms_median\": %.3f, \"ms_mean\": %.3f, \"ms_min\": %.3f, \"ms_max\": %.3f, \"mb\": %.1f, \"gbytes_per_s\": %.2f, \"jobs\": %.1f}\n",
               cell, kv, 1e3 * t[steps / 2], 1e3 * sum / steps, 1e3 * t[0], 1e3 * t[steps - 1], bytes / 1e6, bytes / t[steps / 2] / 1e9, (double)g_runs / (steps + 8));
    }
    ExitProcess(0);
}
