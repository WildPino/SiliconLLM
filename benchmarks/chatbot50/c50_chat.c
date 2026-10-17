/*
 * C50-A8 interactive chatbot: Ling-mini-2.0 on the A3 engine (c50_engine.c, included unchanged; its main is
 * renamed), the C tokenizer (c50_tok.h) and the chat template (c50_chat_template.h), thinking off.
 *
 * Interactive:  c50_chat --model GGUF [--system TEXT] [--max-new N] [--temp T] [--top-p P] [--seed S]
 *               commands: /reset (new conversation), /exit
 * Batch (A8 T6): c50_chat --model GGUF --batch PROMPTS.txt --out PREFIX [--max-new N]
 *               one user prompt per line, each a fresh single-turn conversation; writes
 *               PREFIX.jsonl  {"item":i,"prompt_tokens":n,"stop":"EOS"|"MAX","gen":[ids]} (the stop ID is in gen)
 *               PREFIX.prompts.tsv  "<item>\t<ids>" of each rendered prompt (input format of c50_llama_generate)
 * Decoding: each turn is rendered by the template, encoded with special tokens recognised and fed token by
 * token; generation stops on <|role_end|> (156895) or <|endoftext|> (156892), as generation_config.json says.
 * Greedy (default, --temp 0) takes the argmax with ties to the lowest ID, like c50_llama_generate.  The KV
 * cache keeps the generated IDs across turns (no re-encode); the next turn feeds <|role_end|> and the new
 * '<role>HUMAN</role>...<|role_end|><role>ASSISTANT</role>' segment, which is the template's own text.
 * Prints one CONFIG line.
 *
 * Build: clang -O2 -mavx2 -mfma -mf16c -Wno-unused-function -o bin/c50_chat.exe c50_chat.c
 */
#define main c50_engine_main
#include "c50_engine.c"
#undef main

#include "c50_tok.h"
#include "c50_chat_template.h"

#define ID_ROLE_END 156895
#define ID_EOT 156892

typedef struct { model_t *md; state_t *st; c50_tok *tk; unsigned pos; float temp, top_p; uint64_t rng; } chat_t;

static uint64_t chat_rand(chat_t *c) { c->rng ^= c->rng >> 12; c->rng ^= c->rng << 25; c->rng ^= c->rng >> 27; return c->rng * 2685821657736338717ULL; }

typedef struct { float p; int id; } cand_t;
static int cand_cmp(const void *a, const void *b) {
    const cand_t *x = a, *y = b;
    return x->p > y->p ? -1 : x->p < y->p ? 1 : x->id - y->id;
}

static int pick(chat_t *c) {
    const float *lg = c->st->logits;
    unsigned v, n = 0, arg = 0;
    float mx = -INFINITY;
    for (v = 0; v < VOCAB; ++v) if (lg[v] > mx) { mx = lg[v]; arg = v; }
    if (c->temp <= 0.0f) return (int)arg;
    {
        static cand_t cand[VOCAB];
        double sum = 0.0, acc = 0.0, r;
        for (v = 0; v < VOCAB; ++v)
            if ((mx - lg[v]) / c->temp < 30.0f) { cand[n].id = (int)v; cand[n].p = expf((lg[v] - mx) / c->temp); sum += cand[n].p; ++n; }
        qsort(cand, n, sizeof *cand, cand_cmp);
        for (v = 0; v < n; ++v) { acc += cand[v].p / sum; if (acc >= c->top_p) { n = v + 1; break; } }
        for (sum = 0.0, v = 0; v < n; ++v) sum += cand[v].p;
        r = (double)(chat_rand(c) >> 11) / 9007199254740992.0 * sum;
        for (acc = 0.0, v = 0; v < n; ++v) { acc += cand[v].p; if (r < acc) return cand[v].id; }
        return cand[n - 1].id;
    }
}

static int feed(chat_t *c, const int32_t *ids, uint32_t n) {
    uint32_t i;
    if (c->pos + n >= MAXCTX) return -1;
    for (i = 0; i < n; ++i) forward(c->md, c->st, ids[i], c->pos++);
    return 0;
}

/* streaming UTF-8 printer: emits complete sequences, U+FFFD for invalid ones, waits on a truncated tail */
typedef struct { uint8_t b[256]; uint32_t n; FILE *f; } printer_t;
static void pr_emit(printer_t *p, int final) {
    uint32_t i = 0;
    while (i < p->n) {
        uint32_t k = c50_utf8_valid_len(p->b + i, p->n - i), bl;
        if (k) { fwrite(p->b + i, 1, k, p->f); i += k; continue; }
        bl = c50_utf8_bad_len(p->b + i, p->n - i);
        if (!final && i + bl == p->n && p->b[i] >= 0xC2 && p->b[i] <= 0xF4) break;  /* may still complete */
        fwrite("\xEF\xBF\xBD", 1, 3, p->f); i += bl;
    }
    memmove(p->b, p->b + i, p->n - i); p->n -= i;
    fflush(p->f);
}
static void pr_token(printer_t *p, const c50_tok *t, int id) {
    uint8_t raw[512];
    uint32_t l = c50_token_raw(t, id, raw);
    if (p->n + l > sizeof p->b) pr_emit(p, 1);
    memcpy(p->b + p->n, raw, l); p->n += l;
    pr_emit(p, 0);
}

/* generate until a stop ID or max_new; returns the number of generated IDs (stop ID included) in *gen */
static uint32_t generate(chat_t *c, uint32_t max_new, c50_ids *gen, int *stopped, printer_t *pr) {
    uint32_t k;
    *stopped = 0;
    gen->n = 0;
    for (k = 0; k < max_new; ++k) {
        int id = pick(c);
        c50_ids_push(gen, id);
        if (id == ID_ROLE_END || id == ID_EOT) { *stopped = 1; break; }
        if (pr) pr_token(pr, c->tk, id);
        if (c->pos + 1 >= MAXCTX) break;
        forward(c->md, c->st, id, c->pos++);
    }
    if (pr) pr_emit(pr, 1);
    return gen->n;
}

static int read_line(char *buf, int cap) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode, got;
    if (GetConsoleMode(h, &mode)) {
        static wchar_t w[8192];
        int n;
        if (!ReadConsoleW(h, w, 8191, &got, NULL) || !got) return -1;
        while (got && (w[got - 1] == L'\n' || w[got - 1] == L'\r')) --got;
        n = WideCharToMultiByte(CP_UTF8, 0, w, (int)got, buf, cap - 1, NULL, NULL);
        buf[n] = 0;
        return n;
    }
    if (!fgets(buf, cap, stdin)) return -1;
    {
        int n = (int)strlen(buf);
        while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
        return n;
    }
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *system = NULL, *batch = NULL, *out = NULL;
    unsigned max_new = 512;
    int i;
    chat_t c;
    model_t *md = calloc(1, sizeof *md);
    state_t *st;
    c50_tok *tk = calloc(1, sizeof *tk);
    char types[256];
    memset(&c, 0, sizeof c);
    c.temp = 0.0f; c.top_p = 1.0f; c.rng = 0x9E3779B97F4A7C15ULL;
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--system") && i + 1 < argc) system = argv[++i];
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) c.temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) c.top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) c.rng = strtoull(argv[++i], NULL, 10) | 1;
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) batch = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model_path || (batch && !out) || !max_new) {
        fprintf(stderr, "usage: c50_chat --model GGUF [--system TEXT] [--max-new N] [--temp T] [--top-p P] [--seed S]\n"
                        "       c50_chat --model GGUF --batch PROMPTS.txt --out PREFIX [--max-new N]\n");
        return 2;
    }
    SetConsoleOutputCP(CP_UTF8);
    load_model(md, model_path);
    c50_tok_load(tk, &md->g);
    st = calloc(1, sizeof *st);
    st->kc = VirtualAlloc(NULL, (size_t)NL * MAXCTX * NKV * HD * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    st->vc = VirtualAlloc(NULL, (size_t)NL * MAXCTX * NKV * HD * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    st->logits = malloc(sizeof(float) * VOCAB);
    st->eps = gguf_f(&md->g, "bailingmoe2.attention.layer_norm_rms_epsilon");
    st->rope_base = gguf_f(&md->g, "bailingmoe2.rope.freq_base");
    st->wscale = gguf_f(&md->g, "bailingmoe2.expert_weights_scale");
    if (!st->kc || !st->vc || !st->logits) { fprintf(stderr, "allocation failed\n"); return 2; }
    start_pool();
    type_summary(md, types, sizeof types);
    c.md = md; c.st = st; c.tk = tk;
    printf("CONFIG chat=c50_chat model=%s threads=%d kv=f16 ctx=%u max_new=%u temp=%g top_p=%g mode=%s tokens=%u "
           "unidata=%s template=thinking_off types=[%s]\n", model_path, NT, MAXCTX, max_new, c.temp, c.top_p,
           batch ? "batch" : "interactive", tk->n, C50_UNIDATA_VERSION, types);
    fflush(stdout);

    if (batch) {
        FILE *fi = fopen(batch, "rb"), *fj, *ft;
        char path[1024], line[16384];
        unsigned item = 0;
        snprintf(path, sizeof path, "%s.jsonl", out); fj = fopen(path, "w");
        snprintf(path, sizeof path, "%s.prompts.tsv", out); ft = fopen(path, "w");
        if (!fi || !fj || !ft) { fprintf(stderr, "cannot open batch files\n"); return 2; }
        while (fgets(line, sizeof line, fi)) {
            c50_msg m = {C50_USER, line, 0};
            c50_buf b = {0};
            c50_ids ids = {0}, gen = {0};
            const char *why;
            uint32_t k;
            int stopped;
            m.len = (uint32_t)strlen(line);
            while (m.len && (line[m.len - 1] == '\n' || line[m.len - 1] == '\r')) line[--m.len] = 0;
            if (!m.len) continue;
            if (c50_render(&m, 1, 1, &b, &why)) { fprintf(stderr, "template: %s\n", why); return 2; }
            c50_encode(tk, (const uint8_t *)b.s, b.n, &ids);
            c.pos = 0;
            if (feed(&c, ids.v, ids.n)) { fprintf(stderr, "prompt longer than the context\n"); return 2; }
            generate(&c, max_new, &gen, &stopped, NULL);
            fprintf(ft, "%u\t", item);
            for (k = 0; k < ids.n; ++k) fprintf(ft, k ? ",%d" : "%d", ids.v[k]);
            fprintf(ft, "\n");
            fprintf(fj, "{\"item\":%u,\"prompt_tokens\":%u,\"stop\":\"%s\",\"gen\":[", item, ids.n, stopped ? "EOS" : "MAX");
            for (k = 0; k < gen.n; ++k) fprintf(fj, k ? ",%d" : "%d", gen.v[k]);
            fprintf(fj, "]}\n");
            {
                uint8_t *txt;
                c50_decode(tk, gen.v, gen.n, &txt);
                printf("ITEM %u prompt_tokens=%u gen=%u stop=%s\n%s\n", item, ids.n, gen.n, stopped ? "EOS" : "MAX", txt);
                free(txt);
            }
            fflush(stdout);
            free(b.s); free(ids.v); free(gen.v);
            ++item;
        }
        fclose(fi); fclose(fj); fclose(ft);
        printf("DONE items=%u\n", item);
        return 0;
    }

    {
        static char line[32768];
        int first = 1;
        printf("Ling-mini-2.0 (c50 engine). /reset starts over, /exit quits.\n");
        for (;;) {
            c50_buf b = {0};
            c50_ids ids = {0}, gen = {0};
            printer_t pr;
            int stopped;
            double t0;
            printf("\n> "); fflush(stdout);
            if (read_line(line, sizeof line) < 0 || !strcmp(line, "/exit")) break;
            if (!strcmp(line, "/reset")) { c.pos = 0; first = 1; printf("(new conversation)\n"); continue; }
            if (first) {
                c50_msg m[2];
                const char *why;
                int nm = 0;
                if (system) { m[nm].role = C50_SYSTEM; m[nm].content = system; m[nm].len = (uint32_t)strlen(system); ++nm; }
                m[nm].role = C50_USER; m[nm].content = line; m[nm].len = (uint32_t)strlen(line); ++nm;
                if (c50_render(m, (uint32_t)nm, 1, &b, &why)) { printf("(%s)\n", why); continue; }
            } else {
                C50_ADD(&b, "<|role_end|><role>HUMAN</role>");
                c50_buf_add(&b, line, (uint32_t)strlen(line));
                C50_ADD(&b, "<|role_end|><role>ASSISTANT</role>");
            }
            c50_encode(tk, (const uint8_t *)b.s, b.n, &ids);
            t0 = now_s();
            if (feed(&c, ids.v, ids.n)) { printf("(context full at %u tokens: /reset to continue)\n", c.pos); free(b.s); free(ids.v); continue; }
            pr.n = 0; pr.f = stdout;
            generate(&c, max_new, &gen, &stopped, &pr);
            printf("\n[%u prompt + %u new tokens, %.1f s, context %u/%u%s]\n", ids.n, gen.n, now_s() - t0, c.pos, MAXCTX,
                   stopped ? "" : ", stopped at --max-new");
            first = 0;
            free(b.s); free(ids.v); free(gen.v);
        }
    }
    return 0;
}
