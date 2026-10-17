/*
 * C50-A8 tokenizer driver for the verification (c50_a8_verify.py).
 *   --model GGUF --encode IN --out OUT   IN: records [u32 len][len bytes UTF-8]; OUT: [u32 n][n x i32 ids]
 *   --model GGUF --decode IN --out OUT   IN: [u32 n][n x i32 ids];            OUT: [u32 len][len bytes]
 *   --model GGUF --template IN --out OUT IN: [u32 n_msgs]{[u32 role][u32 len][bytes]}; OUT: [u32 len][rendered
 *                                        bytes][u32 n][ids], rendered with add_generation_prompt (c50_chat_template.h)
 * Prints one CONFIG line (tokens, added tokens, merges, Unicode data version) and DONE with the record count.
 * Build: clang -O2 -o bin/c50_tok_cli.exe c50_tok_cli.c
 */
#include "c50_tok.h"
#include "c50_chat_template.h"

int main(int argc, char **argv) {
    const char *model = NULL, *enc = NULL, *dec = NULL, *tpl = NULL, *out = NULL;
    gguf g;
    c50_tok t;
    FILE *fi, *fo;
    uint32_t n, recs = 0, i, merges = 0;
    for (i = 1; i < (uint32_t)argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < (uint32_t)argc) model = argv[++i];
        else if (!strcmp(argv[i], "--encode") && i + 1 < (uint32_t)argc) enc = argv[++i];
        else if (!strcmp(argv[i], "--decode") && i + 1 < (uint32_t)argc) dec = argv[++i];
        else if (!strcmp(argv[i], "--template") && i + 1 < (uint32_t)argc) tpl = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < (uint32_t)argc) out = argv[++i];
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model || !out || (!!enc + !!dec + !!tpl) != 1) {
        fprintf(stderr, "usage: --model GGUF (--encode IN | --decode IN | --template IN) --out OUT\n");
        return 2;
    }
    gguf_open(&g, model);
    c50_tok_load(&t, &g);
    for (i = 0; i < t.mcap; ++i) merges += t.mkey[i] != UINT64_MAX;
    printf("CONFIG tokenizer=c50_tok model=%s tokens=%u added=%u merges=%u unidata=%s mode=%s\n", model, t.n, t.n_added,
           merges, C50_UNIDATA_VERSION, enc ? "encode" : dec ? "decode" : "template");
    fi = fopen(enc ? enc : dec ? dec : tpl, "rb");
    fo = fopen(out, "wb");
    if (!fi || !fo) { fprintf(stderr, "cannot open files\n"); return 2; }
    while (fread(&n, 4, 1, fi) == 1) {
        if (tpl) {
            c50_msg *m = calloc((size_t)n + 1, sizeof *m);
            c50_buf b = {0};
            c50_ids ids = {0};
            const char *why = "";
            uint32_t k;
            for (k = 0; k < n; ++k) {
                uint32_t role, len;
                char *c;
                if (fread(&role, 4, 1, fi) != 1 || fread(&len, 4, 1, fi) != 1) { fprintf(stderr, "short record\n"); return 2; }
                c = malloc((size_t)len + 1);
                if (len && fread(c, 1, len, fi) != len) { fprintf(stderr, "short record\n"); return 2; }
                m[k].role = (int)role; m[k].content = c; m[k].len = len;
            }
            if (c50_render(m, n, 1, &b, &why)) { fprintf(stderr, "template: %s\n", why); return 2; }
            c50_encode(&t, (const uint8_t *)b.s, b.n, &ids);
            fwrite(&b.n, 4, 1, fo);
            if (b.n) fwrite(b.s, 1, b.n, fo);
            fwrite(&ids.n, 4, 1, fo);
            if (ids.n) fwrite(ids.v, 4, ids.n, fo);
            for (k = 0; k < n; ++k) free((void *)m[k].content);
            free(m); free(b.s); free(ids.v);
        } else if (enc) {
            uint8_t *s = malloc((size_t)n + 1);
            c50_ids ids = {0};
            if (n && fread(s, 1, n, fi) != n) { fprintf(stderr, "short record\n"); return 2; }
            c50_encode(&t, s, n, &ids);
            fwrite(&ids.n, 4, 1, fo);
            if (ids.n) fwrite(ids.v, 4, ids.n, fo);
            free(s); free(ids.v);
        } else {
            int32_t *ids = malloc(4 * ((size_t)n + 1));
            uint8_t *s;
            uint32_t l, k;
            if (n && fread(ids, 4, n, fi) != n) { fprintf(stderr, "short record\n"); return 2; }
            for (k = 0; k < n; ++k) if (ids[k] < 0 || (uint32_t)ids[k] >= t.n) { fprintf(stderr, "id out of range\n"); return 2; }
            l = c50_decode(&t, ids, n, &s);
            fwrite(&l, 4, 1, fo);
            if (l) fwrite(s, 1, l, fo);
            free(ids); free(s);
        }
        ++recs;
    }
    fclose(fi); fclose(fo);
    printf("DONE records=%u\n", recs);
    return 0;
}
