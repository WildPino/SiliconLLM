/*
 * C50-A8 chat template: the Ling-mini-2.0 Jinja chat_template (tokenizer_config.json @ a810f64) for system, user
 * and assistant messages with thinking_option = 'off', as rendered by transformers apply_chat_template:
 *   '<role>SYSTEM</role>' [+ first system content + '\n'] + 'detailed thinking off<|role_end|>'
 *   user:                     '<role>HUMAN</role>' + content + '<|role_end|>'
 *   system (not the first):   '<role>SYSTEM</role>' + content + '<|role_end|>'
 *   assistant:                '<role>ASSISTANT</role>' + content + '<|role_end|>'
 *   add_generation_prompt:    '<role>ASSISTANT</role>'
 * Out of scope (refused, not approximated): tools, tool calls and responses, and assistant content containing
 * '</think>' (the template would move it into a reasoning block).
 */
#ifndef C50_CHAT_TEMPLATE_H
#define C50_CHAT_TEMPLATE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { C50_SYSTEM = 0, C50_USER = 1, C50_ASSISTANT = 2 };
typedef struct { int role; const char *content; uint32_t len; } c50_msg;
typedef struct { char *s; uint32_t n, cap; } c50_buf;

static void c50_buf_add(c50_buf *b, const char *s, uint32_t n) {
    if (b->n + n + 1 > b->cap) { while (b->n + n + 1 > b->cap) b->cap = b->cap ? 2 * b->cap : 256; b->s = realloc(b->s, b->cap); }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}
#define C50_ADD(b, lit) c50_buf_add((b), (lit), (uint32_t)(sizeof(lit) - 1))

static int c50_has(const char *s, uint32_t n, const char *needle) {
    uint32_t k = (uint32_t)strlen(needle), i;
    for (i = 0; i + k <= n; ++i) if (!memcmp(s + i, needle, k)) return 1;
    return 0;
}

/* returns 0, or -1 with *why set when the conversation is outside the supported scope */
static int c50_render(const c50_msg *m, uint32_t n, int add_generation_prompt, c50_buf *out, const char **why) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        if (m[i].role < C50_SYSTEM || m[i].role > C50_ASSISTANT) { *why = "unsupported role"; return -1; }
        if (m[i].role == C50_ASSISTANT && c50_has(m[i].content, m[i].len, "</think>")) { *why = "assistant content with </think>"; return -1; }
    }
    C50_ADD(out, "<role>SYSTEM</role>");
    if (n && m[0].role == C50_SYSTEM) { c50_buf_add(out, m[0].content, m[0].len); C50_ADD(out, "\n"); }
    C50_ADD(out, "detailed thinking off<|role_end|>");
    for (i = 0; i < n; ++i) {
        if (m[i].role == C50_USER) { C50_ADD(out, "<role>HUMAN</role>"); }
        else if (m[i].role == C50_SYSTEM) { if (i == 0) continue; C50_ADD(out, "<role>SYSTEM</role>"); }
        else C50_ADD(out, "<role>ASSISTANT</role>");
        c50_buf_add(out, m[i].content, m[i].len);
        C50_ADD(out, "<|role_end|>");
    }
    if (add_generation_prompt) C50_ADD(out, "<role>ASSISTANT</role>");
    return 0;
}

#endif
