#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/see_codec.h"

static void usage(const char* prog) {
    fprintf(stderr,
        "Silicon Entropy Engine — SEE3 codec\n\n"
        "Usage:\n"
        "  %s encode <input> <output.see> --weights <w.bin> [options]\n"
        "  %s decode <input.see> <output>  --weights <w.bin>\n"
        "  %s audit  <input>               --weights <w.bin> [options]\n\n"
        "Encode/Audit options:\n"
        "  --blend moe|<lambda>       blend mode (default: moe)\n"
        "  --eta   <float>            MoE learning rate       (default 0.03)\n"
        "  --share <float>            MoE fixed-share coeff   (default 0.001)\n"
        "  --topk  <int>              SEE top-k candidates    (default 256)\n"
        "  --tail-mode <0|1|2>        tail compensation mode  (default 0)\n"
        "  --speed <name>             speed profile: full | accurate | fast\n"
        "  --profile <name>           alias for --speed (backward compat)\n"
        "  --expert-profile <name>    expert set: general | prose | experimental\n"
        "    general:      LZ6 + TOKPFX  (stable, all domains)\n"
        "    prose:        LZ6 + TOKPFX + TOK_PREV_ELIG  (modern prose: word-transition patterns)\n"
        "    text:         alias for prose (backward compat)\n"
        "    experimental: use manual --tok-* flags\n"
        "  --no-lz               3-expert mode (SEE+UNI+BI only)\n"
        "  --lz-mute             mute LZ arm (ablation)\n"
        "  --lz-key <4|6|8>      LZ context key width in bytes (default 6)\n"
        "  --lz-dual             5-expert mode: LZ4 + LZ8 as separate MoE experts\n"
        "  --tok-prefix          add inside-token prefix expert (TOKPFX)\n"
        "  --tok-prev            add token-transition expert (TOK_PREV)\n"
        "  --tok-prev-elig       gated TOK_PREV (only eligible outside ALNUM/MACRO)\n"
        "  --span-pfx            inline-span expert: backtick/dollar gated (event-driven)\n"
        "  --span-pfx-mute       mute span expert (ablation — uniform distribution)\n\n"
        "Audit-only options:\n"
        "  --eval-start <pct>    start percent (default 0)\n"
        "  --eval-len   <pct>    length percent (default 100)\n"
        "  --telemetry  <path>   write per-byte CSV\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    const char* cmd = argv[1];

    SeeCodecConfig cfg;
    see_codec_config_defaults(&cfg);

    const char* input_path   = NULL;
    const char* output_path  = NULL;
    const char* weights_path = NULL;

    // positional args differ by command
    if (strcmp(cmd, "encode") == 0 || strcmp(cmd, "audit") == 0) {
        if (argc < 3) { usage(argv[0]); return 1; }
        input_path = argv[2];
        if (strcmp(cmd, "encode") == 0) {
            if (argc < 4) { usage(argv[0]); return 1; }
            output_path = argv[3];
        }
    } else if (strcmp(cmd, "decode") == 0) {
        if (argc < 4) { usage(argv[0]); return 1; }
        input_path  = argv[2];  // archive
        output_path = argv[3];
    } else {
        fprintf(stderr, "see: unknown command '%s'\n", cmd);
        usage(argv[0]); return 1;
    }

    // parse options (starting after positional args)
    int opt_start = (strcmp(cmd, "encode") == 0) ? 4
                  : (strcmp(cmd, "decode") == 0) ? 4
                  : 3; // audit

    for (int i = opt_start; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--blend") == 0 && i+1 < argc) {
            const char* v = argv[++i];
            if (strcmp(v, "moe") == 0)  { cfg.use_moe = 1; cfg.blend_lambda = 0; }
            else if (strcmp(v, "auto") == 0) cfg.blend_lambda = -1.0f;
            else cfg.blend_lambda = (float)atof(v);
        } else if (strcmp(argv[i], "--eta") == 0 && i+1 < argc) {
            cfg.moe_eta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--share") == 0 && i+1 < argc) {
            cfg.moe_share = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--topk") == 0 && i+1 < argc) {
            cfg.req_topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tail-mode") == 0 && i+1 < argc) {
            cfg.tail_mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--speed") == 0 && i+1 < argc) {
            cfg.speed_profile = argv[++i];
        } else if (strcmp(argv[i], "--profile") == 0 && i+1 < argc) {
            cfg.profile = argv[++i];  // legacy alias for --speed
        } else if (strcmp(argv[i], "--expert-profile") == 0 && i+1 < argc) {
            cfg.expert_profile = argv[++i];
        } else if (strcmp(argv[i], "--no-lz") == 0) {
            cfg.no_lz = 1;
        } else if (strcmp(argv[i], "--lz-mute") == 0) {
            cfg.lz_mute = 1;
        } else if (strcmp(argv[i], "--lz-key") == 0 && i+1 < argc) {
            cfg.lz_key_bytes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lz-dual") == 0) {
            cfg.lz_dual = 1;
        } else if (strcmp(argv[i], "--tok-prefix") == 0) {
            cfg.tok_prefix = 1;
        } else if (strcmp(argv[i], "--tok-prev") == 0) {
            cfg.tok_prev = 1;
        } else if (strcmp(argv[i], "--tok-prev-mute") == 0) {
            cfg.tok_prev = 1;
            cfg.tok_prev_mute = 1;
        } else if (strcmp(argv[i], "--tok-prev-elig") == 0) {
            cfg.tok_prev = 1;
            cfg.tok_prev_elig = 1;
        } else if (strcmp(argv[i], "--span-pfx") == 0) {
            cfg.span_pfx = 1;
        } else if (strcmp(argv[i], "--span-pfx-mute") == 0) {
            cfg.span_pfx = 1;
            cfg.span_pfx_mute = 1;
        } else if (strcmp(argv[i], "--eval-start") == 0 && i+1 < argc) {
            cfg.eval_start_pct = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--eval-len") == 0 && i+1 < argc) {
            cfg.eval_len_pct = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--telemetry") == 0 && i+1 < argc) {
            cfg.telemetry_path = argv[++i];
        } else if (strcmp(argv[i], "--regime-prior") == 0) {
            cfg.regime_prior = 1;
        } else if (strcmp(argv[i], "--regime-prior-mute") == 0) {
            cfg.regime_prior = 1;
            cfg.regime_prior_mute = 1;
        } else if (strcmp(argv[i], "--regime-credit") == 0) {
            cfg.regime_credit = 1;
        } else {
            fprintf(stderr, "see: unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!weights_path) { fprintf(stderr, "see: --weights is required\n"); return 1; }
    cfg.weights_path = weights_path;

    if (strcmp(cmd, "encode") == 0) {
        // default to MoE if no blend specified
        if (!cfg.use_moe && cfg.blend_lambda == 0.0f) cfg.use_moe = 1;
        printf("Encoding %s -> %s\n", input_path, output_path);
        return see_codec_encode_file(input_path, output_path, &cfg);
    }

    if (strcmp(cmd, "decode") == 0) {
        printf("Decoding %s -> %s\n", input_path, output_path);
        return see_codec_decode_file(input_path, output_path, weights_path);
    }

    // audit
    if (!cfg.use_moe && cfg.blend_lambda == 0.0f) cfg.use_moe = 1;
    SeeAuditResult result;
    memset(&result, 0, sizeof(result));
    int ret = see_codec_audit_file(input_path, &cfg, &result);
    if (ret == 0) see_audit_result_print(&result, input_path);
    return ret;
}
