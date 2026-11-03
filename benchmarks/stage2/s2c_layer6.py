"""Stage 2 part 1, test 3: where the error carried into layer 6 of stage 1 (b) comes from.  CPU only, fp32.

(b) logged the mean square of the error carried into each layer: 7.5 at layers 2-5, 33.5 at layer 6.  The routed
experts' target at layer 5 is 79.3 against 7.5 before it.  This tool follows the teacher (the original bf16 model, in
fp32) and a student (Ling-mini with the routed experts of layers 1-19 replaced by stage 1 (b)'s layer_<i>.pt) over the
first --heldout tokens of stage 1's held-out rows.  Per layer i (D = the residual stream, n(x) = x / rms(x), what every
later RMSNorm reads before its gain):
    carried, carried_n   MS(h_s - h_t), MS(n(h_s) - n(h_t)) at the layer's input (replay: carried must reproduce (b)'s
                         carried_ms within 25%);
    total, total_n       the same at the layer's output;
  layers 1-7 only (DECOMP):
    own, own_n           f_s(h_t) - f_t(h_t): the layer's own error at the teacher's input (the router is unchanged, so
                         both select the same experts with the same weights);
    prop                 MS(f_t(h_s) - f_t(h_t)): the carried error through the original layer;
    experts              own = sum_e d_e, d_e = w_e (E_s,e(u) - E_t,e(u)); expert e's share sum <d_e, own> / sum |own|^2
                         (the shares add to 1), the top 16;
    dims                 each dimension's share of the own error and of the error out of the layer, and the teacher's
                         largest dimensions at the output, the top 16;
    rows                 for the top expert and its top dimension: the down_proj row's group scales against the original
                         groups' max |w| (stage 1 (b)'s outlier rule keeps a group at its max);
  every layer:
    route_same           the fraction of tokens whose 8 experts at the student's input equal those at the teacher's;
  after layer 19:        the final norm and the head on both streams: top-1 agreement and KL(teacher || student), nats.
Arms, in order:
    plant  the teacher with one down_proj row multiplied by 3 at layer 3: the most selected expert at the teacher's
           input and that expert's row of largest sum |w|; layers 1-4; the decomposition must name both (C1);
    tern   C:/SiliconLLM_fast/s1b_tern (stage 1 (b), trained), layers 1-19;
    ph     C:/SiliconLLM_fast/s1b_ph (post-hoc, no training), layers 1-19;
    fix5   tern with layer 5's top expert (tern's own error, this run) restored to the original weights;
    fix15  fix5 with layer 1's top expert restored too.
Numbers only: s2c_verdict.py.

Usage: python -I benchmarks/stage2/s2c_layer6.py --json benchmarks/stage2/results/s2c_layer6.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "stage1"))
import s1t_model as M  # noqa: E402
from s1t_train import EXP, load_corpus, student_pre  # noqa: E402

STUDENTS = {"tern": Path("C:/SiliconLLM_fast/s1b_tern"), "ph": Path("C:/SiliconLLM_fast/s1b_ph")}
FIX = {"fix5": (5,), "fix15": (1, 5)}
DECOMP, PLANT_LAYER, PLANT_FACTOR, TOP, CHUNK = 7, 3, 3.0, 16, 512


def f32(w: dict) -> dict:
    return {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in w.items()}


def nrm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + M.EPS)


def student_layer(w: dict, path: Path, restore: list[int]) -> dict:
    st = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(st["mlp.gate.weight"].float(), w["mlp.gate.weight"]), "the router changed: shares need both"
    out = {**w}
    for p in EXP:
        x = st[f"mlp.experts.{p}"].float()
        for e in restore:
            x[e] = w[f"mlp.experts.{p}"][e]
        out[f"mlp.experts.{p}"] = x
    return out


def top(v: torch.Tensor, total: float) -> list[tuple[int, float]]:
    return [(int(j), round(float(v[j]) / total, 5)) for j in v.argsort(descending=True)[:TOP]] if total else []


def analyse(i: int, wt: dict, ws: dict, ht: list, hs: list, cs, full: bool) -> tuple[dict, list, list]:
    E, D = M.N_EXPERT, M.D
    acc = dict.fromkeys(("carried", "carried_n", "total", "total_n", "own", "own_n", "prop"), 0.0)
    dim_own, dim_out, dim_t = (torch.zeros(D, dtype=torch.float64) for _ in range(3))
    share, uses = torch.zeros(E, dtype=torch.float64), torch.zeros(E, dtype=torch.long)
    same = n_tok = 0
    nt, ns = [], []
    with torch.no_grad():
        for x_t, x_s in zip(ht, hs):
            n = len(x_t)
            c = cs(n)
            acc["carried"] += float((x_s - x_t).pow(2).sum())
            acc["carried_n"] += float((nrm(x_s) - nrm(x_t)).pow(2).sum())
            y_t = M.layer_forward(i, wt, x_t, c, M.Form()).float()
            y_s = M.layer_forward(i, ws, x_s, c, M.Form()).float()
            acc["total"] += float((y_s - y_t).pow(2).sum())
            acc["total_n"] += float((nrm(y_s) - nrm(y_t)).pow(2).sum())
            dim_out += (y_s - y_t).double().pow(2).sum(0)
            dim_t += y_t.double().pow(2).sum(0)
            _, u_t, _ = student_pre(wt, x_t, c)
            _, u_s, _ = student_pre(wt, x_s, c)
            ids_t, wts_t = M.route(u_t, wt["mlp.gate.weight"], wt["mlp.gate.expert_bias"], M.Form())
            ids_s, _ = M.route(u_s, wt["mlp.gate.weight"], wt["mlp.gate.expert_bias"], M.Form())
            same += int((ids_t.sort(-1)[0] == ids_s.sort(-1)[0]).all(-1).sum())
            n_tok += n
            flat = ids_t.reshape(-1)
            uses += torch.bincount(flat, minlength=E)
            if full:
                y_st = M.layer_forward(i, ws, x_t, c, M.Form()).float()
                own = y_st - y_t
                acc["own"] += float(own.pow(2).sum())
                acc["own_n"] += float((nrm(y_st) - nrm(y_t)).pow(2).sum())
                acc["prop"] += float((M.layer_forward(i, wt, x_s, c, M.Form()).float() - y_t).pow(2).sum())
                dim_own += own.double().pow(2).sum(0)
                tok, wf = torch.arange(n).repeat_interleave(ids_t.shape[1]), wts_t.reshape(-1)
                for e in flat.unique().tolist():
                    m = flat == e
                    r = tok[m]
                    d = wf[m].unsqueeze(-1) * (M.mlp(u_t[r], *(ws[f"mlp.experts.{p}"][e] for p in EXP))
                                               - M.mlp(u_t[r], *(wt[f"mlp.experts.{p}"][e] for p in EXP))).float()
                    share[e] += float((d * own[r]).sum())
            nt.append(y_t)
            ns.append(y_s)
    N = n_tok * D
    rec = {k: v / N for k, v in acc.items()}
    rec.update(teacher_ms=float(dim_t.sum()) / N, route_same=same / n_tok, most_used_expert=int(uses.argmax()),
               dims_out_top=top(dim_out, float(dim_out.sum())), dims_teacher_top=top(dim_t, float(dim_t.sum())))
    if full:
        rec.update(experts_top=top(share, acc["own"]), dims_own_top=top(dim_own, float(dim_own.sum())),
                   experts_top4_share=round(float(share.sort(descending=True)[0][:4].sum()) / acc["own"], 5)
                   if acc["own"] else 0.0)
    return rec, nt, ns


def rows_check(path: Path, w: dict, expert: int, dim: int) -> dict:
    s = torch.load(path, map_location="cpu", weights_only=True)["ternary_scale.down_proj"][expert, dim].float()
    mx = w["mlp.experts.down_proj"][expert, dim].abs().view(s.shape[-1], -1).amax(-1)
    return {"expert": expert, "dim": dim, "scale": s.tolist(), "group_max": mx.tolist(),
            "kept_at_max": [bool(abs(float(a) - float(b)) <= 1e-2 * float(b)) for a, b in zip(s, mx)]}


def head_metrics(W, ht: list, hs: list) -> dict:
    g = W.get(["model.norm.weight", "lm_head.weight"])
    gain, head = g["model.norm.weight"].float(), g["lm_head.weight"].float()
    kl = agree = n = 0.0
    with torch.no_grad():
        for x_t, x_s in zip(ht, hs):
            for s in range(0, len(x_t), CHUNK):
                lt = (M.rms(x_t[s:s + CHUNK], gain) @ head.t()).log_softmax(-1)
                ls = (M.rms(x_s[s:s + CHUNK], gain) @ head.t()).log_softmax(-1)
                kl += float((lt.exp() * (lt - ls)).sum())
                agree += float((lt.argmax(-1) == ls.argmax(-1)).sum())
                n += len(lt)
    return {"kl_nats": kl / n, "top1_agreement": agree / n, "tokens": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--heldout", type=int, default=25_000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--arms", default="plant,tern,ph,fix5,fix15")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print("CONFIG", json.dumps({"heldout": a.heldout, "threads": a.threads, "arms": a.arms, "decomp": DECOMP,
                                "layers": M.N_LAYER - 1, "plant": [PLANT_LAYER, PLANT_FACTOR], "fix": FIX,
                                "students": {k: str(v) for k, v in STUDENTS.items()}, "torch": torch.__version__}),
          flush=True)
    _, he = load_corpus(1_500_000, a.heldout)
    W = M.Weights()
    emb = W.get(["model.word_embeddings.weight"])["model.word_embeddings.weight"]
    h0 = [emb[torch.tensor(s)].float() for s in he]
    del emb
    css: dict = {}

    def cs(n):
        if n not in css:
            css[n] = M.rope(n, "cpu")
        return css[n]

    out = {"config": {**vars(a), "json": str(a.json)}, "heldout_tokens": sum(map(len, he)), "arms": {}}
    with torch.no_grad():
        w0 = f32(W.layer(0))
        h0 = [M.layer_forward(0, w0, h, cs(len(h)), M.Form()).float() for h in h0]
        del w0
    for arm in a.arms.split(","):
        t0 = time.time()
        ht, hs, layers = list(h0), list(h0), {}
        restore = {}
        if arm in FIX:
            restore = {i: [out["arms"]["tern"]["layers"][i]["experts_top"][0][0]] for i in FIX[arm]}
            out["arms"][arm] = {"restored": restore}
        last = PLANT_LAYER + 1 if arm == "plant" else M.N_LAYER - 1
        for i in range(1, last + 1):
            wt = f32(W.layer(i))
            if arm == "plant":
                ws = wt
                if i == PLANT_LAYER:
                    probe, _, _ = analyse(i, wt, wt, ht[:8], ht[:8], cs, False)
                    e = probe["most_used_expert"]
                    dn = wt["mlp.experts.down_proj"].clone()
                    k = int(dn[e].abs().sum(-1).argmax())
                    dn[e, k] *= PLANT_FACTOR
                    ws = {**wt, "mlp.experts.down_proj": dn}
                    out["plant"] = {"layer": i, "expert": e, "dim": k, "factor": PLANT_FACTOR}
            else:
                src = STUDENTS["tern" if arm in FIX else arm]
                ws = student_layer(wt, src / f"layer_{i}.pt", restore.get(i, []))
            rec, ht, hs = analyse(i, wt, ws, ht, hs, cs, i <= DECOMP)
            if arm != "plant" and rec.get("experts_top") and rec.get("dims_own_top"):
                e = rec["experts_top"][0][0]
                k = int((ws["mlp.experts.down_proj"][e] - wt["mlp.experts.down_proj"][e]).abs().sum(-1).argmax())
                rec["rows"] = rows_check(STUDENTS["tern" if arm in FIX else arm] / f"layer_{i}.pt", wt, e,
                                         rec["dims_own_top"][0][0])
                rec["rows"]["largest_changed_row"] = k
            layers[i] = rec
            print("LAYER", arm, i, json.dumps({k: rec.get(k) for k in ("carried", "carried_n", "total_n", "own",
                                                                       "own_n", "teacher_ms", "route_same",
                                                                       "experts_top4_share")}),
                  "top_expert", (rec.get("experts_top") or [])[:3], "top_dims", rec["dims_out_top"][:3], flush=True)
            del wt, ws
        res = {**out["arms"].get(arm, {}), "layers": layers}
        if arm != "plant":
            res["head"] = head_metrics(W, ht, hs)
            print("HEAD", arm, json.dumps(res["head"]), flush=True)
        res["seconds"] = round(time.time() - t0, 1)
        out["arms"][arm] = res
        a.json.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
