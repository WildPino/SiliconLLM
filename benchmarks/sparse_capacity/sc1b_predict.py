#!/usr/bin/env python3
# Sparse Capacity, phase 1b (D-M): desk prediction of r* from the byte model (sc1b_model.py).
#   python benchmarks/sparse_capacity/sc1b_predict.py            desk 1: the assumptions registered in the protocol
# The runner calls predict() with the MEASURED bandwidths (desk 2) right after the controls and before any arm is
# timed, so a miss can be split into "wrong bandwidth" and "wrong model".
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc1b_model as M  # noqa: E402

F_CPU = 200.0                                   # FLOP/ns effective on the arm shapes: an assumption (LITERATURE.md §3)
RS = [2 ** (k / 8) for k in range(0, 57)]       # 1 .. 128, fine grid for the desk crossover
H_ALL = (128, 8, 1)


def table(bw_ram, bw_h2d, bw_bidir, f_cpu=F_CPU):
    rows = []
    for pair, (ca, sa) in M.PAIRS.items():
        for h in H_ALL:
            rho = M.rho_of(ca, h)                       # h1 reads density 1 (review obj. 4)
            ts = [M.desk_str(sa, h, r, rho, bw_ram, bw_h2d, bw_bidir) for r in RS]
            ti = [M.desk_str(sa, h, r, rho, bw_ram, bw_h2d, bw_bidir, ideal=True) for r in RS]
            out = dict(pair=pair, h=h)
            for model, ov in (("add", False), ("ovl", True)):
                tc = [M.desk_cpu(ca, h, r, rho, bw_ram, f_cpu, overlap=ov) for r in RS]
                out[model] = M.crossover(RS, tc, ts)
            out["ideal"] = M.crossover(RS, [M.desk_cpu(ca, h, r, rho, bw_ram, f_cpu) for r in RS], ti)
            out["cpu_r1"] = M.desk_cpu(ca, h, 1, rho, bw_ram, f_cpu)
            out["str_r1"] = ts[0]
            rows.append(out)
    return rows


def predict(ctl):
    """desk 2: measured host RAM (read-modify-write), pinned H2D and bidirectional PCIe; GB/s == bytes/ns."""
    return table(ctl["ram"]["rmw"]["med"], ctl["pcie"]["h2d"]["med"], ctl["pcie"]["bidir_per_dir"]["med"])


def fmt_r(v):
    val, tag = v
    return {"<": "<1", ">": ">128"}.get(tag, f"{val:.1f}")


def print_table(rows, title="DESK"):
    print(f"{title}: r* = tokens per expert per step below which the CPU path is cheaper (add = memory+compute, "
          f"ovl = max of the two)")
    print(f"  {'pair':<16} {'h':>4} {'r* add':>7} {'r* ovl':>7} {'add/ideal':>9} {'cpu@1':>7} {'str@1':>7}  ns/param")
    for o in rows:
        idl = fmt_r(tuple(o["ideal"])) if "ideal" in o else "-"
        print(f"  {o['pair']:<16} {o['h']:>4} {fmt_r(o['add']):>7} {fmt_r(o['ovl']):>7} {idl:>9} {o['cpu_r1']:7.3f} "
              f"{o['str_r1']:7.3f}")


def runtime_estimate(bw_ram=30.0, bw_h2d=25.0, bw_bidir=20.0, p_bank=201326592, steps=5, rounds=3, overhead_s=4.0):
    """Hours for the full grid from the desk model (+ a fixed per-cell overhead for allocation and problem setup)."""
    tot = 0.0
    for arm, (path, fmt, opt, fused, rk) in M.ARMS.items():
        for h in H_ALL:
            if h == 1 and arm in ("C3d", "S2d"):
                continue
            rho = M.rho_of(arm, h)
            for r in (1, 2, 4, 8, 16, 32, 64, 128):
                ns = (M.desk_cpu(arm, h, r, rho, bw_ram, F_CPU) if path == "cpu"
                      else M.desk_str(arm, h, r, rho, bw_ram, bw_h2d, bw_bidir))
                tot += ns * p_bank * 1e-9 * steps + overhead_s
    return tot * rounds / 3600.0


if __name__ == "__main__":
    for name, bw_pcie in (("PCIe 4.0 x16 (25 GB/s per direction, 20 both ways)", (25.0, 20.0)),
                          ("PCIe 3.0 x16 (12 GB/s per direction, 10 both ways)", (12.0, 10.0))):
        print_table(table(30.0, *bw_pcie), title=f"DESK 1, host RAM 30 GB/s, CPU 200 GFLOP/s, {name}")
    print(f"runtime estimate (desk, gen4): {runtime_estimate():.2f} h; (gen3): {runtime_estimate(bw_h2d=12, bw_bidir=10):.2f} h")
