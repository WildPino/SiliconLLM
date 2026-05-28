"""
Cartografia del Segnale di Transizione — Phase 40 Test 1
Analizza tmp_trigger_tel.csv e calcola il trigger T = EMA_tok - EMA_bi
"""
import csv
import sys

CSV_PATH = "tmp_trigger_tel.csv"
ALPHA = 0.05
FRAG_WINDOW = 5  # bytes

def compute_ema(values, alpha):
    ema = []
    e = values[0]
    for v in values:
        e = alpha * v + (1.0 - alpha) * e
        ema.append(e)
    return ema

def main():
    loss_bi = []
    loss_tok = []

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss_bi.append(float(row["loss_bi"]))
            loss_tok.append(float(row["loss_lz8"]))

    n = len(loss_bi)

    ema_bi  = compute_ema(loss_bi,  ALPHA)
    ema_tok = compute_ema(loss_tok, ALPHA)

    T = [ema_tok[i] - ema_bi[i] for i in range(n)]

    sorted_T = sorted(T)
    p85_idx  = int(0.85 * n)
    threshold = sorted_T[p85_idx]

    above = [v > threshold for v in T]
    eligible_count = sum(above)

    # Fragmentation: times the signal fires ON then goes OFF within FRAG_WINDOW bytes
    frag_count = 0
    i = 0
    while i < n:
        if above[i]:
            # find the end of this "on" run
            j = i
            while j < n and above[j]:
                j += 1
            run_len = j - i
            if run_len < FRAG_WINDOW:
                frag_count += 1
            i = j
        else:
            i += 1

    print(f"Total bytes analyzed : {n}")
    print(f"T  85th-percentile threshold : {threshold:.6f} BPB")
    print(f"Eligible bytes (above threshold): {eligible_count}  ({100*eligible_count/n:.1f}%)")
    print(f"Fragmented bursts (run < {FRAG_WINDOW} bytes): {frag_count}")

if __name__ == "__main__":
    main()
