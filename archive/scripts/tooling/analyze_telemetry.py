import csv
import sys
from collections import defaultdict

def analyze(csv_path):
    print(f"\\n--- Analyzing {csv_path} ---")
    
    total_see = 0.0
    total_dyn = 0.0
    total_oracle = 0.0
    count = 0
    
    dyn_wins_h_see = 0.0
    dyn_wins_h_dyn = 0.0
    dyn_wins_count_bi = 0.0
    dyn_wins_tail_mass = 0.0
    dyn_wins_count = 0
    
    see_wins_h_see = 0.0
    see_wins_h_dyn = 0.0
    see_wins_count_bi = 0.0
    see_wins_tail_mass = 0.0
    see_wins_count = 0
    
    # Bins for count_bi: 0, 1-5, 6-20, 21-50, 51-200, 200+
    bi_bins = [(0,0), (1,5), (6,20), (21,50), (51,200), (201, 1000000)]
    bi_bin_counts = {b:0 for b in bi_bins}
    bi_bin_dyn_wins = {b:0 for b in bi_bins}
    
    # Bins for H_diff = H_see - H_dyn
    # < -2, -2 to -0.5, -0.5 to 0, 0 to 0.5, 0.5 to 2, > 2
    h_bins = [(-100, -2), (-2, -0.5), (-0.5, 0), (0, 0.5), (0.5, 2), (2, 100)]
    h_bin_counts = {b:0 for b in h_bins}
    h_bin_dyn_wins = {b:0 for b in h_bins}
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loss_see = float(row['loss_see'])
            loss_dyn = float(row['loss_dyn'])
            h_see = float(row['H_see'])
            h_dyn = float(row['H_dyn'])
            count_bi = int(row['count_bi'])
            tail_mass = float(row['tail_mass'])
            
            total_see += loss_see
            total_dyn += loss_dyn
            total_oracle += min(loss_see, loss_dyn)
            count += 1
            
            dyn_won = loss_dyn < loss_see
            if dyn_won:
                dyn_wins_count += 1
                dyn_wins_h_see += h_see
                dyn_wins_h_dyn += h_dyn
                dyn_wins_count_bi += count_bi
                dyn_wins_tail_mass += tail_mass
            else:
                see_wins_count += 1
                see_wins_h_see += h_see
                see_wins_h_dyn += h_dyn
                see_wins_count_bi += count_bi
                see_wins_tail_mass += tail_mass
                
            for b in bi_bins:
                if b[0] <= count_bi <= b[1]:
                    bi_bin_counts[b] += 1
                    if dyn_won: bi_bin_dyn_wins[b] += 1
                    break
                    
            h_diff = h_see - h_dyn
            for b in h_bins:
                if b[0] <= h_diff <= b[1]:
                    h_bin_counts[b] += 1
                    if dyn_won: h_bin_dyn_wins[b] += 1
                    break

    if count == 0: return
    
    print(f"SEE Only BPB: {total_see/count:.4f}")
    print(f"Dyn Only BPB: {total_dyn/count:.4f}")
    print(f"Oracle BPB:   {total_oracle/count:.4f}")
    
    print("\\n--- Averages when Dyn Wins ---")
    if dyn_wins_count > 0:
        print(f"H_see:     {dyn_wins_h_see/dyn_wins_count:.4f}")
        print(f"H_dyn:     {dyn_wins_h_dyn/dyn_wins_count:.4f}")
        print(f"count_bi:  {dyn_wins_count_bi/dyn_wins_count:.2f}")
        print(f"tail_mass: {dyn_wins_tail_mass/dyn_wins_count:.4f}")
        print(f"Samples:   {dyn_wins_count} ({(dyn_wins_count/count)*100:.1f}%)")
        
    print("\\n--- Averages when SEE Wins ---")
    if see_wins_count > 0:
        print(f"H_see:     {see_wins_h_see/see_wins_count:.4f}")
        print(f"H_dyn:     {see_wins_h_dyn/see_wins_count:.4f}")
        print(f"count_bi:  {see_wins_count_bi/see_wins_count:.2f}")
        print(f"tail_mass: {see_wins_tail_mass/see_wins_count:.4f}")
        print(f"Samples:   {see_wins_count} ({(see_wins_count/count)*100:.1f}%)")
        
    print("\\n--- Dyn Win Rate by count_bi bins ---")
    for b in bi_bins:
        if bi_bin_counts[b] > 0:
            rate = bi_bin_dyn_wins[b] / bi_bin_counts[b] * 100
            print(f"{b[0]:>3} - {b[1]:>6}: {rate:5.1f}% (N={bi_bin_counts[b]})")
            
    print("\\n--- Dyn Win Rate by (H_see - H_dyn) bins ---")
    for b in h_bins:
        if h_bin_counts[b] > 0:
            rate = h_bin_dyn_wins[b] / h_bin_counts[b] * 100
            print(f"{b[0]:>5.1f} to {b[1]:>5.1f}: {rate:5.1f}% (N={h_bin_counts[b]})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        analyze("telemetry_prose.csv")
