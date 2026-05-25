import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_telemetry(csv_path, manifest_path, out_png):
    print(f"Plotting {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate rolling BPB (window = 4096 bytes)
    df['rolling_bpb'] = df['loss_actual'].rolling(window=4096, min_periods=100).mean()
    
    # Read manifest for vertical lines
    offsets = []
    with open(manifest_path, "r") as f:
        for line in f:
            if "->" in line:
                parts = line.split(":")
                name = parts[0].strip()
                range_str = parts[1].strip()
                start_str, end_str = range_str.split("->")
                start = int(start_str.strip())
                offsets.append((name, start))
                
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Subplot 1: Weights
    ax1.plot(df['i'], df['w_see'], label='w_see (Static)', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(df['i'], df['w_uni'], label='w_uni (Dynamic Unigram)', color='red', alpha=0.7, linewidth=1)
    ax1.plot(df['i'], df['w_bi'], label='w_bi (Dynamic Bigram)', color='green', alpha=0.7, linewidth=1)
    ax1.set_ylabel('MoE Weight')
    ax1.set_title(f'Online Expert Mixing Weights ({os.path.basename(csv_path)})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Subplot 2: Rolling BPB
    ax2.plot(df['i'], df['rolling_bpb'], label='Rolling BPB (4KB)', color='black', linewidth=1.5)
    ax2.set_ylabel('BPB')
    ax2.set_xlabel('Byte Index')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 10)
    
    # Vertical lines for domain shifts
    for name, start in offsets:
        ax1.axvline(x=start, color='black', linestyle='--', alpha=0.8)
        ax2.axvline(x=start, color='black', linestyle='--', alpha=0.8)
        # Add text label slightly after the line
        ax1.text(start + 50000, 0.9, name, rotation=0, verticalalignment='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    import glob
    os.makedirs("results", exist_ok=True)
    plot_telemetry("results/multi_domain_1.bin_telemetry.csv", "results/multi_domain_1.bin_manifest.txt", "results/multi_domain_1_plot.png")
    plot_telemetry("results/multi_domain_2.bin_telemetry.csv", "results/multi_domain_2.bin_manifest.txt", "results/multi_domain_2_plot.png")
