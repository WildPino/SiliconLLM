import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_moe(csv_path, output_path, title):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Use rolling average to smooth out the weights
    window_size = 4096
    
    df['w_see_smooth'] = df['w_see'].rolling(window=window_size, min_periods=1).mean()
    df['w_uni_smooth'] = df['w_uni'].rolling(window=window_size, min_periods=1).mean()
    df['w_bi_smooth'] = df['w_bi'].rolling(window=window_size, min_periods=1).mean()
    if 'w_lz' in df.columns:
        df['w_lz_smooth'] = df['w_lz'].rolling(window=window_size, min_periods=1).mean()
    else:
        df['w_lz_smooth'] = 0.0

    df['bpb'] = df['loss_actual'].rolling(window=window_size, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top Plot: BPB
    ax1.plot(df['i'], df['bpb'], color='black', linewidth=1.5, label=f'Rolling BPB ({window_size} bytes)')
    ax1.set_title(f"Compression Performance: {title}")
    ax1.set_ylabel("Bits Per Byte")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bottom Plot: Expert Weights
    ax2.plot(df['i'], df['w_see_smooth'], color='blue', linewidth=1.5, label='W_SEE (Static Model)')
    ax2.plot(df['i'], df['w_uni_smooth'], color='red', linewidth=1.5, label='W_UNI (Global Unigram)')
    ax2.plot(df['i'], df['w_bi_smooth'], color='green', linewidth=1.5, label='W_BI (Local Bigram)')
    if 'w_lz' in df.columns:
        ax2.plot(df['i'], df['w_lz_smooth'], color='purple', linewidth=1.5, label='W_LZ (Dictionary Matcher)')
    
    ax2.set_title("Mixture of Experts: Credit Assignment")
    ax2.set_ylabel("Weight (Credit)")
    ax2.set_xlabel("Byte Position")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Multi-domain segment boundaries for data/multi_domain.bin (812000 bytes total).
    # Layout: c_code(235277) + natural_text(218555) + shuffled(229048) + markdown(129120)
    # Offsets verified against file SHA-256=43112b80f6a1f6362b3a97ca45db15207f02301f929658283f8f339449e9ddb9
    DOMAIN_SEGMENTS = [
        ("c_code",       0,      235277),
        ("natural_text", 235277, 453832),
        ("shuffled",     453832, 682880),
        ("markdown",     682880, 812000),
    ]

    if "multi_domain" in csv_path:
        seg_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
        for idx, (name, start, end) in enumerate(DOMAIN_SEGMENTS):
            color = seg_colors[idx % len(seg_colors)]
            mid = (start + end) / 2
            for ax in (ax1, ax2):
                ax.axvspan(start, end, alpha=0.06, color=color)
                if start > 0:
                    ax.axvline(x=start, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
            label_y_ax2 = ax2.get_ylim()[1] * 0.97 if ax2.get_ylim()[1] > 0 else 0.97
            ax2.text(mid, 0.97, name, ha='center', va='top', fontsize=8, color=color,
                     transform=ax2.get_xaxis_transform(), fontweight='bold')
            ax1.text(mid, 0.98, name, ha='center', va='top', fontsize=8, color=color,
                     transform=ax1.get_xaxis_transform(), fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_moe.py <input.csv> <output.png> [title]")
        sys.exit(1)
    
    title = sys.argv[3] if len(sys.argv) > 3 else "MoE Analysis"
    plot_moe(sys.argv[1], sys.argv[2], title)
