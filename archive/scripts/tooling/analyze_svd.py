import struct
import numpy as np
import os
import sys

def analyze_svd(weights_path):
    print(f"Loading {weights_path}...")
    with open(weights_path, "rb") as f:
        data = f.read()
        
    offset = 0
    header_fmt = "<IIIIffI" # uint32 magic, uint32 version, uint32 feature_dim, uint32 chunk_size, float decay, uint32 seed, float alpha
    # Wait, decay is float, seed is uint32, alpha is float. Let's check struct.
    # struct WeightsFileHeader { uint32_t magic; uint32_t version; uint32_t feature_dim; uint32_t chunk_size; float decay; uint32_t codebook_seed; float alpha; }
    header_fmt = "<IIIIff" # actually 4 uint32, 1 float, 1 uint32, 1 float
    header_fmt = "<IIIIIfI" # wait: magic (I), version (I), feature_dim (I), chunk_size (I), decay (f), seed (I), alpha (f).
    header_fmt = "<IIIIfIf"
    header_size = struct.calcsize(header_fmt)
    header = struct.unpack(header_fmt, data[:header_size])
    offset += header_size
    
    # tringram
    trigram_size = 256 * 256 * 256 * 4
    trigram_logits = np.frombuffer(data[offset:offset+trigram_size], dtype=np.float32).copy()
    offset += trigram_size
    
    # means
    means_size = 192 * 4
    means = np.frombuffer(data[offset:offset+means_size], dtype=np.float32).copy()
    offset += means_size
    
    # stds
    stds_size = 192 * 4
    stds = np.frombuffer(data[offset:offset+stds_size], dtype=np.float32).copy()
    offset += stds_size
    
    # W
    W_size = 256 * 192 * 4
    W = np.frombuffer(data[offset:offset+W_size], dtype=np.float32).copy().reshape(256, 192)
    offset += W_size
    
    # B
    B_size = 256 * 4
    B = np.frombuffer(data[offset:offset+B_size], dtype=np.float32).copy()
    offset += B_size
    
    print("Computing SVD of W...")
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    
    print("\nTop 20 Singular Values:")
    print(S[:20])
    
    total_var = np.sum(S**2)
    for rank in [8, 16, 32, 64]:
        var_explained = np.sum(S[:rank]**2) / total_var
        print(f"Rank {rank}: explains {var_explained*100:.2f}% of variance")
        
        # Create factors
        B_proj = U[:, :rank]  # shape (256, rank)
        A_proj = np.diag(S[:rank]) @ Vh[:rank, :]  # shape (rank, 192)
        
        # Save new file with appended factors
        out_path = f"entropy_weights_factors_r{rank}.bin"
        with open(out_path, "wb") as fout:
            # write original file content exactly as is (so full W is available)
            fout.write(data)
            # append rank (uint32)
            fout.write(struct.pack("<I", rank))
            # append A_proj (rank x 192)
            fout.write(A_proj.astype(np.float32).tobytes())
            # append B_proj (256 x rank)
            fout.write(B_proj.astype(np.float32).tobytes())
        print(f"  Saved {out_path}")

if __name__ == "__main__":
    analyze_svd("entropy_weights.bin")
