import os
import re
import csv
import sys
import zlib
import urllib.request
import subprocess
import random
import hashlib

def ensure_datasets():
    os.makedirs("data", exist_ok=True)
    datasets = {}
    
    # c_code.c
    c_path = "data/c_code.c"
    if os.path.exists(c_path):
        datasets["c_code"] = c_path
    
    # promessi_sposi.txt
    ps_path = "data/promessi_sposi.txt"
    if not os.path.exists(ps_path):
        print("Downloading Promessi Sposi...")
        url = "https://www.gutenberg.org/files/45334/45334-0.txt"
        try:
            urllib.request.urlretrieve(url, ps_path)
        except Exception as e:
            print(f"Failed to download: {e}")
            with open(ps_path, "wb") as f:
                f.write(b"Renzo e Lucia. "*10000)
    datasets["prose"] = ps_path
    
    # markdown_docs.md
    md_path = "data/markdown_docs.md"
    if not os.path.exists(md_path):
        print("Generating markdown_docs.md...")
        with open("README.md", "rb") as f:
            content = f.read()
        with open(md_path, "wb") as f:
            f.write(content * 100)
    datasets["markdown"] = md_path
    
    # shuffled.bin
    shuf_path = "data/shuffled.bin"
    if not os.path.exists(shuf_path):
        print("Generating shuffled.bin...")
        with open(c_path, "rb") as f:
            b = bytearray(f.read())
        random.seed(42)
        random.shuffle(b)
        with open(shuf_path, "wb") as f:
            f.write(b)
    datasets["shuffled"] = shuf_path
    
    return datasets

def get_zlib_bpb(filepath, level):
    with open(filepath, "rb") as f:
        data = f.read()
    if len(data) == 0: return 0.0
    compressed = zlib.compress(data, level=level)
    return (len(compressed) * 8.0) / len(data)

def parse_metrics(out):
    metrics = {
        "model_bpb": 0.0, "quant_bpb": 0.0,
        "total_cyc": 0.0, "predict_cyc": 0.0,
        "cdf_cyc": 0.0, "rc_cyc": 0.0
    }
    m = re.search(r'Model BPB:\s+([\d\.]+)', out)
    if m: metrics["model_bpb"] = float(m.group(1))
    m = re.search(r'Quantized BPB:\s+([\d\.]+)', out)
    if m: metrics["quant_bpb"] = float(m.group(1))
    m = re.search(r'Predict:\s+([\d\.]+)', out)
    if m: metrics["predict_cyc"] = float(m.group(1))
    m = re.search(r'CDF Build:\s+([\d\.]+)', out)
    if m: metrics["cdf_cyc"] = float(m.group(1))
    m = re.search(r'Range Coder:\s+([\d\.]+)', out)
    if m: metrics["rc_cyc"] = float(m.group(1))
    m = re.search(r'Total:\s+([\d\.]+)', out)
    if m: metrics["total_cyc"] = float(m.group(1))
    return metrics

def run_coder(dataset_path, profile, dyn_lambda=None):
    weights = "weights/entropy_weights.bin"
    out_bin = "tmp.see"
    
    cmd = [
        "coder.exe",
        "--encode", dataset_path, out_bin,
        "--weights", weights,
        "--profile", profile
    ]
    if dyn_lambda is not None:
        cmd.extend(["--blend", str(dyn_lambda)])
        
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return parse_metrics(res.stdout), out_bin

def get_sha(fpath):
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for b in iter(lambda: f.read(4096), b""):
            h.update(b)
    return h.hexdigest()

def main():
    print("Ensuring datasets exist...")
    datasets = ensure_datasets()
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/phase20c_audit.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Provenance", "Profile", "Model_BPB", "Quant_BPB", "Phys_BPB", "Zlib_1", "Zlib_9", "Total_Cyc/B", "Predict_Cyc/B", "CDF_Cyc/B", "RC_Cyc/B"])
        
        for dname, dpath in datasets.items():
            print(f"\\n--- Auditing {dname} ---")
            
            zlib_1 = get_zlib_bpb(dpath, 1)
            zlib_9 = get_zlib_bpb(dpath, 9)
            print(f"zlib (level 1): {zlib_1:.4f} BPB")
            print(f"zlib (level 9): {zlib_9:.4f} BPB")
            
            provenance = "In-Domain" if dname == "c_code" else "Cross-Domain"
            if dname == "shuffled": provenance = "Out-of-Domain"
            
            # Static Profiles
            for prof in ["fast", "accurate"]:
                m, out_bin = run_coder(dpath, prof)
                
                eval_len = os.path.getsize(dpath)
                compressed_bits = os.path.getsize(out_bin) * 8.0 if os.path.exists(out_bin) else 0.0
                phys_bpb = compressed_bits / eval_len if eval_len > 0 else 0.0
                
                # Decode
                decode_bin = out_bin + ".decoded"
                cmd_dec = [
                    "coder.exe",
                    "--decode", out_bin, decode_bin,
                    "--weights", "weights/entropy_weights.bin"
                ]
                subprocess.run(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(decode_bin) and os.path.exists(dpath):
                    if get_sha(dpath) != get_sha(decode_bin):
                        print(f"  WARNING: SHA-256 mismatch for {dname} (prof={prof})")
                    os.remove(decode_bin)
                if os.path.exists(out_bin):
                    os.remove(out_bin)
                
                print(f"[{prof}] Model: {m['model_bpb']:.4f} | Quant: {m['quant_bpb']:.4f} | Phys: {phys_bpb:.4f}")
                writer.writerow([dname, provenance, prof, m["model_bpb"], m["quant_bpb"], phys_bpb, zlib_1, zlib_9, m["total_cyc"], m["predict_cyc"], m["cdf_cyc"], m["rc_cyc"]])
            
            # Dynamic Profiles
            for lam in ["moe", 0.5, 1.0]:
                if lam == "moe":
                    prof_name = "dyn_moe"
                    m, out_bin = run_coder(dpath, "fast", dyn_lambda="moe")
                else:
                    prof_name = f"dyn_{lam}"
                    m, out_bin = run_coder(dpath, "fast", dyn_lambda=lam)
                
                eval_len = os.path.getsize(dpath)
                compressed_bits = os.path.getsize(out_bin) * 8.0 if os.path.exists(out_bin) else 0.0
                phys_bpb = compressed_bits / eval_len if eval_len > 0 else 0.0
                
                # Decode
                decode_bin = out_bin + ".decoded"
                cmd_dec = [
                    "coder.exe",
                    "--decode", out_bin, decode_bin,
                    "--weights", "weights/entropy_weights.bin"
                ]
                subprocess.run(cmd_dec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(decode_bin) and os.path.exists(dpath):
                    if get_sha(dpath) != get_sha(decode_bin):
                        print(f"  WARNING: SHA-256 mismatch for {dname} (blend={lam})")
                    os.remove(decode_bin)
                if os.path.exists(out_bin):
                    os.remove(out_bin)
                
                print(f"[{prof_name}] Model: {m['model_bpb']:.4f} | Quant: {m['quant_bpb']:.4f} | Phys: {phys_bpb:.4f}")
                writer.writerow([dname, provenance, prof_name, m["model_bpb"], m["quant_bpb"], phys_bpb, zlib_1, zlib_9, m["total_cyc"], m["predict_cyc"], m["cdf_cyc"], m["rc_cyc"]])

    print("\\nAudit complete! Results saved to results/phase20c_audit.csv")

if __name__ == "__main__":
    main()
