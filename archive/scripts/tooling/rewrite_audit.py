import os

with open("scripts/audit_compression.py", "r") as f:
    content = f.read()

# 1. Update zlib to compress the entire file (excluding first 2 bytes if we want to be EXACT, but zlib usually does whole file)
zlib_idx = content.find('def get_zlib_bpb(filepath, level):')
run_coder_idx = content.find('def run_coder(dataset_path, profile):')

new_zlib = """def get_zlib_bpb(filepath, level):
    with open(filepath, "rb") as f:
        data = f.read()
    # Compress the whole file
    compressed = zlib.compress(data, level=level)
    return (len(compressed) * 8.0) / len(data) if len(data) > 0 else 0.0

"""
content = content[:zlib_idx] + new_zlib + content[run_coder_idx:]

# 2. Update run_coder and run_coder_dynamic
run_coder_dyn_idx = content.find('def run_coder_dynamic(dataset_path, lam):')
main_idx = content.find('def main():')

new_coders = """def run_coder(dataset_path, profile):
    weights = "weights/entropy_weights.bin"
    out_bin = "tmp.see"
    
    cmd = [
        "coder.exe",
        "--encode", dataset_path, out_bin,
        "--weights", weights,
        "--profile", profile
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout
    
    # Parse metrics
    metrics = {
        "model_bpb": 0.0,
        "quant_bpb": 0.0,
        "total_cyc": 0.0,
        "predict_cyc": 0.0,
        "cdf_cyc": 0.0,
        "rc_cyc": 0.0
    }
    
    m = re.search(r'Model BPB:\\s+([\\d\\.]+)', out)
    if m: metrics["model_bpb"] = float(m.group(1))
    
    m = re.search(r'Quantized BPB:\\s+([\\d\\.]+)', out)
    if m: metrics["quant_bpb"] = float(m.group(1))
    
    m = re.search(r'Predict:\\s+([\\d\\.]+)', out)
    if m: metrics["predict_cyc"] = float(m.group(1))
    
    m = re.search(r'CDF Build:\\s+([\\d\\.]+)', out)
    if m: metrics["cdf_cyc"] = float(m.group(1))
    
    m = re.search(r'Range Coder:\\s+([\\d\\.]+)', out)
    if m: metrics["rc_cyc"] = float(m.group(1))
    
    m = re.search(r'Total:\\s+([\\d\\.]+)', out)
    if m: metrics["total_cyc"] = float(m.group(1))
    
    return metrics, out_bin

def run_coder_dynamic(dataset_path, lam):
    weights = "weights/entropy_weights.bin"
    out_bin = "tmp_dyn.see"
    
    cmd = [
        "coder.exe",
        "--encode", dataset_path, out_bin,
        "--weights", weights,
        "--profile", "fast",
        "--blend", str(lam)
    ]
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout
    
    metrics = {
        "model_bpb": 0.0,
        "quant_bpb": 0.0,
        "total_cyc": 0.0,
        "predict_cyc": 0.0,
        "cdf_cyc": 0.0,
        "rc_cyc": 0.0
    }
    
    m = re.search(r'Model BPB:\\s+([\\d\\.]+)', out)
    if m: metrics["model_bpb"] = float(m.group(1))
    
    m = re.search(r'Quantized BPB:\\s+([\\d\\.]+)', out)
    if m: metrics["quant_bpb"] = float(m.group(1))
    
    m = re.search(r'Predict:\\s+([\\d\\.]+)', out)
    if m: metrics["predict_cyc"] = float(m.group(1))
    
    m = re.search(r'CDF Build:\\s+([\\d\\.]+)', out)
    if m: metrics["cdf_cyc"] = float(m.group(1))
    
    m = re.search(r'Range Coder:\\s+([\\d\\.]+)', out)
    if m: metrics["rc_cyc"] = float(m.group(1))
    
    m = re.search(r'Total:\\s+([\\d\\.]+)', out)
    if m: metrics["total_cyc"] = float(m.group(1))
    
    return metrics, out_bin

"""
content = content[:run_coder_idx] + new_coders + content[main_idx:]

# Update the hash validation inside main
dec_idx = content.find('cmd_dec = [')
dec_end = content.find('res_dec = subprocess.run(cmd_dec')

new_dec = """cmd_dec = [
                "coder.exe",
                "--decode", out_bin, decode_bin,
                "--weights", "weights/entropy_weights.bin"
            ]
            
            """
content = content[:dec_idx] + new_dec + content[dec_end:]

# Update dynamic decoder validation
dec_dyn_idx = content.find('cmd_dec = [', content.find('def run_coder_dynamic'))
if dec_dyn_idx != -1:
    dec_dyn_end = content.find('res_dec = subprocess.run(cmd_dec', dec_dyn_idx)
    content = content[:dec_dyn_idx] + new_dec + content[dec_dyn_end:]

# Add provenance labeling to CSV
csv_header_idx = content.find('writer.writerow(["Dataset", "Profile", "Model_BPB"')
csv_header_end = content.find('for dname, dpath in datasets.items():')

new_csv_header = """writer.writerow(["Dataset", "Provenance", "Profile", "Model_BPB", "Quant_BPB", "Phys_BPB", "Zlib_1", "Zlib_9", "Total_Cyc/B", "Predict_Cyc/B", "CDF_Cyc/B", "RC_Cyc/B"])
        """
content = content[:csv_header_idx] + new_csv_header + content[csv_header_end:]

# Provenance logic in static loop
write_row_idx = content.find('writer.writerow([dname, prof,')
write_row_end = content.find(']', write_row_idx) + 1

new_write_row = """provenance = "In-Domain" if dname == "c_code" else "Cross-Domain"
                writer.writerow([dname, provenance, prof, m["model_bpb"], m["quant_bpb"], m["phys_bpb"], zlib_1, zlib_9, m["total_cyc"], m["predict_cyc"], m["cdf_cyc"], m["rc_cyc"]])"""
content = content[:write_row_idx] + new_write_row + content[write_row_end:]

# Provenance logic in dynamic loop
write_row_dyn_idx = content.find('writer.writerow([dname, f"dyn_{lam}",')
write_row_dyn_end = content.find(']', write_row_dyn_idx) + 1

new_write_row_dyn = """provenance = "In-Domain" if dname == "c_code" else "Cross-Domain"
                if dname == "shuffled": provenance = "Out-of-Domain"
                writer.writerow([dname, provenance, f"dyn_{lam}", m["model_bpb"], m["quant_bpb"], m["phys_bpb"], zlib_1, zlib_9, m["total_cyc"], m["predict_cyc"], m["cdf_cyc"], m["rc_cyc"]])"""
content = content[:write_row_dyn_idx] + new_write_row_dyn + content[write_row_dyn_end:]

with open("scripts/audit_compression.py", "w") as f:
    f.write(content)
print("Audit script updated")
