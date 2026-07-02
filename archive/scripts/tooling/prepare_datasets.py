import os
import random

def prepare_c_code():
    out_path = 'data/c_code.c'
    print(f"Preparing {out_path}...")
    source_dirs = ['src', 'benchmarks']
    extensions = ['.c', '.h']
    
    with open(out_path, 'wb') as outfile:
        for d in source_dirs:
            if not os.path.exists(d): continue
            for root, _, files in os.walk(d):
                for f in files:
                    # Ignore executables and other binaries
                    if not any(f.endswith(ext) for ext in extensions): continue
                    
                    filepath = os.path.join(root, f)
                    with open(filepath, 'rb') as infile:
                        outfile.write(infile.read())
                        outfile.write(b'\n\n')
    print(f"Done. Size: {os.path.getsize(out_path)} bytes")

def prepare_markdown():
    out_path = 'data/markdown_docs.md'
    print(f"Preparing {out_path}...")
    source_dirs = ['.']
    extensions = ['.md']
    
    with open(out_path, 'wb') as outfile:
        for d in source_dirs:
            for root, _, files in os.walk(d):
                # Ignore .gemini or .git
                if '.git' in root or '.gemini' in root: continue
                for f in files:
                    if not any(f.endswith(ext) for ext in extensions): continue
                    
                    filepath = os.path.join(root, f)
                    with open(filepath, 'rb') as infile:
                        outfile.write(infile.read())
                        outfile.write(b'\n\n')
    print(f"Done. Size: {os.path.getsize(out_path)} bytes")

def prepare_shuffled():
    in_path = 'data/promessi_sposi.txt'
    out_path = 'data/promessi_sposi_shuffled.txt'
    print(f"Preparing {out_path}...")
    
    with open(in_path, 'rb') as f:
        content = bytearray(f.read())
        
    random.seed(42) # fixed seed
    random.shuffle(content)
    
    with open(out_path, 'wb') as f:
        f.write(content)
        
    print(f"Done. Size: {os.path.getsize(out_path)} bytes")

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
        
    prepare_c_code()
    prepare_markdown()
    prepare_shuffled()
