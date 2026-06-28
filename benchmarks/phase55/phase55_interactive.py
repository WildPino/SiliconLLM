#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN_DIR = os.path.join(ROOT, "bin")
EXE_PATH = os.path.join(BIN_DIR, "phase55_generator.exe")
SRC_PATH = os.path.join(ROOT, "benchmarks", "phase55", "phase55_generator.c")

def ensure_compiled():
    """Ensure the C inference engine is compiled."""
    if not os.path.exists(EXE_PATH):
        print("C inference binary not found. Compiling phase55_generator.c...", flush=True)
        if not os.path.exists(BIN_DIR):
            os.makedirs(BIN_DIR, exist_ok=True)
        
        # Build command using gcc
        cmd = [
            "gcc", "-O3", "-march=native", "-mavx2", "-mfma",
            SRC_PATH, "-o", EXE_PATH, "-lm"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Compilation successful!", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"Error compiling C engine: {e.stderr.decode('utf-8', errors='ignore')}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("Error: 'gcc' compiler not found. Please install gcc (MinGW/Clang) or make sure it is in your PATH.", file=sys.stderr)
            sys.exit(1)

def generate_continuation_via_c(prompt, max_bytes=500, temp=0.65, rep_pen=1.2, rep_win=128):
    """Run the C generator sub-process to generate and stream text in real-time."""
    ensure_compiled()
    
    cmd = [
        EXE_PATH,
        "--prompt", prompt,
        "--gen-bytes", str(max_bytes),
        "--temp", str(temp),
        "--rep", str(rep_pen),
        "--rep-win", str(rep_win),
        "--exp", "fast"
    ]
    
    print("\n--- CONTINUATION ---")
    sys.stdout.write(prompt)
    sys.stdout.flush()
    
    # We run the process and stream its stdout line/char by char
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, # Suppress the load/footprint warnings printed to stderr
            bufsize=0 # Unbuffered
        )
        
        while True:
            # Read single bytes from the C program output
            char_bytes = proc.stdout.read(1)
            if not char_bytes:
                break
            try:
                sys.stdout.write(char_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                # If it's a partial UTF-8 sequence, write raw to buffer
                sys.stdout.buffer.write(char_bytes)
            sys.stdout.flush()
            
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[Generation interrupted]")
        
    print("\n--------------------\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="", help="Prompt to start generation from. If empty, starts interactive mode.")
    ap.add_argument("--temp", type=float, default=0.65, help="Sampling temperature")
    ap.add_argument("--rep-pen", type=float, default=1.2, help="Repetition penalty (1.0 = off)")
    ap.add_argument("--rep-win", type=int, default=128, help="Repetition penalty window size")
    ap.add_argument("--max-bytes", type=int, default=1000, help="Maximum bytes to generate")
    a = ap.parse_args()
    
    # Compile C executable if needed
    ensure_compiled()
    
    if a.prompt:
        generate_continuation_via_c(a.prompt, max_bytes=a.max_bytes, temp=a.temp, rep_pen=a.rep_pen, rep_win=a.rep_win)
    else:
        print("=== SiliconLLM Interactive Prompt (Pure CPU, C Engine) ===")
        print("Type a prompt and press Enter. Type 'exit' or Ctrl+C to stop.")
        print(f"Config: temp={a.temp}, rep_pen={a.rep_pen}, rep_win={a.rep_win}, max_bytes={a.max_bytes}")
        print("=========================================================")
        while True:
            try:
                prompt = input("\nPrompt > ")
                if prompt.strip().lower() == "exit":
                    break
                if not prompt.strip():
                    continue
                generate_continuation_via_c(prompt, max_bytes=a.max_bytes, temp=a.temp, rep_pen=a.rep_pen, rep_win=a.rep_win)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()
