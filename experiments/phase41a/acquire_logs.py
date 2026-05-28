"""
Phase 41a.2.3: Acquire Logs corpus (64MB sample, real + synthetic mix).

Combines:
  - Real samples from LogHub (16 system types, ~4.25 MB total)
  - Synthetic logs generated with 4 realistic formats (Apache common,
    syslog, JSON structured, nginx access) with randomized parameters.

The synthetic portion is generated deterministically (seed=42) so the
sample is reproducible. Mix is documented in the verdict.

Output: experiments/phase41a/corpora/logs_64mb.txt
"""
import os
import sys
import hashlib
import random
from datetime import datetime, timedelta

LOG_DIRS = ["data/external/log_corpus"]
SAMPLE_DIR = "experiments/phase41a/corpora"
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "logs_64mb.txt")
SAMPLE_SIZE = 64 * 1024 * 1024

os.makedirs(SAMPLE_DIR, exist_ok=True)
random.seed(42)

# ── Synthetic generators ──────────────────────────────────────────────────────

IPS = [f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}" for _ in range(500)]
PATHS = ["/", "/index.html", "/api/users", "/api/products/42", "/login", "/logout",
         "/static/css/main.css", "/static/js/app.js", "/favicon.ico", "/about",
         "/contact", "/api/v1/orders", "/api/v1/items/123", "/search?q=test",
         "/admin/dashboard", "/health", "/metrics", "/docs", "/api/auth/token"]
STATUS = [200]*70 + [304]*10 + [301]*5 + [404]*8 + [500]*4 + [403]*3
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "curl/7.88.1", "Python-urllib/3.11", "Go-http-client/1.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) Safari/17.2",
]
HOSTS = [f"host-{i:03d}" for i in range(50)]
PROCS = ["sshd", "systemd", "kernel", "nginx", "postgres", "kubelet",
         "containerd", "cron", "dbus", "NetworkManager"]
LEVELS = ["INFO"]*60 + ["WARN"]*20 + ["ERROR"]*10 + ["DEBUG"]*10
COMPONENTS = ["auth", "router", "db", "cache", "queue", "scheduler", "api", "worker"]
MESSAGES = [
    "request processed", "connection established", "user authenticated",
    "cache miss", "cache hit", "transaction committed", "transaction rolled back",
    "job started", "job completed", "worker idle", "timeout exceeded",
    "retry attempt", "circuit breaker opened", "rate limit applied",
    "invalid token", "session expired", "config reloaded",
]

def gen_apache(ts):
    ip = random.choice(IPS)
    path = random.choice(PATHS)
    status = random.choice(STATUS)
    size = random.randint(100, 50000) if status == 200 else 0
    ua = random.choice(USER_AGENTS)
    return f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "GET {path} HTTP/1.1" {status} {size} "-" "{ua}"\n'

def gen_syslog(ts):
    host = random.choice(HOSTS)
    proc = random.choice(PROCS)
    pid = random.randint(100, 65000)
    msg = random.choice(MESSAGES)
    return f'{ts.strftime("%b %d %H:%M:%S")} {host} {proc}[{pid}]: {msg}\n'

def gen_json(ts):
    level = random.choice(LEVELS)
    comp = random.choice(COMPONENTS)
    msg = random.choice(MESSAGES)
    uid = random.randint(1000, 999999)
    lat = random.randint(1, 5000)
    return f'{{"ts":"{ts.isoformat()}","level":"{level}","component":"{comp}","user_id":{uid},"latency_ms":{lat},"msg":"{msg}"}}\n'

def gen_nginx(ts):
    ip = random.choice(IPS)
    path = random.choice(PATHS)
    status = random.choice(STATUS)
    size = random.randint(100, 50000) if status == 200 else 0
    ref = random.choice(["-"] + PATHS[:5])
    ua = random.choice(USER_AGENTS)
    rt = random.randint(1, 500) / 1000
    return f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "GET {path} HTTP/2.0" {status} {size} "{ref}" "{ua}" {rt:.3f}\n'

GENERATORS = [gen_apache, gen_syslog, gen_json, gen_nginx]

# ── Step 1: collect real LogHub samples ───────────────────────────────────────

print("Collecting real LogHub samples...")
log_files = []
for base in LOG_DIRS:
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            if ".git" in dirs:
                dirs.remove(".git")
            for f in files:
                if f.endswith(".log"):
                    log_files.append(os.path.join(root, f))
log_files.sort()
print(f"  Found {len(log_files)} real .log files")

# ── Step 2: build sample ──────────────────────────────────────────────────────

print(f"Building {SAMPLE_SIZE}-byte sample at {SAMPLE_FILE}...")
total = 0
real_bytes = 0
synth_bytes = 0

with open(SAMPLE_FILE, "wb") as out:
    # First, write real LogHub samples
    for path in log_files:
        try:
            with open(path, "rb") as f:
                content = f.read()
        except Exception:
            continue
        out.write(content)
        out.write(b"\n\n")
        total += len(content) + 2
        real_bytes += len(content) + 2
        if total >= SAMPLE_SIZE:
            break

    # Then, fill the rest with synthetic logs (rotating 4 formats every ~5000 lines)
    ts = datetime(2026, 1, 1, 0, 0, 0)
    fmt_idx = 0
    lines_in_block = 0
    block_size = 5000  # lines per format block (gives regime structure)
    while total < SAMPLE_SIZE:
        ts += timedelta(seconds=random.randint(0, 3))
        line = GENERATORS[fmt_idx](ts).encode("utf-8")
        out.write(line)
        total += len(line)
        synth_bytes += len(line)
        lines_in_block += 1
        if lines_in_block >= block_size:
            fmt_idx = (fmt_idx + 1) % len(GENERATORS)
            lines_in_block = 0
            out.write(b"\n")  # block separator
            total += 1
            synth_bytes += 1

# Trim to exact size
if total > SAMPLE_SIZE:
    with open(SAMPLE_FILE, "rb") as f:
        data = f.read(SAMPLE_SIZE)
    with open(SAMPLE_FILE, "wb") as f:
        f.write(data)

actual = os.path.getsize(SAMPLE_FILE)
with open(SAMPLE_FILE, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

print(f"\n=== Logs 64MB sample (real + synthetic mix) ===")
print(f"  File:        {SAMPLE_FILE}")
print(f"  Size:        {actual} bytes ({actual/1024/1024:.2f} MB)")
print(f"  Real bytes:  {real_bytes} ({100*real_bytes/actual:.1f}%) from LogHub")
print(f"  Synth bytes: {synth_bytes} ({100*synth_bytes/actual:.1f}%) generated (seed=42)")
print(f"  Formats:     LogHub mix + Apache/syslog/JSON/nginx (rotating blocks)")
print(f"  SHA-256:     {sha[:32]}...")
print("Done.")
