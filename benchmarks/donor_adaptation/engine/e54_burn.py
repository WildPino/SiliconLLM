"""A deliberate all-core load, used ONLY as the known-positive for E54's throttle witness.

Not a benchmark and not part of any measurement: its single job is to make the CPU hot and
busy so that a witness which cannot see that is exposed as dead before its nulls are trusted
(the planted-control law).  Runs until killed.
"""
import multiprocessing as mp, sys


def spin():
    x = 1.0
    while True:
        for _ in range(1000000):
            x = x * 1.0000001 + 1.0
        if x > 1e300:
            x = 1.0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else mp.cpu_count()
    ps = [mp.Process(target=spin, daemon=True) for _ in range(n)]
    for p in ps:
        p.start()
    print("burning on %d processes" % n, flush=True)
    for p in ps:
        p.join()
