r"""E54's clock witness: the CPU's actual/nominal frequency ratio, sampled while a cell runs.

WHY THIS EXISTS.  E53 run 2 measured a 4.64% drop between two identical cells forty minutes
apart on a box at 1.68% foreign occupancy, with load correction making the gap BIGGER rather
than smaller (E53 addendum C.3).  Time-in-run is the surviving explanation and nothing here
measured it.  This is the instrument that does.

WHAT IT READS.  `\Processor Information(_Total)\% Processor Performance` -- the ratio of the
actual clock to the nominal clock, in percent.  Added through PdhAddEnglishCounterW so the
ENGLISH path works on a localised Windows: this box reports its counters in Italian
("Informazioni processore", "Prestazioni processore %") and hardcoding either spelling would
be a hardfit to one machine.

WHAT IT DOES NOT READ, AND WHY THAT IS RECORDED HERE.  Four counters were candidates.  Under a
deliberate 12-process all-core burn on 2026-09-13 (e54_burn.py, the planted control):

    idle          105.44 .. 107.23 %    4008 .. 4069 MHz
    burn   5 s    102.47 %              3887 MHz
    burn  60 s    100.73 %              3821 MHz      <- still falling, not saturated
    recovery 5 s  106.20 %              4029 MHz

    % Performance Limit    100.0 throughout    DEAD on this part
    Performance Limit Flags    0 throughout    DEAD on this part

So the witness is `% Processor Performance` (equivalently `Actual Frequency`), and the two
"why am I limited" counters carry no signal on this Zen 2 part.  That is only known because the
known-positive was run first.

THE TWO COMPONENTS ARE DIFFERENT THINGS.  The step (-3.9 points, instant, recovered in under
five seconds) is all-core versus single-core boost and is a LOAD effect.  The decay after it
(-1.70 points over the next 55 seconds) is the soak.  A reading taken between cells measures
the first and tells you nothing about the second, so this samples DURING a cell only.
"""
import ctypes
from ctypes import wintypes

pdh = ctypes.WinDLL("pdh.dll")

PDH_FMT_DOUBLE = 0x00000200
COUNTER_PATH = r"\Processor Information(_Total)\% Processor Performance"


class _FmtUnion(ctypes.Union):
    _fields_ = [("longValue", wintypes.LONG),
                ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_longlong),
                ("AnsiStringValue", ctypes.c_char_p),
                ("WideStringValue", ctypes.c_wchar_p)]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("u", _FmtUnion)]


class ClockWitness(object):
    """Open once, sample many times.  The counter is a rate and needs two collections."""

    def __init__(self, path=COUNTER_PATH):
        self.q = wintypes.HANDLE()
        self.c = wintypes.HANDLE()
        st = pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.q))
        if st != 0:
            raise OSError("PdhOpenQueryW failed 0x%08X" % (st & 0xFFFFFFFF))
        # English path on a localised box -- see the module docstring.
        st = pdh.PdhAddEnglishCounterW(self.q, path, 0, ctypes.byref(self.c))
        if st != 0:
            pdh.PdhCloseQuery(self.q)
            raise OSError("PdhAddEnglishCounterW(%s) failed 0x%08X" % (path, st & 0xFFFFFFFF))
        pdh.PdhCollectQueryData(self.q)      # prime; the first collection has no delta

    def sample(self):
        """Percent of nominal clock, or nan if this collection could not be formatted."""
        if pdh.PdhCollectQueryData(self.q) != 0:
            return float("nan")
        v = PDH_FMT_COUNTERVALUE()
        st = pdh.PdhGetFormattedCounterValue(self.c, PDH_FMT_DOUBLE, None, ctypes.byref(v))
        if st != 0:
            return float("nan")
        return float(v.u.doubleValue)

    def close(self):
        if self.q:
            pdh.PdhCloseQuery(self.q)
            self.q = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == "__main__":
    import sys, time
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with ClockWitness() as w:
        for i in range(n):
            time.sleep(1.0)
            print("%2d  %7.2f %% of nominal" % (i + 1, w.sample()))
