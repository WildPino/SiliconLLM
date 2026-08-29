invocations aggregated: [1, 2, 3, 4, 5] (n=5)
rows: 200

### d5cd control - RESIDENT (single replica, block fits L3)

| shape | stride pad | thr (ach) | arm | reps/inv | replicas | matvecs/s (mean of inv means) | between-inv CV | within-inv CV | moved B | **moved GB/s** | charged B | charged GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama70b_kv | +0 | 1 (1) | byte | 4768 | 1 | **3099.6** | 8.2% | 25.1% | 4194304 | **13.00** | 4194304 | 13.00 |
| llama70b_kv | +0 | 1 (1) | nibble | 9536 | 1 | **3666.3** | 6.0% | 20.2% | 2097152 | **7.69** | 2097152 | 7.69 |
| llama70b_kv | +0 | 6 (6) | byte | 4768 | 1 | **9627.5** | 13.3% | 52.6% | 4194304 | **40.38** | 4194304 | 40.38 |
| llama70b_kv | +0 | 6 (6) | nibble | 9536 | 1 | **10531.5** | 21.0% | 72.6% | 2097152 | **22.09** | 2097152 | 22.09 |

**Nibble / byte, per cell — the A1.3 discriminator:**

| shape | stride pad | thr | matvecs/s ratio | (range over inv) | moved GB/s ratio | (range over inv) | reading |
|---|---|---|---|---|---|---|---|
| llama70b_kv | +0 | 1 | **1.183x** | 1.032-1.368 | **0.591x** | 0.516-0.684 | JOINTLY limited (intermediate) |
| llama70b_kv | +0 | 6 | **1.094x** | 0.673-1.482 | **0.547x** | 0.337-0.741 | compute/port-limited |

### d5cd control - STREAMED (pool >> L3)

| shape | stride pad | thr (ach) | arm | reps/inv | replicas | matvecs/s (mean of inv means) | between-inv CV | within-inv CV | moved B | **moved GB/s** | charged B | charged GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama70b_kv | +0 | 1 (1) | byte | 4768 | 128 | **1503.5** | 6.5% | 13.1% | 4194304 | **6.31** | 4194304 | 6.31 |
| llama70b_kv | +0 | 1 (1) | nibble | 9536 | 256 | **2318.8** | 8.9% | 16.1% | 2097152 | **4.86** | 2097152 | 4.86 |
| llama70b_kv | +0 | 6 (6) | byte | 4768 | 128 | **4445.9** | 21.0% | 56.5% | 4194304 | **18.65** | 4194304 | 18.65 |
| llama70b_kv | +0 | 6 (6) | nibble | 9536 | 256 | **6842.2** | 17.8% | 99.0% | 2097152 | **14.35** | 2097152 | 14.35 |

**Nibble / byte, per cell — the A1.3 discriminator:**

| shape | stride pad | thr | matvecs/s ratio | (range over inv) | moved GB/s ratio | (range over inv) | reading |
|---|---|---|---|---|---|---|---|
| llama70b_kv | +0 | 1 | **1.542x** | 1.279-1.798 | **0.771x** | 0.639-0.899 | JOINTLY limited (intermediate) |
| llama70b_kv | +0 | 6 | **1.539x** | 1.012-2.412 | **0.769x** | 0.506-1.206 | JOINTLY limited (intermediate) |

### Main sweep - stride pad +0 B (every Mpad a power of two / multiple of 4096)

| shape | stride pad | thr (ach) | arm | reps/inv | replicas | matvecs/s (mean of inv means) | between-inv CV | within-inv CV | moved B | **moved GB/s** | charged B | charged GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama70b_down | +0 | 1 (1) | byte | 170 | 5 | **17.1** | 9.3% | 9.4% | 117440512 | **2.01** | 117440512 | 2.01 |
| llama70b_down | +0 | 1 (1) | nibble | 340 | 10 | **22.7** | 17.2% | 16.3% | 58720256 | **1.33** | 58720256 | 1.33 |
| llama70b_down | +0 | 6 (6) | byte | 170 | 5 | **62.2** | 19.7% | 15.6% | 117440512 | **7.30** | 117440512 | 7.30 |
| llama70b_down | +0 | 6 (6) | nibble | 340 | 10 | **79.7** | 28.1% | 25.7% | 58720256 | **4.68** | 58720256 | 4.68 |
| llama70b_gate_up | +0 | 1 (1) | byte | 170 | 5 | **17.3** | 9.2% | 9.1% | 117440512 | **2.03** | 117440512 | 2.03 |
| llama70b_gate_up | +0 | 1 (1) | nibble | 340 | 10 | **71.3** | 12.1% | 14.2% | 58720256 | **4.19** | 58720256 | 4.19 |
| llama70b_gate_up | +0 | 6 (6) | byte | 170 | 5 | **61.5** | 18.5% | 17.2% | 117440512 | **7.22** | 117440512 | 7.22 |
| llama70b_gate_up | +0 | 6 (6) | nibble | 340 | 10 | **173.3** | 10.3% | 16.8% | 58720256 | **10.17** | 58720256 | 10.17 |
| llama70b_kv | +0 | 1 (1) | byte | 4768 | 128 | **1473.5** | 7.2% | 15.9% | 4194304 | **6.18** | 4194304 | 6.18 |
| llama70b_kv | +0 | 1 (1) | nibble | 9536 | 256 | **2312.0** | 9.5% | 22.5% | 2097152 | **4.85** | 2097152 | 4.85 |
| llama70b_kv | +0 | 6 (6) | byte | 4768 | 128 | **4207.2** | 24.0% | 106.4% | 4194304 | **17.65** | 4194304 | 17.65 |
| llama70b_kv | +0 | 6 (6) | nibble | 9536 | 256 | **6605.0** | 28.7% | 136.7% | 2097152 | **13.85** | 2097152 | 13.85 |
| llama70b_qo | +0 | 1 (1) | byte | 596 | 16 | **66.2** | 10.9% | 10.2% | 33554432 | **2.22** | 33554432 | 2.22 |
| llama70b_qo | +0 | 1 (1) | nibble | 1192 | 32 | **135.0** | 14.2% | 14.9% | 16777216 | **2.26** | 16777216 | 2.26 |
| llama70b_qo | +0 | 6 (6) | byte | 596 | 16 | **224.2** | 21.8% | 17.2% | 33554432 | **7.52** | 33554432 | 7.52 |
| llama70b_qo | +0 | 6 (6) | nibble | 1192 | 32 | **489.6** | 19.8% | 32.3% | 16777216 | **8.21** | 16777216 | 8.21 |

**Nibble / byte, per cell — the A1.3 discriminator:**

| shape | stride pad | thr | matvecs/s ratio | (range over inv) | moved GB/s ratio | (range over inv) | reading |
|---|---|---|---|---|---|---|---|
| llama70b_down | +0 | 1 | **1.328x** | 0.889-1.712 | **0.664x** | 0.444-0.856 | JOINTLY limited (intermediate) |
| llama70b_down | +0 | 6 | **1.282x** | 0.545-2.223 | **0.641x** | 0.273-1.111 | JOINTLY limited (intermediate) |
| llama70b_gate_up | +0 | 1 | **4.127x** | 3.339-5.227 | **2.064x** | 1.669-2.614 | OFF-TABLE: byte arm access-limited, not byte-limited |
| llama70b_gate_up | +0 | 6 | **2.818x** | 2.216-4.291 | **1.409x** | 1.108-2.146 | OFF-TABLE: byte arm access-limited, not byte-limited |
| llama70b_kv | +0 | 1 | **1.569x** | 1.276-1.861 | **0.785x** | 0.638-0.930 | JOINTLY limited (intermediate) |
| llama70b_kv | +0 | 6 | **1.570x** | 0.732-2.993 | **0.785x** | 0.366-1.496 | JOINTLY limited (intermediate) |
| llama70b_qo | +0 | 1 | **2.039x** | 1.580-2.646 | **1.020x** | 0.790-1.323 | bandwidth-limited |
| llama70b_qo | +0 | 6 | **2.184x** | 1.426-3.726 | **1.092x** | 0.713-1.863 | bandwidth-limited |

### Main sweep - stride pad +64 B (A1.5 stride-conflict separation)

| shape | stride pad | thr (ach) | arm | reps/inv | replicas | matvecs/s (mean of inv means) | between-inv CV | within-inv CV | moved B | **moved GB/s** | charged B | charged GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama70b_down | +64 | 1 (1) | byte | 170 | 5 | **25.8** | 12.9% | 9.0% | 117440512 | **3.03** | 117440512 | 3.03 |
| llama70b_down | +64 | 1 (1) | nibble | 340 | 10 | **38.0** | 9.3% | 9.2% | 58720256 | **2.23** | 58720256 | 2.23 |
| llama70b_down | +64 | 6 (6) | byte | 170 | 5 | **101.6** | 16.6% | 10.8% | 117440512 | **11.93** | 117440512 | 11.93 |
| llama70b_down | +64 | 6 (6) | nibble | 340 | 10 | **160.7** | 13.8% | 20.0% | 58720256 | **9.43** | 58720256 | 9.43 |
| llama70b_gate_up | +64 | 1 (1) | byte | 170 | 5 | **24.7** | 8.8% | 9.9% | 117440512 | **2.90** | 117440512 | 2.90 |
| llama70b_gate_up | +64 | 1 (1) | nibble | 340 | 10 | **67.8** | 10.8% | 10.7% | 58720256 | **3.98** | 58720256 | 3.98 |
| llama70b_gate_up | +64 | 6 (6) | byte | 170 | 5 | **103.4** | 8.7% | 10.1% | 117440512 | **12.14** | 117440512 | 12.14 |
| llama70b_gate_up | +64 | 6 (6) | nibble | 340 | 10 | **179.2** | 13.9% | 16.0% | 58720256 | **10.52** | 58720256 | 10.52 |
| llama70b_kv | +64 | 1 (1) | byte | 4768 | 121 | **1370.0** | 19.5% | 31.8% | 4194304 | **5.75** | 4194304 | 5.75 |
| llama70b_kv | +64 | 1 (1) | nibble | 9536 | 241 | **2213.7** | 9.5% | 15.6% | 2097152 | **4.64** | 2097152 | 4.64 |
| llama70b_kv | +64 | 6 (6) | byte | 4768 | 121 | **3999.8** | 33.0% | 109.6% | 4194304 | **16.78** | 4194304 | 16.78 |
| llama70b_kv | +64 | 6 (6) | nibble | 9536 | 241 | **6752.5** | 24.9% | 66.1% | 2097152 | **14.16** | 2097152 | 14.16 |
| llama70b_qo | +64 | 1 (1) | byte | 596 | 16 | **103.4** | 8.9% | 9.9% | 33554432 | **3.47** | 33554432 | 3.47 |
| llama70b_qo | +64 | 1 (1) | nibble | 1192 | 32 | **339.7** | 14.6% | 18.3% | 16777216 | **5.70** | 16777216 | 5.70 |
| llama70b_qo | +64 | 6 (6) | byte | 596 | 16 | **375.6** | 18.3% | 26.7% | 33554432 | **12.60** | 33554432 | 12.60 |
| llama70b_qo | +64 | 6 (6) | nibble | 1192 | 32 | **718.0** | 19.9% | 42.2% | 16777216 | **12.05** | 16777216 | 12.05 |

**Nibble / byte, per cell — the A1.3 discriminator:**

| shape | stride pad | thr | matvecs/s ratio | (range over inv) | moved GB/s ratio | (range over inv) | reading |
|---|---|---|---|---|---|---|---|
| llama70b_down | +64 | 1 | **1.474x** | 1.206-2.054 | **0.737x** | 0.603-1.028 | JOINTLY limited (intermediate) |
| llama70b_down | +64 | 6 | **1.581x** | 1.220-2.329 | **0.791x** | 0.610-1.164 | JOINTLY limited (intermediate) |
| llama70b_gate_up | +64 | 1 | **2.747x** | 2.183-3.311 | **1.373x** | 1.091-1.655 | OFF-TABLE: byte arm access-limited, not byte-limited |
| llama70b_gate_up | +64 | 6 | **1.734x** | 1.257-2.164 | **0.867x** | 0.629-1.082 | JOINTLY limited (intermediate) |
| llama70b_kv | +64 | 1 | **1.616x** | 1.241-2.613 | **0.808x** | 0.620-1.306 | JOINTLY limited (intermediate) |
| llama70b_kv | +64 | 6 | **1.688x** | 0.806-4.397 | **0.844x** | 0.403-2.199 | JOINTLY limited (intermediate) |
| llama70b_qo | +64 | 1 | **3.286x** | 2.575-4.100 | **1.643x** | 1.287-2.050 | OFF-TABLE: byte arm access-limited, not byte-limited |
| llama70b_qo | +64 | 6 | **1.912x** | 1.209-2.925 | **0.956x** | 0.604-1.462 | bandwidth-limited |

### A1.5 stride-conflict separation: +64 B vs +0 B, same arm, same shape, same threads

| shape | thr | arm | matvecs/s +0 | matvecs/s +64 | +64 / +0 | moved GB/s +0 | moved GB/s +64 | +64 / +0 |
|---|---|---|---|---|---|---|---|---|
| llama70b_kv | 1 | byte | 1473.5 | 1370.0 | **0.930x** | 6.18 | 5.75 | 0.930x |
| llama70b_kv | 1 | nibble | 2312.0 | 2213.7 | **0.958x** | 4.85 | 4.64 | 0.958x |
| llama70b_kv | 6 | byte | 4207.2 | 3999.8 | **0.951x** | 17.65 | 16.78 | 0.951x |
| llama70b_kv | 6 | nibble | 6605.0 | 6752.5 | **1.022x** | 13.85 | 14.16 | 1.022x |
| llama70b_qo | 1 | byte | 66.2 | 103.4 | **1.562x** | 2.22 | 3.47 | 1.562x |
| llama70b_qo | 1 | nibble | 135.0 | 339.7 | **2.517x** | 2.26 | 5.70 | 2.517x |
| llama70b_qo | 6 | byte | 224.2 | 375.6 | **1.675x** | 7.52 | 12.60 | 1.675x |
| llama70b_qo | 6 | nibble | 489.6 | 718.0 | **1.466x** | 8.21 | 12.05 | 1.466x |
| llama70b_gate_up | 1 | byte | 17.3 | 24.7 | **1.429x** | 2.03 | 2.90 | 1.429x |
| llama70b_gate_up | 1 | nibble | 71.3 | 67.8 | **0.951x** | 4.19 | 3.98 | 0.951x |
| llama70b_gate_up | 6 | byte | 61.5 | 103.4 | **1.681x** | 7.22 | 12.14 | 1.681x |
| llama70b_gate_up | 6 | nibble | 173.3 | 179.2 | **1.034x** | 10.17 | 10.52 | 1.034x |
| llama70b_down | 1 | byte | 17.1 | 25.8 | **1.506x** | 2.01 | 3.03 | 1.506x |
| llama70b_down | 1 | nibble | 22.7 | 38.0 | **1.671x** | 1.33 | 2.23 | 1.671x |
| llama70b_down | 6 | byte | 62.2 | 101.6 | **1.634x** | 7.30 | 11.93 | 1.634x |
| llama70b_down | 6 | nibble | 79.7 | 160.7 | **2.016x** | 4.68 | 9.43 | 2.016x |

### d5cd meter-honesty: resident vs streamed, same arm, donor shape k/v 1024x8192

| arm | thr | matvecs/s resident | matvecs/s streamed | resident / streamed | moved GB/s resident | moved GB/s streamed |
|---|---|---|---|---|---|---|
| byte | 1 | 3099.6 | 1503.5 | **2.06x** | 13.00 | 6.31 |
| byte | 6 | 9627.5 | 4445.9 | **2.17x** | 40.38 | 18.65 |
| nibble | 1 | 3666.3 | 2318.8 | **1.58x** | 7.69 | 4.86 |
| nibble | 6 | 10531.5 | 6842.2 | **1.54x** | 22.09 | 14.35 |
