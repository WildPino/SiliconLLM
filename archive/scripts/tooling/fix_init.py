import os

with open("benchmarks/benchmark18_coder.c", "r") as f:
    lines = f.readlines()

content = "".join(lines)

# Find where base_probs is declared
base_probs_idx = content.find('float (*base_probs)[CLASSES][CLASSES] = NULL;')

# Find where data load starts
data_load_idx = content.find('uint8_t* data = NULL;')

# Find where it ends
data_load_end = content.find('SiliconEntropyState see;')

# Extract the data_load / mode logic
mode_logic = content[data_load_idx:data_load_end]

# Remove mode_logic from its original place
content = content[:data_load_idx] + content[data_load_end:]

# Insert mode_logic BEFORE base_probs_idx
base_probs_idx_new = content.find('float (*base_probs)[CLASSES][CLASSES] = NULL;')
content = content[:base_probs_idx_new] + mode_logic + "\n" + content[base_probs_idx_new:]

with open("benchmarks/benchmark18_coder.c", "w") as f:
    f.write(content)
print("Initialization order fixed")
