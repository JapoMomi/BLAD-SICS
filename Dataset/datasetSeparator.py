# Path to your local dataset file
input_file = "IanRawDataset.txt"

# Prepare containers
normal_lines = []
attack_lines = []

with open(input_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue  # skip malformed lines

        # parts structure:
        # 0 = hex frame
        # 1 = attack category
        # 2 = attack number
        # 3 = source
        # 4 = destination
        # 5 = timestamp
        attack_category = int(parts[1])
        attack_number = int(parts[2])

        if attack_category == 0 and attack_number == 0:
            normal_lines.append(line)
        else:
            attack_lines.append(line)

# Save outputs
with open("normal_traffic.txt", "w") as f:
    f.writelines(normal_lines)

with open("attack_traffic.txt", "w") as f:
    f.writelines(attack_lines)

print(f"Finished processing {input_file}")
print(f"Normal traffic lines: {len(normal_lines)}")
print(f"Attack traffic lines: {len(attack_lines)}")
