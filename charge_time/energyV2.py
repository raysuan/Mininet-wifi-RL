import re
import matplotlib.pyplot as plt

# Specify the paths to your log files
log_file_path = "no_rl.log"
rl_file_path = "rl.log"

# Lists to hold the extracted recharge timestamps
recharge_times = []
rl_recharge_times = []

# Regular expression pattern to capture the timestamp after "Current time:"
pattern = r"Current time:\s*([0-9]+\.[0-9]+)"

# Read from the no_rl.log file
with open(log_file_path, 'r') as file:
    for line in file:
        if "Recharged to full energy" in line:
            match = re.search(pattern, line)
            if match:
                recharge_times.append(float(match.group(1)))

# Read from the rl.log file
with open(rl_file_path, 'r') as file:
    for line in file:
        if "Recharged to full energy" in line:
            match = re.search(pattern, line)
            if match:
                rl_recharge_times.append(float(match.group(1)))

# Provide feedback if no events are found
if not recharge_times:
    print(f"No recharge events found in {log_file_path}.")
if not rl_recharge_times:
    print(f"No recharge events found in {rl_file_path}.")

# Define the max energy level and consumption per event
energy_per_recharge = 80  # Assume each recharge restores 80 units

# Set relative start times
start_time_no_rl = 1739524691.9815652
start_time_rl = 1739526340.2888646

# Compute energy consumption over time (positive values)
def compute_energy_consumption(timestamps, start_time):
    if not timestamps:
        return [], []

    relative_times = [ts - start_time for ts in timestamps]  # Make time relative
    energy_consumed = [i * energy_per_recharge for i in range(len(timestamps))]  # Cumulative consumption

    return relative_times, energy_consumed

# Compute energy consumption data
time_no_rl, energy_no_rl = compute_energy_consumption(recharge_times, start_time_no_rl)
time_rl, energy_rl = compute_energy_consumption(rl_recharge_times, start_time_rl)

# Plotting energy consumption over time
plt.figure(figsize=(12, 6))

plt.plot(time_no_rl, energy_no_rl, marker='o', linestyle='-', label='no_rl.log')
plt.plot(time_rl, energy_rl, marker='s', linestyle='-', label='rl.log')

plt.xlabel("Time (seconds, relative to start)")
plt.ylabel("Cumulative Energy Consumption")
plt.title("Energy Consumption Over Time")
plt.legend()
plt.grid(True)
plt.show()
