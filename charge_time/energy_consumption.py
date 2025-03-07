import re
import matplotlib.pyplot as plt
import numpy as np

def process_log(log_file_path, start_time, rl=False):
    """
    Process a log file to match each sensor's energy warning with its subsequent recharge event.
    Returns two lists: relative times (in seconds, with time zero at the given start_time)
    and the corresponding cumulative energy consumption.
    """
    pending_consumption = {}  # key: (cluster, sensor), value: list of pending consumption values
    events = []  # list of tuples (timestamp, consumption)
    
    # Regex pattern for "energy below threshold" lines.
    energy_pattern = re.compile(
        r"Cluster\s+(\d+),\s+Sensor\s+(\d+):\s+Energy below threshold\s*\((\d+)\)"
    )
    energy_pattern_rl = re.compile(
        r"Cluster\s+(\d+),\s+Sensor\s+(\d+):\s+Energy below threshold\s*\(([0-9]*\.?[0-9]+)\)\. Recharging\.\.\."
    )
    if rl:
        energy_pattern = energy_pattern_rl

    # Regex pattern for recharge events.
    recharge_pattern = re.compile(
        r"Cluster\s+(\d+),\s+Sensor\s+(\d+):\s+Recharged to full energy\s*\(\d+\),\s*Current time:\s*([0-9]+\.[0-9]+)"
    )

    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for an energy warning event.
            energy_match = energy_pattern.search(line)
            if energy_match:
                cluster = energy_match.group(1)
                sensor = energy_match.group(2)
                energy_left = float(energy_match.group(3))
                consumption = 100 - energy_left  # energy used during this cycle
                sensor_key = (cluster, sensor)
                pending_consumption.setdefault(sensor_key, []).append(consumption)
            
            # Look for a recharge event.
            recharge_match = recharge_pattern.search(line)
            if recharge_match:
                cluster = recharge_match.group(1)
                sensor = recharge_match.group(2)
                timestamp = float(recharge_match.group(3))
                sensor_key = (cluster, sensor)
                # Match with the pending energy warning for the same sensor.
                if sensor_key in pending_consumption and pending_consumption[sensor_key]:
                    consumption = pending_consumption[sensor_key].pop(0)
                else:
                    consumption = 0  # No matching warning found, assume zero consumption.
                events.append((timestamp, consumption))
    
    # Sort all events by their timestamp.
    events.sort(key=lambda x: x[0])
    
    # If no events were found, return empty lists.
    if not events:
        return [], []
    
    # Convert timestamps to relative times using the given start_time.
    relative_times = [timestamp - start_time for timestamp, _ in events]
    consumptions = [consumption for _, consumption in events]
    
    # Compute the cumulative energy consumption.
    cumulative_consumption = np.cumsum(consumptions)
    
    return relative_times, cumulative_consumption

# Given start time
start_time_no_rl = 1739524691.9815652
start_time_rl = 1739526340.2888646
# File paths for the logs.
no_rl_log_path = "no_rl.log"
rl_log_path = "rl.log"

# Process each log file with the given start time.
no_rl_time, no_rl_cumulative = process_log(no_rl_log_path, start_time_no_rl, rl=False)
rl_time, rl_cumulative = process_log(rl_log_path, start_time_rl, rl=True)

# Plot the results.
plt.figure(figsize=(12, 6))
plt.plot(no_rl_time, no_rl_cumulative, marker='o', linestyle='-', label='no_rl.log')
plt.plot(rl_time, rl_cumulative, marker='s', linestyle='-', label='rl.log')

plt.xlabel("Time (seconds, relative to given start time)")
plt.ylabel("Cumulative Energy Consumption")
plt.title("Cumulative Energy Consumption Over Time")
plt.legend()
plt.grid(True)
plt.show()
