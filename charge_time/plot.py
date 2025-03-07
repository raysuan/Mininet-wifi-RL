# import re
# import matplotlib.pyplot as plt

# # Specify the paths to your log files
# log_file_path = "no_rl.log"
# rl_file_path = "rl.log"

# # Lists to hold the extracted recharge timestamps
# recharge_times = []
# rl_recharge_times = []

# # Regular expression pattern to capture the timestamp after "Current time:"
# pattern = r"Current time:\s*([0-9]+\.[0-9]+)"

# # Read from the no_rl.log file
# with open(log_file_path, 'r') as file:
#     for line in file:
#         if "Recharged to full energy" in line:
#             match = re.search(pattern, line)
#             if match:
#                 recharge_times.append(float(match.group(1)))

# # Read from the rl.log file
# with open(rl_file_path, 'r') as file:
#     for line in file:
#         if "Recharged to full energy" in line:
#             match = re.search(pattern, line)
#             if match:
#                 rl_recharge_times.append(float(match.group(1)))

# # Provide feedback if no events are found
# if not recharge_times:
#     print(f"No recharge events found in {log_file_path}.")
# if not rl_recharge_times:
#     print(f"No recharge events found in {rl_file_path}.")

# # Create relative time lists by subtracting the first event's timestamp from each timestamp.
# # This makes each series start at time = 0.
# if recharge_times:
#     start_time_no_rl = recharge_times[0]
#     relative_no_rl = [ts - start_time_no_rl for ts in recharge_times]
# else:
#     relative_no_rl = []

# if rl_recharge_times:
#     start_time_rl = rl_recharge_times[0]
#     relative_rl = [ts - start_time_rl for ts in rl_recharge_times]
# else:
#     relative_rl = []

# # Prepare cumulative event counts for each file
# no_rl_event_numbers = range(1, len(relative_no_rl) + 1)
# rl_event_numbers = range(1, len(relative_rl) + 1)

# # Plotting both series on the same plot with relative time on the x-axis.
# plt.figure(figsize=(12, 6))

# plt.plot(relative_no_rl, no_rl_event_numbers, marker='o', linestyle='-', label='no_rl.log')
# plt.plot(relative_rl, rl_event_numbers, marker='s', linestyle='-', label='rl.log')

# plt.xlabel("Time (seconds, relative to first event)")
# plt.ylabel("Cumulative Recharge Count")
# plt.title("Recharge Events Over Time (Relative Time)")
# plt.legend()
# plt.grid(True)
# plt.show()
import re
import matplotlib.pyplot as plt
import numpy as np

# Specify the paths to your log files
log_file_path = "no_rl_v4.log"
rl_file_path = "rl_v4.log"

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

# Convert absolute times to relative times (starting at 0)
if recharge_times:
    start_time_no_rl = recharge_times[0]
    relative_no_rl = [ts - start_time_no_rl for ts in recharge_times]
else:
    relative_no_rl = []

if rl_recharge_times:
    start_time_rl = rl_recharge_times[0]
    relative_rl = [ts - start_time_rl for ts in rl_recharge_times]
else:
    relative_rl = []

# Define the interval length (5 minutes = 300 seconds)
interval = 300

# Determine the maximum time across both logs
max_time_no_rl = relative_no_rl[-1] if relative_no_rl else 0
max_time_rl = relative_rl[-1] if relative_rl else 0
max_time = max(max_time_no_rl, max_time_rl)

# Create bin edges for non-overlapping intervals from time 0 up to max_time
bin_edges = np.arange(0, max_time + interval, interval)

# Count events in each 5-minute bin using numpy's histogram
counts_no_rl, _ = np.histogram(relative_no_rl, bins=bin_edges)
counts_rl, _ = np.histogram(relative_rl, bins=bin_edges)

# Compute cumulative counts: each bin's value is the sum of all counts up to that bin.
cum_counts_no_rl = np.cumsum(counts_no_rl)
cum_counts_rl = np.cumsum(counts_rl)

# Use the right edge of each bin (i.e., 5, 10, 15, ... minutes) for plotting.
time_points_minutes = bin_edges[1:] / 60  # convert seconds to minutes

# Plot the cumulative recharge counts
plt.figure(figsize=(12, 6))
plt.plot(time_points_minutes, cum_counts_no_rl, marker='o', linestyle='-', label='no_rl.log')
plt.plot(time_points_minutes, cum_counts_rl, marker='s', linestyle='-', label='rl.log')

plt.xlabel("Time (minutes, relative to first event)")
plt.ylabel("Cumulative Recharge Count")
plt.title("Cumulative Recharge Events over 5-Minute Intervals")
plt.legend()
plt.grid(True)
plt.show()