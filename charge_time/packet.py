# import re
# from datetime import datetime
# import matplotlib.pyplot as plt

# def process_unique_log(log_file_path, start_time):
#     """
#     Process a log file where each event is represented by two lines:
#     - The first line contains "Total rows: ..." and "Total unique packets: ..."
#     - The second line starts with a timestamp.
    
#     Parameters:
#       log_file_path: path to the log file.
#       start_time: a float representing the start time in seconds since the epoch.
      
#     Returns:
#       A tuple of two lists: relative times (in seconds from start_time) and unique portions.
#     """
#     # Regex pattern to extract total rows and total unique packets.
#     stats_pattern = re.compile(
#         r"Total rows:\s*(\d+).*Total unique packets:\s*(\d+)"
#     )
#     # Regex pattern to extract the timestamp at the beginning of the line.
#     timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)')
    
#     relative_times = []
#     unique_portions = []
    
#     with open(log_file_path, 'r') as f:
#         lines = f.readlines()
        
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
#         stats_match = stats_pattern.search(line)
#         if stats_match:
#             total_rows = int(stats_match.group(1))
#             total_unique = int(stats_match.group(2))
#             # Calculate the unique portion.
#             unique_portion = total_unique / total_rows if total_rows != 0 else 0
            
#             # Assume the next line holds the timestamp (with possible extra text).
#             if i + 1 < len(lines):
#                 timestamp_line = lines[i + 1].strip()
#                 timestamp_match = timestamp_pattern.match(timestamp_line)
#                 if timestamp_match:
#                     timestamp_str = timestamp_match.group(1)
#                     # Try parsing with fractional seconds first.
#                     try:
#                         timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
#                     except ValueError:
#                         try:
#                             timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
#                         except ValueError as e:
#                             print(f"Error parsing timestamp '{timestamp_str}' in file {log_file_path}: {e}")
#                             i += 2
#                             continue
#                     # Convert timestamp to seconds since epoch.
#                     timestamp_epoch = timestamp.timestamp()
#                     # Compute relative time.
#                     rel_time = timestamp_epoch - start_time
#                     # Optionally, print to verify
#                     # print(f"Relative time: {rel_time} seconds")
#                     relative_times.append(rel_time)
#                     unique_portions.append(unique_portion)
#                 else:
#                     print(f"No valid timestamp found in line: {timestamp_line}")
#                 i += 2
#                 continue
#         i += 1
    
#     return relative_times, unique_portions

# # Define start times as strings.
# start_time_no_rl_str = "2025-02-21 01:07:39"
# start_time_rl_str    = "2025-02-21 00:32:15"

# # Parse the start times into datetime objects and convert to epoch seconds.
# try:
#     start_dt_no_rl = datetime.strptime(start_time_no_rl_str, "%Y-%m-%d %H:%M:%S.%f")
# except ValueError:
#     start_dt_no_rl = datetime.strptime(start_time_no_rl_str, "%Y-%m-%d %H:%M:%S")
# start_time_no_rl = start_dt_no_rl.timestamp()

# try:
#     start_dt_rl = datetime.strptime(start_time_rl_str, "%Y-%m-%d %H:%M:%S.%f")
# except ValueError:
#     start_dt_rl = datetime.strptime(start_time_rl_str, "%Y-%m-%d %H:%M:%S")
# start_time_rl = start_dt_rl.timestamp()

# # File paths for the two logs.
# no_rl_file = "no_rl_v3.log"   # Replace with your actual file path
# rl_file    = "rl_v3.log"      # Replace with your actual file path

# # Process each log file with the corresponding start time.
# no_rl_times, no_rl_unique = process_unique_log(no_rl_file, start_time_no_rl)
# rl_times, rl_unique       = process_unique_log(rl_file, start_time_rl)

# # Optional: filter out any negative relative times (if needed)
# no_rl_times, no_rl_unique = zip(*[(t, u) for t, u in zip(no_rl_times, no_rl_unique) if t >= 0]) if no_rl_times else ([], [])
# rl_times, rl_unique = zip(*[(t, u) for t, u in zip(rl_times, rl_unique) if t >= 0]) if rl_times else ([], [])

# rl_times = rl_times[:-5]

# rl_unique = rl_unique[:-5]
# # Plot both lines on the same plot.
# plt.figure(figsize=(12, 6))
# plt.plot(no_rl_times, no_rl_unique, marker='o', linestyle='-', label='No RL')
# plt.plot(rl_times, rl_unique, marker='s', linestyle='-', label='With RL')

# plt.xlabel("Relative Time (seconds)")
# plt.ylabel("Unique Portion (Total Unique Packets / Total Rows)")
# plt.title("Unique Portion Over Time (Relative to Start Time)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import re
from datetime import datetime
import matplotlib.pyplot as plt

def process_unique_log(log_file_path, start_time, bin_interval=300):
    """
    Process a log file where each event is represented by two lines:
    - The first line contains "Total rows: ...", "New unique packets: ...", and "Total unique packets: ..."
    - The second line starts with a timestamp.
    
    This function aggregates events into bins of 'bin_interval' seconds (default 5 minutes).
    For each bin, it computes the ratio:
    
         (Sum of new unique packets in the bin) / (final total unique packets)
    
    Parameters:
      log_file_path: Path to the log file.
      start_time: A float representing the start time in seconds since the epoch.
      bin_interval: Time bin size in seconds (default: 300 seconds, i.e. 5 minutes).
      
    Returns:
      A tuple of two lists: binned relative times (in seconds from start_time) and binned unique portions.
    """
    # Updated regex pattern to extract Total rows, New unique packets, and Total unique packets.
    stats_pattern = re.compile(
        r"Total rows:\s*(\d+).*New unique packets:\s*(\d+),\s*Total unique packets:\s*(\d+)"
    )
    # Regex pattern to extract the timestamp at the beginning of the line.
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)')
    
    relative_times = []
    new_uniques = []
    total_uniques = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        stats_match = stats_pattern.search(line)
        if stats_match:
            new_unique = int(stats_match.group(2))
            total_unique = int(stats_match.group(3))
            
            # Assume the next line holds the timestamp.
            if i + 1 < len(lines):
                timestamp_line = lines[i + 1].strip()
                timestamp_match = timestamp_pattern.match(timestamp_line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError:
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError as e:
                            print(f"Error parsing timestamp '{timestamp_str}' in file {log_file_path}: {e}")
                            i += 2
                            continue
                    timestamp_epoch = timestamp.timestamp()
                    rel_time = timestamp_epoch - start_time
                    relative_times.append(rel_time)
                    new_uniques.append(new_unique)
                    total_uniques.append(total_unique)
                else:
                    print(f"No valid timestamp found in line: {timestamp_line}")
                i += 2
                continue
        i += 1

    # Determine the final total unique packets from the last event.
    if not total_uniques:
        return [], []
    final_total = total_uniques[-1]

    # Group events into bins of bin_interval seconds.
    bins = {}
    for t, n in zip(relative_times, new_uniques):
        bin_index = int(t // bin_interval)
        if bin_index not in bins:
            bins[bin_index] = {'times': [], 'new_unique_sum': 0}
        bins[bin_index]['times'].append(t)
        bins[bin_index]['new_unique_sum'] += n

    # Compute binned times and unique portions.
    binned_times = []
    binned_unique_portions = []
    for bin_index in sorted(bins.keys()):
        # Use the average time of the bin as the representative time.
        avg_time = sum(bins[bin_index]['times']) / len(bins[bin_index]['times'])
        ratio = bins[bin_index]['new_unique_sum'] / final_total if final_total != 0 else 0
        binned_times.append(avg_time)
        binned_unique_portions.append(ratio)
    
    return binned_times, binned_unique_portions

# Define start times as strings.
start_time_no_rl_str = "2025-02-21 01:07:39"
start_time_rl_str    = "2025-02-21 00:32:15"

# Parse the start times into datetime objects and convert to epoch seconds.
try:
    start_dt_no_rl = datetime.strptime(start_time_no_rl_str, "%Y-%m-%d %H:%M:%S.%f")
except ValueError:
    start_dt_no_rl = datetime.strptime(start_time_no_rl_str, "%Y-%m-%d %H:%M:%S")
start_time_no_rl = start_dt_no_rl.timestamp()

try:
    start_dt_rl = datetime.strptime(start_time_rl_str, "%Y-%m-%d %H:%M:%S.%f")
except ValueError:
    start_dt_rl = datetime.strptime(start_time_rl_str, "%Y-%m-%d %H:%M:%S")
start_time_rl = start_dt_rl.timestamp()

# File paths for the two logs.
no_rl_file = "no_rl_v3.log"   # Replace with your actual file path
rl_file    = "rl_v3.log"      # Replace with your actual file path

# Process each log file with the corresponding start time and bin interval of 5 minutes.
no_rl_times, no_rl_unique = process_unique_log(no_rl_file, start_time_no_rl, bin_interval=300)
rl_times, rl_unique       = process_unique_log(rl_file, start_time_rl, bin_interval=300)

# Optional: filter out any negative relative times.
if no_rl_times:
    no_rl_times, no_rl_unique = zip(*[(t, u) for t, u in zip(no_rl_times, no_rl_unique) if t >= 0])
if rl_times:
    rl_times, rl_unique = zip(*[(t, u) for t, u in zip(rl_times, rl_unique) if t >= 0])

# Plot both lines on the same plot.
plt.figure(figsize=(12, 6))
plt.plot(no_rl_times, no_rl_unique, marker='o', linestyle='-', label='No RL')
plt.plot(rl_times, rl_unique, marker='s', linestyle='-', label='With RL')

plt.xlabel("Relative Time (seconds)")
plt.ylabel("Unique Portion (Sum of New Unique Packets / Final Total Unique Packets)")
plt.title("Unique Portion Over Time (Binned in 5-Minute Intervals)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
