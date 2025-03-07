import re
import ast
import matplotlib.pyplot as plt

# Read log lines from file
with open('trust.log', 'r') as f:
    log_lines = f.readlines()

# Lists to hold extracted data
times = []
iterations = []
mean_0_4 = []  # Mean of indices 0-4
mean_5_7 = []  # Mean of indices 5-7
score_8 = []   # Score for index 8
score_9 = []   # Score for index 9

# Regex pattern to capture iteration, time, and the dictionary of trust scores
pattern = r"Trust scores at iteration (\d+) and time ([\d\.]+): (.*)"

for line in log_lines:
    match = re.search(pattern, line)
    if match:
        # Extract iteration and time values
        iteration = int(match.group(1))
        t = float(match.group(2))
        trust_str = match.group(3).strip()
        # Convert the trust score string to a dictionary using ast.literal_eval
        trust_dict = ast.literal_eval(trust_str)
        # Assuming the scores are stored under key 0
        scores = trust_dict[0]
        
        # Compute the means for indices 0-4 and 5-7
        group0_4 = [scores[i] for i in range(5)]
        group5_7 = [scores[i] for i in range(5, 8)]
        
        iterations.append(iteration)
        times.append(t)
        mean_0_4.append(sum(group0_4) / len(group0_4))
        mean_5_7.append(sum(group5_7) / len(group5_7))
        score_8.append(scores[8])
        score_9.append(scores[9])

# Plotting the data versus time
plt.figure(figsize=(10, 6))
plt.plot(times, mean_0_4, label='Mean scores 0-4', marker='o',markersize=1)
plt.plot(times, mean_5_7, label='Mean scores 5-7', marker='o',markersize=1)
# plt.plot(times, score_8, label='Score 8', marker='o',markersize=1)
# plt.plot(times, score_9, label='Score 9', marker='o',markersize=1)
plt.xlabel('Time')
plt.ylabel('Trust Scores')
plt.title('Trust Scores vs Time')
plt.legend()
plt.grid(True)
plt.show()
