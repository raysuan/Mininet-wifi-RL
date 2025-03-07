import matplotlib.pyplot as plt
import numpy as np

# File paths
non_noise_file1_path = "/mydata/mydata/RL_agent/noise_vs_dataset/sensor_1_data.csv"
individual_noise_file2_path = "/mydata/mydata/RL_agent/noise_vs_dataset/sensor_10_data.csv"
group_noise_file2_path = "/mydata/mydata/RL_agent/noise_vs_dataset/sensor_7_data.csv"

# Reading data from files
def read_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        return [float(line.strip()) for line in lines if not line.strip().isalpha()]

relative_humidity_1 = read_data(non_noise_file1_path)[:500]
relative_humidity_2 = read_data(individual_noise_file2_path)[:500]
relative_humidity_3 = read_data(group_noise_file2_path)[:500]

# Create the plot for individual noise
plt.figure(figsize=(8, 6))
plt.plot(range(len(relative_humidity_1)), relative_humidity_1, label='Relative Humidity - No Noise', linestyle='-', linewidth=2, color='blue')
plt.plot(range(len(relative_humidity_2)), relative_humidity_2, label='Relative Humidity - Individual Noise', linestyle=':', linewidth=2, color='orange')

# Add labels, title, and legend
plt.xlabel('Index', fontsize=14)
plt.ylabel('Relative Humidity', fontsize=14)
plt.title('Relative Humidity vs Index (Individual Noise)', fontsize=16)
plt.legend(fontsize=14, loc='upper right')

# Customize tick parameters
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot
plt.tight_layout()
plt.savefig('relative_humidity_individual_noise.png', dpi=300)

# Show the plot
plt.show()

# Create the plot for group noise
plt.figure(figsize=(8, 6))
plt.plot(range(len(relative_humidity_1)), relative_humidity_1, label='Relative Humidity - No Noise', linestyle='-', linewidth=2, color='blue')
plt.plot(range(len(relative_humidity_3)), relative_humidity_3, label='Relative Humidity - Group Noise', linestyle=':', linewidth=2, color='green')

# Add labels, title, and legend
plt.xlabel('Time', fontsize=14)
plt.ylabel('Relative Humidity', fontsize=14)
plt.title('Relative Humidity vs time (Group Noise)', fontsize=16)
plt.legend(fontsize=14, loc='upper right')

# Customize tick parameters
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot
plt.tight_layout()
plt.savefig('relative_humidity_group_noise.png', dpi=300)

# Show the plot
plt.show()