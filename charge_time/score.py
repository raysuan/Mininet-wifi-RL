import re
import matplotlib.pyplot as plt
import numpy as np

def extract_and_plot_grouped_trust_scores(file_path):
    # Initialize variables for all trust scores
    iterations = []
    trust_scores = []
    
    # Regular expression for trust scores
    trust_pattern = re.compile(r'Trust scores at iteration (\d+): \[([-\d.,\s]+)\]')
    
    # Open the file and extract data
    with open(file_path, 'r') as file:
        for line in file:
            match = trust_pattern.search(line)
            if match:
                iteration = int(match.group(1))
                scores = list(map(float, match.group(2).split(',')))
                iterations.append(iteration)
                trust_scores.append(scores)
    
    # Transpose trust scores to get each sensor's scores
    trust_scores = list(zip(*trust_scores))
    
    # Calculate mean values for each group at each timestamp
    group_0_14_means = [np.mean(scores[:14]) for scores in zip(*trust_scores)]
    group_15_18_means = [np.mean(scores[14:18]) for scores in zip(*trust_scores)]
    sensor_17_values = [scores[17] for scores in zip(*trust_scores)]
    sensor_18_values = [scores[18] for scores in zip(*trust_scores)]
    
    # Define a custom color palette
    

    plt.figure(figsize=(8, 6))
    
    # Plot with improved styling
    line1, = plt.plot(iterations, group_0_14_means, 
                      label='No Noise Group Mean', 
                      linestyle='-', color='orange', 
                      linewidth=2, alpha=0.8)
    
    line2, = plt.plot(iterations, group_15_18_means, 
                      label='Same Noise Group Mean', 
                      linestyle='-', color='Blue', 
                      linewidth=2, alpha=0.8)
    
    line3, = plt.plot(iterations, sensor_17_values, 
                      label='Random Noise Sensor 1', 
                      linestyle='--', color='orange', 
                      linewidth=2, alpha=0.8)
    
    line4, = plt.plot(iterations, sensor_18_values, 
                      label='Random Noise Sensor 2', 
                      linestyle='--', color='Blue', 
                      linewidth=2, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Trust Scores', fontsize=14)
    plt.title('Grouped Trust Scores Over Iterations', fontsize=16)
    
    # Retrieve and sort legend entries by their label
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_entries = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles = zip(*sorted_entries)
    plt.legend(handles, labels, loc='best', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig('/mydata/mydata/RL_agent/plot_trust/samefile_grouped_trust_scores_no_noise_v2_modified_style.png', dpi=300)
    plt.show()

# Example usage:
extract_and_plot_grouped_trust_scores("/mydata/mydata/RL_agent/newequation_th_0.2_s1s2_only_v1.log")
