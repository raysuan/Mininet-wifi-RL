# import re
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from collections import defaultdict

# def extract_data(file_path):
#     """Extract data and metadata from the log file."""
#     filename = os.path.basename(file_path)
#     match_params = re.search(r'alpha([\d.]+)_th([\d.]+)', filename)
#     alpha = match_params.group(1) if match_params else "unknown"
#     th = match_params.group(2) if match_params else "unknown"

#     iterations = []
#     trust_scores = []
#     trust_pattern = re.compile(r'Trust scores at iteration (\d+): \[([-\d.,\s]+)\]')

#     with open(file_path, 'r') as file:
#         for line in file:
#             match = trust_pattern.search(line)
#             if match:
#                 iteration = int(match.group(1))
#                 scores = list(map(float, match.group(2).split(',')))
#                 iterations.append(iteration)
#                 trust_scores.append(scores)

#     return alpha, th, iterations, trust_scores

# def calculate_group_means(trust_scores):
#     """Calculate mean trust scores for each group."""
#     trust_scores = list(zip(*trust_scores))
#     group_0_4_means = [np.mean(scores[:4]) for scores in zip(*trust_scores)]
#     group_5_7_means = [np.mean(scores[5:7]) for scores in zip(*trust_scores)]
#     group_8_9_means = [np.mean(scores[8:9]) for scores in zip(*trust_scores)]
#     return group_0_4_means, group_5_7_means, group_8_9_means

# def create_enhanced_plot(title, data_dict, x_label="Iteration", y_label="Trust Scores"):
#     """Create an enhanced plot with improved styling."""
#     plt.figure(figsize=(12, 8))
    
#     # Color palette for different lines
#     colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
#     for idx, (key, (iterations, group_5_7_means, group_8_9_means)) in enumerate(data_dict.items()):
#         color = colors[idx % len(colors)]
#         # Plot group 5-7 with solid line
#         plt.plot(iterations, group_5_7_means, 
#                 label=f'Group 5-7 ({key})', 
#                 linestyle='-', 
#                 linewidth=2, 
#                 color=color)
        
#         # Plot group 8-9 with dotted line
#         plt.plot(iterations, group_8_9_means, 
#                 label=f'Group 8-9 ({key})', 
#                 linestyle=':', 
#                 linewidth=2, 
#                 color=color)

#     # Customize labels and title
#     plt.xlabel(x_label, fontsize=14)
#     plt.ylabel(y_label, fontsize=14)
#     plt.title(title, fontsize=16)
    
#     # Customize legend
#     plt.legend(fontsize=12, loc='upper right')
    
#     # Customize tick parameters
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
    
#     # Add grid for better readability
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Adjust layout
#     plt.tight_layout()

# def plot_by_th(data_by_th, output_dir):
#     """Plot data grouped by threshold value with enhanced styling."""
#     for th, data in data_by_th.items():
#         create_enhanced_plot(
#             title=f'Trust Scores Evolution (Threshold = {th})',
#             data_dict=data
#         )
#         output_file = os.path.join(output_dir, f'trust_scores_th{th}.png')
#         plt.savefig(output_file, dpi=300)
#         plt.close()

# def plot_by_alpha(data_by_alpha, output_dir):
#     """Plot data grouped by alpha value with enhanced styling."""
#     for alpha, data in data_by_alpha.items():
#         create_enhanced_plot(
#             title=f'Trust Scores Evolution (Alpha = {alpha})',
#             data_dict=data
#         )
#         output_file = os.path.join(output_dir, f'trust_scores_alpha{alpha}.png')
#         plt.savefig(output_file, dpi=300)
#         plt.close()

# def process_all_logs(directory, output_dir):
#     """Process all log files and generate enhanced plots."""
#     os.makedirs(output_dir, exist_ok=True)

#     data_by_th = defaultdict(lambda: defaultdict(tuple))
#     data_by_alpha = defaultdict(lambda: defaultdict(tuple))

#     for file_name in os.listdir(directory):
#         if file_name.endswith('.log'):
#             file_path = os.path.join(directory, file_name)
#             alpha, th, iterations, trust_scores = extract_data(file_path)
#             group_0_4_means, group_5_7_means, group_8_9_means = calculate_group_means(trust_scores)

#             data_by_th[th][alpha] = (iterations, group_5_7_means, group_8_9_means)
#             data_by_alpha[alpha][th] = (iterations, group_5_7_means, group_8_9_means)

#     plot_by_th(data_by_th, output_dir)
#     plot_by_alpha(data_by_alpha, output_dir)

# # Directory paths
# log_dir = "/mydata/mydata/RL_agent/result_plot_seed42_v2"
# output_dir = "/mydata/mydata/RL_agent/result_plot_seed42_v2/result_v2"

# # Process all logs and generate enhanced plots
# process_all_logs(log_dir, output_dir)
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def extract_data(file_path):
    """Extract data and metadata from the log file."""
    filename = os.path.basename(file_path)
    match_params = re.search(r'alpha([\d.]+)_th([\d.]+)', filename)
    alpha = match_params.group(1) if match_params else "unknown"
    th = match_params.group(2) if match_params else "unknown"

    iterations = []
    trust_scores = []
    trust_pattern = re.compile(r'Trust scores at iteration (\d+): \[([-\d.,\s]+)\]')

    with open(file_path, 'r') as file:
        for line in file:
            match = trust_pattern.search(line)
            if match:
                iteration = int(match.group(1))
                scores = list(map(float, match.group(2).split(',')))
                iterations.append(iteration)
                trust_scores.append(scores)

    return alpha, th, iterations, trust_scores

def calculate_group_means(trust_scores):
    """Calculate mean trust scores for each group, iteration-based."""
    # trust_scores: list of length N_iterations
    # each element is a list of trust scores of M_agents at that iteration
    # Example: trust_scores[i] = [score_agent_0, score_agent_1, ..., score_agent_M-1] at iteration i

    # Group 0-4 means: mean of agents 0,1,2,3
    group_0_4_means = [np.mean(scores[:4]) for scores in trust_scores]

    # Group 8-9 means: mean of agents 8,9
    group_8_9_means = [np.mean(scores[8:10]) for scores in trust_scores]

    return group_0_4_means, group_8_9_means


# def create_enhanced_plot(title, data_dict, groups_to_plot, x_label="Iteration", y_label="Trust Scores"):
#     """Create an enhanced plot with improved styling, allowing selection of which groups to plot."""
#     plt.figure(figsize=(12, 8))
    
#     # Color palette for different lines
#     colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
#     # Plot each line
#     for idx, (key, (iterations, group_5_7_means, group_8_9_means)) in enumerate(data_dict.items()):
#         color = colors[idx % len(colors)]

#         if '5-7' in groups_to_plot:
#             plt.plot(iterations, group_5_7_means, 
#                      label=f'Group 5-7 ({key})', 
#                      linestyle='-', 
#                      linewidth=2, 
#                      color=color)

#         if '8-9' in groups_to_plot:
#             # Use a different linestyle to distinguish from 5-7
#             plt.plot(iterations, group_8_9_means, 
#                      label=f'Group 8-9 ({key})', 
#                      linestyle=':', 
#                      linewidth=2, 
#                      color=color)

#     # Customize labels and title
#     plt.xlabel(x_label, fontsize=14)
#     plt.ylabel(y_label, fontsize=14)
#     plt.title(title, fontsize=16)
    
#     # Customize legend
#     plt.legend(fontsize=12, loc='upper right')
    
#     # Customize tick parameters
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
    
#     # Add grid for better readability
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Adjust layout
#     plt.tight_layout()
import re
import matplotlib.pyplot as plt

def create_enhanced_plot(title, data_dict, groups_to_plot, x_label="Iteration", y_label="Trust Scores"):
    """Create an enhanced plot with improved styling, allowing selection of which groups to plot."""
    plt.figure(figsize=(8, 6))
    
    # Color palette for different lines
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    # Plot each line
    for idx, (key, (iterations, group_0_4_means, group_8_9_means)) in enumerate(data_dict.items()):
        color = colors[idx % len(colors)]

        # Generate labels
        if '0-4' in groups_to_plot:
            plt.plot(iterations, group_0_4_means, 
                     label=f'No Noise ({key})', 
                     linestyle='-', 
                     linewidth=2, 
                     color=color)

        if '8-9' in groups_to_plot:
            plt.plot(iterations, group_8_9_means, 
                     label=f'Random Noise ({key})', 
                     linestyle='-', 
                     linewidth=2, 
                     color=color)

    # Retrieve current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Function to extract numeric value from labels of the form "Group 5-7 (0.1)"
    def extract_numeric_value(label):
        match = re.search(r'\(([\d.]+)\)', label)
        if match:
            return float(match.group(1))
        return float('inf')  # If not found, put it at the end

    # Sort the legend entries by the numeric value
    sorted_entries = sorted(zip(handles, labels), key=lambda x: extract_numeric_value(x[1]))
    handles, labels = zip(*sorted_entries)

    # Customize labels and title
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    
    # Set legend with sorted entries
    plt.legend(handles, labels, fontsize=12, loc='lower left')
    
    # Customize tick parameters
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()


def plot_by_th(data_by_th, output_dir):
    """Plot data grouped by threshold value with enhanced styling, separate plots for each group."""
    for th, data in data_by_th.items():
        # Plot only group 5-7
        create_enhanced_plot(
            title=f'Trust Scores Evolution (Threshold = {th}, No Noise)',
            data_dict=data,
            groups_to_plot=['0-4']
        )
        output_file = os.path.join(output_dir, f'trust_scores_th{th}_no_noise.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

        # Plot only group 8-9
        create_enhanced_plot(
            title=f'Trust Scores Evolution (Threshold = {th}, Random Noise)',
            data_dict=data,
            groups_to_plot=['8-9']
        )
        output_file = os.path.join(output_dir, f'trust_scores_th{th}_random_noise.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

def plot_by_alpha(data_by_alpha, output_dir):
    """Plot data grouped by alpha value with enhanced styling, separate plots for each group."""
    for alpha, data in data_by_alpha.items():
        # Plot only group 5-7
        create_enhanced_plot(
            title=f'Trust Scores Evolution (Alpha = {alpha}, No Noise)',
            data_dict=data,
            groups_to_plot=['0-4']
        )
        output_file = os.path.join(output_dir, f'trust_scores_alpha{alpha}_no_noise.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

        # Plot only group 8-9
        create_enhanced_plot(
            title=f'Trust Scores Evolution (Alpha = {alpha}, Random Noise)',
            data_dict=data,
            groups_to_plot=['8-9']
        )
        output_file = os.path.join(output_dir, f'trust_scores_alpha{alpha}_random_noise.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

def process_all_logs(directory, output_dir):
    """Process all log files and generate enhanced plots."""
    os.makedirs(output_dir, exist_ok=True)

    data_by_th = defaultdict(lambda: defaultdict(tuple))
    data_by_alpha = defaultdict(lambda: defaultdict(tuple))

    for file_name in os.listdir(directory):
        if file_name.endswith('.log'):
            file_path = os.path.join(directory, file_name)
            alpha, th, iterations, trust_scores = extract_data(file_path)
            group_0_4_means, group_8_9_means = calculate_group_means(trust_scores)

            data_by_th[th][alpha] = (iterations, group_0_4_means, group_8_9_means)
            data_by_alpha[alpha][th] = (iterations, group_0_4_means, group_8_9_means)

    plot_by_th(data_by_th, output_dir)
    plot_by_alpha(data_by_alpha, output_dir)

# Directory paths
log_dir = "/mydata/mydata/RL_agent/result_samefile_plot_seed42_v2"
output_dir = "/mydata/mydata/RL_agent/result_samefile_plot_seed42_v2/result_v2"

# Process all logs and generate enhanced plots (separated by groups 5-7 and 8-9)
process_all_logs(log_dir, output_dir)
