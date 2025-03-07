# import re
# import matplotlib.pyplot as plt

# def extract_and_plot_trust_scores(file_path):
#     # Initialize variables for all trust scores
#     iterations = []
#     trust_scores = []
#     trust_scores_s1 = []
#     trust_scores_s2 = []
    
#     # Regular expressions for each trust score type
#     trust_pattern = re.compile(r'Trust scores at iteration (\d+): \[([-\d.,\s]+)\]')
#     trust_s1_pattern = re.compile(r'Trust scores_s1 at iteration (\d+): \[([-\d.,\s]+)\]')
#     trust_s2_pattern = re.compile(r'Trust scores_s2 at iteration (\d+): \[([-\d.,\s]+)\]')
    
#     # Open the file and extract data
#     with open(file_path, 'r') as file:
#         for line in file:
#             match = trust_pattern.search(line)
#             match_s1 = trust_s1_pattern.search(line)
#             match_s2 = trust_s2_pattern.search(line)
            
#             if match:
#                 iteration = int(match.group(1))
#                 scores = list(map(float, match.group(2).split(',')))
#                 iterations.append(iteration)
#                 trust_scores.append(scores)
            
#             if match_s1:
#                 scores_s1 = list(map(float, match_s1.group(2).split(',')))
#                 trust_scores_s1.append(scores_s1)
            
#             if match_s2:
#                 scores_s2 = list(map(float, match_s2.group(2).split(',')))
#                 trust_scores_s2.append(scores_s2)
    
#     # Transpose trust scores to get each series
#     trust_scores = list(zip(*trust_scores))
#     trust_scores_s1 = list(zip(*trust_scores_s1))
#     trust_scores_s2 = list(zip(*trust_scores_s2))
    
#     # Plotting Trust Scores
#     plt.figure(figsize=(12, 6))
#     for i, scores in enumerate(trust_scores):
#         linestyle = '--' if (0 <= i < 5) else '-'
#         plt.plot(iterations, scores, label=f'Trust Score {i+1}', linestyle=linestyle)
#     plt.xlabel('Iteration')
#     plt.ylabel('Trust Score')
#     plt.title('Trust Scores Over Iterations')
#     plt.legend(loc='center left')
#     plt.grid(True)
#     plt.savefig('/mydata/mydata/RL_agent/plot_trust/no_samefile_trustscore_no_noise_s2_v1.png')
#     plt.show()
    
#     # Plotting Trust Scores S1
#     plt.figure(figsize=(12, 6))
#     for i, scores_s1 in enumerate(trust_scores_s1):
#         linestyle = '--' if (0 <= i < 5) else '-'
#         plt.plot(iterations, scores_s1, label=f'Trust Score S1 {i+1}', linestyle=linestyle)
#     plt.xlabel('Iteration')
#     plt.ylabel('Trust Score S1')
#     plt.title('Trust Scores S1 Over Iterations')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('/mydata/mydata/RL_agent/plot_trust/no_samefile_trustscore_no_noise_s1_v1.png')
#     plt.show()
    
#     # Plotting Trust Scores S2
#     plt.figure(figsize=(12, 6))
#     for i, scores_s2 in enumerate(trust_scores_s2):
#         linestyle = '--' if (0 <= i < 5) else '-'
#         plt.plot(iterations, scores_s2, label=f'Trust Score S2 {i+1}', linestyle=linestyle)
#     plt.xlabel('Iteration')
#     plt.ylabel('Trust Score S2')
#     plt.title('Trust Scores S2 Over Iterations(th=0.2)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('/mydata/mydata/RL_agent/plot_trust/no_samefile_trustscore_no_noise_s2_v1.png')
#     plt.show()

# # Example usage:
# extract_and_plot_trust_scores("/mydata/mydata/RL_agent/no_samefile_trustscore_no_noise_v1.log")
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
    group_0_4_means = [np.mean(scores[:4]) for scores in zip(*trust_scores)]
    group_5_7_means = [np.mean(scores[5:8]) for scores in zip(*trust_scores)]
    group_8_9_means = [np.mean(scores[8:10]) for scores in zip(*trust_scores)]
    
    # Plotting all four lines
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, group_0_4_means, label='Group 0-4 Mean', linestyle='-', marker='o',markersize=1)
    plt.plot(iterations, group_5_7_means, label='Group 5-7 Mean', linestyle='--', marker='x',markersize=1)
    plt.plot(iterations, group_8_9_means, label='Group 8-9 Mean', linestyle='-.', marker='s',markersize=1)

    
    plt.xlabel('Iteration')
    plt.ylabel('Trust Scores')
    plt.title('Grouped Trust Scores Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('/mydata/mydata/RL_agent/result_plot_seed42_v2/result_v2/alpha0.9_th0.2_difffile_trustscore_noise_v1.png')
    plt.show()

# Example usage:
extract_and_plot_grouped_trust_scores("/mydata/mydata/RL_agent/result_plot_seed42_v2/alpha0.9_th0.2_difffile_trustscore_noise_v1.log")