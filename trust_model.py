import numpy as np
from itertools import combinations
import os
from concurrent.futures import ProcessPoolExecutor


import matplotlib.pyplot as plt


# Define similarity function for Pearson correlation
def similar(x, y):
    return np.corrcoef(x, y)[0, 1]

# Trust calculation function
# def trust(X, i, th):
#     m = X.shape[0]
#     s_1 = (1 / m) * sum(similar(X[i], X[j]) for j in range(m) if j != i)

#     largest_subset_size = 0
#     G_i = []

#     for subset_size in range(2, m + 1):
#         for subset in combinations(range(m), subset_size):
#             if i in subset and all(similar(X[j], X[k]) >= th for j, k in combinations(subset, 2)):
#                 if len(subset) > largest_subset_size:
#                     largest_subset_size = len(subset)
#                     G_i = subset
#     print(len(G_i))
#     s_2 = 1 / len(G_i) if G_i else 0
#     return s_1, s_2
def trust(X, i, th):
    m = X.shape[0]
    # Vectorized similarity calculation for s_1
    similarities = np.array([similar(X[i], X[j]) for j in range(m) if j != i])
    s_1 = similarities.mean()

    # Pre-calculate similarities for subset calculations
    sim_cache = {}
    for j, k in combinations(range(m), 2):
        if j != k:
            sim = similar(X[j], X[k])
            sim_cache[(j, k)] = sim
            sim_cache[(k, j)] = sim

    largest_subset_size = 0
    G_i = []


    for subset_size in range(2, m + 1):
        for subset in combinations(range(m), subset_size):
            if i in subset:
                # Use cached similarities
                if all(sim_cache.get((j, k), 0) >= th for j, k in combinations(subset, 2)):
                    if len(subset) > largest_subset_size:
                        largest_subset_size = len(subset)
                        G_i = subset
    
    s_2 = 1 / len(G_i) if G_i else 0
    return s_1, s_2

# Extract RelativeHumidity_2m column data from file and chunk it
def preprocess_relative_humidity(file_path, chunk_size, add_noise=False, noise_min=-0.1, noise_max=0.1):
    relative_humidity_data = []
    try:
        with open(file_path, 'r') as file:
            next(file)  # Skip header
            for line in file:
                columns = line.strip().split(",")  # Assumes CSV format
                relative_humidity = float(columns[9])  # 8th index is RelativeHumidity_2m
                relative_humidity_data.append(relative_humidity)

        if add_noise:
            # Add uniform random noise using numpy
            relative_humidity_data = np.array(relative_humidity_data)
            noise = np.random.uniform(noise_min, noise_max, relative_humidity_data.shape)
            relative_humidity_data = np.clip(relative_humidity_data + noise, a_min=0, a_max=None)  # Clip to non-negative

        # Chunk the data
        return [relative_humidity_data[i:i + chunk_size] for i in range(0, len(relative_humidity_data), chunk_size)]
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []


def load_datasets_with_clusters(dataset_directory, chunk_size):
    available_datasets = []
    noisy_towers_same = [7,8,9]  # Pairs of towers with the same noise
    noisy_towers_random = [10,11]  # Towers with independent random noise
    no_noise_towers = [2, 3, 4, 5, 6]  # Towers without noise
    np.random.seed(42)
    # Generate noise for clusters with same noise
    # same_noise = np.random.uniform(15, 30, (21, 5))
    same_noise = np.random.normal(loc=0, scale=70, size=(21, 5))

    for tower_number in range(2, 12):  # Tower2 to Tower11
        file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
        
        if os.path.exists(file_path):
            chunks = preprocess_relative_humidity(
                file_path, chunk_size, add_noise=False
            )

            if tower_number in noisy_towers_same:
                # Apply the same noise to this cluster
                # chunks = [np.clip(chunk + same_noise[:len(chunk)], a_min=0, a_max=None) for chunk in chunks]
                for i in range(len(chunks)):
                    if 60 <= i <= 80:
                        # chunks[i] = np.clip(chunks[i] + same_noise[i - 60], a_min=0, a_max=None)
                        chunks[i] = np.clip(chunks[i] + same_noise[i - 60], a_min=0, a_max=None)
                noise_info = " with same noise (chunks 60-80)"
            elif tower_number in noisy_towers_random:

                random_noise = np.random.uniform(15, 70, (21, 5))
                for i in range(len(chunks)):
                    if 60 <= i <= 80:
                        chunks[i] = np.clip(chunks[i] + random_noise[i - 60], a_min=0, a_max=None)
                noise_info = " with random noise (chunks 60-80)"

            # Towers without noise will not modify the chunks
            available_datasets.append(chunks)
            noise_info = " with same noise" if tower_number in noisy_towers_same else (
                " with random noise" if tower_number in noisy_towers_random else " without noise"
            )
            print(f"Loaded dataset for Tower {tower_number}{noise_info}: {len(chunks)} chunks")
        else:
            print(f"Warning: Dataset file not found: {file_path}")
            available_datasets.append([])  # Add an empty list for missing datasets

    return available_datasets
# Define variables

def parallel_trust(args):
    X, i, th = args
    return trust(X, i, th)

def save_datasets(available_datasets, save_directory):
    """
    Save processed datasets for different sensors into separate files.
    
    Parameters:
        available_datasets (list of list): A list containing data chunks for each sensor.
        save_directory (str): Directory to save the datasets.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for i, sensor_data in enumerate(available_datasets, start=1):
        # Construct the file path for each sensor
        file_path = os.path.join(save_directory, f"sensor_{i}_data.csv")
        
        # Flatten the chunks into a single array for saving
        if sensor_data:
            flattened_data = np.concatenate(sensor_data)
            np.savetxt(file_path, flattened_data, delimiter=',', header='RelativeHumidity', comments='')
            print(f"Saved data for Sensor {i} to {file_path}")
        else:
            print(f"No data available for Sensor {i}. Skipping.")



dataset_directory = "/mydata/mydata/actuallyuse/towerdataset"
chunk_size = 5  # Number of events per chunk (per node)
available_datasets = load_datasets_with_clusters(dataset_directory, chunk_size)
# save_directory = "/mydata/mydata/RL_agent/noise_vs_dataset/"
# save_datasets(available_datasets, save_directory)
m = 10  # Number of nodes (sensors)
T = 100  # Duration of trust calculation loop
W, alpha, th = 0.5, 0.5, 0.1
Tr = [0 for _ in range(m)]  # Initialize trust scores


print("W, alpha, th", W, alpha, th,flush=True)
for t in range(T):
    # Populate matrix X with one chunk from each dataset for each sensor node
    X = np.zeros((m, chunk_size))
    for i in range(m):
        # Select a specific chunk for each node from `available_datasets`
        if available_datasets[i]:
            chunk = available_datasets[i][t % len(available_datasets[i])]  # Cycle through chunks
            X[i] = chunk  # Directly assign the chunk as relative humidity values
    print("X.shape", X.shape,flush=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        args_list = [(X, i, th) for i in range(m)]
        results = list(executor.map(parallel_trust, args_list))
    
    S_1, S_2 = zip(*results)
    S_1, S_2 = list(S_1), list(S_2)

    
    print("s1 and s2 score at iteration", t,flush=True)
    print("S_1", S_1,flush=True)
    print("S_2", S_2,flush=True)
    # Update overall trust scores
    for i in range(m):
        Tr[i] = alpha * Tr[i] + (1 - alpha) * ((S_1[i] / max(S_1)) - (S_2[i] / max(S_2)))
    print(f"Trust scores at iteration {t + 1}: {Tr}",flush=True)
