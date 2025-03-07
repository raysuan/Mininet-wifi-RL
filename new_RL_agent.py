import os
import gym
from gym import spaces
from torch import nn
import torch
from collections import deque, defaultdict
import itertools
import numpy as np
import random
from scipy.stats import poisson
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import msgpack
import matplotlib.pyplot as plt
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()
import pickle
import dill

# TOTAL_ENERGY = 100
log_directory = "/mydata/mydata/RL_agent/output_multi_cluster"



def load_q_table(filename):
    with open(filename, 'rb') as f:
        q_values = dill.load(f)
    return q_values
def rmse(e1,e2):
    return np.sqrt(np.mean((e1 - e2) ** 2))

# def read_temperature_data(file_path):
#     with open(file_path, 'r') as file:
#         # Skip the header line
#         next(file)
#         # Read only the temperature data (9th column, index 8)
#         data = [float(line.strip().split(',')[8]) for line in file]
#     return np.array(data)
def read_temperature_data(file_path):
    with open(file_path, 'r') as file:
        # Skip the first 4 lines
        
        data = []
        for line in file:
            try:
                # Split the line and try to convert the temperature value (9th column, index 8) to float
                temperature = float(line.strip().split(',')[8])
                data.append(temperature)
            except (ValueError, IndexError):
                # If conversion fails or the line doesn't have enough columns, skip this line
                continue
    
    return np.array(data)



class RLScheduler:
    def __init__(self, num_sensors, state_size, action_size, num_levels):
        self.num_sensors = num_sensors
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = defaultdict(lambda: np.random.rand(action_size) * 0.01)
        self.num_levels = num_levels

    def get_action(self, state, training=True, epsilon=0.2):
        if training and np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state, alpha=0.1, gamma=0.95):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += alpha * (target - self.q_table[state][action])

class WSNEnvironment(gym.Env):

    metadata = {"render_modes": ["console"]}
    def _load_temperature_data(self, data_directory):
        temperature_data = []
        
        for i in range(self.num_sensors):
            file_path = os.path.join(data_directory, f'ch_received_from_sensor_cluster_{self.cluster_id}_sensor_{i}.txt')
            if os.path.exists(file_path):
                sensor_data = read_temperature_data(file_path)
                temperature_data.append(sensor_data)
            else:
                print(f"Warning: Temperature data file not found: {file_path}")
        
        if not temperature_data:
            print("Warning: No temperature data loaded")
            return np.array([[]])  # Return an empty 2D array
        
        # Find the length of the shortest data array
        min_length = min(len(data) for data in temperature_data)
        
        # Truncate all arrays to the shortest length
        truncated_data = [data[:min_length] for data in temperature_data]
        
        result = np.array(truncated_data)
        print(f"Final temperature data shape: {result.shape}")
    
        return result
    
    def __init__(self, num_sensors=10, cluster_id=0,sensor_coverage=0.4, sampling_freq = 2, max_steps=50, threshold_prob=0.3):
        super(WSNEnvironment, self).__init__()

        # Environment parameters
        self.num_sensors = num_sensors
        self.sensor_coverage = sensor_coverage
        self.max_steps = max_steps
        self.threshold_prob = threshold_prob
        self.alpha=0.6
        self.beta=0.3
        self.cluster_id = cluster_id
        self.cluster_id = 0
        cluster_id = 0

        ## this is where we should load the temprature data
        # self.temprature = np.random.randint(10, 12, size=(num_sensors-1, 100))
        self.temprature = self._load_temperature_data(log_directory)
        # self.remaining_energy = [energy] * self.num_sensors

        # Initialize the environment
        self.sensor_information, self.num_points = self._generate_sensor_positions(self.temprature[:,0])

        # Define observation space
        # self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_points, 3), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_points, 1), dtype=np.float32)

        self.sampling_freq = sampling_freq

        ## Define the action_space
        self.action_space = spaces.MultiDiscrete([self.sampling_freq] * self.num_points)



        # Internal state variables
        self.step_count = 0

        # information dictionary
        self.info = {}

        # the generated event
        self.event = None

        # Number of generated events
        self.generated_events = 0
        self.similarity=0
        self.similarity_penalty=0
        
        
        self.global_time = 0
        self.scheduler = None

    def initialize_scheduler(self, state_size, action_size, num_levels):
        self.scheduler = RLScheduler(10, state_size, action_size, num_levels)
        
        
    def discretize_value(self, value, threshold=0.3):
        return 1 if value > threshold else 0

    def calculate_jaccard_similarity(self, step):
        similarities = {}
        similarity_log_file = f"/mydata/mydata/RL_agent/similarity_gpu/similarity_log_cluster_{self.cluster_id}.txt"
        with open(similarity_log_file, 'a') as log_file:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"Step {self.step_count} at {current_time}:\n")
                
                # Calculate Jaccard similarities
            for i in range(len(self.sensor_information)):
                for j in range(i + 1, len(self.sensor_information)):
                        # Round values for comparison
                    value_i = round(float(self.sensor_information[i,0]), 3)
                    value_j = round(float(self.sensor_information[j,0]), 3)
                        
                        # Convert to sets for Jaccard calculation
                    set_i = {value_i}
                    set_j = {value_j}
                        
                        # Calculate Jaccard similarity
                    intersection = len(set_i.intersection(set_j))
                    union = len(set_i.union(set_j))
                    similarity = intersection / union if union > 0 else 0
                    similarities[(value_i, value_j)] = similarity
                    log_file.write(f"\tJaccard similarity between sensor {i} and sensor {j}: {similarity}\n")
                        # Penalize pairs with high similarity and high frequency rates
                    

        return similarities
    
    def calculate_similarity_score(self, sensor_id, jaccard_similarities, alpha, beta, energy_data, frequency_data):
        similarity = 0.0
        n = len(jaccard_similarities)
        for (i, j), jaccard in jaccard_similarities.items():
            if sensor_id == i or sensor_id == j:
                similarity += sigmoid(alpha * jaccard) * sigmoid(beta * frequency_data[0][sensor_id])
        return similarity / max(n, 1)
    
    def calculate_reward(self, f_i, L_i_prev, lambda_param):
        return -lambda_param * f_i * L_i_prev
    
    def determine_usage(self, n_steps,energy_data,frequency_data, alpha=1, beta=1, lambda_param=0.1, training=True):
        num_sensors = self.sensor_information.shape[0]
        sensor_data = self.sensor_information
        # num_rows = min(len(df) for df in sensor_data.values())
        num_rows = self.temprature.shape[1] if self.temprature.ndim > 1 else 1
        iterations = 1
        usage_array = np.zeros((num_sensors, iterations), dtype=int)
        cluster_id = 0
        for step in tqdm(range(iterations), desc="Determining sensor usage", unit="step"):
            state = tuple((energy_data[cluster_id][sensor_id], frequency_data[cluster_id][sensor_id])
              for sensor_id in range(self.num_points))
            
            jaccard_similarities = self.calculate_jaccard_similarity(step)
            
            actions = []
            for sensor_id in range(num_sensors):
                similarity = self.calculate_similarity_score(sensor_id, jaccard_similarities, alpha, beta, energy_data, frequency_data)
                similarity_level = self.discretize_value(similarity)
                
                action = self.scheduler.get_action((state, similarity_level), training=training)
                actions.append(action)
                
                # self.sensors[sensor_id].update_energy(self.global_time)
                # self.sensors[sensor_id].previous_frequency = action
                energy_data[cluster_id][sensor_id] -= action
                frequency_data[cluster_id][sensor_id] = action
                
                
                usage_array[sensor_id, step] = action

            reward = 0
            for sensor_id, action in enumerate(actions):
                f_i = action
                L_i_prev = frequency_data[cluster_id][sensor_id]
                reward += self.calculate_reward(f_i, L_i_prev, lambda_param)
                # if (sensor_id == 4):
                #     print(reward)
                #     reward += 3
                #     print("After: ", reward)

            next_state = tuple((energy_data[cluster_id][sensor_id], frequency_data[cluster_id][sensor_id]) 
                             for sensor_id in range(num_sensors))
            
            
            for sensor_id, action in enumerate(actions):
                self.scheduler.update_q_value((state, similarity_level), action, reward, next_state)
        return usage_array
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))



    def _generate_sensor_positions(self,temp):

        # Set a fixed random seed for reproducibility
        np.random.seed(42)

        num_points = 10

        # Generate the sensor positions uniformly at random within the space
        sensor_information = np.zeros((num_points, 1))

        sensor_information[:, 0] = temp[0]

   


        return sensor_information.astype(np.float32), num_points


