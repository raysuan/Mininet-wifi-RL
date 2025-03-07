import sys
import os
import time
import threading
from tqdm import tqdm
from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from new_RL_agent import WSNEnvironment, RLScheduler
from collections import deque
import numpy as np
from itertools import combinations
import os
from concurrent.futures import ProcessPoolExecutor

total_start_time = time.time()
print("Total start time: ", total_start_time)
f = 10  # Number of sensors per cluster
num_clusters = 1  # Number of clusters
log_directory = "/mydata/mydata/RL_agent/output_multi_cluster"
dataset_directory = "/mydata/mydata/actuallyuse/towerdataset"

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

for cluster_id in range(num_clusters):
    for sensor_id in range(f):
        rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_rate.txt'
        with open(rate_file, 'w') as file:
            file.write("1")
            
for cluster_id in range(num_clusters):
    for sensor_id in range(f):
        rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_energy.txt'
        with open(rate_file, 'w') as file:
            file.write("100")

tower_locations = {
    1: {"name": "Twr 1", "coordinates": (-106.7417173, 32.59018526)},
    2: {"name": "Twr 2", "coordinates": (-106.7659365, 32.59022433)},
    3: {"name": "Twr 3", "coordinates": (-106.7535646, 32.60023873)},
    4: {"name": "Twr 4", "coordinates": (-106.7533307, 32.58088351)},
    5: {"name": "Twr 5", "coordinates": (-106.7534346, 32.59521617)},
    6: {"name": "Twr 6", "coordinates": (-106.7534466, 32.58535606)},
    7: {"name": "Twr 7", "coordinates": (-106.7590889, 32.59018652)},
    8: {"name": "Twr 8", "coordinates": (-106.7488613, 32.59018626)},
    9: {"name": "Twr 9", "coordinates": (-106.7576863, 32.59382962)},
    10: {"name": "Twr 10", "coordinates": (-106.7495726, 32.59377137)},
    11: {"name": "Twr 11", "coordinates": (-106.7497557, 32.58723514)},
    12: {"name": "Twr 12", "coordinates": (-106.7574373, 32.58735384)}
}


E_0 = 0.1  # Base energy consumption
K = 5      # Scaling factor
a = 2      # Sigmoid steepness
d_0 = 0.5  # Reference distance

def calculate_energy(d, t):
    """
    Calculate energy consumption for a single node based on distance and time.
    
    Parameters:
        d (float): Distance of the node from the base station (meters).
        t (float): Transmission time (seconds).
    
    Returns:
        float: Energy consumption rounded to two decimal places.
    """
    # Compute energy using the updated formula
    energy_value = (E_0 + K * (np.sqrt(d) / (1 + np.exp(-a * (d - d_0))))) * t
    energy_value = energy_value / 100
    return round(energy_value, 2)  

energy_data = {}
frequency_data = {}
for cluster_id in range(num_clusters):
    energy_data[cluster_id] = {}
    frequency_data[cluster_id] = {}
    for sensor_id in range(f):
         energy_data[cluster_id][sensor_id] = 100  
         frequency_data[cluster_id][sensor_id] = 1   
            
print(f"Created rate files for {f} sensors in {num_clusters} clusters in {log_directory}")

chunk_size = 30

def check_received_data(base_output_file, num_sensors, num_clusters, min_lines=100):
    for cluster_id in range(num_clusters):
        for i in range(num_sensors):
            file_path = f'{base_output_file}_cluster_{cluster_id}_sensor_{i}.txt'
            if not os.path.exists(file_path):
                return False
            with open(file_path, 'r') as file:
                if sum(1 for _ in file) < min_lines:
                    return False
    return True

# def preprocess_dataset_into_chunks(dataset_path, chunk_size):
#     try:
#         with open(dataset_path, 'r') as file:
#             lines = file.readlines()
#         return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
#     except Exception as e:
#         info(f"Error processing file {dataset_path}: {str(e)}")
#         return []
def preprocess_dataset_into_chunks(dataset_path, chunk_size):
    try:
        with open(dataset_path, 'r') as file:
            lines = file.readlines()
        
        # Convert each chunk of text lines into a NumPy array
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            # Parse each line and convert to a NumPy array
            # Assuming CSV data with comma as separator
            data = []
            for line in chunk_lines:
                try:
                    values = [float(val) for val in line.strip().split(',')]
                    data.append(values)
                except ValueError:
                    # Skip header lines or lines with non-numeric values
                    continue
            
            if data:  # Only add chunk if it has valid data
                chunks.append(np.array(data))
            else:
                chunks.append(np.zeros((chunk_size, 9)))  # Or an appropriate empty array size
        
        return chunks
    except Exception as e:
        info(f"Error processing file {dataset_path}: {str(e)}")
        return []

buffer_size = 100
sensor_buffers = {}  # Dictionary to store buffers for each sensor
flush_size = 10
# Initialize buffers for all sensors in all clusters
for cluster_id in range(num_clusters):
    sensor_buffers[cluster_id] = {}
    for sensor_id in range(f):
        sensor_buffers[cluster_id][sensor_id] = deque(maxlen=buffer_size)



datasets = {}
available_datasets = []

# # Load all available datasets
# for tower_number in range(2, 12):  # Tower2 to Tower11
#     file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
#     if os.path.exists(file_path):
#         chunks = preprocess_dataset_into_chunks(file_path, chunk_size)
#         available_datasets.append(chunks)
#         info(f"Loaded dataset for Tower {tower_number}: {len(chunks)} chunks")
#     else:
#         info(f"Warning: Dataset file not found: {file_path}")
#         available_datasets.append([])  # Add an empty list for missing datasets


available_datasets = []
    # Define tower groups for noise assignment.
noisy_towers_same = [7, 8, 9]      # Towers that share the same noise
noisy_towers_random = [10, 11]     # Towers with independent random noise
no_noise_towers = [2, 3, 4, 5, 6]   # Towers without noise

np.random.seed(42)
    # Generate noise for towers with same noise.
    # We generate one noise value per affected chunk (chunks 60 to 80: 21 values).
noise_length = 100000
same_noise = np.random.normal(loc=0, scale=70, size=(noise_length,))

for tower_number in range(2, 12):  # Tower2 to Tower11
    file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
        
    if os.path.exists(file_path):
            # Use a temperature-specific preprocessing function.
            # This function should return a list of NumPy arrays, each array representing a chunk.
            # It is assumed that column index 8 corresponds to Temperature_2m.
        chunks = preprocess_dataset_into_chunks(file_path, chunk_size)
        if tower_number in noisy_towers_same:
                # Apply the same noise vector to the Temperature_2m column in chunks 60-80.
            for i in range(len(chunks)):
                if i>=50:
                        # Modify only the Temperature_2m column (index 8) for all rows in the chunk.
                    chunks[i][:, 8] = np.clip(chunks[i][:, 8] + same_noise[i - 50],
                                                   a_min=0, a_max=None)
            noise_info = " with same noise (chunks 60-80)"
        elif tower_number in noisy_towers_random:
                # Generate independent random noise for these towers.
            random_noise = np.random.uniform(15, 70, size=(len(chunks),))
            for i in range(len(chunks)):
                if i>=50:
                    chunks[i][:, 8] = np.clip(chunks[i][:, 8] + random_noise[i - 50],
                                                   a_min=0, a_max=None)
            noise_info = " with random noise (chunks 60-80)"
        else:
            noise_info = " without noise"
            
        available_datasets.append(chunks)
        print(f"Loaded dataset for Tower {tower_number}{noise_info}: {len(chunks)} chunks")
    else:
        print(f"Warning: Dataset file not found: {file_path}")
        available_datasets.append([]) 

# Distribute datasets across clusters



for cluster_id in range(num_clusters):
    datasets[cluster_id] = {}
    for i in range(f):
        datasets[cluster_id][i] = available_datasets[i] if i < len(available_datasets) else []
        info(f"Cluster {cluster_id}, Sensor {i}: Using data from Tower {i+2}, chunk size {len(datasets[cluster_id][i])}")

# def send_messages(sensor, ch_ip, cluster_id, sensor_id):
#     chunks = datasets[cluster_id][sensor_id]
#     info(f"Cluster {cluster_id}, Sensor {sensor_id}: chunk size {len(chunks)}\n")
#     rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_rate.txt'
#     energy_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_energy.txt'
#     if not chunks:
#         info(f"Cluster {cluster_id}, Sensor {sensor_id}: No data available. Skipping send_messages.\n")
#         return
    
#     with open(energy_file, 'r') as file:
#             energy = float(file.read().strip())
    
    
#     energy_v2 = energy_data[cluster_id][sensor_id]
#     rate_v2 = frequency_data[cluster_id][sensor_id]
#     packetnumber = 0
#     port = 5001 + sensor_id
#     rate = 2  # Initial rate
#     threshold = 20
#     recharge = 3
#     full = 100
#     chargecount = 0
    
#     for chunk in chunks:
        
#         if energy < threshold:
#             chargecount +=1
#             recharge_time = time.time()
#             rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
#             info(f"Cluster {cluster_id}, Sensor {sensor_id}: Energy below threshold ({energy}). Recharging...\n")
#             time.sleep(recharge)
#             energy = full
#             energy_v2 = full
#             info(f"Cluster {cluster_id}, Sensor {sensor_id}: Recharged to full energy ({energy}),Current time: {recharge_time},chargecount{chargecount}. Resuming operations.\n")
    
        
        
#         # packet_data = ''.join(chunk)
#         if isinstance(chunk, np.ndarray):
#             # Convert the NumPy array to a list of comma-separated strings
#             rows = []
#             for row in chunk:
#                 rows.append(','.join(str(val) for val in row))
#             packet_data = '\n'.join(rows)
#         else:
#             # Handle case where chunk is still a list of strings
#             packet_data = ''.join(chunk)
#         packet_size_kb = len(packet_data) / 1024.0
        
#         with open(rate_file, 'r') as file:
#             rate = float(file.read().strip())
        
#         current_time = time.time()
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
#         ms = int((current_time - int(current_time)) * 1000)
        
#         if rate > 0:
#             sensor.cmd(f'echo "{packet_data}" | nc -q 1 -u {ch_ip} {port}')
#             energy = energy - 4
#             energy_v2 = energy_v2 - 4
#             info(f"Cluster {cluster_id}, Sensor {sensor_id}: Sent packet {packetnumber} of size {packet_size_kb:.2f} KB at {timestamp}.{ms:03d}\n")
#         else:
#             energy = energy - 0.3
#             energy_v2 = energy_v2 - 0.3
#             info(f"Cluster {cluster_id}, Sensor {sensor_id}: Skipped sending packet {packetnumber} due to rate 0 at {timestamp}.{ms:03d}\n")
        
#         packetnumber += 1
#         # if packetnumber == 100:
#         #     break
#         if time.time() - total_start_time > 1800:
#             break
        
#         if rate > 0:
#             time.sleep(1.0 / rate)
#         else:
#             time.sleep(2)
        
#         with open(energy_file, 'w') as file:
#             file.write(str(energy))
#         frequency_data[cluster_id][sensor_id] = rate
#         energy_data[cluster_id][sensor_id] = energy_v2
#     info(f"Cluster {cluster_id}, Sensor {sensor_id}: Finished sending messages\n")
def send_messages(sensor, ch_ip, cluster_id, sensor_id):
    chunks = datasets[cluster_id][sensor_id]
    info(f"Cluster {cluster_id}, Sensor {sensor_id}: chunk size {len(chunks)}\n")
    rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_rate.txt'
    energy_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_energy.txt'
    if not chunks:
        info(f"Cluster {cluster_id}, Sensor {sensor_id}: No data available. Skipping send_messages.\n")
        return

    with open(energy_file, 'r') as file:
        energy = float(file.read().strip())

    energy_v2 = energy_data[cluster_id][sensor_id]
    packetnumber = 0
    port = 5001 + sensor_id
    rate = 2  # Initial rate
    threshold = 20
    recharge = 3
    full = 100
    chargecount = 0
    
    
    
    tower1_loc = tower_locations[1]["coordinates"]  # Tower 1 coordinates
    normalized_distance = 0 
    # Get sensor's tower location (for source position)
    tower_number = sensor_id + 2  # Since towers start from 2
    if tower_number in tower_locations:
        sensor_tower_loc = tower_locations[tower_number]["coordinates"]
        # Calculate distance from sensor's tower to Tower 1
        distance = np.sqrt((sensor_tower_loc[0] - tower1_loc[0])**2 + (sensor_tower_loc[1] - tower1_loc[1])**2)
        # Calculate absolute distance value
        normalized_distance = abs(distance)
    else:
        # Default distance if tower not found
        normalized_distance = 0.5
    

    for chunk in chunks:
        # Recharge if energy is below the threshold
        if energy < threshold:
            chargecount += 1
            recharge_time = time.time()
            rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Energy below threshold ({energy}). Recharging...\n")
            time.sleep(recharge)
            energy = full
            energy_v2 = full
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Recharged to full energy ({energy}), Current time: {recharge_time}, chargecount: {chargecount}. Resuming operations.\n")

        # Prepare packet data from the chunk
        if isinstance(chunk, np.ndarray):
            rows = []
            for row in chunk:
                rows.append(','.join(str(val) for val in row))
            packet_data = '\n'.join(rows)
        else:
            packet_data = ''.join(chunk)
        packet_size_kb = len(packet_data) / 1024.0

        with open(rate_file, 'r') as file:
            rate = float(file.read().strip())

        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        ms = int((current_time - int(current_time)) * 1000)

        # Record the start time of this iteration
        iteration_start = time.time()

        if rate > 0:
            sensor.cmd(f'echo "{packet_data}" | nc -q 1 -u {ch_ip} {port}')
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Sent packet {packetnumber} of size {packet_size_kb:.2f} KB at {timestamp}.{ms:03d}\n")
            consumption_rate = 4  # Energy consumption (units/sec) when sending
        else:
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Skipped sending packet {packetnumber} due to rate 0 at {timestamp}.{ms:03d}\n")
            consumption_rate = 0.3  # Energy consumption (units/sec) when idle

        # Sleep based on the current rate
        if rate > 0:
            sleep_time = 1.0 / rate
        else:
            sleep_time = 2
        time.sleep(sleep_time)

        # Calculate elapsed time and update energy based on time consumed
        iteration_end = time.time()
        dt = iteration_end - iteration_start
        if rate > 0:
            energy_consumed = calculate_energy(normalized_distance, dt)
        else:
            energy_consumed = consumption_rate * dt
        energy -= energy_consumed
        energy_v2 -= energy_consumed

        packetnumber += 1

        # Stop after a set total run time (e.g., 30 minutes)
        if time.time() - total_start_time > 1800:
            break

        # Update energy file and shared energy data
        with open(energy_file, 'w') as file:
            file.write(str(energy))
        frequency_data[cluster_id][sensor_id] = rate
        energy_data[cluster_id][sensor_id] = energy_v2

    info(f"Cluster {cluster_id}, Sensor {sensor_id}: Finished sending messages\n")




def train_agent(env, agent, n_episodes=1000):
    episodes = 20
    for episode in tqdm(range(n_episodes)):
        obs, env_info = env.reset()
        m_obs = tuple(tuple(row) for row in obs)
        done = False

        while not done:
            action = agent.get_action(m_obs)
            if episode % 20 == 0:
                new_rates = [rate + 1 for rate in action]  # Convert to 1-3 range
                info(f"action: {action}")
                info(f"Cluster {cluster_id} Head: New rates: {new_rates}")
                
                for i, rate in enumerate(new_rates):
                    rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{i}_rate.txt'
                    history_file = f'{log_directory}/cluster_{cluster_id}_sensor_{i}_history.txt'
                    with open(rate_file, 'w') as file:
                        if rate == 1:
                            file.write(str(0))
                        elif rate == 2:
                            file.write(str(2))
                        elif rate == 3:
                            file.write(str(1))
                            
                    with open(history_file, 'a') as history:
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        history.write(f"{current_time}: Rate {rate}\n")
                    
            next_obs, reward, terminated, truncated, env_info = env.step(action)

            m_next_obs = tuple(tuple(row) for row in next_obs)
            agent.update(m_obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            m_obs = m_next_obs

        agent.decay_epsilon()
    for episode in range(1, episodes+1):
        state, env_info = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            n_state, reward, terminated, truncated, env_info = env.step(action)
            score += reward
            done = terminated or truncated
            state = n_state
        if (episode==n_episodes-1):
            filename = f'q_table_cluster_{env.cluster_id}.pkl'
            agent.save_q_table(filename)
        info(f'Cluster {env.cluster_id} - Episode:{episode}\t Score:{score:.2f} \t{env_info}')
    info(f"Training completed for Cluster {env.cluster_id}. Final epsilon: {agent.epsilon}")

def receive_messages(node, cluster_id):
    base_output_file = f'{log_directory}/ch_received_from_sensor'
    ch_received = f'{log_directory}/ch_received_from_sensors_cluster_{cluster_id}.txt'
    
    for i in range(f):
        output_file = f'{base_output_file}_cluster_{cluster_id}_sensor_{i}.txt'
        node.cmd(f'touch {output_file}')
        port = 5001 + i
        # node.cmd(f'while true; do nc -ul -p {port} >> {output_file} & done &')
        node.cmd(f'while true; do nc -ul -p {port} | tee -a {output_file} >> {ch_received} & done &')
        ##TODO::
        # node.cmd(f'do nc -ul -k -p {port} | tee -a {output_file} >> {ch_received} & done &')
        info(f"Cluster {cluster_id} Receiver: Started listening on port {port} for sensor {i}\n")

    pcap_file = f'{log_directory}/capture_cluster_{cluster_id}.pcap'
    node.cmd(f'tcpdump -i {node.defaultIntf().name} -n udp portrange 5001-5010 -w {pcap_file} &')
    info(f"Cluster {cluster_id} Receiver: Started tcpdump capture on ports 5001-5010\n")

def train_and_run_agent(env, sensors, cluster_head, cluster_id):
    # First, train the agent
    info(f"Training the RL agent for Cluster {cluster_id}...")
    # train_agent(env, agent)
    
    # After training, start the RL agent process
    rl_agent_process(env, sensors, cluster_head, cluster_id)

def rl_agent_process(env, sensors, cluster_head, cluster_id):
    # step = 0
    # training_interval = 50
    # training_episodes = 20
    env.initialize_scheduler(state_size=100, action_size=2, num_levels=2)
    while True:
        usage_array = env.determine_usage(n_steps=100, energy_data=energy_data, frequency_data=frequency_data, alpha=1, beta=1, lambda_param=0.1, training=True)
        for i in range(len(sensors)):
            print(f"Cluster {cluster_id} Sensor {i}: Usage {usage_array[i]}")   
            rate = usage_array[i]
            rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{i}_rate.txt'
            history_file = f'{log_directory}/cluster_{cluster_id}_sensor_{i}_history.txt'
            with open(rate_file, 'w') as file:
                if rate == 1:
                    file.write(str(1))
                else:
                    file.write(str(0))
                        
            with open(history_file, 'a') as history:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                history.write(f"{current_time}: Rate {rate}\n")
                    
            sensor_ip = sensors[i].params['ip'].split('/')[0]
            cluster_head.cmd(f'echo "{rate}" | nc -q 1 -u {sensor_ip} 6001')
            time.sleep(1)
    # while True:
    #     time.sleep(5)
    #     obs, env_info = env.reset()
    #     action = agent.get_action(obs)

    #     new_rates = [rate + 1 for rate in action]  # Convert to 1-3 range
    #     info(f"action: {action}")
    #     info(f"Cluster {cluster_id} Head: New rates: {new_rates}")
        
    #     for i, rate in enumerate(new_rates):
    #         rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{i}_rate.txt'
    #         history_file = f'{log_directory}/cluster_{cluster_id}_sensor_{i}_history.txt'
    #         with open(rate_file, 'w') as file:
    #             if rate == 1:
    #                 file.write(str(0))
    #             elif rate == 2:
    #                 file.write(str(2))
    #             elif rate == 3:
    #                 file.write(str(1))
                    
    #         with open(history_file, 'a') as history:
    #             current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #             history.write(f"{current_time}: Rate {rate}\n")
                
    #         sensor_ip = sensors[i].params['ip'].split('/')[0]
    #         cluster_head.cmd(f'echo "{rate}" | nc -q 1 -u {sensor_ip} 6001')

    #     time.sleep(5)
            
    #     if step % 100 == 0:
    #         agent.save_q_table(f'q_table_cluster_{cluster_id}.pkl')
    #         agent.decay_epsilon()
    #         info(f"Cluster {cluster_id}, Step {step}: Saved Q-table and decayed epsilon to {agent.epsilon}")
    #     step += 1

def count_unique_packets(cluster_id, interval=3):
    """
    Continuously monitor and count unique packets based on temperature values
    with a threshold of 0.001
    """
    file_path = f'{log_directory}/ch_received_from_sensors_cluster_{cluster_id}.txt'
    last_processed_temps = set()
    
    while True:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            total_rows = len(lines)
            current_temps = set()
            for line in lines:
                try:
                    # Extract temperature from the line
                    temp = float(line.strip().split(',')[8])
                    # Check if this temperature is unique considering the threshold
                    is_unique = all(abs(temp - existing_temp) > 0.001 for existing_temp in current_temps)
                    if is_unique:
                        current_temps.add(temp)
                except (IndexError, ValueError):
                    continue
            
            new_packets = len(current_temps - last_processed_temps)
            total_unique = len(current_temps)
            
            # Log the results
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            info(f"Total rows: {total_rows}")
            info(f"Cluster {cluster_id} - New unique packets: {new_packets}, "
                 f"Total unique packets: {total_unique}\n",current_time)
            
            # Update the set of processed temperatures
            last_processed_temps = current_temps
            
            # Sleep for the specified interval
            time.sleep(interval)
            
            if time.time() - total_start_time > 1800:
                break
            
        except Exception as e:
            info(f"Error in count_unique_packets for cluster {cluster_id}: {str(e)}\n")
            time.sleep(interval)
            

# Global dictionary to store trust scores for each sensor (keyed by cluster and sensor id)
trust_scores = {cluster_id: {sensor_id: 1.0 for sensor_id in range(f)} for cluster_id in range(num_clusters)}

def build_observation_matrix(t):
    """
    Build an observation matrix X of shape (f, chunk_size) by reading data from 
    each sensor file. For cluster 0, each sensor file is located at:
      /mydata/mydata/RL_agent/output_multi_cluster/ch_received_from_sensor_cluster_0_sensor_{i}.txt
    Only the Temperature_2m values (column index 8) are extracted.
    """
    matrix_chunk_size = 15
    X = np.zeros((f, matrix_chunk_size))
    for i in range(f):
        file_path = f"/mydata/mydata/RL_agent/output_multi_cluster/ch_received_from_sensor_cluster_0_sensor_{i}.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            if lines:
                lines = lines[1:]
                
            total_lines = len(lines)
            if total_lines == 0:
                # If the file is empty, keep zeros.
                continue

            # Calculate starting index for the chunk.
            start_index = (t * matrix_chunk_size) % total_lines

            # Wrap-around if there aren’t enough lines until the end.
            if start_index + matrix_chunk_size <= total_lines:
                chunk_lines = lines[start_index:start_index + matrix_chunk_size]
            else:
                chunk_lines = lines[start_index:] + lines[:(start_index + matrix_chunk_size) - total_lines]

            # Extract Temperature_2m values from each line.
            try:
                values = []
                for line in chunk_lines:
                    fields = line.strip().split(',')
                    if len(fields) > 8:  # Ensure there is a Temperature_2m column.
                        value = float(fields[8])
                    else:
                        value = 0.0
                    values.append(value)
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")
                values = [0.0] * matrix_chunk_size

            # Ensure we have exactly 'chunk_size' values.
            if len(values) < matrix_chunk_size:
                values += [0.0] * (matrix_chunk_size - len(values))
            elif len(values) > matrix_chunk_size:
                values = values[:matrix_chunk_size]
            X[i] = values
        else:
            print(f"File not found for sensor {i}: {file_path}")
    print("X.shape:", X.shape, flush=True)
    return X

# Example trust function and parallel wrapper (as provided earlier)
def similar(x, y):
    corr = np.corrcoef(x, y)[0, 1]
    # Option 1: Using np.isnan to check and replace
    if np.isnan(corr):
        return 0.0
    return corr

def trust(X, i, th):
    m = X.shape[0]
    # Compute s1 for sensor i (average similarity to all others)
    similarities = np.array([similar(X[i], X[j]) for j in range(m) if j != i])
    s1 = similarities.mean() if similarities.size > 0 else 0

    # Cache pairwise similarities for subset calculation
    sim_cache = {}
    for j in range(m):
        for k in range(j + 1, m):
            sim = similar(X[j], X[k])
            sim_cache[(j, k)] = sim
            sim_cache[(k, j)] = sim

    largest_subset_size = 0
    G_i = []
    from itertools import combinations
    for subset_size in range(2, m + 1):
        for subset in combinations(range(m), subset_size):
            if i in subset:
                if all(sim_cache.get((j, k), 0) >= th for j, k in combinations(subset, 2)):
                    if len(subset) > largest_subset_size:
                        largest_subset_size = len(subset)
                        G_i = subset
    s2 = 1 / len(G_i) if G_i else 0
    return s1, s2

# def parallel_trust(args):
#     X, i, th = args
#     return trust(X, i, th)

def update_trust_scores():
    W, alpha, th = 0.5, 0.5, 0.1
    t = 0
    while True:
        # Build observation matrix X from sensor files
        X = build_observation_matrix(t)
        
        # Compute trust metrics in parallel for each sensor (for cluster 0)
        # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     args_list = [(X, i, th) for i in range(f)]
        #     results = list(executor.map(parallel_trust, args_list))
    
        # S1, S2 = zip(*results)
        # S1, S2 = list(S1), list(S2)
        S1 = []
        S2 = []
        for i in range(f):
            s1, s2 = trust(X, i, th)
            S1.append(s1)
            S2.append(s2)
    
        print(f"s1 and s2 scores at iteration {t}", flush=True)
        print("S1:", S1, flush=True)
        print("S2:", S2, flush=True)
        
        # Update trust scores for sensors in cluster 0
        for i in range(f):
            max_S1 = max(S1) if max(S1) != 0 else 1
            max_S2 = max(S2) if max(S2) != 0 else 1
            new_trust = alpha * trust_scores[0][i] + (1 - alpha) * ((S1[i] / max_S1) - (S2[i] / max_S2))
            trust_scores[0][i] = new_trust
        print(f"Trust scores at iteration {t} and time {time.time()}: {trust_scores}", flush=True)
        
        t += 1  # Increment the iteration counter
        time.sleep(1)  # Adjust update frequency as needed
        if time.time() - total_start_time > 1800:
            break



def stop_receivers(node):
    node.cmd('pkill -f "nc -ul"')
    node.cmd('pkill tcpdump')
    info("Stopped all nc receivers and tcpdump\n")

def topology(args):
    start_time = time.time()
    print("start time: ", start_time)
    net = Mininet_wifi(controller=Controller, link=wmediumd,
                       wmediumd_mode=interference)

    info("*** Creating nodes\n")
    aps = []
    for i in range(num_clusters):
        aps.append(net.addAccessPoint(f'ap{i}', ssid=f'cluster{i}-ssid', mode='g', channel='1', position=f'{50+i*250},{50+i*250},0'))

    sensors = []
    cluster_heads = []
    for cluster_id in range(num_clusters):
        cluster_sensors = []
        for i in range(f):
            ip_address = f'192.168.{cluster_id}.{i + 1}/24'
            sensor = net.addStation(f'c{cluster_id}s{i}', ip=ip_address,
                                    position=f'{i*30 + cluster_id*250},{cluster_id*250},0')
            cluster_sensors.append(sensor)
            
        sensors.append(cluster_sensors)
        
        cluster_head = net.addStation(f'ch{cluster_id}', ip=f'192.168.{cluster_id}.100/24',
                                      position=f'{150 + cluster_id*250},{30 + cluster_id*250},0')
        cluster_heads.append(cluster_head)

    info("*** Adding Controller\n")
    net.addController('c0')

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()

    info("*** Creating links\n")
    for i, ap in enumerate(aps):
        for sensor in sensors[i]:
            net.addLink(sensor, ap)
        net.addLink(cluster_heads[i], ap)

    info("*** Starting network\n")
    net.build()
    net.start()

    info("*** Setting up communication flow\n")
    try:
        info("*** Starting receivers\n")
        receive_threads = []
        unique_packet_threads = []
        for cluster_id, cluster_head in enumerate(cluster_heads):
            receive_thread = threading.Thread(target=receive_messages, args=(cluster_head, cluster_id))
            receive_thread.start()
            receive_threads.append(receive_thread)
            
            
            unique_thread = threading.Thread(target=count_unique_packets, args=(cluster_id,))
            unique_thread.start()
            unique_packet_threads.append(unique_thread)
        
        time.sleep(2)  # Give receivers time to start

        info("*** Starting senders\n")
        sender_threads = []
        for cluster_id, cluster_sensors in enumerate(sensors):
            for i, sensor in enumerate(cluster_sensors):
                tcpdump_file = f'{log_directory}/tcpdump_sender_cluster{cluster_id}_sensor{i}.pcap'
                sensor.cmd(f'tcpdump -i {sensor.defaultIntf().name} -w {tcpdump_file} &')
                
                ch_ip = f'192.168.{cluster_id}.100'
                thread = threading.Thread(target=send_messages, args=(sensor, ch_ip, cluster_id, i))
                thread.start()
                sender_threads.append(thread)
        
        print("Waiting for initial data before starting RL agents")
        while not check_received_data(f'{log_directory}/ch_received_from_sensor', f, num_clusters):
            time.sleep(1)
        print("Sufficient data received. Starting RL agents.")

        trust_thread = threading.Thread(target=update_trust_scores, daemon=True)
        trust_thread.start()
        
        info("*** Starting RL agents\n")
        envs = []
        agents = []
        rl_threads = []
        for cluster_id in range(num_clusters):
            env = WSNEnvironment(num_sensors=f, cluster_id=cluster_id)
            envs.append(env)
            
            learning_rate = 0.01
            n_episodes = 10000
            start_epsilon = 1.0
            epsilon_decay = start_epsilon / (n_episodes / 2)
            final_epsilon = 0.1
            rl_thread = threading.Thread(
                target=train_and_run_agent,args=(env, sensors[cluster_id], cluster_heads[cluster_id], cluster_id)
            )
            rl_thread.start()
            rl_threads.append(rl_thread)



        trust_thread.join()

        for thread in sender_threads:
            thread.join()

        for cluster_head in cluster_heads:
            stop_receivers(cluster_head)
        
        for thread in receive_threads:
            thread.join()
        
        # for thread in rl_threads:
        #     thread.join()
        for thread in unique_packet_threads:
            thread.join()
        for cluster_sensors in sensors:
            for sensor in cluster_sensors:
                sensor.cmd('pkill tcpdump')
    
    except Exception as e:
        info(f"*** Error occurred during communication: {str(e)}\n")

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology(sys.argv)