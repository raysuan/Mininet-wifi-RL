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
from multi_agent import WSNEnvironment, WSNEnvironmentAgent

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
            file.write("2")
            
for cluster_id in range(num_clusters):
    for sensor_id in range(f):
        rate_file = f'{log_directory}/cluster_{cluster_id}_sensor_{sensor_id}_energy.txt'
        with open(rate_file, 'w') as file:
            file.write("100")
            
print(f"Created rate files for {f} sensors in {num_clusters} clusters in {log_directory}")

chunk_size = 5000

def check_received_data(base_output_file, num_sensors, num_clusters, min_lines=1000):
    for cluster_id in range(num_clusters):
        for i in range(num_sensors):
            file_path = f'{base_output_file}_cluster_{cluster_id}_sensor_{i}.txt'
            if not os.path.exists(file_path):
                return False
            with open(file_path, 'r') as file:
                if sum(1 for _ in file) < min_lines:
                    return False
    return True

def preprocess_dataset_into_chunks(dataset_path, chunk_size):
    try:
        with open(dataset_path, 'r') as file:
            lines = file.readlines()
        return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    except Exception as e:
        info(f"Error processing file {dataset_path}: {str(e)}")
        return []




datasets = {}
available_datasets = []

# Load all available datasets
for tower_number in range(2, 12):  # Tower2 to Tower11
    file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
    if os.path.exists(file_path):
        chunks = preprocess_dataset_into_chunks(file_path, chunk_size)
        available_datasets.append(chunks)
        info(f"Loaded dataset for Tower {tower_number}: {len(chunks)} chunks")
    else:
        info(f"Warning: Dataset file not found: {file_path}")
        available_datasets.append([])  # Add an empty list for missing datasets

# Distribute datasets across clusters
for cluster_id in range(num_clusters):
    datasets[cluster_id] = {}
    for i in range(f):
        datasets[cluster_id][i] = available_datasets[i] if i < len(available_datasets) else []
        info(f"Cluster {cluster_id}, Sensor {i}: Using data from Tower {i+2}, chunk size {len(datasets[cluster_id][i])}")

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
    
    packetnumber = 0
    port = 5001 + sensor_id
    rate = 2  # Initial rate
    threshold = 20
    recharge = 3
    full = 100
    chargecount = 0
    
    for chunk in chunks:
        
        if energy < threshold:
            chargecount +=1
            recharge_time = time.time()
            rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Energy below threshold ({energy}). Recharging...\n")
            time.sleep(recharge)
            energy = full
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Recharged to full energy ({energy}),Current time: {recharge_time},chargecount{chargecount}. Resuming operations.\n")
            sensor.energy = energy  # Update sensor's energy attribute
        
        
        packet_data = ''.join(chunk)
        packet_size_kb = len(packet_data) / 1024.0
        
        with open(rate_file, 'r') as file:
            rate = float(file.read().strip())
        
        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        ms = int((current_time - int(current_time)) * 1000)
        
        if rate > 0:
            sensor.cmd(f'echo "{packet_data}" | nc -q 1 -u {ch_ip} {port}')
            energy = energy - 4
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Sent packet {packetnumber} of size {packet_size_kb:.2f} KB at {timestamp}.{ms:03d}\n")
        else:
            energy = energy - 0.3
            info(f"Cluster {cluster_id}, Sensor {sensor_id}: Skipped sending packet {packetnumber} due to rate 0 at {timestamp}.{ms:03d}\n")
        
        packetnumber += 1
        if packetnumber == 500:
            break
        
        if rate > 0:
            time.sleep(1.0 / rate)
        else:
            time.sleep(1)
        
        with open(energy_file, 'w') as file:
            file.write(str(energy))

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

def train_and_run_agent(env, agent, sensors, cluster_head, cluster_id):
    # First, train the agent
    info(f"Training the RL agent for Cluster {cluster_id}...")
    train_agent(env, agent)
    
    # After training, start the RL agent process
    rl_agent_process(env, agent, sensors, cluster_head, cluster_id)

def rl_agent_process(env, agent, sensors, cluster_head, cluster_id):
    step = 0
    training_interval = 50
    training_episodes = 20

    while True:
        time.sleep(5)
        obs, env_info = env.reset()
        action = agent.get_action(obs)

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
                
            sensor_ip = sensors[i].params['ip'].split('/')[0]
            cluster_head.cmd(f'echo "{rate}" | nc -q 1 -u {sensor_ip} 6001')

        time.sleep(5)
            
        if step % 100 == 0:
            agent.save_q_table(f'q_table_cluster_{cluster_id}.pkl')
            agent.decay_epsilon()
            info(f"Cluster {cluster_id}, Step {step}: Saved Q-table and decayed epsilon to {agent.epsilon}")
        step += 1

def stop_receivers(node):
    node.cmd('pkill -f "nc -ul"')
    node.cmd('pkill tcpdump')
    info("Stopped all nc receivers and tcpdump\n")

def topology(args):
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
        for cluster_id, cluster_head in enumerate(cluster_heads):
            receive_thread = threading.Thread(target=receive_messages, args=(cluster_head, cluster_id))
            receive_thread.start()
            receive_threads.append(receive_thread)
        
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
            agent = WSNEnvironmentAgent(
                env=env,
                learning_rate=learning_rate,
                initial_epsilon=start_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
            )
            agents.append(agent)

            # print(f"Training the RL agent for Cluster {cluster_id}...")
            # train_agent(env, agent)

            # rl_thread = threading.Thread(target=rl_agent_process, args=(env, agent, sensors[cluster_id], cluster_heads[cluster_id], cluster_id))
            rl_thread = threading.Thread(
                target=train_and_run_agent,args=(env, agent, sensors[cluster_id], cluster_heads[cluster_id], cluster_id)
            )
            rl_thread.start()
            rl_threads.append(rl_thread)

        for thread in sender_threads:
            thread.join()

        for cluster_head in cluster_heads:
            stop_receivers(cluster_head)
        
        for thread in receive_threads:
            thread.join()
        
        for thread in rl_threads:
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