import sys
import os
import time
import threading
import random
import json
from tqdm import tqdm
from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from RL_agent import WSNEnvironment, WSNEnvironmentAgent

f = 10  # Number of sensors
log_directory = "/mydata/mydata/RL_agent/output"
dataset_directory = "/mydata/mydata/actuallyuse/towerdataset"


if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Due to limitation of thread control in Mininet-Wifi, so I use a rate file to represent the change in rate that send by cluster
for sensor_id in range(f):
    rate_file = f'{log_directory}/sensor_{sensor_id}_rate.txt'
    with open(rate_file, 'w') as file:
        file.write("2")

print(f"Created rate files for {f} sensors in {log_directory}")    

#size for each chunk
chunk_size = 5000


def check_received_data(base_output_file, num_sensors, min_lines=10):
    #To run the RL agent, we need to receive at least 10 line in cluster head.
    for i in range(num_sensors):
        file_path = f'{base_output_file}_{i}.txt'
        if not os.path.exists(file_path):
            return False
        with open(file_path, 'r') as file:
            if sum(1 for _ in file) < min_lines:
                return False
    return True

def preprocess_dataset_into_chunks(dataset_path, chunk_size):
    # cache the dataset into memory first
    with open(dataset_path, 'r') as file:
        lines = file.readlines()
    # chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    # return chunks + chunks
    return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

datasets = {}
    # Each sensor will be stored different dataset
for i in range(f):
    tower_number = i + 2  # Start from tower2 to tower11
    file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
    if os.path.exists(file_path):
        datasets[i] = preprocess_dataset_into_chunks(file_path, chunk_size)
    else:
        print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
        datasets[i] = []
        
def send_messages(sensor, ch_ip, sensor_id):
    #function used to send message from sensor to cluster head

    chunks = datasets[sensor_id]
    info(f"chunk size {len(chunks)}\n")
    rate_file = f'/mydata/mydata/RL_agent/output/sensor_{sensor_id}_rate.txt'
    if not chunks:
        info(f"Sensor {sensor_id}: No data available. Skipping send_messages.\n")
        return
    
    packetnumber = 0
    port = 5001 + sensor_id
    rate = 2  # Initial rate
    
    for chunk in chunks:
        packet_data = ''.join(chunk)
        packet_size_kb = len(packet_data) / 1024.0
        
        #read rate from file to reflect rate change
        with open(rate_file, 'r') as file:
            rate = float(file.read().strip())
        
        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        ms = int((current_time - int(current_time)) * 1000)
        
        if rate > 0:
            sensor.cmd(f'echo "{packet_data}" | nc -q 1 -u {ch_ip} {port}')
            info(f"Sensor {sensor_id}: Sent packet {packetnumber} of size {packet_size_kb:.2f} KB at {timestamp}.{ms:03d}\n")
        else:
            info(f"Sensor {sensor_id}: Skipped sending packet {packetnumber} due to rate 0 at {timestamp}.{ms:03d}\n")
        
        packetnumber += 1
        if packetnumber == 300:
            break
            
        if rate > 0:
            time.sleep(1.0 / rate)
        else:
            time.sleep(1) 

    info(f"Sensor {sensor_id}: Finished sending messages\n")

def train_agent(env, agent, n_episodes=1000):
    #train the model
    
    episodes = 20
    for episode in tqdm(range(n_episodes)):
        obs, env_info = env.reset()
        m_obs = tuple(tuple(row) for row in obs)
        done = False

        while not done:
            action = agent.get_action(m_obs)
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
            filename = 'q_table.pkl'
            agent.save_q_table(filename)
        print('Episode:{}\t Score:{:.2f} \t{}'.format(episode, score, env_info))
    print(f"Training completed. Final epsilon: {agent.epsilon}")

def receive_messages(node):
    #function use for cluster head to receive message
    base_output_file = f'{log_directory}/ch_received_from_sensor'
    
    for i in range(f):
        #save the data that receive from different sensor to different file
        output_file = f'{base_output_file}_{i}.txt'
        node.cmd(f'touch {output_file}')
        node.cmd(f'while true; do nc -ul -p {5001 + i} >> {output_file} & done &')
        info(f"Receiver: Started listening on port {5001 + i} for sensor {i}\n")
    #capture the network by pcap
    pcap_file = f'{log_directory}/capture.pcap'
    node.cmd(f'tcpdump -i {node.defaultIntf().name} -n udp portrange 5001-{5001+f-1} -w {pcap_file} &')
    info(f"Receiver: Started tcpdump capture on ports 5001-{5001+f-1}\n")

    while True:
        time.sleep(1)

def rl_agent_process(env, agent, sensors, cluster_head):
    #RL agent make decision on rate for each sensor based on their similarity
    step = 0
    training_interval = 300  # Train every 300 steps
    training_episodes = 20

    while True:
        time.sleep(5)  # Update every 5 seconds
        obs, env_info = env.reset()
        action = agent.get_action(obs)

        new_rates = [rate + 1 for rate in action]  # Convert to 1-3 range
        info(f"Cluster Head: New rates: {new_rates}")
        
        for i, rate in enumerate(new_rates):
            #save the rate that decide by RL_agent to file, 1 means stop, 2 means 2 packet per second, 3 means 1 packet per second 
            rate_file = f'/mydata/mydata/RL_agent/output/sensor_{i}_rate.txt'
            with open(rate_file, 'w') as file:
                if rate == 1:
                    file.write(str(0))
                elif rate == 2:
                    file.write(str(2))
                elif rate == 3:
                    file.write(str(1))
            sensor_ip = sensors[i].params['ip'].split('/')[0]
            cluster_head.cmd(f'echo "{rate}" | nc -q 1 -u {sensor_ip} 6001')

        time.sleep(5)

        if step % training_interval == 0:
            print(f"Starting training at step {step}")
            train_agent(env, agent)
            print(f"Finished training at step {step}")
            
        if step % 100 == 0:
            agent.save_q_table('q_table.pkl')
            agent.decay_epsilon()
            print(f"Step {step}: Saved Q-table and decayed epsilon to {agent.epsilon}")
        step += 1

def stop_receivers(node):
    #stop receiver
    node.cmd('pkill -f "nc -ul"')
    node.cmd('pkill tcpdump')
    info("Stopped all nc receivers and tcpdump\n")

def topology(args):
    #build network
    net = Mininet_wifi(controller=Controller, link=wmediumd,
                       wmediumd_mode=interference)

    info("*** Creating nodes\n")
    #create accesspoint
    ap23 = net.addAccessPoint('ap23', ssid='new-ssid', mode='g', channel='5', position='50,50,0')

    
    #create 10 sensors
    sensors = []
    for i in range(f):
        ip_address = f'192.168.0.{i + 1}/24'
        sensors.append(net.addStation(f's{i}', ip=ip_address,
                                      range='116', position=f'{30 + i},30,0'))
    #create cluster head
    cluster_head = net.addStation('ch', ip='192.168.0.100/24',
                                  range='150', position='70,70,0')

    info("*** Adding Controller\n")
    net.addController('c0')

    info("*** Configuring wifi nodes\n")
    net.configureWifiNodes()


    info("*** Starting network\n")
    net.build()
    net.start()

    info("*** Setting up communication flow\n")
    try:
        info("*** Starting receivers\n")
        
        #let cluster starter to listen to all the sensor
        receive_thread = threading.Thread(target=receive_messages, args=(cluster_head,))
        receive_thread.start()
        
        time.sleep(2)  # Give receivers time to start


        info("*** Starting senders\n")
        sender_threads = []
        listener_threads = []
        for i, sensor in enumerate(sensors):
            tcpdump_file = f'{log_directory}/tcpdump_sender_sensor{i}.pcap'
            sensor.cmd(f'tcpdump -i s{i}-wlan0 -w {tcpdump_file} &')
            
            ch_ip = '192.168.0.100'
            thread = threading.Thread(target=send_messages, args=(sensor, ch_ip, i))
            thread.start()
            sender_threads.append(thread)
            
            # listener_thread = threading.Thread(target=continuous_rate_listener, args=(sensor, i))
            # listener_thread.daemon = True
            # listener_thread.start()
            # listener_threads.append(listener_thread)
            
        print("Waiting for initial data before starting RL agent")
        while not check_received_data(f'{log_directory}/ch_received_from_sensor', f):
            time.sleep(5)
        print("Sufficient data received. Starting RL agent.")
        info("*** Starting RL agent\n")
        env = WSNEnvironment(num_sensors=f)
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

    
        print("Training the RL agent...")
        train_agent(env, agent)
        # else:
        #     print("Loading pre-trained RL agent...")
        #     agent.load_q_table('q_table.pkl')

        rl_thread = threading.Thread(target=rl_agent_process, args=(env, agent, sensors, cluster_head))
        rl_thread.start()   

        for thread in sender_threads:
            thread.join()

        stop_receivers(cluster_head)
        receive_thread.join()
        rl_thread.join()
        # for thread in listener_threads:
        #     thread.join()
    
        for sensor in sensors:
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