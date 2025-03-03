from Environment import SCIONEnvironment
# For Dijkstra routing, uncomment the next line:
# from Dijkstra_routing import DijkstraRouter
from DQNAgent import DQNAgent
import torch
import csv
import requests
import random
import numpy as np

# Set up the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def send_data_and_record(env, action, data_size, max_rate):
    """
    Sends data using POST request and records the results.
    
    Args:
        env (SCIONEnvironment): The simulation environment.
        action (int): The selected action/path.
        data_size (int): The size of data to send.
        max_rate (int): The maximum data rate.
        
    Returns:
        dict: A dictionary containing the results of the data transmission.
    """
    try:
        response = requests.post(
            f"{env.sender_url}/send",
            json={"duration": 1.0, "size": data_size}
        )
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data: {e}")
        response_data = {}

    return {
        "timestamp": env.current_time,
        "path": action,
        "data_size": data_size,
        "average_delay": response_data.get("average_delay", None),
        "elapsed_time": response_data.get("elapsed_time", None),
        "goodput_received_mbps": response_data.get("goodput_received_mbps", None),
        "goodput_sent_mbps": response_data.get("goodput_sent_mbps", None),
        "packet_loss": response_data.get("packet_loss", None),
        "total_bytes_received": response_data.get("total_bytes_received", None),
        "total_bytes_sent": response_data.get("total_bytes_sent", None)
    }


def calculate_reward(path):
    """
    Calculate reward using 70% bandwidth and 30% link trust.
    
    Note: This function uses a global 'env' variable. Ensure that 'env' is defined before calling.
    
    Args:
        path (dict): Path information containing bandwidth and latency data.
        
    Returns:
        float: Calculated reward.
    """
    bandwidth = path.get("bandwidth_mbps", 0)
    link_trust = 1 - (
        0.5 * path.get("loss_percent", 0) +
        0.5 * path.get("latency_ms", 0)
    )
    # Using env.get_range() to normalize values (env must be defined globally)
    bandwidth_range = env.get_range(bandwidth / 50, 1)
    link_trust_range = env.get_range(link_trust, 1)
    link_trust = max(0, min(link_trust, 1))  # Ensure link trust is between 0 and 1
    return 0.7 * bandwidth_range + 0.3 * link_trust_range


def find_path_reward(env):
    """
    Determine the path rewards and return sorted paths.
    
    Args:
        env (SCIONEnvironment): The simulation environment.
        
    Returns:
        tuple: A dictionary mapping each path ID to its reward, and a list of path IDs sorted by reward (descending).
    """
    paths = env.get_paths()
    sorted_paths = sorted(
        paths.items(),
        key=lambda x: calculate_reward(x[1]),
        reverse=True
    )
    paths_and_rewards = {path[0]: calculate_reward(path[1]) for path in sorted_paths}
    return paths_and_rewards, [path[0] for path in sorted_paths]


def send_path_change_request(path_id, env):
    """
    Send a request to the environment to change the selected path.
    
    Args:
        path_id (str): The path ID to switch to.
        env (SCIONEnvironment): The simulation environment.
    """
    try:
        response = requests.get(f"{env.path_selection_url}{path_id}")
        response.raise_for_status()
        print(f"Path changed successfully to path ID: {path_id}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to change path: {e}")


def evaluate_model(model_type, env, output_file, model_path="dqn_model_16.pth"):
    """
    Evaluate the performance of the selected model and save the results.
    
    Args:
        model_type (str): Either 'dqn', 'dijkstra', 'random', or 'optimal'.
        env (SCIONEnvironment): The simulation environment.
        output_file (str): The output file name to save summary results.
        model_path (str): Path to the DQN model file (only used if model_type is "dqn").
    """
    # Advance the traffic model to initialize env.current_time
    env.advance_traffic_model()

    if model_type == "dqn":
        # Load the pre-trained DQN model using a parameterized model path
        agent = DQNAgent(env.state_size, env.action_size)
        agent.load(model_path)

        def select_action(state):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                act_values = agent.model(state_tensor).cpu().numpy()
            chosen_action = np.argmax(act_values)
            with open('path_state_statistics.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow((state, chosen_action))
            send_path_change_request(path_id=str(chosen_action), env=env)
            return str(chosen_action)

    elif model_type == "dijkstra":
        # Initialize Dijkstra routing (ensure to uncomment the DijkstraRouter import if needed)
        router = DijkstraRouter(env, metric="latency")

        def select_action(state):
            best_path, best_path_id, best_cost = router.find_shortest_path()
            if best_cost < router.current_best_cost or best_path_id != router.current_best_path_id:
                router.logger.info(
                    f"Path updated! New best path based on {router.metric}: {(best_path_id, best_path)} with cost: {best_cost}"
                )
                router.current_best_path_id = best_path_id
                router.current_best_cost = best_cost
                router.send_path_change_request(best_path_id)
            return best_path_id

    elif model_type == "random":
        def select_action(state):
            chosen_action = random.randrange(env.action_size)
            send_path_change_request(path_id=str(chosen_action), env=env)
            return str(chosen_action)

    elif model_type == "optimal":
        def select_action(state):
            # Find the path with the maximum reward
            chosen_action = max(find_path_reward(env).items(), key=lambda x: x[1])
            send_path_change_request(path_id=str(chosen_action), env=env)
            return str(chosen_action)

    else:
        raise ValueError("Invalid model type. Choose 'dqn', 'dijkstra', 'random', or 'optimal'.")

    # Initialize counters for statistics
    total_decisions = 0
    counter = 0
    episode = 1 
    max_rate = 50
    data_sizes = [10, 20, 30, 50]

    while not env.simulation_completed:
        for data_size in data_sizes:
            # Write best paths to CSV every 40 iterations
            if counter % 40 == 0:
                counter = 0
                with open("best_paths.csv", 'a') as file:
                    writer = csv.writer(file)
                    file.write(f"Episode {episode}\n")
                    episode += 1
                    writer.writerow(find_path_reward(env))
            state = env.get_state(response=True)
            action = select_action(state)
            counter += 1

            # Record statistics on path selections
            paths_dict, top_3_paths = find_path_reward(env)
            total_decisions += 1
            paths_count[int(action)] += 1
            top_counts[top_3_paths.index(action)] += 1

            print(f"\nData sent information\nData Size: {data_size} | Path: {action} | Time: {env.current_time}")
            print(f"Top 3 paths: {top_3_paths}")

            # Break the loop if simulation is completed
            if env.simulation_completed:
                break

    # Calculate and save selection percentages to a summary file
    paths_percentage = [(count / total_decisions) * 100 for count in paths_count]
    top_paths_percentage = [(count / total_decisions) * 100 for count in top_counts]

    with open(output_file.replace(".csv", "_summary.txt"), "w") as summary_file:
        summary_file.write("Path selection\n")
        for index, path_percent in enumerate(paths_percentage):
            summary_file.write(f"path {index} selection percentage: {path_percent:.2f}%\n")
        summary_file.write("Top paths percentage\n")
        for index, path_percent in enumerate(top_paths_percentage, start=1):
            summary_file.write(f"Top {index} selection percentage: {path_percent:.2f}%\n")
    print(f"Simulation with {model_type} completed. Results saved to {output_file}.")


if __name__ == "__main__":
    # Initialize the simulation environment
    env = SCIONEnvironment(
        sender_url="http://10.105.0.71:5000",
        receiver_url="http://10.106.0.71:5002",
        paths_url="http://10.101.0.71:8050/get_paths",
        path_selection_url="http://10.105.0.71:8010/paths/",
        step_update_frequency = 1
    )

    # Evaluate the DQN model with a general model path
    dqn_output_file = "dqn_simulation_results.csv"
    paths_count = np.zeros(12)
    top_counts = np.zeros(12)
    # You can now change the model file by passing a different path if needed.
    evaluate_model("dqn", env, dqn_output_file, model_path="dqn_model_16.pth")

    # To evaluate other models, uncomment the desired lines below:
    # evaluate_model("dijkstra", env, "dijkstra_simulation_results.csv")
    # evaluate_model("random", env, "random_simulation_results.csv")
    # evaluate_model("optimal", env, "optimal_simulation_results.csv")
