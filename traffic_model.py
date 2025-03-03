import requests
import json
import datetime
import random
import numpy as np

# Configuration
API_URL = 'http://10.101.0.71:8050/set_link'

# Link Definitions
LINKS = [
    {'id': 'ix200', 'category': 'edge', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix201', 'category': 'edge', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix202', 'category': 'edge', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix205', 'category': 'edge', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix206', 'category': 'core', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix207', 'category': 'core', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix208', 'category': 'core', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix209', 'category': 'core', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix203', 'category': 'access', "bw": 0, "packet_loss": 0, "latency": 0},
    {'id': 'ix204', 'category': 'access', "bw": 0, "packet_loss": 0, "latency": 0},
]

# Simulation Parameters
SIMULATION_DURATION = 7 * 24 * 60 * 60   # Simulate 7 days (in seconds)
TIME_STEP = 15 * 60  # 15-minute increments (in simulated seconds)

# Random Seed for Repeatability
SEED = 42  # You can change this seed to get different repeatable runs
local_random = random.Random()
local_np_random = np.random.RandomState()
local_random.seed(SEED)
local_np_random.seed(SEED)

paths = None

# Start time in simulated time
SIMULATED_TIME_START = datetime.datetime(2023, 10, 1, 0, 0, 0)  # Arbitrary start date
current_simulated_time = SIMULATED_TIME_START


def load_paths_from_json(file_path="./topo/paths.json"):
    """
    Reads the JSON file and loads the paths into a global variable.
    Adjust the parsing logic if your JSON structure is different.
    """
    global paths
    with open(file_path, "r") as f:
        paths = json.load(f)


def get_path_data(path):
    """
    Retrieves aggregated link data for a given path.
    """
    global paths
    path_links = list(paths.get(f'{path}', {})["links"])
    links_on_path = [link for link in LINKS if link['id'] in path_links]
    bw = min([link['bw'] for link in links_on_path])
    latency = sum([link['latency'] for link in links_on_path]) / 1e3
    loss = sum([link['packet_loss'] for link in links_on_path])
    return {"goodput_received_mbps": bw, "average_delay": latency, "packet_loss": loss}


def get_current_simulated_time():
    """Returns the current simulated time."""
    return current_simulated_time


# Link Category Configurations
LINK_CATEGORIES = {
    'core': {
        'min_bw': 20, 'max_bw': 50,  # in Mbps
        'min_latency': 5, 'max_latency': 200,  # in ms
        'min_jitter': 0, 'max_jitter': 0,  # in ms
        'min_loss': 0.0, 'max_loss': 0.1,  # in %
    },
    'edge': {
        'min_bw': 10, 'max_bw': 50,
        'min_latency': 5, 'max_latency': 500,
        'min_jitter': 0, 'max_jitter': 0,
        'min_loss': 0.0, 'max_loss': 1,
    },
    'access': {
        'min_bw': 7, 'max_bw': 35,
        'min_latency': 5, 'max_latency': 100,
        'min_jitter': 0, 'max_jitter': 0,
        'min_loss': 0.0, 'max_loss': 5,
    },
}

# Markov Model States and Transitions
STATES = ['Normal', 'Congested', 'Overloaded']
STATE_TRANSITIONS = {
    'Normal': {'Normal': 0.85, 'Congested': 0.10, 'Overloaded': 0.05},
    'Congested': {'Normal': 0.10, 'Congested': 0.80, 'Overloaded': 0.10},
    'Overloaded': {'Normal': 0.05, 'Congested': 0.15, 'Overloaded': 0.80},
}

# Initialize link states randomly
link_states = {link['id']: local_random.choice(STATES) for link in LINKS}


def update_link_state(link_id, is_peak):
    """
    Update the state of a link using a Markov chain.
    Adjusts transition probabilities during peak times.
    """
    current_state = link_states[link_id]
    transitions = STATE_TRANSITIONS[current_state].copy()

    if is_peak:
        if current_state == 'Normal':
            transitions = {'Normal': 0.80, 'Congested': 0.15, 'Overloaded': 0.05}
        elif current_state == 'Congested':
            transitions = {'Normal': 0.05, 'Congested': 0.85, 'Overloaded': 0.10}
        elif current_state == 'Overloaded':
            transitions = {'Normal': 0.05, 'Congested': 0.10, 'Overloaded': 0.85}

    next_state = local_random.choices(
        population=list(transitions.keys()),
        weights=list(transitions.values())
    )[0]
    link_states[link_id] = next_state
    return next_state


def is_peak_time(link_category, sim_time):
    """
    Determines whether the given simulated time falls within peak hours for a link category.
    """
    weekday = sim_time.weekday()
    hour = sim_time.hour
    if link_category == 'access':
        return True if weekday >= 5 else (18 <= hour < 23)
    elif link_category == 'core':
        return 8 <= hour < 18 if weekday < 5 else False
    elif link_category == 'edge':
        return 8 <= hour < 23 if weekday < 5 else (12 <= hour < 23)
    else:
        return False


def calculate_link_metrics(link):
    """
    Calculate and adjust link metrics (bandwidth, latency, jitter, loss)
    based on the current simulated time, peak conditions, and link state.
    """
    global current_simulated_time
    category = link['category']
    link_id = link['id']
    peak = is_peak_time(category, current_simulated_time)
    state = update_link_state(link_id, peak)
    config = LINK_CATEGORIES[category]

    time_factor = 1.0 if peak else 0.5

    # Base metrics calculations
    base_bw = config['max_bw'] * time_factor
    base_latency = config['min_latency'] / time_factor
    base_jitter = config['min_jitter'] / time_factor
    base_loss = config['min_loss'] * time_factor

    if state == 'Normal':
        state_factor = 1.0
    elif state == 'Congested':
        state_factor = 0.6
    elif state == 'Overloaded':
        state_factor = 0.3

    adjusted_bw = base_bw * state_factor
    adjusted_latency = base_latency / state_factor
    adjusted_jitter = base_jitter / state_factor
    adjusted_loss = base_loss * (1 / state_factor)

    # Ensure metrics are within defined bounds
    adjusted_bw = max(config['min_bw'], min(config['max_bw'], adjusted_bw))
    adjusted_latency = max(config['min_latency'], min(config['max_latency'], adjusted_latency))
    adjusted_jitter = max(config['min_jitter'], min(config['max_jitter'], adjusted_jitter))
    adjusted_loss = max(config['min_loss'], min(config['max_loss'], adjusted_loss))

    # Introduce randomness
    adjusted_bw += local_np_random.normal(0, (config['max_bw'] - config['min_bw']) * 0.05)
    adjusted_latency += local_np_random.normal(0, (config['max_latency'] - config['min_latency']) * 0.05)
    adjusted_jitter += local_np_random.normal(0, (config['max_jitter'] - config['min_jitter']) * 0.05)
    adjusted_loss += local_np_random.normal(0, (config['max_loss'] - config['min_loss']) * 0.05)

    # Clamp metrics after randomness
    adjusted_bw = max(config['min_bw'], min(config['max_bw'], adjusted_bw))
    adjusted_latency = max(config['min_latency'], min(config['max_latency'], adjusted_latency))
    adjusted_jitter = max(config['min_jitter'], min(config['max_jitter'], adjusted_jitter))
    adjusted_loss = max(config['min_loss'], min(config['max_loss'], adjusted_loss))

    return {
        'id': link['id'],
        'bw': adjusted_bw,
        'latency': adjusted_latency,
        'jitter': adjusted_jitter,
        'loss': adjusted_loss
    }


def apply_metrics_to_link(link, metrics):
    """
    Applies the calculated metrics to a link and sends an API request to update it.
    """
    payload = {
        'link': link['id'],
        'bw': metrics['bw'],
        'latency': metrics['latency'],
        'jitter': metrics['jitter'],
        'loss': metrics['loss']
    }
    # Update local link dictionary
    link["bw"] = metrics['bw']
    link["latency"] = metrics['latency']
    link["jitter"] = metrics['jitter']
    link["packet_loss"] = metrics['loss']

    # Send API request to update the link metrics
    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=2  # seconds
        )
        if response.status_code != 200:
            print(f"[{current_simulated_time}] Failed to update link {link['id']}: {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f"[{current_simulated_time}] Exception when updating link {link['id']}: {e}")


def run_simulation():
    """
    Main simulation loop.
    Loads path data, then iteratively updates link metrics based on the simulated time.
    """
    load_paths_from_json()
    global current_simulated_time

    simulated_time_seconds = 0

    while simulated_time_seconds < SIMULATION_DURATION:
        # Update the current simulated time
        current_simulated_time = SIMULATED_TIME_START + datetime.timedelta(seconds=simulated_time_seconds)
        print(f"\n\n######### Current simulated time: {current_simulated_time} #########")

        # Calculate and apply metrics for each link periodically
        for link in LINKS:
            metrics = calculate_link_metrics(link)
            if simulated_time_seconds % 12000 == 0:
                apply_metrics_to_link(link, metrics)

        # Yield the current simulated time (if further processing is needed)
        yield current_simulated_time

        simulated_time_seconds += TIME_STEP


if __name__ == "__main__":
    # Iterate through the simulation generator to run the simulation
    for _ in run_simulation():
        pass
