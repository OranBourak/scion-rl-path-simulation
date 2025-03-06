import requests
import numpy as np
import logging
from traffic_model import run_simulation, get_current_simulated_time, get_path_data

MAXIMUM_BANDWIDTH = 50

class SCIONEnvironment:
    def __init__(self, sender_url, receiver_url, paths_url, path_selection_url, step_update_frequency=1):
        self.sender_url = sender_url
        self.receiver_url = receiver_url
        self.paths_url = paths_url
        self.path_selection_url = path_selection_url
        # Using only 2 features: normalized day-of-week and normalized time-of-day.
        self.state_size = 2  
        self.paths = self.get_paths()
        self.action_size = len(self.paths)
        self.last_action = None
        self.simulation_completed = False

        # Frequency to update the traffic model state in steps
        self.step_update_frequency = step_update_frequency
        self.steps_counter = 0

        # Initialize the traffic model generator
        self.traffic_model = run_simulation()
        self.current_time = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_paths(self):
        try:
            response = requests.get(self.paths_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch paths: {e}")
            return {}

    def reset(self):
        self.advance_traffic_model()
        return self.get_state()

    def advance_traffic_model(self):
        try:
            self.current_time = next(self.traffic_model)
        except StopIteration:
            self.logger.info("Traffic model simulation completed.")
            self.simulation_completed = True
            self.current_time = get_current_simulated_time()

    def get_state(self, response=None, path_index=None):
        """
        Returns the current state as a NumPy array:
        [day_of_week_normalized, time_of_day_normalized]
        """
        if response is None:
            return np.array([0, 0])
        
        self.logger.info(f'Stats: {response}')
        
        self.steps_counter += 1
        if self.steps_counter % self.step_update_frequency == 0:
            self.advance_traffic_model()
            self.steps_counter = 0

        day_of_week = self.current_time.weekday()  # 0 (Monday) to 6 (Sunday)
        current_hour = self.current_time.hour
        current_minute = self.current_time.minute
        time_of_day = self.get_time_of_day(current_hour, current_minute)  # Currently returns hour

        # Normalize the features:
        day_norm = day_of_week / 6.0       # normalized to [0, 1]
        time_norm = time_of_day / 24.0      # normalized to [0, 1]

        state = np.array([day_norm, time_norm])
        self.logger.info(f"State: {state}")
        return state

    def get_time_of_day(self, current_hour, current_minute):
        # Currently, state uses only the hour.
        # Alternative: return current_hour * 2 + (current_minute // 30)
        return current_hour

    def get_range(self, value, max_value, num_ranges=40):
        step = max_value / num_ranges
        for i in range(1, num_ranges + 1):
            if value <= i * step:
                return (i - 1) / num_ranges
        return (num_ranges - 1) / num_ranges

    def calculate_link_trust(self, packet_loss, average_delay, packet_loss_weight=0.5, delay_weight=0.5):
        if average_delay is None:
            average_delay = 0
            self.logger.error("average_delay is None")
        link_trust = 1 - ((packet_loss_weight * packet_loss) + (delay_weight * average_delay))
        return max(0, min(link_trust, 1))

    def normalize_value(self, value, max_value):
        return min(max(value / max_value, 0), 1)

    def calculate_reward(self, response, goodput_weight=0.7, link_trust_weight=0.3):
        goodput = response.get('goodput_received_mbps', 0)
        goodput_normalized = self.normalize_value(goodput, max_value=50)
        goodput_range = self.get_range(goodput_normalized, max_value=1)

        average_delay = response.get("average_delay", 0)
        loss_percent = response.get('packet_loss', 0)
        link_trust = self.calculate_link_trust(loss_percent, average_delay)
        link_trust_normalized = self.normalize_value(link_trust, max_value=1)
        link_trust_range = self.get_range(link_trust_normalized, max_value=1)

        reward = 2 * ((goodput_range * goodput_weight) + (link_trust_range * link_trust_weight)) - 1
        self.logger.info(f"Reward: {reward}")
        return reward

    def step(self, action, duration=0.5):
        path = self.paths.get(f'{action}', {})
        # Change path if a different action is selected
        try:
            if action != self.last_action:
                requests.get(self.path_selection_url + f'{action}')
                self.last_action = action
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to select path: {e}")

        # The send request code is commented out; instead we use get_path_data.
        response = get_path_data(action)
        new_state = self.get_state(response, path_index=action)
        reward = self.calculate_reward(response)
        done = self.is_done(new_state)
        return new_state, reward, done

    def is_done(self, state):
        return False

    def render(self):
        self.logger.info(f"Current State: {self.get_state()}")
