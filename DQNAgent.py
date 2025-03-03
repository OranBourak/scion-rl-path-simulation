import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import csv
from Environment import SCIONEnvironment

# Set up the device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
gamma = 0.7
epsilon = 0.995           # Starting epsilon for each episode
epsilon_min = 0.1         # Minimum epsilon for epsilon-greedy
epsilon_max = 0.995       # Maximum epsilon (reset each episode)
epsilon_increment = 0.35  # How much to bump epsilon if reward drops during exploitation
epsilon_decay = 0.997     # Decay factor for epsilon (applied each step)
batch_size = 256
replay_memory_buffer = 100000
target_update_frequency = 2
previous_reward = 0

## SEED ##
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Soft update parameter for the target network
soft_update_tau = 0.01

# Define the Q-Network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_shape)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # now 2 (day and time)
        self.action_size = action_size
        self.memory_buffer = deque(maxlen=replay_memory_buffer)
        self.recent_rewards = deque(maxlen=10)  # Track last 10 rewards
        self.priority_buffer = deque(maxlen=replay_memory_buffer)  # Priority scores
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.previous_reward = previous_reward
        self.epsilon_decay = epsilon_decay
        self.path_counts = np.zeros(action_size)  # Track path selection counts
        self.epsilon_priority = 1e-5  # Small value to avoid zero probabilities
        self.alpha = 0.6  # Controls how much prioritization is used (0 = uniform sampling)
        # Track whether the last action was exploration (True) or exploitation (False)
        self.action_flag = None

        self.model = QNetwork(state_size, action_size).to(device)
        print(f"Model is on GPU: {next(self.model.parameters()).is_cuda}")
        self.target_model = QNetwork(state_size, action_size).to(device)
        self.update_target_model(hard=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self, hard=False):
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            # Soft update: target = tau * model + (1-tau) * target
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(soft_update_tau * param.data + (1 - soft_update_tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))
        # Assign initial priority
        max_priority = max(self.priority_buffer) if self.priority_buffer else 1.0
        self.priority_buffer.append(max_priority)

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            chosen_action = random.randrange(self.action_size)
            print(f'\nChosen action (exploration): {chosen_action}')
            flag = True
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                act_values = self.model(state_tensor).cpu().numpy()
            chosen_action = np.argmax(act_values)
            flag = False
            print(f'\nChosen action (exploitation): {chosen_action}')
            print(f'Model output values: {act_values}')

        self.path_counts[chosen_action] += 1  # Update path count for chosen action
        self.action_flag = flag  # Save whether it was exploration or exploitation
        return chosen_action, flag

    def replay(self, batch_size):
        if len(self.memory_buffer) < batch_size:
            return

        # Convert priorities to probabilities
        priorities = np.array(list(self.priority_buffer), dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()  # Normalize to sum to 1

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory_buffer), size=batch_size, p=probabilities)
        with open("indicies.csv", "a", newline="") as stats_file:
            writer = csv.writer(stats_file)
            writer.writerow(indices)

        # Gather minibatch
        minibatch = [self.memory_buffer[idx] for idx in indices]
        importance_sampling_weights = (1 / (len(self.memory_buffer) * probabilities[indices])) ** (1 - self.alpha)
        importance_sampling_weights /= importance_sampling_weights.max()  # Normalize weights

        # Process the minibatch
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(device).squeeze(1)
        actions = torch.LongTensor(actions).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device).squeeze(1)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q-values from the online network
        q_values = self.model(states)
        next_actions = q_values.argmax(1)
        current_q_values = q_values.gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        target_q_values = target_q_values.unsqueeze(1)

        # Compute loss and apply importance sampling weights
        loss = self.loss_fn(current_q_values, target_q_values)
        loss = (loss * torch.FloatTensor(importance_sampling_weights).to(device)).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities based on TD error
        td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy().flatten()
        for i, idx in enumerate(indices):
            self.priority_buffer[idx] = float(td_errors[i]) + self.epsilon_priority

        # Dynamic epsilon adjustment:
        # If the most recent reward is significantly lower than the rolling average and the last action was exploitation,
        # then increase epsilon (to encourage more exploration); otherwise, decay epsilon.
        rolling_avg_reward = np.mean(self.recent_rewards)
        last_reward = self.recent_rewards[-1]
        print(f"Last reward: {last_reward}, Rolling average: {rolling_avg_reward}, Epsilon before update: {self.epsilon}")
        if (last_reward < rolling_avg_reward - 0.01) and (self.action_flag == False):
            # If exploitation resulted in a reward much lower than average, increase epsilon to explore more
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
            print(f"Reward dropped during exploitation; increasing epsilon to {self.epsilon}")
        else:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Decaying epsilon to {self.epsilon}")

        # Soft update the target network
        self.update_target_model(hard=False)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":
    # Corrected environment instantiation with step_update_frequency parameter (set to 20 for DQNAgent)
    env = SCIONEnvironment(
        sender_url="http://10.105.0.71:5000",
        receiver_url="http://10.106.0.71:5002",
        paths_url="http://10.101.0.71:8050/get_paths",
        path_selection_url="http://10.105.0.71:8010/paths/",
        step_update_frequency=20
    )
    state_size = env.state_size  # Now state_size is 2 (day_of_week_normalized, time)
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    episodes = 1000

    with open("episode_stats.csv", "w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(["Episode", "Total_Actions", "Actions_Taken", "Path Counts"])

        for e in range(episodes):
            # Reset path counts and environment state at the start of each episode
            agent.path_counts = np.zeros(action_size)
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            actions_taken = []  # Track the sequence of actions

            for t in range(800):
                print(state)
                action, action_type = agent.act(state)
                actions_taken.append((action, action_type, state[0][0], state[0][1]))
                next_state, reward, done = env.step(action)
                agent.recent_rewards.append(reward)
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory_buffer) > batch_size:
                    agent.replay(batch_size)

                # Use simulation_completed flag to end training if simulation finishes
                if env.simulation_completed:
                    agent.save(f"dqn_model_{e}.pth")
                    total_actions = len(actions_taken)
                    writer.writerow([e, total_actions, actions_taken, agent.path_counts])
                    print("Simulation completed")
                    exit()

                if done:
                    break

            total_actions = len(actions_taken)
            writer.writerow([e, total_actions, actions_taken, agent.path_counts])
            print(f"Episode {e}: Total Actions = {total_actions}")
            # Optionally, update the target model every few episodes
            # if e % target_update_frequency == 0:
            #     agent.update_target_model(hard=False)
            agent.save(f"dqn_model_{e}.pth")
