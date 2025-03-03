# SCION Path Selection and Traffic Simulation Project

## Overview

This project implements an advanced reinforcement learning (RL) framework designed for network path selection and traffic simulation within a SCION-based environment. It uses a Deep Q-Network (DQN) agent—built with PyTorch—to dynamically select the best network path under varying traffic conditions. The system consists of several interconnected components:

- **DQN Agent:** Uses deep reinforcement learning to learn optimal path selection. It incorporates a Q-network architecture, experience replay, epsilon-greedy action selection, and both soft and hard target network updates. The agent adjusts its exploration rate dynamically based on recent rewards, ensuring effective learning in a non-stationary environment.
- **Environment Interface:** The `SCIONEnvironment` class manages interactions with the SCION network simulation. It retrieves available paths via API calls, computes the current state based on normalized features (day-of-week and time-of-day), and calculates rewards derived from key network performance metrics such as goodput, delay, and packet loss.
- **Traffic Simulation:** A traffic model module simulates realistic network conditions by updating link metrics (bandwidth, latency, jitter, and packet loss) through a Markov chain model. It continuously adjusts link states (Normal, Congested, Overloaded) and communicates these changes via API calls.
- **Analysis Module:** Evaluates and compares various path selection strategies including DQN-based selection, Dijkstra routing (when enabled), random selection, and optimal strategies. It logs detailed performance statistics and exports results to CSV files.
- **Base SCION Testbed Integration:** The project relies on a base SCION testbed (seed-emulator) to simulate the network environment. Detailed setup instructions—including cloning, starting Docker containers, and running helper scripts—are provided in the accompanying PDF (`instructions (2).pdf`).

This comprehensive system is designed for network researchers and engineers who wish to experiment with RL-driven network optimization and to simulate realistic traffic scenarios in a controlled environment.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Cloning the Base Project](#cloning-the-base-project)
- [Usage](#usage)
- [API Commands](#api-commands)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **Python 3.x** – Ensure that Python is installed.
- **Virtual Environment** – Use `venv` (or similar) to manage dependencies.
- **Git** – Required for cloning repositories and version control.
- **Docker** – For running the SCION testbed components.
- **Required Python Packages** – All dependencies are listed in the `requirements.txt` file (includes PyTorch, TorchVision, NumPy, Requests, and more).  
  (See the `requirements.txt` file for full details.)

## Installation

1. **Clone This Repository:**

   ```bash
   git clone "https://github.com/OranBourak/scion-rl-path-simulation.git"
   cd <your-project-directory>
   ```

2. **Set Up the Virtual Environment:**

   Create and activate a virtual environment:
   - **On Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **On macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**

   Install all required packages with:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- **DQNAgent.py**  
  **Responsibility:**  
  Implements the DQN agent responsible for learning the optimal network path. Key features include:
  - Deep Q-network architecture with PyTorch.
  - Experience replay and epsilon-greedy action selection.
  - Dynamic adjustment of exploration rate based on recent rewards.
  - Logging of path selections and model checkpoints.
  
  **Usage:** Run this script to begin training the DQN agent.
  
- **Environment.py**  
  **Responsibility:**  
  Contains the `SCIONEnvironment` class that:
  - Interfaces with the SCION network simulation.
  - Retrieves available paths via API calls.
  - Computes the current state from normalized features.
  - Calculates rewards based on network metrics (goodput, delay, packet loss).
  - Advances the simulation by updating the traffic model.
  
  **Usage:** This module is used by both the DQN agent and analysis scripts to simulate network interactions.
  
- **path_selection_analysis.py**  
  **Responsibility:**  
  Provides functions for evaluating different routing strategies:
  - Compares DQN, Dijkstra (if enabled), random, and optimal path selection methods.
  - Sends API requests to change network paths.
  - Logs performance statistics and outputs CSV reports.
  
  **Usage:** Run this script to assess and compare the performance of various path selection strategies.
  
- **traffic_model.py**  
  **Responsibility:**  
  Simulates network traffic conditions by:
  - Using a Markov chain to update link states (Normal, Congested, Overloaded).
  - Calculating dynamic link metrics (bandwidth, latency, jitter, packet loss).
  - Sending API requests to update network conditions.
  - Yielding the current simulated time to synchronize with the environment.
  
  **Usage:** This module is automatically used by the environment to provide up-to-date traffic simulation.
  
- **requirements.txt**  
  **Responsibility:**  
  Lists all Python package dependencies needed to run the project.
  
  **Usage:** Run `pip install -r requirements.txt` to install all required packages.
  
- **instructions.pdf**  
  **Responsibility:**  
  Contains detailed instructions for setting up and running the SCION testbed environment. Topics include:
  - Cloning the base testbed repository (`seed-emulator`).
  - Running the testbed script and Docker containers.
  - Starting helper scripts for nodes and gateway.
  - Troubleshooting common errors.
  
  **Usage:** Refer to this document for complete setup details and troubleshooting guidance.

## Cloning the Base Project

This project depends on a base SCION testbed. Follow these steps (see `instructions (2).pdf` for full details):

1. **Clone the Base Repository:**

   ```bash
   git clone -b tjohn327/testbed https://github.com/netsys-lab/seed-emulator.git
   ```

2. **Navigate to the Testbed Directory:**

   ```bash
   cd /home/oran/Desktop/ScionProject/seed-emulator/examples/scion/path-selection-testbed
   ```

3. **Run the Testbed Script:**

   ```bash
   sudo python3 testbed.py
   ```

4. **Start Docker Compose:**

   In a separate terminal window:
   ```bash
   sudo docker-compose up --build
   ```

5. **Start Helper Scripts:**

   Navigate to the `helper_scripts` directory and run:
   ```bash
   sudo ./start_nodes.sh
   sudo ./start_gateway.sh
   ```

## Usage

### Running the DQN Agent

1. **Activate your virtual environment.**
2. **Run the Agent Script:**

   ```bash
   python DQNAgent.py
   ```

   The DQN agent will:
   - Initialize the SCION environment.
   - Begin training over multiple episodes.
   - Log statistics and save model checkpoints.

### Running the Analysis Module

1. **Activate your virtual environment.**
2. **Run the Analysis Script:**

   ```bash
   python path_selection_analysis.py
   ```

   This script will:
   - Evaluate various path selection strategies.
   - Send API requests to update network paths.
   - Record and output statistics in CSV format.

### Running the Traffic Simulation

The traffic simulation is integrated with the environment. The `traffic_model.py` module continuously updates link metrics, allowing the simulation to reflect real-time network conditions.

## Seed Emulator API Commands

Here are some sample API commands (also detailed in the PDF):

- **Gateway API – Path Selection:**
  ```bash
  curl -X GET "http://10.105.0.71:8010/paths/0_1"
  ```
- **Gateway API – Get Paths:**
  ```bash
  curl -X GET "http://10.101.0.71:8050/get_paths"
  ```
- **Sender App API – Send Data:**
  ```bash
  curl -X POST "http://10.105.0.71:5000/send" -H 'Content-Type: application/json' -d '{"rate": 10, "duration": 3}'
  ```
- **Sender/Receiver API – Get Statistics:**
  ```bash
  curl -X GET "http://10.105.0.71:5000/stats"
  curl -X GET "http://10.106.0.71:5002/stats"
  ```

## Troubleshooting

For common errors (e.g., missing modules like `seedemu` or `web3`, Docker configuration issues, or ensuring `scion-pki` is in your PATH), please refer to the `instructions.pdf`.


