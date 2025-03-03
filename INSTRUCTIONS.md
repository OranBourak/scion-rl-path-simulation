# SCION Testbed Setup Instructions

This document provides detailed instructions to set up and run the SCION testbed environment. All commands that require elevated privileges must be run with `sudo`.

---

## 1. Cloning and Initial Setup

### 1.1. Clone the Base Repository
Clone the seed-emulator repository using the specified branch:
```bash
git clone -b tjohn327/testbed https://github.com/netsys-lab/seed-emulator.git
```

### 1.2. Navigate to the Testbed Directory
Change directory to the testbed folder:
```bash
cd /path to your poject/seed-emulator/examples/scion/path-selection-testbed
```

---

## 2. Running the Testbed Environment

### 2.1. Start the Testbed
Run the testbed Python script:
```bash
sudo python3 testbed.py
```

### 2.2. Check the Output Folder
After running the testbed, navigate to the output folder to review generated files:
```bash
cd /home/oran/Desktop/ScionProject/seed-emulator/examples/scion/path-selection-testbed/output
```

### 2.3. Start Docker Compose
Launch the Docker containers using Docker Compose:
```bash
sudo docker-compose up --build
```

---

## 3. Running Helper Scripts

### 3.1. Start Nodes
Open a new terminal window, navigate to the helper scripts directory, and run:
```bash
cd /home/oran/Desktop/ScionProject/seed-emulator/examples/scion/path-selection-testbed/helper_scripts
sudo ./start_nodes.sh
```
*Note: Running without `sudo` may cause errors.*

### 3.2. Start Gateway
In the same or a new terminal window, run:
```bash
sudo ./start_gateway.sh
```

### 3.3. Access the Dashboard
After starting nodes and gateway, look for the dashboard link printed in the terminal. You might need to click the "use scion" switch on the dashboard.

### 3.4. Stop the Testbed
When finished, shut down all resources with:
```bash
sudo docker-compose down
```

---

## 4. Troubleshooting Common Errors

### 4.1. Error: ModuleNotFoundError: No module named 'seedemu'
If you encounter this error when running `testbed.py`:
- **Solution A:** Install seedemu by running:
  ```bash
  cd ~/Desktop/Scion\ Project/seed-emulator/
  pip3 install .
  ```
- **Solution B:** Alternatively, run:
  ```bash
  cd ~/Desktop/Scion\ Project/seed-emulator/
  sudo python3 setup.py install
  ```

### 4.2. Error: ModuleNotFoundError: No module named 'web3'
Install the missing module with:
```bash
sudo pip3 install web3
```

### 4.3. Error: "scion-pki not found in PATH"
If you see an error related to `scion-pki` not being found, install SCION tools on Ubuntu as follows:
```bash
sudo apt-get install apt-transport-https ca-certificates
echo "deb [trusted=yes] https://packages.netsec.inf.ethz.ch/debian all main" | sudo tee /etc/apt/sources.list.d/scionlab.list
sudo apt-get update
sudo apt-get install scion-tools
```

---

## 5. Compiling SCION from Source

If you prefer to compile SCION from source:

1. **Clone the SCION Repository:**
   ```bash
   git clone https://github.com/scionproto/scion.git
   ```
2. **Build scion-pki:**
   ```bash
   cd scion
   go build -o bin ./scion-pki/cmd/scion-pki
   ```

---

## 6. API Commands

Use the following API commands to interact with various SCION components:

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

- **Sender App API – Get Statistics:**
  ```bash
  curl -X GET "http://10.105.0.71:5000/stats"
  ```

- **Receiver App API – Get Statistics:**
  ```bash
  curl -X GET "http://10.106.0.71:5002/stats"
  ```

- **Additional API Endpoints:**
  - `http://10.105.0.71:8010/paths/0_1`
  - `http://10.105.0.71:5000/stats`
  - `http://10.101.0.71:8050/get_paths`

---

## Notes

- **Run All Commands with Sudo:**  
  When prompted or when necessary, run commands using `sudo` to avoid permission issues.
- **Ensure Proper PATH Configuration:**  
  For SCION-specific tools like `scion-pki`, ensure they are in your system's PATH after installation.

---

By following these instructions, you should be able to set up, run, and troubleshoot the SCION testbed environment effectively.
