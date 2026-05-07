# 🏥 HealthAIoT — Cloud Computing Lab Implementation

> **Lab implementation** of the paper:  
> *"HealthAIoT: AIoT-driven smart healthcare system for sustainable cloud computing environments"*  
> Published in **Elsevier Internet of Things Journal, 2025**  
> DOI: [10.1016/j.iot.2025.101555](https://doi.org/10.1016/j.iot.2025.101555)

---

## 📌 Overview

HealthAIoT is an **AIoT + Cloud Computing** framework that creates a smart healthcare system. This implementation focuses on:

- **Diabetes Risk Prediction** — Assesses an individual's risk of developing Diabetes Mellitus based on personal health metrics and medical history using a Multilayer Perceptron (MLP) neural network.
- **AI-Powered Cloud Scheduler** — Intelligently routes incoming IoT/user requests to the most optimal Worker VM (deployed on Google Cloud Platform), minimising energy consumption, cost, and latency using a second MLP model.
- **Web Interface** — A Flask-hosted questionnaire that collects user health data and returns a prediction, demonstrating an end-to-end patient-facing IoT request pipeline.

The framework is **disease-agnostic by design** and can be extended to other diseases beyond diabetes.

---

## 🏗️ System Architecture

```
User (Browser)
      │
      ▼
 Broker / Scheduler (Local Machine — scheduler.py on port 5002)
      │
      │  MLP Scheduler picks optimal Worker
      │
  ┌───┴───┐
  ▼       ▼
Worker 1  Worker 2
(GCP VM)  (GCP VM)
  │         │
  └────┬────┘
       │
  MLP Diabetes Predictor (best_model.pth)
       │
  Result returned to User
```

- **Broker/Scheduler** runs on your local machine and hosts the web interface.
- **Workers** are Flask apps running on GCP VMs that perform the actual diabetes prediction.
- The Scheduler uses a trained MLP (`vm_selector_model.pth`) to decide which Worker handles each request based on real-time system metrics (CPU, memory, etc.).

---


## 🧠 Models

| Model | Architecture | Task | Accuracy |
|-------|-------------|------|----------|
| Diabetes Predictor | MLP (PyTorch) | Binary classification — diabetes risk | **78.30%**, F1: 0.7719 |
| Cloud Scheduler | MLP (PyTorch) | Worker VM selection | **93.6%** |

**Datasets:**
- **Diabetes Predictor**: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (via `ucimlrepo`)
- **Cloud Scheduler**: [Bitbrains GWA-T-12](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)

---

## ⚙️ Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.8+
- Two GCP VM instances (or equivalent cloud VMs)
- SSH access configured to both VMs

---

## 🚀 Setup & Running

### Step 1 — Set Up Local Environment

Clone the repository and create the conda environment:

```bash
git clone https://github.com/shrav-p24/cc-lab-HealthAIoT.git
cd cc-lab-HealthAIoT/HealthAIoT

conda env create -f environment.yml
conda activate HealthAIoT
```

---

### Step 2 — Set Up AWS Worker VMs

#### 2a. Create VMs on AWS

Create **two VM instances** with the following specs:
- **vCPU**: 2 (1 shared core)
- **RAM**: 2 GB
- **Disk**: 15 GB Balanced Persistent Disk
- **OS**: Ubuntu 20.04 LTS

> ⚠️ **Important**: Add a firewall rule on **VPC Firewall** to allow TCP traffic on **port 5000** for both VMs.

#### 2b. Generate SSH Key Pair

Run this command in the main application directory on your local machine:

```bash
ssh-keygen -t rsa -f ./id_rsa -C your_username
```

Copy the contents of `id_rsa.pub` and add it to each VM's **SSH keys** section in the GCP Console (VM → Edit → SSH Keys).

#### 2c. Install Dependencies on Both VMs

SSH into each VM and install the required packages:

```bash
ssh -i ./id_rsa your_username@VM_EXTERNAL_IP
```

Then run:

```bash
sudo apt update
sudo apt-get install -y python3 python3-pip
pip3 install Flask psutil
pip3 install torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install joblib numpy
pip3 install ucimlrepo imblearn
```

#### 2d. Deploy Worker Code to VMs

From your local machine (in the main application directory), run:

```bash
tar -czvf worker.tar.gz -C worker app.py best_model.pth main_training.py scaler.pkl model_utils.py
scp -i ./id_rsa worker.tar.gz your_username@VM_EXTERNAL_IP:/home/your_username/

ssh -i ./id_rsa your_username@VM_EXTERNAL_IP
cd /home/your_username/
tar -xzvf worker.tar.gz
```

Repeat for **both VMs**.

> **Alternative**: After step 2c, you can manually create the `.py` files on each VM using `vim app.py`, `vim main_training.py`, `vim model_utils.py` and paste the code from the repository.

#### 2e. (Optional) Train the Diabetes Predictor on the VMs

If `best_model.pth` and `scaler.pkl` are not pre-trained, train the model on each VM:

```bash
python3 main_training.py
```

This will generate:
- `best_model.pth` — trained predictor model weights
- `scaler.pkl` — fitted StandardScaler
- `diabetic_model_train_val_test_log.txt` — training logs

#### 2f. Start the Flask Worker App on Both VMs

```bash
python3 app.py
```

The Flask app will start on **port 5000** and begin generating `system_stats.json` every 10 seconds with live CPU/memory metrics.

---

### Step 3 — Run the Main Scheduler (Local Machine)

Once both Worker VMs are running, start the scheduler from your local machine:

```bash
python3 scheduler.py
```

The application will be available at:

```
http://<your-local-IP>:5002
```

(Accessible from any device on the same WLAN/network.)

---

## 🌐 Using the Web Interface

1. Open the web app in a browser at `http://<local-IP>:5002`
2. Fill in the **health questionnaire** (`form.html`) — it collects metrics like BMI, blood pressure, cholesterol levels, physical activity, and medical history.
3. Submit the form — the scheduler routes your request to the optimal Worker VM.
4. View the **prediction result** (`result.html`) showing your diabetes risk assessment.

After each request, two JSON files are updated in the main directory:
- `worker_system_metric_stats.json` — system metrics of the Worker that handled the request
- `temporal_stats.json` — timing and latency stats

> **Fallback**: If the optimal Worker is unreachable, the Broker itself handles the request and its stats are saved.

---

## 📊 Visualising Results

Scripts in the `plot_figures/` directory can be used to visualise performance metrics such as energy consumption, carbon-free energy usage, cost, execution time, and latency across different configurations.

---

## 🗂️ Output Files

| File | Description |
|------|-------------|
| `worker_system_metric_stats.json` | Live system metrics from the Worker VM that handled the request |
| `temporal_stats.json` | Request latency and timing statistics |
| `diabetic_model_train_val_test_log.txt` | Training/validation/test logs for the diabetes predictor |
| `scheduler_train_test_log.txt` | Training/test logs for the scheduler model |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `Flask` | Web framework for Broker and Worker APIs |
| `PyTorch 1.13.1` | MLP model training and inference |
| `psutil` | System metric collection on Worker VMs |
| `ucimlrepo` | Fetching the CDC Diabetes dataset |
| `imblearn` | Handling class imbalance in diabetes dataset |
| `joblib` | Saving/loading scaler objects |
| `numpy`, `pandas` | Data processing |

Full dependency list: see `environment.yml`.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{HealthAIoT,
    title   = {{HealthAIoT: AIoT-driven smart healthcare system for sustainable cloud computing environments}},
    year    = {2025},
    journal = {Internet of Things},
    author  = {Wang, Han and Anurag, Kumar Ankur and Benamer, Amira Rayane and Arora, Priyansh
               and Wander, Gurleen and Johnson, Mark R. and Anjana, Ranjit Mohan
               and Mohan, Viswanathan and Gill, Sukhpal Singh and Uhlig, Steve and Buyya, Rajkumar},
    month   = {3},
    pages   = {101555},
    doi     = {10.1016/j.iot.2025.101555},
    issn    = {25426605}
}
```

---

## 📄 License

This project is licensed under **Attribution–NonCommercial 4.0 (CC BY-NC 4.0)**.  
See the [LICENSE](./LICENSE.txt) file for details.

---

## 🔗 References

- Original HealthAIoT Repository: [HTXW/HealthAIoT](https://github.com/HTXW/HealthAIoT)
- Paper DOI: [10.1016/j.iot.2025.101555](https://doi.org/10.1016/j.iot.2025.101555)
- CDC Diabetes Health Indicators Dataset: [UCI ML Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- Bitbrains Scheduler Dataset: [GWA-T-12](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)
- CloudAIBus Testbed: Used for performance evaluation
