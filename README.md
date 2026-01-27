# ICVF - PyTorch

This repository contains a PyTorch implementation of **ICVF (Reinforcement Learning from Passive Data via Latent Intentions)**, based on the paper:

> **Reinforcement Learning from Passive Data via Latent Intentions**
> [https://arxiv.org/abs/2304.04782](https://arxiv.org/abs/2304.04782)

The official code release is available at:
[https://github.com/dibyaghosh/icvf_release](https://github.com/dibyaghosh/icvf_release)

This README explains how to run the PyTorch training script for learning ICVF value functions from offline datasets.

---

## 1. Environment Setup

### 1.1 Python & PyTorch

Use Python **3.8+** and install PyTorch with CUDA support if available:

```bash
pip install torch torchvision torchaudio
```

(Install the CUDA-enabled version matching your system if needed.)

### 1.2 Required Dependencies

Install core dependencies:

```bash
pip install numpy wandb absl-py ml-collections
```

### 1.3 D4RL and MuJoCo

This script relies on **D4RL** datasets.

```bash
pip install d4rl
```

You also need a working MuJoCo installation. Follow the official instructions:

* [https://github.com/Farama-Foundation/D4RL](https://github.com/Farama-Foundation/D4RL)

Make sure environments like `hopper-medium-v2` can be created successfully.

---

## 2. Script Overview

The training script performs the following steps:

1. Initializes a D4RL environment (e.g., Hopper)
2. Loads an offline dataset
3. Wraps the dataset using `GCSDataset`
4. Builds an ensemble ICVF value network
5. Trains the agent using offline batches
6. Logs metrics to **Weights & Biases (wandb)**
7. Periodically saves learned `phi` network checkpoints

The value function is trained **without environment interaction** (pure offline RL).

---

## 3. Running the Training Script

### 3.1 Basic Command

Run the script using `absl.app`:

```bash
python train_icvf.py
```

---

### 3.2 Common Flags

You can override default hyperparameters via command-line flags:

```bash
python train_icvf.py \
  --env_name=hopper-medium-v2 \
  --batch_size=256 \
  --max_steps=400000 \
  --seed=0
```

Key flags:

| Flag            | Description               | Default            |
| --------------- | ------------------------- | ------------------ |
| `env_name`      | D4RL environment name     | `hopper-medium-v2` |
| `batch_size`    | Training batch size       | `256`              |
| `max_steps`     | Total gradient steps      | `400000`           |
| `hidden_dims`   | MLP hidden layers         | `[256, 256]`       |
| `save_interval` | Model checkpoint interval | `100000`           |
| `log_interval`  | wandb logging interval    | `100`              |
| `seed`          | Random seed               | random             |

---

## 4. Model Checkpoints

During training, the script saves the learned **phi network**:

```python
torch.save(
  agent.value_fn.model_1.phi_net.state_dict(),
  experiment_output/<env_name>/phi_<step>.pt
)
```

Checkpoints are saved to:

```
experiment_output/<env_name>/
```

These weights can later be reused for following training
