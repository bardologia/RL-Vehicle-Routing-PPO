# Reinforcement Learning for Vehicle Routing Problems

## Abstract

This repository implements a Proximal Policy Optimization (PPO) algorithm for solving Vehicle Routing Problems (VRP) under conditions of uncertainty. The system uses Graph Neural Networks (GNN) to encode problem states and employs a hierarchical action space with four operators: job insertion, job removal, no-operation, and full reoptimization.

## System Architecture

### 1. Problem Formulation

The system addresses VRP with the following characteristics:
- Multiple vehicles with capacity constraints
- Jobs with location coordinates, service time, and priority values
- Dynamic events: job insertion, job removal, vehicle insertion, vehicle removal
- Objective: minimize total distance while reducing unassigned jobs and idle vehicles

### 2. State Representation

States are represented as heterogeneous graphs with three node types:

**Job Nodes** (7 features):
- Longitude, latitude coordinates
- Service time
- Priority level
- Capacity demand
- Assignment status (binary)
- Time window constraints

**Vehicle Nodes** (5 features):
- Longitude, latitude coordinates
- Capacity
- Speed factor
- Current load

**Path Nodes** (5 features):
- Longitude, latitude coordinates from OSRM polylines
- Segment distance
- Segment duration
- Route membership

**Edge Types** (4 attributes per edge):
- Distance (km)
- Duration (hours)
- Same route indicator (binary)
- Edge type identifier

### 3. Neural Network Architecture

#### 3.1 Graph Encoder
- GATv2Conv layers with multi-head attention
- Layer normalization
- Cross-attention mechanisms between node types
- Self-attention pooling for global context

#### 3.2 Policy Network
- **Operator Selection**: 4-way categorical distribution over operators
- **Vehicle Selection**: Conditional distribution given operator
- **Job Selection**: Pointer network with glimpse and pointer mechanisms
- Tanh clipping (C=10.0) for logit stabilization

#### 3.3 Value Network
- Separate GNN encoder
- MLP critic head
- Outputs state value estimate

### 4. Action Space

Four operators define the action space:

| Operator | Index | Description |
|----------|-------|-------------|
| Insert Job | 0 | Assign job to vehicle route |
| Remove Job | 1 | Remove job from vehicle route |
| No Action | 2 | Maintain current state |
| Reoptimize | 3 | Full VROOM optimization |

Each action requires:
1. Operator selection (4 choices)
2. Vehicle selection (V choices, masked based on operator)
3. Job selection (J choices, masked based on operator and vehicle)

### 5. Reward Function

Reward components at timestep t:

```
R_t = R_distance + R_unassigned + R_idle + R_priority + R_action
```

Where:
- **R_distance**: Negative change in total route distance (weighted by 1.5)
- **R_unassigned**: Negative change in unassigned job count (weighted by 1.0)
- **R_idle**: Negative change in idle vehicle count (weighted by 0.5)
- **R_priority**: Negative sum of unassigned job priorities (weighted by 0.5)
- **R_action**: Operator-specific penalty {Insert: -0.5, Remove: -1.5, No-op: 0.0, Reopt: 1.5}

### 6. Training Algorithm

**PPO Configuration:**
- Clip parameter: ε = 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: scheduled from 0.02 to 0.001 over 50,000 steps
- GAE lambda: 0.95
- Discount factor: γ = 0.99
- Gradient clipping: max norm 3.0
- KL divergence threshold: 0.015

**Learning Rate Schedule:**
- Warmup: 1,000 steps (linear increase)
- Decay: cosine annealing over 100,000 steps
- Minimum LR: 1e-5
- Component-specific rates:
  - Operator actor: 3e-4
  - Vehicle actor: 3e-4
  - Job pointer: 3e-4
  - Critic: 5e-4
  - GNN encoder: 2e-4

**Training Protocol:**
1. Load episode from dataset
2. Apply random event (job/vehicle insertion or removal)
3. Agent performs 1-N actions until episode termination
4. Store experience in memory buffer
5. After each chunk (1024 episodes), perform PPO update with 4 epochs
6. Minibatch size: 128

### 7. Dataset Generation

Episodes are generated with the following distribution:
- Jobs per instance: N(μ=16, σ=4), minimum 4
- Vehicles per instance: N(μ=4, σ=1), minimum 2
- Outlier probability: 1/8 with 2× radius multiplier
- Center coordinates: (-46.63°, -23.55°)
- Sampling radius: 25 km
- Service time per job: 300 seconds
- Job priorities: uniform discrete {1, 2, 3, 4, 5}

### 8. External Dependencies

**OSRM (Open Source Routing Machine):**
- Algorithm: Multi-Level Dijkstra (MLD)
- Road network: Southeast Brazil region
- Function: provides distance and duration matrices

**VROOM:**
- Version: 1.14.0
- Function: generates initial solutions and handles reoptimization operator
- Threads: 8

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Docker and Docker Compose

### Setup

1. Start routing services:
```powershell
docker-compose up -d
```

2. Verify services:
- OSRM: http://localhost:5000
- VROOM: http://localhost:3000

3. Install Python dependencies:
```powershell
conda create -n routing-ppo python=3.10
conda activate routing-ppo
pip install torch torch-geometric numpy scipy pandas requests polyline tqdm tensorboard
```

## Dataset Generation

Generate training data with specified parameters:

```python
from core.dataset import Dataset

dataset = Dataset("data")
dataset.append(
    num_events=100000,
    output_dir="data",
    batch_size=128,
    chunk_size=1024,
    seed=42
)
```

This produces files: `data/chunk_00000.pt`, `data/chunk_00001.pt`, etc.

## Training

Execute training with configuration:

```powershell
cd main
python train.py
```

Training outputs:
- Tensorboard logs: `runs/run_YYYYMMDD-HHMMSS/`
- Model checkpoints: `graph_ppo_policy.pt`
- Training logs: text file in run directory

## Configuration

Configuration parameters are defined in `tools/config.py` with dataclass structure:

| Component | Parameter | Default |
|-----------|-----------|---------|
| GNN | Hidden channels | 64 |
| GNN | Number of layers | 3 |
| GNN | GAT heads | 4 |
| Pointer | Hidden dimension | 64 |
| Pointer | Number of heads | 4 |
| Training | Batch size | 1024 |
| Training | Minibatch size | 128 |
| Training | PPO epochs | 4 |

## File Structure

```
routing-ppo/
├── core/
│   ├── dataset.py          # Dataset management and generation
│   ├── embedding.py         # GNN and attention modules
│   ├── environment.py       # VRP environment and operators
│   ├── graph.py            # Heterogeneous graph construction
│   ├── mask.py             # Action masking logic
│   ├── ppo.py              # PPO algorithm implementation
│   ├── state.py            # State representation
│   └── training.py         # Training loop and schedulers
├── main/
│   ├── dataset_pipeline.py # Dataset generation script
│   └── train.py            # Training entry point
├── tools/
│   ├── auxiliary.py        # Utility functions (OSRM/VROOM API)
│   ├── config.py           # Configuration dataclasses
│   ├── logger.py           # Logging utilities
│   └── tracker.py          # Tensorboard tracking
├── data/                   # Generated dataset chunks
├── runs/                   # Training outputs and checkpoints
├── osrm-data/              # OSRM map data
└── docker-compose.yml      # Service orchestration
```

## Graph Construction Details

### Node Creation
- Vehicles: one node per vehicle at start location
- Jobs: one node per job (assigned and unassigned)
- Paths: sampled from OSRM polylines with adaptive sampling (minimum angle: 15°)

### Edge Construction
Five edge type categories:
1. **Within-route**: consecutive jobs in route
2. **Vehicle-to-job**: vehicle depot to assigned jobs
3. **Job-to-path**: k-nearest path segments (k=1)
4. **Job-to-vehicle**: k-nearest vehicles (k=1)
5. **Vehicle-to-job**: unassigned vehicles to k-nearest jobs (k=1)

All edges are bidirectional with distance and duration attributes.

### Action Masking
Dynamic masking based on feasibility:
- Operator 0 (Insert): requires unassigned jobs
- Operator 1 (Remove): requires vehicles with assigned jobs
- Operator 2 (No-op): always available
- Operator 3 (Reoptimize): always available

Vehicle masking: constrains selection to feasible vehicles for chosen operator
Job masking: constrains selection to feasible jobs given operator and vehicle

## Results Format

Training produces metrics logged to Tensorboard:

**Episode-level:**
- total_reward
- episode_length
- operator_frequency (per operator)
- operator_avg_reward (per operator)

**Batch-level:**
- policy_loss
- value_loss
- entropy (operator, vehicle, job components)
- KL divergence (operator, vehicle, job components)
- gradient_norm
- learning_rate (per parameter group)

**Step-level:**
- action probabilities (per operator, vehicle, job)
- pointer attention weights (glimpse and pointer)
- state value estimates

## References

This implementation combines concepts from:
1. Proximal Policy Optimization (Schulman et al., 2017)
2. Attention Model for VRP (Kool et al., 2019)
3. Graph Attention Networks (Veličković et al., 2018)
4. Generalized Advantage Estimation (Schulman et al., 2016)

## License

This work is provided for research purposes.
