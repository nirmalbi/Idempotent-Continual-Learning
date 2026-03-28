# IDER: Idempotent Experience Replay — Improved Version

This repository is a fork of [IDER (ICLR 2026)](https://github.com/YutingLi0606/Idempotent-Continual-Learning) with two improvements proposed as part of a course project.

**Original Paper:** IDER: IDempotent Experience Replay for Reliable Continual Learning (ICLR 2026)


## Our Improvements

We modified `models/ider.py` with two contributions:

**1. Adaptive Weight Scaling**
Instead of fixed loss weights α and β, we make them grow across tasks:
`scale = 1.0 + gamma * (task_id / total_tasks)`
This enforces stronger idempotence regularization as more tasks are learned, directly addressing the stability-plasticity tradeoff.

**2. Uncertainty-Based Buffer Sampling**
Instead of random replay, we prioritize buffer samples with the highest idempotence distance (most forgotten samples). This is motivated by the paper's own finding that idempotence distance correlates with prediction error.

New arguments added:
- `--adaptive_weight True/False` — enable adaptive scaling
- `--adaptive_gamma 0.5 or 1.0` — growth rate
- `--uncertainty_sampling True/False` — enable uncertainty sampling

---

## Results

### CIFAR-10 (Buffer=200, Seed=0, 50 epochs)

| Method | CIL | ECE | Task-IL |
|--------|-----|-----|---------|
| ER | 52.61% | 41.68 | 91.98% |
| IDER original | 69.55% | 13.73 | 95.46% |
| IDER + Ours (γ=0.5) | **70.34%** | 14.21 | 96.04% |
| IDER + Ours (γ=1.0) | 69.55% | **13.16** | 95.58% |

### CIFAR-100 (Buffer=500, Seed=0, 50 epochs)

| Method | CIL | ECE | Task-IL |
|--------|-----|-----|---------|
| ER | 21.49% | 64.04 | 75.15% |
| IDER original | 25.02% | 27.47 | 70.77% |
| IDER + Ours (γ=0.5) | 45.49% | 12.55 | 83.58% |
| IDER + Ours (γ=1.0) | **45.67%** | **11.85** | **83.88%** |

Our γ=1.0 variant **exceeds the paper's reported IDER result** on CIFAR-100 (45.67% vs 44.82%).

---

## How To Run

**Setup:**
```bash
git clone <this repo>
cd Idempotent-Continual-Learning
pip install mlflow setproctitle
```

**Run baseline IDER:**
```bash
python main.py --model="ider" --load_best_args --class_balance=True --dataset="seq-cifar10" --seed=0 --n_tasks=5 --buffer_size=200
```

**Run our improved IDER (γ=0.5):**
```bash
python main.py --model="ider" --load_best_args --class_balance=True --dataset="seq-cifar10" --seed=0 --n_tasks=5 --buffer_size=200 --adaptive_weight=True --adaptive_gamma=0.5 --uncertainty_sampling=True
```

**Run our improved IDER (γ=1.0):**
```bash
python main.py --model="ider" --load_best_args --class_balance=True --dataset="seq-cifar10" --seed=0 --n_tasks=5 --buffer_size=200 --adaptive_weight=True --adaptive_gamma=1.0 --uncertainty_sampling=True
```

---

## Only File Changed

`models/ider.py` — all other files remain identical to the original repo.

---

## Original Repo

https://github.com/YutingLi0606/Idempotent-Continual-Learning
