# IDER: Idempotent Experience Replay — Improved Version

This repository is a fork of [IDER (ICLR 2026)](https://github.com/YutingLi0606/Idempotent-Continual-Learning) with two improvements proposed as part of a course project (CSL4020).

**Original Paper:** IDER: IDempotent Experience Replay for Reliable Continual Learning (ICLR 2026)

---

## Our Improvements

We modified `models/ider.py` with two contributions:

**1. Adaptive Weight Scaling**
Instead of fixed loss weights α and β, we make them grow across tasks:
`scale = 1.0 + gamma * (task_id / total_tasks)`
This enforces stronger idempotence regularization as more tasks are learned.

**2. Uncertainty-Based Buffer Sampling**
Instead of random replay, we prioritize buffer samples with the highest idempotence distance (most forgotten samples). Motivated by the paper's own finding that idempotence distance correlates with prediction error.

**New arguments added:**
- `--adaptive_weight True/False` — enable adaptive scaling
- `--adaptive_gamma 0.5 or 1.0` — growth rate
- `--uncertainty_sampling True/False` — enable uncertainty sampling

---

## Results

> **Note:** We compare our improved version against our own reproduced baseline (not the paper's numbers) for a fair comparison. Paper numbers are shown separately for reference. All runs use Seed=0, 50 epochs on Kaggle T4 GPU.

### CIFAR-10 (Buffer=200)

| Method | CIL | ECE | Task-IL |
|--------|-----|-----|---------|
| ER (our run) | 52.61% | 41.68 | 91.98% |
| IDER original (our run) | 69.55% | 13.73 | 95.46% |
| **IDER + Ours (γ=0.5)** | **70.34%** | 14.21 | **96.04%** |
| **IDER + Ours (γ=1.0)** | 69.55% | **13.16** | 95.58% |
| Paper's IDER (5 seeds avg) | 71.02% | 12.36 | — |

### CIFAR-100 (Buffer=500)

| Method | CIL | ECE | Task-IL |
|--------|-----|-----|---------|
| ER (our run) | 21.49% | 64.04 | 75.15% |
| **IDER + Ours (γ=0.5)** | 45.49% | 12.55 | 83.58% |
| **IDER + Ours (γ=1.0)** | **45.67%** | **11.85** | **83.88%** |
| Paper's IDER (5 seeds avg) | 44.82% | 13.65 | — |

**Key finding:** Our γ=1.0 variant exceeds the paper's reported IDER on CIFAR-100 (45.67% vs 44.82% CIL, 11.85 vs 13.65 ECE). On CIFAR-10, γ=0.5 gives the best CIL accuracy improvement over our baseline (+0.79%).

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
python main.py --model="ider" --load_best_args --class_balance=True \
    --dataset="seq-cifar10" --seed=0 --n_tasks=5 --buffer_size=200
```

**Run our improved IDER (γ=0.5):**
```bash
python main.py --model="ider" --load_best_args --class_balance=True \
    --dataset="seq-cifar10" --seed=0 --n_tasks=5 --buffer_size=200 \
    --adaptive_weight=True --adaptive_gamma=0.5 --uncertainty_sampling=True
```

**Run our improved IDER (γ=1.0):**
```bash
python main.py --model="ider" --load_best_args --class_balance=True \
    --dataset="seq-cifar10" --seed=0 --n_tasks=5 --buffer_size=200 \
    --adaptive_weight=True --adaptive_gamma=1.0 --uncertainty_sampling=True
```

---

## Limitations
- Results based on single seed (paper uses 5 seeds)
- TinyImageNet not tested due to compute constraints
- Experiments run on Kaggle T4 GPU (not RTX 4090 as in paper)

---

## Only File Changed
`models/ider.py` — all other files remain identical to the original repo.

---

## Original Repo
https://github.com/YutingLi0606/Idempotent-Continual-Learning
