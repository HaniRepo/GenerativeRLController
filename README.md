# GenerativeRLController

This repository contains the implementation accompanying the IEEE IES Generative AI Challenge 2026 submission:

**Generative Safety Augmentation for Industrial Reinforcement Learning Control Using Conformal Temporal Logic**

We extend safe reinforcement learning with a **generative decision layer** that improves safety and tracking performance under uncertainty.

---

##  Overview

The project builds on the AeroBench F-16 engine benchmark and integrates:

- PPO-based reinforcement learning control
- STL-inspired runtime safety monitoring
- Conformal prediction for uncertainty-aware safety
- **Generative action selection (GENAI)** for safety-aware decision refinement

The key idea is to move from **reactive filtering (conformal shielding)** to **active candidate-based decision making**.

---

##  Repository Structure

```text
code/
├── aerobench/                      # AeroBench simulator
├── f16_engine_env.py               # F-16 Gym environment
├── train_Newppo.py                 # PPO training
├── stress_wrappers.py              # Noise, delay, actuator limits
├── conformal_shield.py             # Conformal STL runtime shield
├── stl_monitor.py                  # STL robustness utilities
├── shield_pack/
│   └── ppo_f16_engine_baseline.zip # Pretrained PPO model
└── genai_hackathon/
    ├── genai_shield.py
    ├── run_paper_figure_nominal_and_stress.py
    ├── run_satisfaction_explain_plot.py
    ├── run_candidate_ablation_hardstress_suite.py
    ├── run_incremental_stress_sweep.py
    └── figures/
```
---

  ## Environment Setup


Recommended: Python 3.10–3.11 in a clean virtual environment.

### 1. Create environment

```bash
python -m venv .venv
```

Activate:

* Windows:

```bash
.venv\Scripts\activate
```

* Linux / macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install numpy scipy matplotlib
pip install gymnasium stable-baselines3 torch
```

Optional (only for AeroBench extras):

```bash
pip install control slycot
```
---
## Quick Start (Main Experiments)
Run all experiments from:

```bash
cd code
```
🔹 1. Nominal vs Stress Results
```bash
python genai_hackathon/run_paper_figure_nominal_and_stress.py
```
🔹 2. STL Satisfaction Analysis
```bash
python genai_hackathon/run_satisfaction_explain_plot.py
```
🔹 3. Candidate Ablation (GENAI Study)
```bash
python genai_hackathon/run_candidate_ablation_hardstress_suite.py
```
##  Key Results

Across stress scenarios:

- **PPO** → frequent STL violations  
- **PPO + Conformal** → improves safety but leaves residual violations  
- **PPO + GENAI** → achieves full STL satisfaction (**100%**)  

### GENAI improves:
- ✅ Safety (STL satisfaction)  
- ✅ Robustness margins  
- ✅ Tracking error (RMSE)  


##  Method Summary

At each control step:

1. PPO proposes an action `u_RL`  
2. GENAI generates candidate actions around it  
3. Each candidate is evaluated using:
   - Conformal prediction (uncertainty-aware bounds)  
   - STL robustness  
4. The best action is selected based on predicted safety margin  


##  Conceptual Shift

This framework transforms runtime safety from:

- ❌ Filtering unsafe actions  
➡️ to  
- ✅ Optimizing decisions under safety constraints  


## 📝 Notes for Reviewers

- Pretrained PPO model is included → **no training required**  
- All experiments are **fully reproducible**  
- GENAI module is **lightweight** (no deep generative model required)  
- Designed as a **proof-of-concept for real-time safety augmentation**  


##  Paper Context

This repository supports the submission:

**"Generative Safety Augmentation for Industrial Reinforcement Learning Control Using Conformal Temporal Logic"**

### Includes:
- Full experimental pipeline  
- Safety evaluation scripts  
- Figure reproduction  
- Ablation studies  

  
