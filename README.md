# Deep Reinforcement Learning — Lab Assignment 1
## Part 1: Multi-Armed Bandit (MAB)
### Adaptive Treatment Recommendation System
<img width="1052" height="808" alt="image" src="https://github.com/user-attachments/assets/265713ca-4656-486e-8c61-34d95dd832bf" />


## Problem Overview

A hospital is evaluating **7 medicines** for treating a chronic disease. Patient responses vary due to hidden factors (age, immunity, genetics, disease severity). Since the best treatment is unknown at the start, an intelligent recommendation system must **learn from patient outcomes over time** and progressively identify the optimal medicine.

Each medicine is modelled as an **arm** in a Multi-Armed Bandit (MAB) problem.

---

## Group 218 — Parameters

| Parameter | Formula | Value |
|---|---|---|
| Group number | G | **218** |
| Number of medicines | K = (G mod 3) + 5 | **7** |
| Random seeds | `random.seed(218)`, `np.random.seed(218)` | |

### Hidden Success Probabilities

| Medicine | Formula | P_i |
|---|---|---|
| M0 | 0.4 + ((218+0) mod 6) × 0.07 | 0.54 |
| M1 | 0.4 + ((218+1) mod 6) × 0.07 | 0.61 |
| M2 | 0.4 + ((218+2) mod 6) × 0.07 | 0.68 |
| **M3** | 0.4 + ((218+3) mod 6) × 0.07 | **0.75 ★ best** |
| M4 | 0.4 + ((218+4) mod 6) × 0.07 | 0.40 |
| M5 | 0.4 + ((218+5) mod 6) × 0.07 | 0.47 |
| M6 | 0.4 + ((218+6) mod 6) × 0.07 | 0.54 |

### Utility Score Formula

```
UtilityScore = clinical_outcome × (1 − severity / 10)
```

| Severity | Recovered | Not Recovered |
|---|---|---|
| 1 (mild) | 0.9 | 0.0 |
| 3 | 0.7 | 0.0 |
| 5 (critical) | 0.5 | 0.0 |

---

## Repository Structure

```
Team_218_MAB/
│
├── Team_218_MAB.py               # Standalone Python script (all 5 tasks)
├── Team_218_MAB.ipynb            # Jupyter Notebook (with inline illustrations)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
│
│   Generated on run:
├── Team_218_dataset_base.csv           # 1000 patients — severity only
├── Team_218_dataset_greedy.csv         # Greedy strategy results
├── Team_218_dataset_eps_greedy_10.csv  # ε-Greedy 10% results
├── Team_218_dataset_eps_greedy_01.csv  # ε-Greedy 1%  results
├── Team_218_dataset_eps_greedy_50.csv  # ε-Greedy 50% results
├── Team_218_dataset_ucb1.csv           # UCB1 strategy results
├── Team_218_summary_all_strategies.csv # 5-row comparison summary
└── Team_218_MAB_comparison.png         # Cumulative reward plot
```

---

## Setup & Installation

### 1. Prerequisites

- Python **3.8 or higher**
- `pip` package manager

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib
```

### 3. Run the Python script

```bash
python Team_218_MAB.py
```

All CSV files and the comparison plot are saved in the **current working directory**.

### 4. Run the Jupyter Notebook

```bash
jupyter notebook Team_218_MAB.ipynb
```

Then: **Kernel → Restart & Run All**

> **For submission:** After running all cells, go to  
> `File → Download as → PDF via LaTeX` (or `Print → Save as PDF`)  
> to produce the required PDF with all outputs and timestamps visible.

---

## Tasks Implemented

### Task 1 — Dataset Design (1 Mark)
- Generates 1000 synthetic patient records using `random.seed(218)` and `np.random.seed(218)`
- Computes K = 7 medicines and their hidden success probabilities
- Prints first 10 rows of the base dataset

### Task 2 — Immediate Exploitation / Greedy 
<img width="1052" height="618" alt="image" src="https://github.com/user-attachments/assets/4b8e30b4-c75f-4636-88bc-28291db0c445" />

- **Warm-up:** Each of the 7 medicines is tested exactly 10 times (70 patients total, round-robin)
- **Exploitation:** All remaining 930 patients receive only the best-performing medicine
- Implements `greedy_strategy(base_df, K, warm_up=10)`

### Task 3 — Controlled Clinical Trial / ε-Greedy 

<img width="1052" height="650" alt="image" src="https://github.com/user-attachments/assets/a71c7b57-9a6c-48aa-ae41-8f6231d0d946" />

- With probability ε → **explore** (random medicine)
- With probability 1−ε → **exploit** (best known medicine)
- Three runs: ε = 10% (main), ε = 1% (analysis), ε = 50% (analysis)
- Implements `epsilon_greedy_strategy(base_df, K, epsilon, label)`

### Task 4 — Confidence-Based / UCB1 (1 Mark)
<img width="1052" height="650" alt="image" src="https://github.com/user-attachments/assets/9b42230a-7900-4310-ad5c-1b6b7fd14f43" />

- Selects the arm with the highest Upper Confidence Bound:

```
UCB_i(t) = avg_utility_i + sqrt(2 × ln(t) / n_i)
```

- Exploration bonus shrinks automatically — no manual tuning required
- Implements `ucb1_strategy(base_df, K)`

### Task 5 — Comparative Analysis (0.5 Marks)
- Matplotlib line chart: Cumulative Reward vs Number of Patients for all 5 runs
- Saved as `Team_218_MAB_comparison.png`
- Answers all 4 comparative questions with a written summary

---

## Results Summary

| Rank | Strategy | Final Cumulative Reward |
|---|---|---|
| 1 | ε-Greedy 10% | **498.6** |
| 2 | ε-Greedy 50% | 460.0 |
| 3 | UCB1 | 441.0 |
| 4 | Greedy | 418.2 |
| 5 | ε-Greedy 1% | 403.4 |

**Recommended strategy for real-world deployment: UCB1**  
UCB1 requires no manual hyperparameter tuning, converges to Medicine 3 (P=0.75) fastest, and provides theoretical regret guarantees — making it the most principled choice for an adaptive clinical treatment system.

---

## CSV Output Schema

Each strategy dataset (`*_dataset_*.csv`) contains 1000 rows with these columns:

| Column | Type | Description |
|---|---|---|
| `patient_id` | int | Sequential index 0–999 |
| `severity_score` | int | Disease severity 1–5; computed as `(patient_id mod 5) + 1` |
| `assigned_medicine` | int | Medicine index 0–6 selected by the algorithm |
| `clinical_outcome` | int | 1 = recovered, 0 = not recovered (Bernoulli draw) |
| `utility_score` | float | `clinical_outcome × (1 − severity/10)` |

The summary file (`Team_218_summary_all_strategies.csv`) contains 5 rows — one per strategy — with columns: `strategy`, `final_cumulative_reward`, `avg_utility_per_patient`, `recovery_rate`, `most_used_medicine`, `total_patients`.

---
<img width="1052" height="650" alt="image" src="https://github.com/user-attachments/assets/5bea6afc-531e-43e4-9ff5-6c1b445dc5c1" />



