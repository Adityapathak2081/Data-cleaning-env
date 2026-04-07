---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
---
# 🧹 Data Cleaning Environment

An OpenEnv-compliant environment where AI agents learn to clean messy
real-world datasets — filling nulls, fixing types, removing duplicates,
handling outliers, and standardizing formatting.

---

## 🌍 Why This Environment?

Data cleaning is one of the most common and time-consuming tasks in
real-world data workflows. Data engineers and analysts spend up to 80%
of their time cleaning data. This environment trains AI agents to do it
automatically.

---

## 📋 Tasks

| Task | Difficulty | Description | Max Score |
|---|---|---|---|
| `fix_nulls` | Easy | Fill missing values | 0.4 |
| `fix_types` | Medium | Fix types + remove duplicates | 0.8 |
| `full_clean` | Hard | Fix everything | 1.0 |

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `dataset_name` | string | Name of the dataset |
| `data` | string (JSON) | The dirty dataset as JSON |
| `issues_hint` | string | Hint about what's wrong |
| `task` | string | Which task is being run |

---

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `cleaned_data` | string (JSON) | The cleaned dataset |
| `steps_taken` | string | Explanation of what was fixed |

---

## 🏆 Reward Function

- `+0.4` for fixing all null values
- `+0.2` for fixing data types (medium + hard only)
- `+0.2` for removing duplicates (medium + hard only)
- `+0.1` for removing outliers (hard only)
- `+0.1` for fixing formatting (hard only)

---

## 🚀 Setup & Usage

### 1. Install dependencies
```bash
pip install fastapi uvicorn pydantic openai pandas openenv-core
```

### 2. Run the server
```bash
python server.py
```

### 3. Run baseline inference
```bash
$env:HF_TOKEN="your_token_here"
python inference.py
```

### 4. Using Docker
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

---

## 📊 Baseline Scores

| Task | Score |
|---|---|
| fix_nulls | 0.40 |
| fix_types | 0.80 |
| full_clean | 0.90 |

---

## 📁 Project Structure
```
data-cleaning-env/
├── env.py          # OpenEnv environment
├── datasets.py     # Sample dirty datasets
├── server.py       # FastAPI server
├── inference.py    # Baseline inference script
├── openenv.yaml    # OpenEnv metadata
├── Dockerfile      # Container setup
└── README.md       # This file
```