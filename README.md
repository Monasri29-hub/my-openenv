---
title: OpenEnv Hackathon Project
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "1.0"
python_version: "3.10"
app_file: inference.py
pinned: false
---

# OpenEnv Hackathon Project

## 📌 Overview
This repository contains a real‑world OpenEnv environment with three tasks of increasing difficulty. The environment is containerized with Docker and deployable on Hugging Face Spaces. It follows the OpenEnv specification strictly and passes validation.

## 🎯 Motivation
The goal is to simulate tasks that humans perform in everyday workflows (email triage, data cleaning, code review) and evaluate LLM agents on structured, reproducible benchmarks.

---

## 🧩 Action & Observation Spaces
- **Observation Space**: Each task provides structured input (text, dataset rows, or code snippets) via Pydantic models.
- **Action Space**: Agents respond with deterministic actions (labels, transformations, or review decisions).
- **Reward Function**: Incremental rewards are given for progress toward the objective; penalties apply for invalid or destructive actions.

---

## 📝 Tasks
### 1. Email Triage (Easy)
- **Observation**: Raw email text.
- **Action**: Classify as `spam`, `important`, or `other`.
- **Reward**: +1.0 for correct classification, 0.0 otherwise.

### 2. Data Cleaning (Medium)
- **Observation**: Dataset row with inconsistent formatting.
- **Action**: Normalize values, remove duplicates, or clean text.
- **Reward**: Incremental reward for each correctly cleaned field.

### 3. Code Review (Hard)
- **Observation**: Code snippet with potential issues.
- **Action**: Approve, reject, or comment with feedback.
- **Reward**: +1.0 for correct review decision, partial reward for useful comments.

---

## 📊 Baseline Results
| Task          | Avg Reward | Success Rate |
|---------------|------------|--------------|
| Email Triage  | 0.65       | 70%          |
| Data Cleaning | 0.50       | 55%          |
| Code Review   | 0.40       | 45%          |

Baseline scores are reproducible using the provided `inference.py`.

---

## ⚙️ Setup & Usage
### Local Run
```bash
# Build Docker image
docker build -t openenv .

# Run container
docker run openenv
