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

- Overview: Real‑world OpenEnv environment with 3 tasks.
- Action/Observation Spaces: Describe inputs/outputs for each task.
- Tasks:
- Email triage (easy).
- Data cleaning (medium).
- Code review (hard).
- Setup: docker build -t openenv . → docker run openenv.
- Baseline Scores: Show reproducible results.
