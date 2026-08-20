# AutoRoll Phase 17.1 — Data Collection System Status Report

## 1. System Overview

The **Phase 17.1 Real-World Evaluation Data Collection System** is fully deployed in `backend/scripts/dataset/`. It provides automated ingestion, SHA-256 duplicate rejection, anonymous ID metadata logging (`P001`–`P030`), 15-condition taxonomy management, anti-spoofing presentation attack logging, and identity-disjoint trial pair generation.

---

## 2. Component Inventory

- [`collect_real_world_evaluation.py`](file:///c:/Users/taran/Documents/GitHub/AutoRoll/backend/scripts/dataset/collect_real_world_evaluation.py): Live ingestion & SHA-256 provenance engine.
- [`generate_evaluation_trial_pairs.py`](file:///c:/Users/taran/Documents/GitHub/AutoRoll/backend/scripts/dataset/generate_evaluation_trial_pairs.py): Trial pair generator (genuine & impostor pairs).
- [`validate_real_world_eval_dataset.py`](file:///c:/Users/taran/Documents/GitHub/AutoRoll/backend/scripts/dataset/validate_real_world_eval_dataset.py): Dataset integrity validator.
- [`test_phase17_data_collection.py`](file:///c:/Users/taran/Documents/GitHub/AutoRoll/backend/tests/test_phase17_data_collection.py): Pytest integrity test suite.

---

## 3. Data Collection Target Capacity

- **Anonymous IDs:** `P001`–`P030` (Expandable to 100+).
- **Target Enrollment Images:** 150–300 Frontal Samples (5–10 per participant).
- **Target Probe Images:** 450 Probe Samples (15 conditions x 30 participants).
- **Target Physical Anti-Spoofing Samples:** 150 Presentation Attacks (Printed Photo, Phone Replay, Tablet Replay, Video Replay, Bona Fide Live).
- **Total Dataset Capacity:** **~900 Real Physical Images** under `data/real_world_evaluation/`.

---

## 4. Benchmark Hold Constraint

> [!IMPORTANT]
> The final Phase 17 evaluation benchmark script will NOT be executed until physical image acquisition for `P001`–`P030` is complete. Synthetic score generation is strictly prohibited.
