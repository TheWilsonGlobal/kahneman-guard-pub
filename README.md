# KahnemanGuard (Public Benchmark & Training Backend)

**Master's Graduation Thesis:** Researching risk-aware selective cascading for efficient large language model input moderation.  
**Author:** Pham Ngoc Son (MSSV: 25210182) | **Supervisor:** M.Sc. Nguyen Huu Quyen (CITD, UIT - VNU-HCM)  
**Framework:** KahnemanGuard (RASC-IM)

This public repository contains the minimal components required by the **KahnemanGuard** Colab training pipeline:

## Repository Structure
```
kahneman-guard-pub/
├── data/
│   └── benchmarks/
│       ├── kg_eval_cal.jsonl               # 300 holdout prompts for Temperature Scaling & ECE calibration
│       ├── kg_eval_route.jsonl             # 400 prompts for dual-threshold routing optimization
│       ├── kg_eval_test.jsonl              # 800 frozen prompts for final blind acceptance
│       └── kg_eval_frozen_checksums.sha256 # SHA-256 integrity checksums (0% data leakage)
├── notebooks/
│   └── colab_L4/
│       └── train_l4.py                     # Self-contained training module for ModernBERT on L4 GPU
├── .gitignore
└── README.md
```
