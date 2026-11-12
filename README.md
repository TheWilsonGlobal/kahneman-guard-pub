# KahnemanGuard (Public Benchmark & Colab Training Artifacts)

This public repository contains the minimal components required to run the **System 1 ModernBERT-large (395M)** training and calibration benchmarks on **Google Colab NVIDIA L4 GPU**:

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
│       ├── kahneman-guard-colab-l4.ipynb   # Official Google Colab L4 training & quantization notebook
│       ├── train_l4.py                     # Self-contained training module for ModernBERT with RLCD
│       └── README.md                       # Colab instructions
├── .gitignore
└── README.md
```

## Quick Start on Google Colab
Open `notebooks/colab_L4/kahneman-guard-colab-l4.ipynb` on Google Colab with an **NVIDIA L4 GPU** runtime:
1. Validates Ada Lovelace GPU hardware and Bfloat16 support.
2. Clones this repository into `/content/kahneman-guard`.
3. Downloads and tokenizes 25,000 alignment prompts from Hugging Face (`ToxicChat` and `BeaverTails`).
4. Trains both **Baseline Cross-Entropy** and **RLCD Calibrated ($T=1.1320$)** models in ~40 minutes.
5. Computes Expected Calibration Error (ECE - 10 bins) and Brier Score on `kg_eval_cal.jsonl`.
6. Quantizes model to ONNX Dynamic INT8 (379.6 MB) and benchmarks CPU inference latency.
