
# RCA: Resonant Context Anchoring

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

This repository contains the official implementation of **Resonant Context Anchoring (RCA)**, a novel inference-time intervention method for Large Language Models (LLMs).

RCA addresses the problem of **"Contextual Disregard"**—where LLMs prioritize internal parametric memory over conflicting external evidence. Instead of suppressing internal knowledge (which often degrades fluency), RCA dynamically amplifies the signal of contextual evidence within the residual stream using a **non-linear resonance mechanism**.

---

## 🚀 Key Features

*   **Training-Free**: Plug-and-play module that requires no fine-tuning or weight updates.
*   **Low Latency**: Single-pass inference. Unlike Contrastive Decoding (CD) or DoLa, RCA introduces negligible computational overhead ($O(Ld)$ vs $O(L^2d)$).
*   **SOTA Performance**: Outperforms baselines on faithfulness (XSum) and knowledge conflict (NQ-Swap) tasks across Llama-3-8B and 70B models.
*   **Pareto Improvement**: Improves factual consistency (FactKB) without sacrificing generation quality (ROUGE).
*   **Safe & Robust**: Automatically remains dormant in closed-book tasks (TruthfulQA), preserving general model capabilities.

## 🧠 Methodology: Signal Dynamics

RCA operates on the hypothesis that hallucinations stem from a low **Signal-to-Noise Ratio (SNR)** in the residual stream, where context signals are submerged by parametric noise.

1.  **Resonance Detection**: Uses raw attention scores to detect semantic alignment between the query and context tokens.
2.  **Non-linear Rectification**: Applies a **Softplus** function to filter noise and extract resonance intensity.
3.  **In-Situ Value Modulation**: Dynamically amplifies the norms of Value vectors ($\tilde{v}_i = \lambda_{t,i} \cdot v_i$), decoupling signal strength from routing logic.

![RCA Overview](docs/assets/rca_architecture.png)
*(See paper for detailed mathematical formulation.)*

---

## 📊 Main Results

We evaluate RCA on **Llama-3-8B-Instruct** and **Llama-3-70B-Instruct** across diverse benchmarks.

### Faithfulness & Knowledge Conflict (SOTA)

| Task | Metric | Baseline (8B) | **RCA (8B)** | Baseline (70B) | **RCA (70B)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XSum** | FactKB | 47.61% | **50.14%** (+2.5%) | 61.32% | **61.72%** |
| | ROUGE-L | 19.90 | **20.03** | 22.41 | 22.21 |
| **NQ-Swap** | Exact Match | 60.62% | **64.54%** (+3.9%) | 76.11% | **77.46%** |
| **MemoTrap** | Micro Acc | 64.40% | **65.77%** | 66.52% | **77.35%** |
| **IFEval** | Prompt Acc | 70.24% | **72.09%** | 77.45% | **78.74%** |

> **Highlight**: On NQ-Swap (a strong conflict task), RCA achieves a **+3.92%** improvement in Exact Match, effectively overriding parametric hallucinations.

### Safety Check (Harmlessness)

RCA maintains performance parity on closed-book tasks, proving it does not degrade general knowledge.

| Task | Metric | Baseline (8B) | RCA (8B) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TruthfulQA** | MC1 | 38.92% | **38.92%** | ✅ Safe |
| **TriviaQA** | EM | 56.58% | 56.52% | ✅ Safe |

---

## 🛠️ Quick Start

### 1. Installation

```bash
conda env create -f environment.yaml
conda activate decore  # Environment name inherited from base
pip install -r requirements.txt
```

### 2. Prepare Models

Download Llama-3 models to your local directory (e.g., `/data/my_models/`).
Ensure that `src/utils/modelling_llama.py` contains the RCA implementation (specifically the modified `LlamaAttention` class).

### 3. Run Evaluation

RCA is implemented as a decoder strategy named `aca`.

#### Run XSum (Summarization)
```bash
# Llama-3-8B (Recommended alpha=0.04)
python scripts/main.py \
    experiment=xsum/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    decoder.configs.alpha=0.04 \
```

#### Run NQ-Swap (Knowledge Conflict)
```bash
# Llama-3-8B (Recommended alpha=0.04~0.08)
python scripts/main.py \
    experiment=nq_swap/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    decoder.configs.alpha=0.04
```

#### Run on Llama-3-70B
For 70B models, we recommend slightly higher sensitivity ($\alpha=0.05$). Ensure you have sufficient VRAM (4x A100 40GB or 2x A100 80GB).

```bash
# Llama-3-70B (Recommended alpha=0.05)
python scripts/main.py \
    experiment=xsum/baseline/llama3_70b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data_loader.batch_size=1 \
    decoder.configs.alpha=0.05 \
    model.configs.model_name_or_path="/path/to/70B/model"
```

## ⚙️ Hyperparameters

*   **`alpha` ($\alpha$)**: **Resonance Sensitivity**. Controls the amplification strength.
    *   Default: `0.04` (8B), `0.05` (70B).
## 📂 Repository Structure

```
.
├── configs/             # Hydra configs
│   ├── decoder/aca.yaml # RCA configuration
├── src/
│   ├── models/
│   │   ├── aca.py       # High-level RCA logic
│   └── utils/
│       └── modelling_llama.py # Low-level implementation (Attention & Decoder)
├── scripts/             # Evaluation scripts
└── ...
```

---
*This project is built upon the framework of [DeCoRe](https://github.com/aryopg/DeCoRe).*
