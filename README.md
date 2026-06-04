# Component Ablation for Efficient Hybrid Language Model Architectures: Performance, Resilience, and Compression Implications

This repository contains the code, result tables, and figure-generation materials for the manuscript:

**Component Ablation for Efficient Hybrid Language Model Architectures: Performance, Resilience, and Compression Implications**

Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó — Universitat Politècnica de València

A preprint version is available on arXiv:

**arXiv:2603.22473**

Earlier versions of this work circulated under the title:

**Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures**

The repository supports reproducibility for the final journal-submission version, including processed result tables, figure-generation scripts, benchmark outputs, and the figures used in the manuscript.

## Key Findings

1. **Both component types contribute to performance**  
   Removing either attention or the alternative sequence-processing pathway degrades downstream benchmark performance in the tested hybrid language models.

2. **Likelihood is more sensitive to the linear/SSM pathway**  
   WikiText-2 perplexity increases more strongly when removing the linear-attention pathway in Qwen3.5-0.8B or the SSM pathway in Falcon-H1-0.5B than when removing attention in the corresponding model.

3. **Ablation impact is position-dependent**  
   The strongest single-component effects are concentrated in early or mid-network components, while late components generally produce smaller degradation.

4. **Hybrid and Transformer controls degrade differently under random removal**  
   Matched random controls show different normalized perplexity degradation patterns between the hybrid model and the same-family Transformer baseline.

## Models Studied

| Model | Architecture | Parameters | Layers | Type |
|-------|-------------|-----------|--------|------|
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | Sequential hybrid | 752M | 24 | 18 linear + 6 attention |
| [Falcon-H1-0.5B-Base](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) | Parallel hybrid | 521M | 36 | SSM + attention per block |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Pure Transformer | 490M | 24 | Control baseline |

## Repository Structure

```
├── notebooks/
│   └── hybrid_component_ablation.ipynb   # Complete experimental pipeline
├── results/                               # All CSV result files
│   ├── experiment1_summary.csv            # Exp 1: all ablation conditions
│   ├── experiment1_pivot.csv              # Exp 1: benchmark × condition pivot
│   ├── perplexity_under_ablation.csv      # WikiText-2 perplexity
│   ├── random_control_perplexity.csv      # Random control perplexity (hybrid)
│   ├── transformer_baseline_perplexity.csv # Transformer baseline perplexity
│   ├── experiment3_task_dependent_analysis.csv
│   ├── *_experiment2_metrics.csv          # Hidden-state metrics per model
│   ├── *_architecture_summary.csv         # Architecture details per model
│   └── ...
├── figures/                               # All generated figures
│   ├── paper_radar_group_ablation.png
│   ├── perplexity_under_ablation_bar.png
│   ├── paper_random_control_significance.png
│   ├── paper_component_dominance_by_layer.png
│   ├── paper_dual_layer_sweep_heatmap.png
│   ├── paper_exp1_exp2_correlation.png
│   ├── experiment3_score_drop_heatmap.png
│   └── ...
├── tables/                                # LaTeX tables (.tex)
│   ├── paper_table1_group_ablation_delta.tex
│   ├── perplexity_under_ablation.tex
│   └── ...
└── README.md
```

## Reproducing the Experiments

### Requirements

- Google Colab Pro (L4 GPU recommended, 16 GB VRAM)
- Python 3.10+
- PyTorch 2.1+ with CUDA
- For Falcon: `mamba-ssm` and `causal-conv1d` (optional; without them, a 1024-token cap is applied)

### Quick Start

1. Open `notebooks/hybrid_component_ablation.ipynb` in Google Colab
2. Run Section 0 (setup + config)
3. Run Section 1 (architecture discovery)
4. Run Sections 2–7 sequentially

All results are checkpointed to disk. Sessions can be interrupted and resumed at any point.

### Execution Time

| Section | Description | Time (L4 GPU) |
|---------|------------|---------------|
| 0–2 | Setup + architecture + ablation mechanism | ~15 min (includes model downloads) |
| 3 | Experiment 1 — Qwen ablations | ~3 hrs (Colab L4) |
| 3 | Experiment 1 — Falcon ablations | ~6 hrs (RunPod A100 SXM) |
| 3G | Perplexity under group ablation | ~15 min |
| 3G-extra | Random controls + Transformer baseline | ~15 min |
| 4 | Experiment 2 (hidden-state metrics) | ~30 min |
| 5–7 | Analysis + tables + figures | ~5 min |

**Note on Falcon compute**: Falcon-H1 requires `mamba-ssm` CUDA kernels, which do not compile on Colab's default CUDA 12.8 / PyTorch 2.5+ environment. Falcon experiments were run on a RunPod A100 SXM 80GB spot instance ($0.95/hr) with PyTorch 2.4.0 / CUDA 12.1. Total compute cost for all experiments: approximately $10–15.

## Ablation Methodology

- **Sequential skip (Qwen)**: Layer output replaced with input (identity within residual stream)
- **Parallel zeroing (Falcon)**: Targeted component output zeroed via forward hooks
- **Random controls**: Same number of randomly selected layers/components removed; random trials are reported with standard deviations where applicable
- **Transformer baseline**: Random layer removal on Qwen2.5-0.5B for cross-architecture comparison

## Data and Code Availability

Code, processed result tables, figure-generation scripts, and reproducibility materials are publicly available in this repository:

https://github.com/hecboar/hybrid-component-ablation

The repository includes the evaluation outputs used to generate the paper tables and figures. Pretrained model weights and benchmark datasets are not redistributed; they should be accessed from their original providers under their respective licenses. Large intermediate artifacts and additional execution logs are available from the corresponding author upon reasonable request.

## Citation

```bibtex
@misc{borobia2026componentablation,
  title={Component Ablation for Efficient Hybrid Language Model Architectures: Performance, Resilience, and Compression Implications},
  author={Borobia, Hector and Segu{\'i}-Mas, Elies and Tormo-Carb{\'o}, Guillermina},
  year={2026},
  eprint={2603.22473},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.22473}
}
```

## License

MIT
