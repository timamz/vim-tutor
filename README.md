# Vim-Motion LLM Toolkit

A small end-to-end pipeline that **generates synthetic training data, fine-tunes an open-source LLM with QLoRA, and benchmarks the result** — all focused on writing the shortest possible *vim* motion needed to accomplish a given editing task.

| File | Purpose |
|------|---------|
| **`generating.py`** | Builds the training corpus. It reads `base.txt` (all vim motions) and uses the OpenAI API to create ⟨**natural-language task**, **vim-motion**⟩ pairs, saving them to `data_raw.csv` / `final.csv`. |
| **`finetuning.py`** | Fine-tunes *Qwen-1.5-1B* with 4-bit LoRA on the generated data. Fully reproducible on Kaggle or a single 16 GB GPU. |
| **`benchmark.py`** | Evaluates any number of local or HF-hosted models on a held-out test split. Produces per-model Accuracy & Avg. Levenshtein distance plus a CSV summary. |
| **`analysis.py`** | Tiny notebook that loads `Model_Performance_Summary.csv` and surfaces “top-N” tables for quick inspection. |
