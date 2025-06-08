# Vim-Motion LLM Toolkit

A small end-to-end pipeline that **generates synthetic training data, fine-tunes an open-source LLM with QLoRA, and benchmarks the result** — all focused on writing the shortest possible *vim* motion needed to accomplish a given editing task.

| File | Purpose |
|------|---------|
| **`data_prep/generating_prep.ipynb`** | Builds the training corpus. It reads `base.txt` (all vim motions) and uses the OpenAI API to create ⟨**natural-language task**, **vim-motion**⟩ pairs, saving them to `data_raw.csv` / `final.csv`. |
| **`vim-tutor-fintuning.ipynb`** | Fine-tunes *Qwen-2.5-0.5B-instruct* and *Qwen-2.5-7B-instruct*  with 4-bit LoRA on the generated data. Fully reproducible on Kaggle or a single 16 GB GPU. [Actiual notebook on kaggle](https://www.kaggle.com/code/timofeymazurenko/vim-tutor-fintuning)|
| **`benchmark.ipynb`** | Compares fintuned LLMs with ChatGPT and deepseek. Produces per-model Accuracy & Avg. Levenshtein distance. [Actiual notebook on kaggle](https://www.kaggle.com/code/timofeymazurenko/benchmark)|
| **`analysis.ipynb`** | Tiny notebook that loads `Model_Performance_Summary.csv` and surfaces “top-N” tables for quick inspection. |

## Vim Motion CLI

`vim_motion_cli.py` provides a small terminal helper that queries ChatGPT 4.1 using the extended prompt in `extended_prompt.txt`.

### Example

```bash
OPENAI_API_KEY=your-key python vim_motion_cli.py "delete to end of line"
```

Run without arguments to enter an interactive shell:

```bash
OPENAI_API_KEY=your-key python vim_motion_cli.py
Describe motion> delete to end of line
Motion: d$
```
