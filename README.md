# 5520-GP18-project

This repository provides a simple framework for running **Chain-of-Thought (CoT) evaluations** using local model. It contains two main scripts:

- `model_set.py` – Handles model selection and downloading local model.
- `model_eval.py` – Feeds datasets into the chosen model for evaluation.

---

## Prerequisites

Before running the project, you need to uninstall and reinstall specific Hugging Face libraries to ensure compatibility:

```bash
# Uninstall old versions
pip uninstall -y transformers huggingface_hub accelerate bitsandbytes

# Install required versions
pip install -U "transformers>=4.51.0" "huggingface_hub>=0.34.0" accelerate bitsandbytes safetensors


```
Note: Other dependencies may be required depending on your dataset or code usage. Please install them as needed.

## Usage

1. Select and Download Models
In model_set.py, change the model name and cache dir for the model you want to download first.
Run the following command to select or download a local model:

```bash
python model_set.py
```
The model will be downloaded in the root directory of the repo

2. Evaluate with Dataset
In model_eval.py, change the local_path to the model in your local disk.
Run the evaluation script to feed a dataset into the model:

```bash
python model_eval.py
```
