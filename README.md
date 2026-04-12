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

model_eval.py systematically evaluates Chain-of-Thought prompting effectiveness using the GSM8K dataset by comparing four strategies—Direct zero-shot, CoT zero-shot, Direct few-shot, and CoT few-shot—through automated generation, answer extraction, and accuracy computation. You can set the range of evaluation in dataset by change the last two argument of evaluate_gsm8k(). The produced CSV results will contains per-strategy performance metrics with name "gsm8k_cot_results[xx-xx]" where [xx-xx] repersent the range in tested dataset.

After range setting, run the evaluation script:
Run the evaluation script to feed a dataset into the model:

```bash
python model_eval.py
```

3. Data interpretation

After the csv files generated, you can move them into the /data folder, and use the data_analyze.py to check how many data are tesed and the accuracy for each methods.
```bash
python data_analyze.py
```

