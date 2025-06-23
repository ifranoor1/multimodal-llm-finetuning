# Multimodal Large Language Model Finetuning and Evaluation

## 1. Overview

This repository demonstrates a complete workflow for finetuning and evaluating a multimodal large language model (LLM) using Unsloth’s `FastVisionModel`. The project focuses on radiology images paired with textual instructions or clinical questions. The goal is to enable the model to describe radiology images accurately and answer visual medical questions reliably.

---

## 2. Features

- Load and configure a pretrained multimodal vision-language model.
- Download, process, and merge multiple radiology vision-language datasets.
- Finetune both vision encoder and language decoder using parameter-efficient LoRA.
- Log training metrics at each step and plot loss curves.
- Evaluate the finetuned model using standard generation and QA metrics.
- Log detailed evaluation results for transparency and reproducibility.
- Save and reuse the finetuned model and tokenizer.

---

## 3. Technologies Used

- **Python:** Core scripting language for data handling and model orchestration.
- **PyTorch:** Deep learning framework for tensor operations and GPU-accelerated training.
- **Transformers (Hugging Face):** Provides pretrained large language models and tokenizers.
- **Unsloth (FastVisionModel):** Efficient, flexible framework for multimodal model training and inference.
- **Hugging Face Datasets:** Simplifies loading and managing open-access datasets.
- **NLTK:** Used for BLEU score computation to evaluate text generation accuracy.
- **ROUGE-Score:** Computes ROUGE-L for summarization and sequence overlap quality.
- **Matplotlib:** For visualizing training loss trends.
- **TQDM:** Displays progress bars during dataset iteration and evaluation.

---

## 4. Datasets

The following publicly available datasets were used for training and evaluation:

- **[VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad):** A benchmark dataset for visual question answering in radiology. It contains radiology images paired with clinical questions and corresponding answers, making it suitable for training models to answer medical image-related queries.

- **[Radiology Mini](https://huggingface.co/datasets/unsloth/Radiology_mini):** A compact dataset for radiology image captioning. Each image is paired with an expert caption describing the observations, which helps the model learn to generate accurate medical image reports.

---

## 5. Setup Environment

Run the following commands step by step to prepare your environment:

```bash
# Clone this repository
git clone https://github.com/your_username/your_repo.git
cd your_repo

# (Optional) Create a Python virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# venv\Scripts\activate   # For Windows

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install other required packages
pip install unsloth transformers datasets nltk rouge-score matplotlib tqdm

# Download NLTK tokenizer data
python -c "import nltk; nltk.download('punkt')"

