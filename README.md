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
 ```
---

## 6. How Finetuning is Done

- **Load the pretrained multimodal LLM** using Unsloth’s `FastVisionModel`.
- **Download [Radiology Mini](https://huggingface.co/datasets/unsloth/Radiology_mini)** and **[VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)** datasets directly from Hugging Face.
- **Convert each sample** into a unified conversational format combining text instructions/questions and images.
- **Merge and shuffle** the datasets to ensure balanced training.
- **Enable LoRA-based parameter-efficient finetuning** for both vision and language modules.
- **Train the model** using `SFTTrainer` along with a multimodal-aware data collator.
- **Log training step-wise metrics** to `.jsonl` for detailed tracking of loss and progress.
- **Visualize training loss** over training steps and export the plot as a PNG for reporting.
- **Save the final finetuned model and tokenizer** for future inference and sharing.

---

## 7. Evaluation

The finetuned model is rigorously evaluated on held-out test splits of **VQA-RAD** and **Radiology Mini** datasets. For each test sample, the model generates an answer or descriptive caption, which is then compared with the ground truth reference text. The evaluation uses three key metrics:

- **Exact Match (EM):** Checks whether the generated text exactly matches the reference after text normalization (lowercasing, punctuation removal, and whitespace cleanup).
- **BLEU Score:** Measures n-gram overlap to assess how closely the generated text aligns with the reference text in terms of word choice and phrasing.
- **ROUGE-L:** Computes the longest common subsequence between the generated and reference texts to evaluate content coverage and fluency.

All predictions, scores, and summary statistics are logged to output files for transparency, reproducibility, and future analysis.

---

## 8. Download Finetuned Model and Tokenizer

You can download the finetuned model and tokenizer from Google Drive for local inference or further research:

- [Download Finetuned Model and tokenizer](https://drive.google.com/your_finetuned_model_link)

After downloading, extract the files and update your `from_pretrained` paths in your scripts accordingly.

---

## Acknowledgements

This project makes use of the following open-source tools and datasets:

- [Unsloth](https://github.com/unslothai/unsloth) for efficient multimodal LLM training and inference.
- [Hugging Face](https://huggingface.co/) for dataset hosting and easy API access.
- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) and [Radiology Mini](https://huggingface.co/datasets/unsloth/Radiology_mini) for advancing multimodal medical AI research.
