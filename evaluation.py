# imports
from unsloth import FastVisionModel
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
import nltk
import re
nltk.download('punkt')

#load finetune model and tokenizer
finetuned_model, tokenizer = FastVisionModel.from_pretrained(
    "/home3/ial/lamma3.2_visionmodel/finetune_model", #replace the path with your finetuned model path
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# load test Datasets
test_raddata = load_dataset("unsloth/Radiology_mini", split="test")
test_vqarad = load_dataset("flaviagiammarino/vqa-rad", split="test")

instruction = "You are an expert radiographer. Describe accurately what you see in this image."

# define conversion functions for both test datasets
def convert_radiology(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }
pass

def convert_vqa_rad(sample):
    question = sample["question"]
    answer = sample["answer"]
    image = sample["image"]  # assume this is PIL.Image or compatible

    conversation = [
        { "role": "user",
          "content": [
              {"type": "text", "text": question},
              {"type": "image", "image": image}
          ]
        },
        { "role": "assistant",
          "content": [
              {"type": "text", "text": answer}
          ]
        }
    ]
    return { "messages": conversation }

# convert datasets
test_raddata_converted = [convert_radiology(sample) for sample in test_raddata]
test_vqarad_converted = [convert_vqa_rad(sample) for sample in test_vqarad]

# evaluate model on vqarad
def normalize_answer(s):
    """Lowercase, remove punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return normalize_answer(prediction) == normalize_answer(truth)

def evaluate_model_on_merged(model, tokenizer, dataset, log_file="evaluation_output_vqarad.txt"):
    FastVisionModel.for_inference(model)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_scores = []
    rouge_l_scores = []
    predictions = []
    references = []

    with open(log_file, "w") as f:
        for idx, sample in enumerate(tqdm(dataset)):
            messages = sample["messages"]
            image = None
            for content in messages[0]["content"]:
                if content["type"] == "image":
                    image = content["image"]
                elif content["type"] == "text":
                    input_text = content["text"]

            target_text = messages[1]["content"][0]["text"]

            prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True)
            inputs = tokenizer(image, prompt, return_tensors="pt").to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            bleu_score = sentence_bleu([target_text.split()], pred_text.split())
            rouge_score = scorer.score(target_text, pred_text)["rougeL"].fmeasure

            predictions.append(pred_text)
            references.append(target_text)

            bleu_scores.append(bleu_score)
            rouge_l_scores.append(rouge_score)

            f.write(f"Sample {idx+1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Prediction: {pred_text}\n")
            f.write(f"Ground Truth: {target_text}\n")
            f.write(f"Exact Match: {'Yes' if exact_match(pred_text, target_text) else 'No'}\n")
            f.write(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}\n")
            f.write("-" * 50 + "\n")

        total = len(dataset)
        correct = sum(exact_match(p, r) for p, r in zip(predictions, references))
        em_score = (correct / total) * 100
        avg_bleu = (sum(bleu_scores) / total) * 100
        avg_rouge = (sum(rouge_l_scores) / total) * 100

        f.write(f"\nEvaluation Summary on {total} samples:\n")
        f.write(f"Exact Match: {em_score:.2f}%\n")
        f.write(f"BLEU Score: {avg_bleu:.2f}%\n")
        f.write(f"ROUGE-L Score: {avg_rouge:.2f}%\n")

    print(f"\nEvaluation results saved to: {log_file}")
    print(f"Exact Match: {em_score:.2f}%")
    print(f"BLEU Score: {avg_bleu:.2f}%")
    print(f"ROUGE-L Score: {avg_rouge:.2f}%")

    return {"Exact Match": em_score, "BLEU": avg_bleu, "ROUGE-L": avg_rouge}

# run evaluation and log to file
metrics = evaluate_model_on_merged(finetuned_model, tokenizer, test_vqarad_converted, log_file="evaluation_output_vqarad.txt")



# evaluation on radiologymini
def normalize_answer(s):
    """Lowercase, remove punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return normalize_answer(prediction) == normalize_answer(truth)

def evaluate_model_on_merged(model, tokenizer, dataset, log_file="evaluation_output_radmini.txt"):
    FastVisionModel.for_inference(model)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_scores = []
    rouge_l_scores = []
    predictions = []
    references = []

    with open(log_file, "w") as f:
        for idx, sample in enumerate(tqdm(dataset)):
            messages = sample["messages"]
            image = None
            for content in messages[0]["content"]:
                if content["type"] == "image":
                    image = content["image"]
                elif content["type"] == "text":
                    input_text = content["text"]

            target_text = messages[1]["content"][0]["text"]

            prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True)
            inputs = tokenizer(image, prompt, return_tensors="pt").to("cuda")

            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            bleu_score = sentence_bleu([target_text.split()], pred_text.split())
            rouge_score = scorer.score(target_text, pred_text)["rougeL"].fmeasure

            predictions.append(pred_text)
            references.append(target_text)

            bleu_scores.append(bleu_score)
            rouge_l_scores.append(rouge_score)

            f.write(f"Sample {idx+1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Prediction: {pred_text}\n")
            f.write(f"Ground Truth: {target_text}\n")
            f.write(f"Exact Match: {'Yes' if exact_match(pred_text, target_text) else 'No'}\n")
            f.write(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}\n")
            f.write("-" * 50 + "\n")

        total = len(dataset)
        correct = sum(exact_match(p, r) for p, r in zip(predictions, references))
        em_score = (correct / total) * 100
        avg_bleu = (sum(bleu_scores) / total) * 100
        avg_rouge = (sum(rouge_l_scores) / total) * 100

        f.write(f"\nEvaluation Summary on {total} samples:\n")
        f.write(f"Exact Match: {em_score:.2f}%\n")
        f.write(f"BLEU Score: {avg_bleu:.2f}%\n")
        f.write(f"ROUGE-L Score: {avg_rouge:.2f}%\n")

    print(f"\nEvaluation results saved to: {log_file}")
    print(f"Exact Match: {em_score:.2f}%")
    print(f"BLEU Score: {avg_bleu:.2f}%")
    print(f"ROUGE-L Score: {avg_rouge:.2f}%")

    return {"Exact Match": em_score, "BLEU": avg_bleu, "ROUGE-L": avg_rouge}

# run evaluation and log to file
metrics = evaluate_model_on_merged(finetuned_model, tokenizer, test_raddata_converted, log_file="evaluation_output_radmini.txt")







