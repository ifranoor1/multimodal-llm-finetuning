# ========================================
# multimodal LLM finetuning script
# ========================================
# description: fine-tune llama-3.2 11B vision-instruct with unsloth on radiology datasets
# ========================================

# imports
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer, TrainerCallback
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from random import shuffle
import torch
import json
import matplotlib.pyplot as plt

# load base model & tokenizer
basemodel, tokenizer = FastVisionModel.from_pretrained(
    "/home3/ial/lamma3.2_visionmodel/Llama-3.2-11B-Vision-Instruct", # Replace it with your base model path
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# prepare PEFT finetuned model
finetuned_model = FastVisionModel.get_peft_model(
    basemodel,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers
    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# load Datasets
train_raddata = load_dataset("unsloth/Radiology_mini", split="train")
train_vqarad = load_dataset("flaviagiammarino/vqa-rad", split="train")

instruction = "You are an expert radiographer. Describe accurately what you see in this image."

# define conversion functions for both datasets
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
train_raddata_converted = [convert_radiology(sample) for sample in train_raddata]
train_vqarad_converted = [convert_vqa_rad(sample) for sample in train_vqarad]

# merged datasets
from random import shuffle
train_merged = train_raddata_converted + train_vqarad_converted
shuffle(train_merged)

# sanity check before finetuning: run inference on 1 sample by using base model 
FastVisionModel.for_inference(basemodel) # Enable for inference!

image = train_raddata[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")


text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = basemodel.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# define custom logging callback
class LoggingCallback(TrainerCallback):
    def __init__(self, log_file="step_logs.jsonl"):
        self.log_file = log_file
        open(self.log_file, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a") as f:
                json.dump({"step": state.global_step, **logs}, f)
                f.write("\n")



# prepare trainer & config
FastVisionModel.for_training(finetuned_model) # Enable for training!

trainer = SFTTrainer(
    model = finetuned_model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(finetuned_model, tokenizer), # Must use!
    train_dataset = train_merged,
    callbacks=[LoggingCallback()],
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 2826, #6epochs
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 1e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "vision-instruct-final",
        report_to = "none",     # For Weights and Biases
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
        save_steps=500,  # Save model every 500 steps
        save_total_limit=6
    ),
)


#start training & show GPU stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train(resume_from_checkpoint=False)

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# plot training loss from file
steps, losses = [], []
with open("step_logs.jsonl", "r") as f: #save the loss per steps in to jsonl file
    for line in f:
        log = json.loads(line)
        if "step" in log and "loss" in log:
            if log['step'] % 50 == 0:
                steps.append(log["step"])
                losses.append(log["loss"])

plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o', linestyle='-', color='blue')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_plot.png")
plt.show()

#save the model
finetuned_model.save_pretrained("finetune-model") # Local saving
tokenizer.save_pretrained("finetune-model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving   
  
