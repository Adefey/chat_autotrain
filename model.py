import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    __version__,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import login


class Model:

    def __init__(self, huggingface_key: str, checkpoint: str = "google/gemma-3-4b-it", previous_dialog: str = ""):
        login(huggingface_key)
        self.model = Gemma3ForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.chat_history = previous_dialog

    def interact(self, new_message: str, keep_history:bool=True):
        if not keep_history:
            self.chat_history = ""

        new_message_template = f"<start_of_turn>user\n{new_message}<end_of_turn>\n<start_of_turn>model\n"

        self.chat_history += new_message_template
        
        model_inputs = self.tokenizer(self.chat_history, return_tensors="pt")

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
            generation = generation[0][input_len:]

        result = self.tokenizer.decode(generation, skip_special_tokens=True)

        self.chat_history += f"{result}<end_of_turn>\n"

        with open("chat_history", "w") as file:
            file.write(self.chat_history)

        return result
    
    def train_on_interaction(self):
        dataset = {"text": self.chat_history}

        with open("dataset.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(dataset, ensure_ascii=False) + "\n")
        dataset = load_dataset("json", data_files="dataset.jsonl")

        def preprocess(example):
            return self.tokenizer(example["text"], return_tensors="pt")

        tokenized_dataset = dataset.map(preprocess, batched=True)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            num_train_epochs=3,
            logging_steps=1,
            save_steps=1,
            output_dir="./results",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()

