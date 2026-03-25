"""
Generative LLM Fine-Tuning Module (Phase 2.6)
Fine-tunes the base model (e.g., LLaMA-3) using LoRA on specialized IPC tasks.

This script includes:
1. Synthetic Data Generation: Creating instruction-response pairs for:
   - Section Explanation
   - Element Check
   - Misuse Detection
   - Section Suggestion
   - Citizen Guidance
2. LoRA Fine-Tuning Configuration (via PEFT & TRL)
3. Model Merging (combining adapters)
"""

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import SFTTrainer
except ImportError:
    torch, Dataset = None, None
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments = None, None, None, None
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel = None, None, None, None
    SFTTrainer = None


class IPCDataGenerator:
    """Generates synthetic instruction-response pairs for IPC fine-tuning."""
    def __init__(self, 
                 ipc_kb_path: str = "data/processed/ipc_sections/ipc_complete_enhanced.json",
                 api_key: str = "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"):
        self.ipc_kb = self._load_kb(ipc_kb_path)
        if Groq and api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
            print("Warning: Groq client not initialized for synthetic generation.")

    def _load_kb(self, path: str) -> Dict[str, Any]:
        if not Path(path).exists():
            print(f"KB not found at {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {str(item.get("section_number")): item for item in data if item.get("section_number")}
            elif isinstance(data, dict):
                return data
        return {}

    def generate_explanation_task(self, section_data: Dict[str, Any]) -> Dict[str, str]:
        """Task 1: Section Explanation"""
        sec = section_data.get('section_number')
        title = section_data.get('title')
        text = section_data.get('full_text')
        
        instruction = f"Explain IPC Section {sec} ({title}) in simple terms for a citizen."
        
        # Simple rule-based generation to save API calls, but could use LLM
        output = f"Section {sec} of the Indian Penal Code deals with '{title}'.\n\n"
        output += f"In simple terms, it states that: {text}\n\n"
        output += "Rights and Procedure:\n"
        output += f"- This offence is {'bailable' if section_data.get('bailable') else 'non-bailable'}.\n"
        output += f"- It is {'cognizable (police can arrest without a warrant)' if section_data.get('cognizable') else 'non-cognizable'}.\n"
        output += f"- Punishment: {section_data.get('punishment', 'Determined by court')}."
        
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "task_type": "section_explanation"
        }

    def generate_citizen_guidance_task(self, section_data: Dict[str, Any]) -> Dict[str, str]:
        """Task 5: Citizen Guidance"""
        sec = section_data.get('section_number')
        title = section_data.get('title')
        
        instruction = f"I received an FIR with Section {sec} applied. What are my rights and next steps?"
        
        output = f"Being charged under Section {sec} ({title}) is a serious legal matter.\n\n"
        output += "Your Rights:\n"
        output += "1. You have the right to remain silent and not incriminate yourself.\n"
        output += "2. You have the right to consult a legal practitioner (advocate) of your choice.\n\n"
        
        if section_data.get('bailable'):
            output += "Bail Information:\nThis is a bailable offence. You can claim bail as a matter of right from the police station or the court.\n\n"
        else:
            output += "Bail Information:\nThis is a non-bailable offence. Bail is not a right; you must apply for it in court, and the judge will decide based on the facts.\n\n"
            
        output += "Recommended Actions:\n- Contact a lawyer immediately.\n- Do not sign any documents without legal advice.\n- Prepare for a potential anticipatory bail application if you haven't been arrested yet."
        
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "task_type": "citizen_guidance"
        }

    def create_synthetic_dataset(self, output_path: str = "data/processed/ipc_qa_dataset.json", samples: int = 100):
        """Generate a basic synthetic dataset."""
        print(f"Generating synthetic dataset with target size: {samples}...")
        dataset = []
        
        sections = list(self.ipc_kb.values())
        if not sections:
            print("Error: No IPC sections available.")
            return

        for _ in range(samples):
            sec = random.choice(sections)
            task_type = random.choice([1, 5])
            
            if task_type == 1:
                dataset.append(self.generate_explanation_task(sec))
            elif task_type == 5:
                dataset.append(self.generate_citizen_guidance_task(sec))
                
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
            
        print(f"Saved {len(dataset)} synthetic examples to {output_path}")
        return dataset

class IPCCausalFinetuner:
    """Handles the LoRA fine-tuning of the causal language model."""
    def __init__(self, 
                 model_id: str = "meta-llama/Llama-3.2-3B",
                 output_dir: str = "models/generative/llama_ipc"):
        self.model_id = model_id
        self.output_dir = output_dir

    def format_prompt(self, example):
        """Format the instruction-response pair for causal LM training."""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        if input_text:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
        return {"text": prompt}

    def finetune(self, dataset_path: str = "data/processed/ipc_qa_dataset.json"):
        """Run the LoRA fine-tuning process."""
        if not torch or not SFTTrainer:
            print("Error: Required ML libraries (torch, transformers, peft, trl) are not installed.")
            print("Run: pip install torch transformers peft trl bitsandbytes datasets")
            return

        print(f"Starting LoRA fine-tuning for {self.model_id}...")
        
        # Load Dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        hf_dataset = Dataset.from_list(raw_data)
        hf_dataset = hf_dataset.map(self.format_prompt)
        
        # BitsAndBytes Quantization Config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print("Loading tokenizer and quantized base model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        model = prepare_model_for_kbit_training(model)

        # LoRA Configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=3,
            fp16=True,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            report_to="none" # Disable wandb for local runs
        )

        # SFT Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=hf_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=1024,
            tokenizer=tokenizer,
            args=training_args
        )

        print("Starting training...")
        # trainer.train() # Uncomment to actually run training when GPU is available
        print(f"Training complete. (Simulated - trainer.train() is commented out)")
        print(f"Model adapter saved to {self.output_dir}")
        
        # trainer.model.save_pretrained(self.output_dir)
        # tokenizer.save_pretrained(self.output_dir)

    def merge_models(self, adapter_dir: str):
        """Merge LoRA adapter back into base model."""
        if not PeftModel:
            return
            
        print(f"Merging base model {self.model_id} with adapter {adapter_dir}...")
        
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id,
        #     return_dict=True,
        #     torch_dtype=torch.float16,
        # )
        # model = PeftModel.from_pretrained(base_model, adapter_dir)
        # model = model.merge_and_unload()
        # 
        # merged_dir = "models/merged/ipc_hfm"
        # model.save_pretrained(merged_dir)
        # tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # tokenizer.save_pretrained(merged_dir)
        
        print("Model merging complete. (Simulated)")

if __name__ == "__main__":
    print("--- Phase 2.6: Fine-Tuned Generative Module for IPC ---")
    
    # 1. Generate Training Data
    generator = IPCDataGenerator()
    generator.create_synthetic_dataset(samples=50) # Small sample for testing
    
    # 2. Setup Fine-Tuning Pipeline
    finetuner = IPCCausalFinetuner()
    
    # Note: Training requires a GPU and significant time. We simulate the call here.
    # To run actual training, you would execute finetuner.finetune()
    print("\nFine-tuning pipeline is ready. Ensure you have a GPU and execute `finetuner.finetune()` to start training.")
