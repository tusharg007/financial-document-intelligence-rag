"""
LoRA Fine-Tuning Script for Financial Document QA.

Runnable on Google Colab (free tier with TinyLlama, T4 GPU with Mistral-7B).

Usage (Colab):
    !pip install transformers peft datasets accelerate bitsandbytes trl
    
    from src.finetuning.lora_finetune import run_finetuning
    run_finetuning(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_dir="./lora_findoc")
"""

import json
from pathlib import Path
from typing import Optional


# ============================================================================
# Training Data Generation
# ============================================================================

def build_training_dataset():
    """
    Build instruction-tuning dataset from financial QA pairs.
    
    Format: {"instruction": ..., "input": ..., "output": ...}
    These are realistic SEC filing QA pairs for domain adaptation.
    """
    examples = [
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: Tesla reported total revenues of $96.77 billion for fiscal year 2023, representing a 19% increase from $81.46 billion in 2022. Automotive revenues were $82.42 billion.\nQuestion: What was Tesla's total revenue in 2023?",
            "output": "Tesla reported total revenues of $96.77 billion for fiscal year 2023, a 19% year-over-year increase from $81.46 billion in 2022. Automotive revenues specifically accounted for $82.42 billion of the total."
        },
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: Apple Inc. reported net sales of $383.3 billion for fiscal year 2023. iPhone revenue was $200.6 billion, Services revenue reached $85.2 billion. Gross margin was 44.1%.\nQuestion: What were Apple's key revenue figures in 2023?",
            "output": "Apple reported net sales of $383.3 billion in FY2023. iPhone remained the largest segment at $200.6 billion, while Services revenue reached $85.2 billion. The company maintained a gross margin of 44.1%."
        },
        {
            "instruction": "Based on the SEC filing excerpt, analyze the risk factors.",
            "input": "Context: The company faces significant supply chain risks including semiconductor shortages, single-source suppliers for critical components, and geopolitical tensions affecting manufacturing in Asia. Lithium and cobalt prices remain volatile.\nQuestion: What are the key supply chain risks mentioned?",
            "output": "The filing identifies several critical supply chain risks: (1) semiconductor shortages impacting production capacity, (2) dependency on single-source suppliers for critical components creating concentration risk, (3) geopolitical tensions affecting Asian manufacturing operations, and (4) commodity price volatility, particularly for lithium and cobalt used in battery production."
        },
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: Ford Motor Company reported a net loss of $1.98 billion on its EV segment (Model e) in 2023, while the overall company posted net income of $4.35 billion. The Pro segment generated $7.23 billion in EBIT.\nQuestion: How did Ford's EV business perform in 2023?",
            "output": "Ford's EV segment (Model e) reported a net loss of $1.98 billion in 2023, highlighting the significant investment phase of their electrification strategy. However, the overall company remained profitable with $4.35 billion in net income, supported by the Pro (commercial) segment which generated $7.23 billion in EBIT."
        },
        {
            "instruction": "Based on the SEC filing excerpt, compare the financial metrics.",
            "input": "Context: NVIDIA reported data center revenue of $47.5 billion in FY2024, up 217% year-over-year. Gross margin expanded to 72.7%. Total revenue reached $60.9 billion. Operating income was $32.97 billion.\nQuestion: Summarize NVIDIA's financial performance.",
            "output": "NVIDIA achieved exceptional financial results in FY2024: total revenue of $60.9 billion with data center revenue of $47.5 billion (up 217% YoY), driven by AI chip demand. Gross margin expanded to a robust 72.7%, and operating income reached $32.97 billion, reflecting strong operating leverage."
        },
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: Microsoft's Intelligent Cloud segment revenue was $96.8 billion in FY2024, with Azure growing 29%. Total revenue was $245.1 billion. AI services contributed an estimated $10+ billion annualized run rate.\nQuestion: How is Microsoft's cloud business performing?",
            "output": "Microsoft's Intelligent Cloud segment generated $96.8 billion in FY2024 revenue, with Azure growing at 29%. AI services have reached an estimated $10+ billion annualized run rate. Total company revenue was $245.1 billion, with cloud and AI as primary growth drivers."
        },
        {
            "instruction": "Based on the SEC filing excerpt, analyze the risk factors.",
            "input": "Context: The company's operations are subject to various cybersecurity risks including ransomware attacks, data breaches, and unauthorized access to customer information. In 2023, the company invested $2.1 billion in cybersecurity measures.\nQuestion: What cybersecurity risks does the company disclose?",
            "output": "The company discloses significant cybersecurity risks including (1) ransomware attacks targeting operational systems, (2) potential data breaches exposing customer information, and (3) unauthorized access threats. To mitigate these risks, the company invested $2.1 billion in cybersecurity measures during 2023."
        },
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: JPMorgan Chase reported net income of $49.6 billion in 2023, with net revenue of $162.4 billion. The consumer banking segment generated $17.8 billion in net income. Return on equity was 17%.\nQuestion: What was JPMorgan's profitability in 2023?",
            "output": "JPMorgan Chase achieved record profitability in 2023 with net income of $49.6 billion on net revenue of $162.4 billion. Consumer banking contributed $17.8 billion in net income. The company's return on equity stood at a strong 17%, reflecting efficient capital allocation."
        },
        {
            "instruction": "Based on the SEC filing excerpt, discuss the trends.",
            "input": "Context: Amazon's AWS segment revenue grew from $62.2 billion in 2022 to $90.8 billion in 2023, with operating income increasing from $22.8 billion to $36.8 billion. AWS operating margin expanded to 40.5%.\nQuestion: What are the trends in Amazon's cloud business?",
            "output": "Amazon's AWS shows strong growth trends: revenue grew 46% from $62.2 billion (2022) to $90.8 billion (2023). More notably, operating income grew 61% to $36.8 billion, with operating margins expanding to 40.5%, indicating improving unit economics and scale advantages in cloud infrastructure."
        },
        {
            "instruction": "Based on the SEC filing excerpt, analyze the risk factors.",
            "input": "Context: Climate-related risks include physical risks from extreme weather events impacting facilities and supply chains, transition risks from evolving regulations including carbon pricing, and reputational risks from stakeholder expectations around ESG commitments.\nQuestion: What climate and ESG risks are disclosed?",
            "output": "The filing discloses three categories of climate-related risks: (1) Physical risks from extreme weather events that could damage facilities and disrupt supply chains, (2) Transition risks from evolving regulations including potential carbon pricing mechanisms, and (3) Reputational risks from increasing stakeholder expectations around ESG commitments and climate action."
        },
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: Alphabet reported total revenues of $307.4 billion in 2023, with Google Search and other advertising revenue at $175.0 billion. Google Cloud revenue was $33.1 billion, achieving its first full-year operating profit of $1.7 billion.\nQuestion: How is Alphabet's cloud segment performing?",
            "output": "Alphabet's Google Cloud reached $33.1 billion in 2023 revenue and achieved a milestone by posting its first full-year operating profit of $1.7 billion, signaling the segment's path to sustainable profitability. This was within Alphabet's overall revenue of $307.4 billion."
        },
        {
            "instruction": "Based on the SEC filing excerpt, answer the financial question.",
            "input": "Context: The company's R&D expenses increased by 23% to $14.2 billion in 2023, driven by AI and machine learning investments. Capital expenditure was $32.3 billion, primarily for data center expansion and GPU procurement.\nQuestion: What are the company's key investment areas?",
            "output": "The company is heavily investing in AI infrastructure: R&D expenses rose 23% to $14.2 billion, focused on AI and machine learning capabilities. Capital expenditure reached $32.3 billion, primarily directed toward data center expansion and GPU procurement to support growing AI workloads."
        },
    ]
    
    return examples


def format_for_training(examples: list, tokenizer) -> list:
    """Format examples into the chat template expected by the model."""
    formatted = []
    for ex in examples:
        # Use standard instruction format
        text = f"""### Instruction:
{ex['instruction']}

### Input:
{ex['input']}

### Response:
{ex['output']}"""
        formatted.append({"text": text})
    return formatted


# ============================================================================
# Fine-Tuning
# ============================================================================

def run_finetuning(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir: str = "./lora_findoc",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 512,
    use_4bit: bool = True,
):
    """
    Run LoRA fine-tuning on a small LLM for financial QA.
    
    Args:
        model_name: HuggingFace model ID. Options:
            - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (Colab free tier)
            - "mistralai/Mistral-7B-Instruct-v0.2" (needs T4/A100)
        output_dir: Where to save the LoRA adapter
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        lora_r: LoRA rank (lower = fewer params, higher = more capacity)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout for LoRA layers
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_length: Maximum sequence length
        use_4bit: Whether to use 4-bit quantization (QLoRA)
    
    Returns:
        Path to saved LoRA adapter
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer
    from datasets import Dataset
    
    print(f"🚀 Starting LoRA fine-tuning")
    print(f"   Model: {model_name}")
    print(f"   Output: {output_dir}")
    print(f"   4-bit quantization: {use_4bit}")
    
    # --- 1. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # --- 2. Quantization Config ---
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # --- 3. Load Base Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # --- 4. LoRA Configuration ---
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable, total = model.get_nb_trainable_parameters()
    print(f"   Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    # --- 5. Prepare Dataset ---
    raw_examples = build_training_dataset()
    formatted = format_for_training(raw_examples, tokenizer)
    dataset = Dataset.from_list(formatted)
    
    print(f"   Training examples: {len(dataset)}")
    
    # --- 6. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
        seed=42,
    )
    
    # --- 7. Train ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
    )
    
    print("🏋️ Training started...")
    trainer.train()
    
    # --- 8. Save LoRA Adapter ---
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ LoRA adapter saved to: {output_dir}")
    print(f"   Upload to HuggingFace Hub with:")
    print(f"   model.push_to_hub('your-username/findoc-lora')")
    
    return output_dir
